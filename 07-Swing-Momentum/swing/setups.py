"""Qullamaggie / Stockbee swing setups on a daily US equity panel.

Sources for the rules (see README for links): Kristjan Kullamägi's own
writeups and interviews, and Pradeep Bonde's Stockbee methodology - Bonde
mentored Kullamägi, and the two systems share a spine:

    a big prior move  ->  a quiet, contracting base  ->  range expansion

What separates this from the breakout study in ``03-Alpha-Research`` (which
found nothing) is precisely the parts that were missing there:

* a **prior momentum leg** requirement (+30-100% in 1-3 months);
* **volatility contraction** - the base must be tightening, not merely narrow;
* an **ADR-constrained stop** - skip the trade if the day's low is more than
  one average daily range away, which is Kullamägi's hardest filter;
* **partial profit + moving-average trailing** rather than a fixed ATR target;
* a **breadth regime** layer (Bonde's Market Monitor) computed from the whole
  panel.

Execution model, learned the hard way in the earlier studies: a signal is
computed on the close of day t and **entered at the open of day t+1**. Nothing
is ever decided and priced inside the same bar. Stops are intraday orders
(gaps through them fill at the open); moving-average exits are market-on-close.
"""
from __future__ import annotations

import numpy as np
from numba import njit

EXIT_STOP, EXIT_TRAIL, EXIT_TIME, EXIT_EOD = 1, 2, 3, 4


# ---------------------------------------------------------------------------
# trade simulation
# ---------------------------------------------------------------------------
@njit(cache=True)
def simulate(sig_rows, block_end, open_, high, low, close, ma_trail, adr_frac,
             stop_adr_mult, partial_days, partial_frac, target_r,
             max_hold, breakeven_after_partial):
    """One trade per signal; returns per-trade arrays.

    ``adr_frac`` is the average daily range as a fraction of price.  The
    initial stop is the signal bar's low; if that is further than
    ``stop_adr_mult`` ADRs below the entry the trade is skipped, which is the
    rule Kullamägi describes as the one he will not bend.
    """
    n = len(sig_rows)
    t_sig = np.empty(n, np.int64)
    t_entry = np.empty(n, np.int64)
    t_exit = np.empty(n, np.int64)
    t_epx = np.empty(n, np.float64)
    t_risk = np.empty(n, np.float64)
    t_pnl = np.empty(n, np.float64)       # per share, net of nothing
    t_reason = np.empty(n, np.int8)
    t_bars = np.empty(n, np.int32)
    t_turn = np.empty(n, np.float64)      # gross traded value per share held
    k_out = 0

    last_exit = -1
    last_block = -1

    for q in range(n):
        i = sig_rows[q]
        b = block_end[q]
        if b != last_block:
            last_block = b
            last_exit = -1
        if i <= last_exit:               # already in a position in this name
            continue
        j = i + 1                        # entry bar
        if j >= b:
            continue

        entry = open_[j]
        stop = low[i]
        risk = entry - stop
        if risk <= 0.0 or entry <= 0.0:
            continue
        if risk > stop_adr_mult * adr_frac[i] * entry:
            continue                     # stop too wide relative to ADR

        pos = 1.0
        realized = 0.0
        turn = 1.0                       # bought one unit
        exit_k = -1
        reason = EXIT_TIME
        took_partial = False

        kmax = j + max_hold
        if kmax > b - 1:
            kmax = b - 1

        for k in range(j, kmax + 1):
            # --- intraday stop (gap through fills at the open) -------------
            if low[k] <= stop:
                px = open_[k] if open_[k] < stop else stop
                realized += pos * (px - entry)
                turn += pos
                exit_k = k
                reason = EXIT_STOP
                pos = 0.0
                break

            held = k - j
            # --- scale out on the first burst of strength ------------------
            if (not took_partial) and held >= partial_days:
                r_now = (close[k] - entry) / risk
                if r_now >= target_r or held >= partial_days:
                    realized += partial_frac * (close[k] - entry)
                    turn += partial_frac
                    pos -= partial_frac
                    took_partial = True
                    if breakeven_after_partial and entry > stop:
                        stop = entry

            # --- trail the remainder on the moving average -----------------
            if took_partial and pos > 0.0 and k > j:
                m = ma_trail[k]
                if m == m and close[k] < m:          # not NaN and below
                    realized += pos * (close[k] - entry)
                    turn += pos
                    exit_k = k
                    reason = EXIT_TRAIL
                    pos = 0.0
                    break

        if pos > 0.0:
            kk = kmax
            realized += pos * (close[kk] - entry)
            turn += pos
            exit_k = kk
            reason = EXIT_TIME if kk < b - 1 else EXIT_EOD

        t_sig[k_out] = i
        t_entry[k_out] = j
        t_exit[k_out] = exit_k
        t_epx[k_out] = entry
        t_risk[k_out] = risk
        t_pnl[k_out] = realized
        t_reason[k_out] = reason
        t_bars[k_out] = exit_k - j + 1
        t_turn[k_out] = turn
        k_out += 1
        last_exit = exit_k

    return (t_sig[:k_out], t_entry[:k_out], t_exit[:k_out], t_epx[:k_out],
            t_risk[:k_out], t_pnl[:k_out], t_reason[:k_out], t_bars[:k_out],
            t_turn[:k_out])


# ---------------------------------------------------------------------------
# portfolio: risk-based sizing, concurrency cap, sequential over days
# ---------------------------------------------------------------------------
@njit(cache=True)
def portfolio(order, entry_day, exit_day, epx, risk_ps, pnl_ps, turn_ps,
              n_days, start_equity, risk_frac, max_pos, max_weight,
              max_new_per_day, cost_bps, risk_mult):
    equity = start_equity
    eq = np.empty(n_days, np.float64)
    pending = np.zeros(n_days + 4, np.float64)
    slot_free = np.zeros(max_pos, np.int64)
    taken = np.zeros(len(entry_day), np.uint8)
    p = 0
    n = len(order)
    new_today = 0
    cur = -1

    for d in range(n_days):
        equity += pending[d]
        eq[d] = equity
        if equity <= 0:
            for dd in range(d, n_days):
                eq[dd] = 0.0
            break
        if d != cur:
            cur = d
            new_today = 0

        while p < n:
            t = order[p]
            if entry_day[t] > d:
                break
            if entry_day[t] < d:
                p += 1
                continue
            if new_today >= max_new_per_day:
                break
            slot = -1
            for s in range(max_pos):
                if slot_free[s] <= d:
                    slot = s
                    break
            if slot < 0:
                break

            rm = risk_mult[t]
            if rm <= 0.0:
                p += 1
                continue
            shares = risk_frac * rm * equity / risk_ps[t]
            notional = shares * epx[t]
            cap = max_weight * equity
            if notional > cap:
                notional = cap
                shares = notional / epx[t]
            if notional < 200.0:
                p += 1
                continue

            gross = pnl_ps[t] * shares
            cost = turn_ps[t] * shares * epx[t] * cost_bps / 1e4
            net = gross - cost
            xd = exit_day[t]
            if xd <= d:
                equity += net
            elif xd < n_days:
                pending[xd] += net
            slot_free[slot] = xd + 1
            taken[t] = 1
            new_today += 1
            p += 1

    return eq, taken

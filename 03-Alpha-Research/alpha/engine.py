"""Trade generation and portfolio simulation.

Two clearly separated layers:

1.  ``simulate_trades`` - a numba kernel that turns a boolean *setup* mask into
    a table of trades.  This is where the intraday mechanics live: the resting
    limit order, the gap-through fills, the same-bar stop/target ambiguity.

2.  ``run_portfolio`` - takes the trade table plus a ranking score and turns it
    into a capital-constrained equity curve (concurrency limits, ATR risk
    sizing, participation caps, costs on both sides).

Separating them is what makes brute force affordable: layer 1 is the expensive
part and depends only on the rule parameters, layer 2 is cheap and is where
portfolio/ML choices get swapped in and out.

INTRADAY FILL MODEL
-------------------
The vendor panel is daily OHLC, so the path *within* a bar is unknown.  Every
ambiguity is therefore resolved against the strategy:

* A pullback limit is only considered filled if the bar's low actually trades
  through it (``low <= limit``); touching is not enough on the boundary.
* If the bar opens below the limit, the fill is at the open, not the limit -
  gaps work against us, never for us.
* On any bar where both the stop and the target are inside the range, the stop
  is assumed to have been hit first.
* The entry bar is included in the stop scan, so a fill that is immediately
  followed (or preceded) by a deeper low is treated as a same-day stop-out.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# exit reason codes
EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_EOD = 1, 2, 3, 4


@njit(cache=True)
def simulate_trades(sig, open_, high, low, close, atr, starts, ends, side,
                    pullback_atr, entry_window, stop_atr, target_atr,
                    max_hold, trail_atr, max_paths):
    """Walk each symbol once and materialise every trade the rule produces."""
    cap = 0
    for i in range(len(sig)):
        if sig[i]:
            cap += 1
    if cap == 0:
        cap = 1

    t_sig = np.empty(cap, np.int64)
    t_entry = np.empty(cap, np.int64)
    t_exit = np.empty(cap, np.int64)
    t_epx = np.empty(cap, np.float64)
    t_xpx = np.empty(cap, np.float64)
    t_stop = np.empty(cap, np.float64)
    t_tgt = np.empty(cap, np.float64)
    t_atr = np.empty(cap, np.float64)
    t_reason = np.empty(cap, np.int8)
    t_mae = np.empty(cap, np.float64)
    t_mfe = np.empty(cap, np.float64)
    t_poff = np.empty(cap, np.int64)
    t_plen = np.empty(cap, np.int32)

    path_row = np.empty(max_paths, np.int64)
    pp = 0
    nt = 0

    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        i = a
        while i < b:
            if not sig[i]:
                i += 1
                continue
            A = atr[i]
            if not (A > 0.0) or not np.isfinite(A):
                i += 1
                continue

            limit = close[i] - side * pullback_atr * A

            # ---- resting limit order, live for `entry_window` sessions ----
            filled = -1
            epx = 0.0
            jmax = i + 1 + entry_window
            if jmax > b:
                jmax = b
            for j in range(i + 1, jmax):
                if side > 0:
                    if low[j] <= limit:
                        epx = open_[j] if open_[j] < limit else limit
                        filled = j
                        break
                else:
                    if high[j] >= limit:
                        epx = open_[j] if open_[j] > limit else limit
                        filled = j
                        break
            if filled < 0:
                i += 1          # order expired unfilled; symbol stays available
                continue

            stop = epx - side * stop_atr * A
            tgt = epx + side * target_atr * A
            peak = epx

            exit_k = -1
            xpx = 0.0
            reason = EXIT_TIME
            mae = 0.0
            mfe = 0.0

            kmax = filled + max_hold + 1
            if kmax > b:
                kmax = b
            for k in range(filled, kmax):
                if trail_atr > 0.0 and k > filled:
                    c1 = close[k - 1]
                    if side > 0:
                        if c1 > peak:
                            peak = c1
                        ts = peak - trail_atr * A
                        if ts > stop:
                            stop = ts
                    else:
                        if c1 < peak:
                            peak = c1
                        ts = peak + trail_atr * A
                        if ts < stop:
                            stop = ts

                # On the entry bar the favourable extreme may have printed
                # before the limit filled us, so only the stop is live there.
                first_bar = (k == filled)

                # running excursions, in R units, measured against the entry
                if side > 0:
                    ex_bad = (low[k] - epx) / A
                    ex_good = (high[k] - epx) / A
                else:
                    ex_bad = (epx - high[k]) / A
                    ex_good = (epx - low[k]) / A
                if ex_bad < mae:
                    mae = ex_bad
                if ex_good > mfe and not first_bar:
                    mfe = ex_good

                o = open_[k]
                if side > 0:
                    if not first_bar and o <= stop:
                        xpx = o; exit_k = k; reason = EXIT_STOP; break
                    if not first_bar and o >= tgt:
                        xpx = o; exit_k = k; reason = EXIT_TARGET; break
                    if low[k] <= stop:          # stop assumed first on ties
                        xpx = stop; exit_k = k; reason = EXIT_STOP; break
                    if not first_bar and high[k] >= tgt:
                        xpx = tgt; exit_k = k; reason = EXIT_TARGET; break
                else:
                    if not first_bar and o >= stop:
                        xpx = o; exit_k = k; reason = EXIT_STOP; break
                    if not first_bar and o <= tgt:
                        xpx = o; exit_k = k; reason = EXIT_TARGET; break
                    if high[k] >= stop:
                        xpx = stop; exit_k = k; reason = EXIT_STOP; break
                    if not first_bar and low[k] <= tgt:
                        xpx = tgt; exit_k = k; reason = EXIT_TARGET; break

            if exit_k < 0:
                kk = filled + max_hold
                if kk > b - 1:
                    kk = b - 1
                    reason = EXIT_EOD
                xpx = close[kk]
                exit_k = kk

            plen = exit_k - filled + 1
            if pp + plen > max_paths:
                break
            for k in range(filled, exit_k + 1):
                path_row[pp + k - filled] = k

            t_sig[nt] = i
            t_entry[nt] = filled
            t_exit[nt] = exit_k
            t_epx[nt] = epx
            t_xpx[nt] = xpx
            t_stop[nt] = stop
            t_tgt[nt] = tgt
            t_atr[nt] = A
            t_reason[nt] = reason
            t_mae[nt] = mae
            t_mfe[nt] = mfe
            t_poff[nt] = pp
            t_plen[nt] = plen
            nt += 1
            pp += plen

            i = exit_k + 1      # one position per symbol at a time

    return (t_sig[:nt], t_entry[:nt], t_exit[:nt], t_epx[:nt], t_xpx[:nt],
            t_stop[:nt], t_tgt[:nt], t_atr[:nt], t_reason[:nt], t_mae[:nt],
            t_mfe[:nt], t_poff[:nt], t_plen[:nt], path_row[:pp])


# ---------------------------------------------------------------------------
@njit(cache=True)
def _close_due(d, op_trade, op_shares, op_cost, op_notional, exit_day, exit_px,
               entry_px, side_arr, addv, base_bps, impact_coef, min_cps,
               trade_pnl, cash):
    """Realise every open position whose exit day is `d`.  Returns new cash.

    Called twice per session: once before new entries free up slots, and once
    after, because a position can be opened and stopped out on the same bar.
    """
    for sl in range(len(op_trade)):
        t = op_trade[sl]
        if t < 0 or exit_day[t] != d:
            continue
        sh = op_shares[sl]
        notional = sh * exit_px[t]
        bps = base_bps + impact_coef / np.sqrt(max(addv[t], 1.0) / 1e6)
        cst = notional * bps / 1e4
        fl = sh * min_cps
        if cst < fl:
            cst = fl
        if side_arr[t] > 0:
            cash += sh * exit_px[t] - cst
        else:
            cash += sh * (entry_px[t] - exit_px[t]) - cst
        trade_pnl[t] = side_arr[t] * sh * (exit_px[t] - entry_px[t]) - cst - op_cost[sl]
        op_trade[sl] = -1
        op_shares[sl] = 0.0
        op_cost[sl] = 0.0
        op_notional[sl] = 0.0
    return cash


@njit(cache=True)
def run_portfolio(order, entry_day, exit_day, entry_px, exit_px, stop_px,
                  side_arr, addv, path_off, path_len, path_day, path_close,
                  n_days, start_equity, risk_frac, max_pos, max_w, max_new,
                  base_bps, impact_coef, min_cps, participation, borrow_bps):
    """Capital-constrained simulation over the integer trading calendar.

    ``order`` must already be sorted by (entry_day, -score) so that on a
    crowded day the highest-conviction candidates take the available slots.
    """
    equity = start_equity
    cash = start_equity

    op_trade = np.full(max_pos, -1, np.int64)
    op_shares = np.zeros(max_pos, np.float64)
    op_cost = np.zeros(max_pos, np.float64)
    op_notional = np.zeros(max_pos, np.float64)

    eq_curve = np.empty(n_days, np.float64)
    exposure = np.empty(n_days, np.float64)
    n_open_c = np.empty(n_days, np.int32)
    trade_pnl = np.zeros(len(entry_day), np.float64)
    trade_taken = np.zeros(len(entry_day), np.uint8)
    trade_w = np.zeros(len(entry_day), np.float64)

    p = 0
    n = len(order)

    for d in range(n_days):
        # ---- 1. close whatever exits today -----------------------------
        cash = _close_due(d, op_trade, op_shares, op_cost, op_notional,
                          exit_day, exit_px, entry_px, side_arr, addv,
                          base_bps, impact_coef, min_cps, trade_pnl, cash)

        gross_now = 0.0
        for sl in range(max_pos):
            if op_trade[sl] >= 0:
                gross_now += op_notional[sl]

        # ---- 2. admit new entries -------------------------------------
        n_new = 0
        while p < n:
            t = order[p]
            if entry_day[t] > d:
                break
            if entry_day[t] < d:          # slot was unavailable yesterday
                p += 1
                continue
            if n_new >= max_new:
                break
            slot = -1
            for sl in range(max_pos):
                if op_trade[sl] < 0:
                    slot = sl
                    break
            if slot < 0:
                break

            risk_ps = abs(entry_px[t] - stop_px[t])
            if risk_ps <= 0.0:
                p += 1
                continue
            notional = risk_frac * equity / risk_ps * entry_px[t]

            cap_w = max_w * equity
            if notional > cap_w:
                notional = cap_w
            cap_liq = participation * addv[t]
            if notional > cap_liq:
                notional = cap_liq
            room = equity - gross_now          # no leverage beyond 1x gross
            if notional > room:
                notional = room
            if side_arr[t] > 0 and notional > cash:
                notional = cash
            if notional < 500.0:
                p += 1
                continue
            shares = notional / entry_px[t]

            bps = base_bps + impact_coef / np.sqrt(max(addv[t], 1.0) / 1e6)
            cst = notional * bps / 1e4
            fl = shares * min_cps
            if cst < fl:
                cst = fl
            if side_arr[t] < 0:
                cst += notional * borrow_bps / 1e4

            if side_arr[t] > 0:
                cash -= notional + cst
            else:
                cash -= cst
            op_trade[slot] = t
            op_shares[slot] = shares
            op_cost[slot] = cst
            op_notional[slot] = notional
            gross_now += notional
            trade_taken[t] = 1
            trade_w[t] = notional / equity
            n_new += 1
            p += 1

        # ---- 3. a position can be opened and closed on the same bar ----
        cash = _close_due(d, op_trade, op_shares, op_cost, op_notional,
                          exit_day, exit_px, entry_px, side_arr, addv,
                          base_bps, impact_coef, min_cps, trade_pnl, cash)

        # ---- 4. mark to market ----------------------------------------
        mv = 0.0
        gross_exp = 0.0
        nop = 0
        for sl in range(max_pos):
            t = op_trade[sl]
            if t < 0:
                continue
            nop += 1
            off = path_off[t]
            ln = path_len[t]
            px = entry_px[t]
            for q in range(ln):
                if path_day[off + q] == d:
                    px = path_close[off + q]
                    break
            sh = op_shares[sl]
            if side_arr[t] > 0:
                mv += sh * px
            else:
                mv += sh * (entry_px[t] - px)
            gross_exp += sh * px
        equity = cash + mv
        eq_curve[d] = equity
        exposure[d] = gross_exp / equity if equity > 0 else 0.0
        n_open_c[d] = nop
        if equity <= 0.0:
            for dd in range(d, n_days):
                eq_curve[dd] = 0.0
                exposure[dd] = 0.0
                n_open_c[dd] = 0
            break

    return eq_curve, exposure, n_open_c, trade_pnl, trade_taken, trade_w

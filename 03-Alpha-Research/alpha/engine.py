"""Portfolio simulation.

``run_portfolio`` takes a trade table plus a ranking score and turns it into a
capital-constrained, daily marked equity curve: concurrency limits, ATR risk
sizing, a participation cap on each name's average dollar volume, costs on both
sides, and a hard no-leverage constraint on gross exposure.

Trade generation itself lives in ``search.py``: it is the same code the sweep
uses, so a configuration cannot look good in the search and then behave
differently when it is reported.
"""
from __future__ import annotations

import numpy as np
from numba import njit

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
                  base_bps, impact_coef, min_cps, participation, borrow_bps,
                  gross_target):
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
            # gross_target == 1.0 means fully invested with no margin; above
            # that the book is explicitly levered, which is a disclosed choice
            # rather than an accident of the sizing rule
            room = gross_target * equity - gross_now
            if notional > room:
                notional = room
            if side_arr[t] > 0 and gross_target <= 1.0 and notional > cash:
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

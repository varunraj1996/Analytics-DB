"""One canonical path from parameters to a fully-marked equity curve.

The search stages and the final evaluation deliberately share the *same*
primitives (``search.mask_rows`` -> ``search.find_fills`` -> ``search.eval_exits``)
so that a configuration cannot look good in the sweep and then quietly behave
differently when it is reported.  The only thing that changes at the end is the
portfolio layer: the proxy used for ranking is swapped for the daily
mark-to-market simulation in ``engine.run_portfolio``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from . import config, engine, search, strategy


@njit(cache=True)
def _build_paths(entry_row, exit_row, day, close):
    n = len(entry_row)
    lens = np.empty(n, np.int32)
    tot = 0
    for i in range(n):
        lens[i] = exit_row[i] - entry_row[i] + 1
        tot += lens[i]
    off = np.empty(n, np.int64)
    p_day = np.empty(tot, np.int32)
    p_close = np.empty(tot, np.float64)
    q = 0
    for i in range(n):
        off[i] = q
        for r in range(entry_row[i], exit_row[i] + 1):
            p_day[q] = day[r]
            p_close[q] = close[r]
            q += 1
    return off, lens, p_day, p_close


def build_trades(ws, p: config.StrategyParams, uni: config.UniverseSpec,
                 day_lo: int, day_hi: int, E=None,
                 entry_bar_mode: int = 2) -> pd.DataFrame:
    """Every trade the rule would have taken between two calendar days."""
    P, F = ws.P, ws.F
    E = E or search.Eligible(P, F, uni, ws.member, day_lo, day_hi)
    block_end = P.ends[P.sym_id]

    sig_rows = search.mask_rows(E, p, ws.regime[p.regime_ma] if p.regime_ma else None)
    if len(sig_rows) == 0:
        return pd.DataFrame()

    f_sig, f_row, f_px, f_atr = search.find_fills(
        sig_rows, block_end[sig_rows], P.open, P.high, P.low, P.close,
        F["atr14"], np.int64(p.side), float(p.pullback_atr),
        np.int64(p.entry_window))
    if len(f_sig) == 0:
        return pd.DataFrame()

    x_row, x_px, x_reason, x_mae, x_mfe, keep = search.eval_exits(
        f_sig, f_row, f_px, f_atr, block_end[f_sig], P.open, P.high, P.low,
        P.close, np.int64(p.side), float(p.stop_atr), float(p.target_atr),
        np.int64(p.max_hold), float(p.trail_atr), np.int64(entry_bar_mode))

    k = keep.astype(bool)
    # mode 2 executes market-on-close on the trigger bar, so the price paid is
    # that bar's close rather than the resting limit
    entry_px = P.close[f_row[k]] if entry_bar_mode == 2 else f_px[k]
    t = pd.DataFrame({
        "sym": P.sym_id[f_sig[k]],
        "ticker": P.symbols[P.sym_id[f_sig[k]]],
        "sig_row": f_sig[k], "entry_row": f_row[k], "exit_row": x_row[k],
        "sig_day": P.day[f_sig[k]], "entry_day": P.day[f_row[k]],
        "exit_day": P.day[x_row[k]],
        "sig_date": P.dates[f_sig[k]], "entry_date": P.dates[f_row[k]],
        "entry_px": entry_px, "exit_px": x_px[k], "atr": f_atr[k],
        "reason": x_reason[k], "mae_r": x_mae[k], "mfe_r": x_mfe[k],
        "side": p.side,
    })
    t["stop_px"] = t["entry_px"] - p.side * p.stop_atr * t["atr"]
    t["target_px"] = t["entry_px"] + p.side * p.target_atr * t["atr"]
    t["addv"] = F["addv21"][f_sig[k]]
    t["bars_held"] = t["exit_row"] - t["entry_row"] + 1
    t["wait"] = t["entry_row"] - t["sig_row"]
    t["ret"] = p.side * (t["exit_px"] / t["entry_px"] - 1.0)
    t["r_mult"] = p.side * (t["exit_px"] - t["entry_px"]) / (t["entry_px"] - t["stop_px"]).abs()
    return t.reset_index(drop=True)


def run_portfolio_full(ws, t: pd.DataFrame, score: np.ndarray | None = None,
                       spec: config.PortfolioSpec | None = None,
                       costs: config.CostModel | None = None,
                       day_lo: int = 0, day_hi: int | None = None,
                       participation: float = 0.01, borrow_bps: float = 2.0):
    """Daily mark-to-market portfolio over an existing trade table."""
    spec = spec or config.DEFAULT_PORTFOLIO
    costs = costs or config.DEFAULT_COSTS
    day_hi = ws.n_days - 1 if day_hi is None else day_hi
    if len(t) == 0:
        return None

    sel = (t["entry_day"].to_numpy() >= day_lo) & (t["entry_day"].to_numpy() <= day_hi)
    t = t[sel].reset_index(drop=True)
    if len(t) == 0:
        return None
    score = np.zeros(len(t)) if score is None else np.asarray(score)[sel]

    off, lens, p_day, p_close = _build_paths(
        t["entry_row"].to_numpy().astype(np.int64),
        t["exit_row"].to_numpy().astype(np.int64),
        ws.P.day, ws.P.close)

    order = np.lexsort((-score, t["entry_day"].to_numpy())).astype(np.int64)
    eq, expo, nopen, pnl, taken, w = engine.run_portfolio(
        order,
        t["entry_day"].to_numpy().astype(np.int64),
        t["exit_day"].to_numpy().astype(np.int64),
        t["entry_px"].to_numpy(), t["exit_px"].to_numpy(),
        t["stop_px"].to_numpy(), t["side"].to_numpy().astype(np.float64),
        t["addv"].to_numpy(), off, lens, p_day, p_close,
        ws.n_days, float(spec.starting_equity), float(spec.risk_per_trade),
        int(spec.max_positions), float(spec.max_weight),
        int(spec.max_new_per_day), float(costs.base_bps),
        float(costs.impact_coef), float(costs.min_cents_per_share),
        float(participation), float(borrow_bps), float(spec.gross_target))

    m = strategy.metrics(eq, ws.calendar, day_lo, day_hi)
    tk = taken.astype(bool)
    m.update({
        "n_candidates": len(t),
        "n_taken": int(tk.sum()),
        "take_rate": float(tk.mean()),
        "avg_exposure": float(expo[day_lo:day_hi + 1].mean()),
        "avg_open": float(nopen[day_lo:day_hi + 1].mean()),
        "win_rate_taken": float((pnl[tk] > 0).mean()) if tk.any() else np.nan,
        "avg_bars": float(t.loc[tk, "bars_held"].mean()) if tk.any() else np.nan,
        "trades_per_year": float(tk.sum() / max(m.get("years", 1), 1e-9)),
        "gross_target": float(spec.gross_target),
    })
    return {"metrics": m, "equity": eq, "exposure": expo, "n_open": nopen,
            "pnl": pnl, "taken": tk, "weight": w, "trades": t}


@njit(cache=True)
def _greedy_non_overlap(sym, entry_row, exit_row):
    """Keep, per symbol, only trades that do not overlap an earlier one.

    Each variant already enforces one-position-per-symbol internally, but an
    *ensemble* of variants can propose two overlapping trades in the same name
    (or a long and a short at once).  Rows must arrive sorted by (sym, entry).
    """
    keep = np.zeros(len(sym), np.uint8)
    cur_sym = -1
    last_exit = -1
    for i in range(len(sym)):
        if sym[i] != cur_sym:
            cur_sym = sym[i]
            last_exit = -1
        if entry_row[i] > last_exit:
            keep[i] = 1
            last_exit = exit_row[i]
    return keep


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge several rule variants into one candidate pool.

    Exact duplicates are dropped (the earlier frame in the list wins), then any
    residual overlap within a symbol is resolved greedily in entry order.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    all_t = pd.concat(frames, ignore_index=True)
    all_t = all_t.drop_duplicates(subset=["sym", "entry_row", "side"], keep="first")
    all_t = all_t.sort_values(["sym", "entry_row"], ignore_index=True)
    keep = _greedy_non_overlap(
        all_t["sym"].to_numpy().astype(np.int64),
        all_t["entry_row"].to_numpy().astype(np.int64),
        all_t["exit_row"].to_numpy().astype(np.int64)).astype(bool)
    return all_t[keep].sort_values("entry_day", ignore_index=True)

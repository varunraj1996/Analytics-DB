"""Stage 1 brute force: sweep the rule space on the TRAIN window only.

~2,300 entry-filter combinations x 20 limit-order variants x 128 exit variants
= ~5.9 million backtests.  Everything is evaluated on 2001-2011 and nothing
else is looked at.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, search, workspace

# --------------------------------------------------------------------------
MASK_GRID = dict(
    side=[1, -1],
    base_len=[20, 40, 60, 120],
    breakout_margin=[0.0, 0.5],
    vol_mult=[0.0, 1.5, 2.5],
    trend_ma=[0, 200],
    regime_ma=[0, 200],
    min_close_loc=[0.0, 0.6],
    max_ext_atr=[99.0, 4.0],
    min_base_tightness=[0.0, 8.0],
)
FILL_GRID = dict(
    pullback_atr=[0.25, 0.5, 1.0, 1.5],
    entry_window=[1, 3, 5],
)
EXIT_GRID = dict(
    stop_atr=[1.0, 1.5, 2.5],
    target_atr=[0.75, 1.5, 3.0, 5.0],
    max_hold=[3, 5, 10, 20],
    trail_atr=[0.0, 2.5],
)

MIN_TRADES = 300
TOP_PER_MASK = 40

# Execution model: the intraday pullback is the trigger, the fill is that day's
# close.  Filling at the resting limit instead (mode 0/1) lets the backtest use
# intra-bar prices whose ordering is unknowable, and it showed: under mode 0
# the *validation* window beat the train window, which is not something a real
# edge does.  See scripts/04c_decompose.py.
ENTRY_BAR_MODE = 2

_G: dict = {}


def _combos(grid: dict):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def setup(ws, E, day_lo, day_hi, spec, costs):
    """Populate the module global *before* forking so workers share the panel
    copy-on-write instead of each pickling a multi-gigabyte workspace."""
    _G.update(ws=ws, E=E, day_lo=day_lo, day_hi=day_hi, spec=spec, costs=costs)
    _G["n_days"] = ws.n_days
    _G["block_end"] = ws.P.ends[ws.P.sym_id]


def evaluate_mask(mc: dict) -> list[dict]:
    ws, E = _G["ws"], _G["E"]
    P = ws.P
    spec, costs = _G["spec"], _G["costs"]
    block_end = _G["block_end"]
    n_days = _G["n_days"]

    p = config.StrategyParams(**mc)
    regime = ws.regime[p.regime_ma] if p.regime_ma else None
    sig_rows = search.mask_rows(E, p, regime)
    if len(sig_rows) < MIN_TRADES:
        return []

    sig_be = block_end[sig_rows]
    out = []

    for fc in _combos(FILL_GRID):
        f_sig, f_row, f_px, f_atr = search.find_fills(
            sig_rows, sig_be, P.open, P.high, P.low, P.close, ws.F["atr14"],
            np.int64(p.side), float(fc["pullback_atr"]),
            np.int64(fc["entry_window"]))
        if len(f_sig) < MIN_TRADES:
            continue
        f_be = block_end[f_sig]
        entry_day = P.day[f_row].astype(np.int64)
        addv = ws.F["addv21"][f_sig]

        fill_px = P.close[f_row] if ENTRY_BAR_MODE == 2 else f_px

        for xc in _combos(EXIT_GRID):
            x_row, x_px, x_reason, x_mae, x_mfe, keep = search.eval_exits(
                f_sig, f_row, f_px, f_atr, f_be, P.open, P.high, P.low, P.close,
                np.int64(p.side), float(xc["stop_atr"]), float(xc["target_atr"]),
                np.int64(xc["max_hold"]), float(xc["trail_atr"]),
                np.int64(ENTRY_BAR_MODE))
            k = keep.astype(bool)
            n = int(k.sum())
            if n < MIN_TRADES:
                continue

            epx = fill_px[k]
            xpx = x_px[k]
            atr_k = f_atr[k]
            stop_px = epx - p.side * xc["stop_atr"] * atr_k
            r = p.side * (xpx - epx) / np.abs(epx - stop_px)
            ret = p.side * (xpx / epx - 1.0)

            avg_r = float(np.nanmean(r))
            sd = float(np.nanstd(r, ddof=1))
            tstat = avg_r / sd * np.sqrt(n) if sd > 0 else 0.0

            ed = entry_day[k]
            xd = P.day[x_row[k]].astype(np.int64)
            order = np.lexsort((-addv[k], ed)).astype(np.int64)
            eq, n_taken = search.fast_portfolio(
                order, ed, xd, epx, xpx, stop_px,
                np.full(n, float(p.side)), addv[k], n_days,
                float(spec.starting_equity), float(spec.risk_per_trade),
                int(spec.max_positions), float(spec.max_weight),
                int(spec.max_new_per_day), float(costs.base_bps),
                float(costs.impact_coef), float(costs.min_cents_per_share), 0.01)

            seg = eq[_G["day_lo"]:_G["day_hi"] + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 100 or seg[0] <= 0 or seg[-1] <= 0:
                continue
            yrs = len(seg) / 252.0
            cagr = (seg[-1] / seg[0]) ** (1 / yrs) - 1
            dr = np.diff(seg) / seg[:-1]
            vol = dr.std(ddof=1) * np.sqrt(252)
            sharpe = dr.mean() * 252 / vol if vol > 0 else 0.0
            mdd = float((seg / np.maximum.accumulate(seg) - 1).min())

            out.append(dict(
                **mc, **fc, **xc,
                n_signals=len(sig_rows), n_fills=len(f_sig), n_trades=n,
                win_rate=float((r > 0).mean()), avg_r=avg_r, t_stat=float(tstat),
                expectancy_bps=float(np.nanmean(ret) * 1e4),
                cagr=float(cagr), sharpe=float(sharpe), max_dd=mdd,
                n_taken=int(n_taken),
                avg_bars=float(np.mean(x_row[k] - f_row[k] + 1)),
            ))

    if not out:
        return []
    # keep the best by statistical strength *and* by realised compounding -
    # t-stat alone just rewards whichever variant produced the most trades
    by_t = sorted(out, key=lambda d: -d["t_stat"])[:TOP_PER_MASK]
    by_c = sorted(out, key=lambda d: -d["cagr"])[:TOP_PER_MASK]
    seen, keep = set(), []
    for d in by_t + by_c:
        k = (d["pullback_atr"], d["entry_window"], d["stop_atr"],
             d["target_atr"], d["max_hold"], d["trail_atr"])
        if k not in seen:
            seen.add(k)
            keep.append(d)
    return keep


def main():
    ws = workspace.load()
    day_lo, day_hi = ws.train
    uni = config.DEFAULT_UNIVERSE
    print(f"[search] train days {day_lo}..{day_hi} "
          f"({ws.label(day_lo)} -> {ws.label(day_hi)})")

    t0 = time.time()
    E = search.Eligible(ws.P, ws.F, uni, ws.member, day_lo, day_hi)
    print(f"[search] eligible rows {len(E):,} of {ws.P.n:,} "
          f"({time.time() - t0:.1f}s)")

    setup(ws, E, day_lo, day_hi, config.DEFAULT_PORTFOLIO, config.DEFAULT_COSTS)

    masks = _combos(MASK_GRID)
    if "--quick" in sys.argv:
        masks = masks[::96]
    n_eval = len(masks) * len(_combos(FILL_GRID)) * len(_combos(EXIT_GRID))
    print(f"[search] {len(masks)} mask configs x {len(_combos(FILL_GRID))} fill "
          f"x {len(_combos(EXIT_GRID))} exit = {n_eval:,} backtests")

    t0 = time.time()
    rows: list[dict] = []
    with Pool(4) as pool:
        for i, res in enumerate(pool.imap_unordered(evaluate_mask, masks, chunksize=1)):
            rows.extend(res)
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"[search]   {i + 1}/{len(masks)} masks  {len(rows):,} kept  "
                      f"{el / 60:.1f}m elapsed  eta {el / (i + 1) * (len(masks) - i - 1) / 60:.0f}m",
                      flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, "train_search.parquet")
    df.to_parquet(out, index=False)
    print(f"[search] {len(df):,} results -> {out}  ({(time.time() - t0) / 60:.1f}m)")

    if len(df):
        top = df.sort_values("t_stat", ascending=False).head(20)
        cols = ["side", "base_len", "pullback_atr", "entry_window", "stop_atr",
                "target_atr", "max_hold", "trail_atr", "n_trades", "win_rate",
                "avg_r", "t_stat", "cagr", "sharpe", "max_dd"]
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()

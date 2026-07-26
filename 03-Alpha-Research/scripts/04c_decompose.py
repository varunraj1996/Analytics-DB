"""Where does the per-trade return actually come from?

Every number in a daily-bar backtest of an intraday entry rests on assumptions
about prices that were never observed.  This script strips them away and
measures the setup using *only* closing prices, which are unambiguous:

    leg A   limit fill  ->  close of the fill day     (pure fill assumption)
    leg B   close of the fill day -> close of day +1  (real, observable)
    leg C   close of the fill day -> close of day +3  (real, observable)

If leg A carries the edge and legs B and C are flat, then the strategy is not
predicting anything - it is just buying the intraday low and marking it to the
close of the same bar, which no real limit order reliably achieves.

It also tests fill realism directly: requiring the bar to trade a margin
*through* the limit before counting a fill is the standard way to check whether
a result depends on being at the front of the queue at the day's extreme.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, pipeline, search, workspace

pd.set_option("display.width", 200)

ws = workspace.load()
P, F = ws.P, ws.F
tr_lo, tr_hi = ws.train
uni = config.DEFAULT_UNIVERSE

frozen = json.load(open(os.path.join(config.RESULTS_DIR, "frozen_config.json")))
cfgs = frozen["configs"]
KEYS = list(cfgs[0].keys())
INT = {"side", "base_len", "trend_ma", "regime_ma", "entry_window", "max_hold"}


def to_p(d):
    return config.StrategyParams(**{k: (int(d[k]) if k in INT else float(d[k]))
                                    for k in KEYS})


E = search.Eligible(P, F, uni, ws.member, tr_lo, tr_hi)
pool = pipeline.combine([pipeline.build_trades(ws, to_p(c), uni, tr_lo, tr_hi, E,
                                               entry_bar_mode=0)
                         for c in cfgs])
print(f"pooled train candidates: {len(pool):,}")

side = pool["side"].to_numpy()
epx = pool["entry_px"].to_numpy()
er = pool["entry_row"].to_numpy()
block_end = P.ends[P.sym_id][er]

c_j = P.close[er]
low_j = P.low[er]
high_j = P.high[er]


def close_at(offset):
    idx = np.minimum(er + offset, block_end - 1)
    return P.close[idx]


print()
print("=" * 78)
print("RETURN DECOMPOSITION (train window, sign-adjusted for side)")
print("=" * 78)
legs = {
    "A  fill -> close of fill day": side * (c_j / epx - 1.0),
    "B  close(j) -> close(j+1)": side * (close_at(1) / c_j - 1.0),
    "C  close(j) -> close(j+3)": side * (close_at(3) / c_j - 1.0),
    "D  close(j) -> close(j+5)": side * (close_at(5) / c_j - 1.0),
    "E  fill -> close(j+1)": side * (close_at(1) / epx - 1.0),
    "   full modelled trade": pool["ret"].to_numpy(),
}
rows = []
for name, r in legs.items():
    r = r[np.isfinite(r)]
    rows.append({"leg": name, "mean_bps": r.mean() * 1e4,
                 "median_bps": np.median(r) * 1e4,
                 "win": (r > 0).mean(),
                 "t_stat": r.mean() / r.std(ddof=1) * np.sqrt(len(r))})
print(pd.DataFrame(rows).to_string(index=False))

print("\nHow close is the fill to the bar's low?")
depth = (epx - low_j) / np.maximum(high_j - low_j, 1e-9)
print(f"  fill position in the entry bar's range (0 = the low): "
      f"median {np.median(depth):.3f}, mean {depth.mean():.3f}")
print(f"  fills within 5% of the bar range above the low: {(depth < 0.05).mean():.1%}")
print(f"  fills exactly at the bar low:                   {(depth < 1e-6).mean():.1%}")

print()
print("=" * 78)
print("FILL REALISM: require the bar to trade THROUGH the limit by a margin")
print("=" * 78)
p0 = to_p(cfgs[0])
atr = F["atr14"]
sig_rows = search.mask_rows(E, p0, ws.regime[p0.regime_ma] if p0.regime_ma else None)
be = P.ends[P.sym_id]
for margin in (0.0, 0.05, 0.10, 0.25, 0.50):
    f_sig, f_row, f_px, f_atr = search.find_fills(
        sig_rows, be[sig_rows], P.open, P.high, P.low, P.close, atr,
        np.int64(p0.side), float(p0.pullback_atr + margin), np.int64(p0.entry_window))
    if len(f_sig) == 0:
        continue
    # fill at the ORIGINAL limit, but only when price traded `margin` ATR beyond it
    orig_limit = P.close[f_sig] - p0.side * p0.pullback_atr * f_atr
    px = np.where(P.open[f_row] < orig_limit, P.open[f_row], orig_limit) \
        if p0.side > 0 else np.where(P.open[f_row] > orig_limit, P.open[f_row], orig_limit)
    nxt = np.minimum(f_row + 1, be[f_sig] - 1)
    r1 = p0.side * (P.close[nxt] / px - 1.0)
    r0 = p0.side * (P.close[f_row] / px - 1.0)
    print(f"  through-limit margin {margin:.2f} ATR: fills {len(f_sig):>7,}  "
          f"legA {r0.mean() * 1e4:7.1f} bps   legE(fill->close j+1) {r1.mean() * 1e4:7.1f} bps")

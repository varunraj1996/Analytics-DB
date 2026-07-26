"""Did the effect decay, or did the backtest just get lucky?

The portfolio failed out of sample.  That could mean the rule never had an edge
and the train/validation results were selection noise, or it could mean the edge
was real and stopped paying.  Those have different implications, and they are
distinguishable: measure the raw, assumption-free close-to-close reversal after
a pullback trigger, year by year, over the whole sample.

No portfolio, no ranking, no parameter selection - just the population mean of
every trigger the base rule produces, so nothing here can be an artefact of how
capital was allocated.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, search, workspace

pd.set_option("display.width", 200)

ws = workspace.load()
P, F = ws.P, ws.F
uni = config.DEFAULT_UNIVERSE
block_end = P.ends[P.sym_id]

# One plain, un-tuned configuration on each side.
BASE = dict(base_len=20, breakout_margin=0.0, vol_mult=1.5, trend_ma=200,
            regime_ma=0, min_close_loc=0.0, max_ext_atr=99.0,
            min_base_tightness=0.0, pullback_atr=1.0, entry_window=5)

E = search.Eligible(P, F, uni, ws.member, 0, 10 ** 9)
rows = []
for side in (1, -1):
    p = config.StrategyParams(side=side, **BASE)
    sig = search.mask_rows(E, p, None)
    f_sig, f_row, f_px, f_atr = search.find_fills(
        sig, block_end[sig], P.open, P.high, P.low, P.close, F["atr14"],
        np.int64(side), float(p.pullback_atr), np.int64(p.entry_window))
    be = block_end[f_sig]
    c0 = P.close[f_row]
    for h in (1, 3, 5):
        nxt = np.minimum(f_row + h, be - 1)
        r = side * (P.close[nxt] / c0 - 1.0)
        yr = pd.DatetimeIndex(P.dates[f_row]).year
        for y, idx in pd.Series(np.arange(len(r))).groupby(yr):
            v = r[idx.to_numpy()]
            v = v[np.isfinite(v)]
            if len(v) < 200:
                continue
            rows.append({"side": side, "horizon": h, "year": int(y),
                         "n": len(v), "bps": v.mean() * 1e4,
                         "t": v.mean() / v.std(ddof=1) * np.sqrt(len(v))})

d = pd.DataFrame(rows)
d = d[(d.year >= 2001) & (d.year <= 2017)]

for side in (1, -1):
    name = "LONG  (pullback after breakout)" if side == 1 else "SHORT (rally after breakdown)"
    print("=" * 78)
    print(f"{name}: mean close-to-close return after the trigger, by year")
    print("=" * 78)
    piv = d[d.side == side].pivot(index="year", columns="horizon",
                                  values="bps").round(1)
    cnt = d[(d.side == side) & (d.horizon == 1)].set_index("year")["n"]
    piv["n_triggers"] = cnt
    piv.columns = [f"+{c}d bps" if isinstance(c, (int, np.integer)) else c
                   for c in piv.columns]
    print(piv.to_string())
    print()

print("=" * 78)
print("SPLIT AVERAGES (horizon +1d, equal-weighted across triggers)")
print("=" * 78)
spans = {"train 2001-2011": (2001, 2011), "valid 2012-2014": (2012, 2014),
         "test  2015-2017": (2015, 2017)}
out = []
for side in (1, -1):
    for name, (a, b) in spans.items():
        s = d[(d.side == side) & (d.horizon == 1) & (d.year >= a) & (d.year <= b)]
        if len(s) == 0:
            continue
        w = s["n"].to_numpy()
        out.append({"side": "long" if side == 1 else "short", "split": name,
                    "triggers": int(w.sum()),
                    "bps": float(np.average(s["bps"], weights=w))})
print(pd.DataFrame(out).to_string(index=False))

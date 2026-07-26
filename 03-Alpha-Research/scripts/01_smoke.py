"""Build the workspace cache and sanity-check the engine on the train split."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np

from alpha import config, strategy, workspace

ws = workspace.load(rebuild="--rebuild" in sys.argv)
P, F = ws.P, ws.F
print(f"[smoke] {P.n:,} rows  {len(P.symbols):,} symbols  "
      f"{ws.calendar[0]} -> {ws.calendar[-1]}  {ws.n_days} sessions")
for name, sp in (("train", ws.train), ("valid", ws.valid), ("test", ws.test)):
    print(f"[smoke]   {name}: day {sp[0]}..{sp[1]}  {ws.label(sp[0])} -> {ws.label(sp[1])}")

lo, hi = ws.train
p = config.StrategyParams()
uni = config.DEFAULT_UNIVERSE

t0 = time.time()
mask = strategy.setup_mask(P, F, p, uni, ws.regime[p.regime_ma])
print(f"[smoke] setups in panel: {mask.sum():,}   ({time.time() - t0:.1f}s)")

t0 = time.time()
tr = strategy.generate_trades(P, F, p, uni, ws.regime[p.regime_ma], lo, hi)
print(f"[smoke] trades generated: {len(tr):,}   ({time.time() - t0:.1f}s)")
print(f"[smoke] fill rate: {len(tr) / max(int((mask & (P.day >= lo) & (P.day <= hi)).sum()), 1):.1%}")

st = strategy.trade_stats(tr)
for k, v in st.items():
    print(f"[smoke]   {k:>14}: {v:,.4f}")

t0 = time.time()
res = strategy.portfolio_from_trades(tr, P, ws.n_days, day_lo=lo, day_hi=hi)
print(f"[smoke] portfolio ({time.time() - t0:.1f}s)")
m = strategy.metrics(res["equity"], ws.calendar, lo, hi)
for k, v in m.items():
    print(f"[smoke]   {k:>14}: {v:,.4f}")
print(f"[smoke]   taken {res['taken'].sum():,} / {len(tr):,} "
      f"({res['taken'].mean():.1%})  avg exposure {res['exposure'][lo:hi].mean():.2f}")

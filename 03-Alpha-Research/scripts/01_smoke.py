"""Build the workspace cache and sanity-check the engine on the train split."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

from alpha import config, pipeline, search, strategy, workspace

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
E = search.Eligible(P, F, uni, ws.member, lo, hi)
sig = search.mask_rows(E, p, ws.regime[p.regime_ma])
print(f"[smoke] eligible rows {len(E):,}  setups {len(sig):,}  "
      f"({time.time() - t0:.1f}s)")

t0 = time.time()
tr = pipeline.build_trades(ws, p, uni, lo, hi, E)
print(f"[smoke] trades: {len(tr):,}  ({time.time() - t0:.1f}s)  "
      f"trigger rate {len(tr) / max(len(sig), 1):.1%}")

for k, v in strategy.trade_stats(tr).items():
    print(f"[smoke]   {k:>14}: {v:,.4f}")

t0 = time.time()
res = pipeline.run_portfolio_full(ws, tr, day_lo=lo, day_hi=hi)
print(f"[smoke] portfolio ({time.time() - t0:.1f}s)")
for k, v in res["metrics"].items():
    print(f"[smoke]   {k:>16}: {v:,.4f}")

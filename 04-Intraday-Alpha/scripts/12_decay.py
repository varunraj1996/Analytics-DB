"""Year-by-year raw expectancy of one un-tuned configuration, both sides.

Same diagnostic that settled the daily study: no portfolio, no selection, just
the population mean of every trade the plain rule produces, by year.  Whatever
the sweep says, this is the number that decides whether there is a *pattern*
or just a parameter choice.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/user/Analytics-DB/04-Intraday-Alpha")

import pandas as pd

from intraday import ingest, study
from intraday.config import IntradayParams

pd.set_option("display.width", 200)

panel = ingest.load_panel()
book = study.Book(panel)

for side in (1, -1):
    p = IntradayParams(side=side)     # all defaults, no tuning
    t = book.trades(p)
    t["year"] = t["date"].dt.year
    g = (t.groupby("year")
         .agg(n=("ret", "size"),
              bps_net=("ret", lambda x: x.mean() * 1e4),
              bps_gross=("ret_gross", lambda x: x.mean() * 1e4),
              win=("ret", lambda x: (x > 0).mean()),
              eod_pct=("reason", lambda x: (x == 4).mean())))
    name = "LONG breakout-pullback" if side == 1 else "SHORT breakdown-rally"
    print("=" * 70)
    print(f"{name}  (defaults: OR-3 breakout, 0.5 ATR pullback, "
          f"1.5/3.0 stop/target, EOD flat)")
    print("=" * 70)
    print(g.round(2).to_string())
    print(f"ALL: n={len(t)}  net {t['ret'].mean() * 1e4:.1f} bps  "
          f"gross {t['ret_gross'].mean() * 1e4:.1f} bps  "
          f"win {(t['ret'] > 0).mean():.1%}")
    print()

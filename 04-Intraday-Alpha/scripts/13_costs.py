"""Where does the edge sit relative to the friction?

Gross expectancy is what the pattern produces; net is what a trader keeps.  If
break-even sits above a realistic spread, no amount of modelling saves it.
"""
import sys; sys.path.insert(0, "/home/user/Analytics-DB/04-Intraday-Alpha")
import json, os
import numpy as np, pandas as pd
from intraday import config, ingest, study
from intraday.config import IntradayParams
pd.set_option("display.width", 200)

panel = ingest.load_panel(); book = study.Book(panel)
frozen = json.load(open(os.path.join(config.RESULTS_DIR, "frozen10.json")))

cases = {
    "un-tuned long":  IntradayParams(side=1),
    "un-tuned short": IntradayParams(side=-1),
    "frozen (selected on 2024)": IntradayParams(**frozen["params"]),
}
rows = []
for name, p in cases.items():
    for win, (lo, hi) in (("full", (None, None)), ("test", config.TEST)):
        t = book.trades(p, lo, hi)
        if len(t) < 50:
            continue
        g = t["ret_gross"].mean() * 1e4
        be = g / 2.0            # per-side bps that zeroes it out
        rows.append({"config": name, "window": win, "n": len(t),
                     "gross_bps": round(g, 2),
                     "breakeven_bps_per_side": round(be, 2),
                     "net@5bps": round(g - 10, 2),
                     "net@2bps": round(g - 4, 2),
                     "net@1bps": round(g - 2, 2)})
print(pd.DataFrame(rows).to_string(index=False))
print()
print("Reference: TQQQ trades ~$35 with a 1-cent spread => half-spread ~1.4 bps.")
print("A patient limit/mid execution on these ETFs is ~1-2 bps per side;")
print("5 bps per side (the study default) is deliberately punitive.")

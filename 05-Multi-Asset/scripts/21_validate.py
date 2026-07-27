"""Step 6: select ONE configuration on TRAIN + VALIDATION, then freeze it.

Pre-declared rule (STATE.md): among configurations with train Sharpe >= 0.5
and validation max drawdown within 1.5x the train max drawdown, pick the
highest validation Sharpe.  Family stability is reported alongside so the pick
can be sanity-checked against its neighbours, but the rule is the rule.

The saved per-config return series from the sweep are reused - nothing is
recomputed, and nothing past 2016-12-31 is read.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/05-Multi-Asset")

import pandas as pd

from multiasset import config, portfolio

pd.set_option("display.width", 240)


def main():
    df = pd.read_parquet(os.path.join(config.RESULTS_DIR, "sweep_train.parquet"))
    print(f"[val] {len(df)} sweep configs")

    rows = []
    for _, r in df.iterrows():
        f = os.path.join(config.RESULTS_DIR, "rets", f"{int(r.cfg):04d}.parquet")
        ret = pd.read_parquet(f)["ret"]
        m = portfolio.metrics(ret, "2010-01-01", config.VALID_END)
        if not m:
            continue
        rows.append({**r.to_dict(), **{f"va_{k}": v for k, v in m.items()}})
    v = pd.DataFrame(rows)
    v.to_parquet(os.path.join(config.RESULTS_DIR, "validate.parquet"), index=False)

    ok = v[(v.tr_sharpe >= 0.5) & (v.va_max_dd >= 1.5 * v.tr_max_dd)]
    print(f"[val] {len(ok)} of {len(v)} pass the pre-declared screen "
          f"(train sharpe >= 0.5, valid DD within 1.5x train DD)")

    cols = ["key", "tr_sharpe", "tr_cagr", "tr_max_dd",
            "va_sharpe", "va_cagr", "va_vol", "va_max_dd"]
    print("\n[val] top 20 by VALIDATION sharpe (among screen passers):")
    top = ok.sort_values("va_sharpe", ascending=False)
    print(top.head(20)[cols].to_string(index=False))

    print("\n[val] family stability - validation sharpe by component:")
    for dim in ("trend", "brk", "carry", "skew", "classes"):
        g = ok.groupby(dim)["va_sharpe"].agg(["mean", "median", "size"])
        print(f"  by {dim}:\n{g.round(3).to_string()}")

    pick = top.iloc[0]
    frozen = {
        "cfg": int(pick["cfg"]), "key": str(pick["key"]),
        "trend": str(pick["trend"]), "brk": str(pick["brk"]),
        "carry": float(pick["carry"]), "skew": float(pick["skew"]),
        "classes": str(pick["classes"]),
        "train": {"sharpe": float(pick["tr_sharpe"]), "cagr": float(pick["tr_cagr"]),
                  "max_dd": float(pick["tr_max_dd"])},
        "valid": {"sharpe": float(pick["va_sharpe"]), "cagr": float(pick["va_cagr"]),
                  "max_dd": float(pick["va_max_dd"]), "vol": float(pick["va_vol"])},
    }
    with open(os.path.join(config.RESULTS_DIR, "frozen_ma.json"), "w") as fh:
        json.dump(frozen, fh, indent=2)
    print(f"\n[val] FROZEN: {frozen['key']}")
    print(f"[val]   train sharpe {frozen['train']['sharpe']:.2f} "
          f"cagr {frozen['train']['cagr']:.1%} dd {frozen['train']['max_dd']:.1%}")
    print(f"[val]   valid sharpe {frozen['valid']['sharpe']:.2f} "
          f"cagr {frozen['valid']['cagr']:.1%} dd {frozen['valid']['max_dd']:.1%}")


if __name__ == "__main__":
    main()

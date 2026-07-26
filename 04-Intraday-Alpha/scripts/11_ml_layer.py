"""Walk-forward ML sizing on top of the frozen base configuration.

Run AFTER ``10_run_study.py``.  Reads the frozen config, regenerates its trades
over the whole sample, scores every trade with models fitted only on earlier
trades, and compares three portfolios on each window:

    base     - every trade at 1x risk weight
    model    - risk weight scaled by the walk-forward score rank (0.5x-1.5x)
    shuffle  - the same multipliers randomly permuted (the control)

Decision rule, stated before running: the model earns a place only if it beats
BOTH base and shuffle on validation, and the verdict that matters is then its
test-window comparison, reported either way.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/04-Intraday-Alpha")

import numpy as np
import pandas as pd

from intraday import config, ingest, ml, study
from intraday.config import IntradayParams

pd.set_option("display.width", 220)


def portfolio(t: pd.DataFrame, mult: np.ndarray | None = None) -> dict:
    t = t.copy()
    if mult is not None:
        t["stop_frac"] = t["stop_frac"] / np.where(mult > 0, mult, 1.0)
        # scaling risk weight w = risk/stop_frac by m == dividing stop_frac by m
    dr = study.daily_returns(t)
    return study.metrics(dr)


def main():
    with open(os.path.join(config.RESULTS_DIR, "frozen10.json")) as fh:
        frozen = json.load(fh)
    if not frozen.get("survived"):
        print("[ml] base study did not survive validation - nothing to size. "
              "Running the layer anyway on the best train config would be "
              "selection on noise; refusing.")
        return
    p = IntradayParams(**frozen["params"])
    print(f"[ml] frozen base config: {frozen['params']}")

    panel = ingest.load_panel()
    book = study.Book(panel)
    t = book.trades(p)          # full sample
    t = ml.attach_features(t, book)
    print(f"[ml] {len(t):,} trades in full sample, "
          f"{t[ml.FEATURES].notna().all(axis=1).mean():.1%} fully featured")

    scores = ml.walk_forward_scores(t)
    mult = ml.size_multiplier(scores)
    rng = np.random.default_rng(0)
    mult_shuffled = mult.copy()
    scored = np.isfinite(scores)
    mult_shuffled[scored] = rng.permutation(mult[scored])

    print(f"[ml] scored trades: {scored.sum():,} of {len(t):,} "
          f"(the rest ride at 1x)")

    spans = {"train": config.TRAIN, "valid": config.VALID, "test": config.TEST}
    rows = []
    for name, (lo, hi) in spans.items():
        m = (t["date"] >= lo) & (t["date"] <= hi)
        sub = t[m].reset_index(drop=True)
        if len(sub) < 50:
            continue
        for label, mm in (("base 1x", None),
                          ("model sized", mult[m.to_numpy()]),
                          ("shuffled sized", mult_shuffled[m.to_numpy()])):
            met = portfolio(sub, mm)
            rows.append({"window": name, "policy": label, **{k: round(v, 4)
                         for k, v in met.items()}})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_parquet(os.path.join(config.RESULTS_DIR, "ml_layer10.parquet"),
                   index=False)

    # rank IC per window: does the score predict the return at all?
    print("\n[ml] rank IC (Spearman score vs net return):")
    for name, (lo, hi) in spans.items():
        m = (t["date"] >= lo) & (t["date"] <= hi) & scored
        if m.sum() < 100:
            continue
        sub = t[m]
        ic = pd.Series(scores[m.to_numpy()]).corr(
            sub["ret"].reset_index(drop=True), method="spearman")
        print(f"   {name:6s} n={m.sum():5d}  IC={ic:+.3f}")


if __name__ == "__main__":
    main()

"""Step 7: the frozen configuration meets 2017-01-01 .. 2024-03-28, once.

Also produced here, because a single point estimate is not a result:
cost stress, drop-a-class ablation, calendar-year table, vol-target ladder
(the leverage choice a real allocator would make), and the same frozen blend
run on the completely independent snowguru spot panel.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/05-Multi-Asset")

import numpy as np
import pandas as pd

from multiasset import config, portfolio, runner

pd.set_option("display.width", 240)


def wtable(ret: pd.Series) -> pd.DataFrame:
    r = ret.loc["2017-01-01":config.TEST_END]
    out = (1 + r).groupby(r.index.year).prod() - 1
    return (out * 100).round(1).rename("return_%").to_frame()


def main():
    with open(os.path.join(config.RESULTS_DIR, "frozen_ma.json")) as fh:
        frozen = json.load(fh)
    print(f"[test] frozen config: {frozen['key']}")
    print(f"[test] validation said: sharpe {frozen['valid']['sharpe']:.2f} "
          f"cagr {frozen['valid']['cagr']:.1%} dd {frozen['valid']['max_dd']:.1%}")

    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module
    sweep = import_module("20_sweep")
    w = sweep.make_weights(frozen["trend"], frozen["brk"],
                           frozen["carry"], frozen["skew"])
    classes = sweep.CLASS_SETS[frozen["classes"]]

    U = runner.load_futures()
    res = U.evaluate(w, classes=classes)
    r = res["ret"]

    for win, lo, hi in (("train", None, config.TRAIN_END),
                        ("valid", "2010-01-01", config.VALID_END),
                        ("test", "2017-01-01", config.TEST_END)):
        m = portfolio.metrics(r, lo, hi)
        print(f"[test] {win:5s}: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"vol {m['vol']:6.2%}  dd {m['max_dd']:7.2%}  skew {m['skew']:+.2f}")

    print("\n[test] calendar years (test window):")
    print(wtable(r).to_string())

    te = portfolio.metrics(r, "2017-01-01", config.TEST_END)

    print("\n[test] cost stress (test window):")
    for cm in (1.0, 2.0, 4.0):
        rr = U.evaluate(w, classes=classes, cost_mult=cm)["ret"]
        m = portfolio.metrics(rr, "2017-01-01", config.TEST_END)
        print(f"   costs x{cm:.0f}: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}")

    print("\n[test] drop-a-class ablation (test window):")
    meta = U.meta
    all_classes = sorted(meta["asset_class"].unique()) if classes is None else classes
    for drop in all_classes:
        keep = [c for c in (sorted(meta["asset_class"].unique())
                            if classes is None else classes) if c != drop]
        rr = U.evaluate(w, classes=keep)["ret"]
        m = portfolio.metrics(rr, "2017-01-01", config.TEST_END)
        print(f"   -{drop:8s}: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}")

    print("\n[test] vol-target ladder (the leverage dial, test window):")
    for vt in (0.15, 0.25, 0.35, 0.50):
        spec = config.PortfolioSpec(vol_target=vt)
        rr = U.evaluate(w, classes=classes, spec=spec)["ret"]
        m = portfolio.metrics(rr, "2017-01-01", config.TEST_END)
        print(f"   {vt:.0%} target: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"realised vol {m['vol']:6.2%}  dd {m['max_dd']:7.2%}")

    print("\n[test] independent cross-check: same blend on the spot panel "
          "(FX/metals/indices/crypto, 2007-2023):")
    Us = runner.load_sg()
    w_sg = {k: v for k, v in w.items() if k != "carry"}   # spot panel has no carry
    tot = sum(w_sg.values())
    w_sg = {k: v / tot for k, v in w_sg.items()}
    rs = Us.evaluate(w_sg)["ret"]
    for win, lo, hi in (("train", None, config.TRAIN_END),
                        ("valid", "2010-01-01", config.VALID_END),
                        ("test", "2017-01-01", None)):
        m = portfolio.metrics(rs, lo, hi)
        if m:
            print(f"   {win:5s}: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
                  f"dd {m['max_dd']:7.2%}")

    r.rename("ret").to_frame().to_parquet(
        os.path.join(config.RESULTS_DIR, "frozen_returns.parquet"))
    with open(os.path.join(config.RESULTS_DIR, "test_metrics.json"), "w") as fh:
        json.dump(te, fh, indent=2)
    print("\n[test] saved frozen return series and test metrics")


if __name__ == "__main__":
    main()

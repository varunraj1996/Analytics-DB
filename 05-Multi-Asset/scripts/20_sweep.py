"""Step 5: brute-force the signal-blend space on TRAIN (<= 2009) only.

Nothing after 2009-12-31 is printed or ranked here.  The full return series is
saved per config so 21_validate / 22_test can slice later windows without
re-running, but this script's output deliberately stops at the train boundary.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

sys.path.insert(0, "/home/user/Analytics-DB/05-Multi-Asset")

import numpy as np
import pandas as pd

from multiasset import config, portfolio, runner

pd.set_option("display.width", 240)

TREND_SETS = {
    "fast4": ["ewmac2_8", "ewmac4_16", "ewmac8_32", "ewmac16_64"],
    "slow4": ["ewmac8_32", "ewmac16_64", "ewmac32_128", "ewmac64_256"],
    "mid2": ["ewmac16_64", "ewmac32_128"],
    "all6": ["ewmac2_8", "ewmac4_16", "ewmac8_32", "ewmac16_64",
             "ewmac32_128", "ewmac64_256"],
    "none": [],
}
BRK_SETS = {
    "none": [],
    "mid2": ["brk80", "brk160"],
    "all4": ["brk40", "brk80", "brk160", "brk320"],
}
CARRY_W = [0.0, 0.2, 0.4]
SKEW_W = [0.0, 0.1]
CLASS_SETS = {
    "all": None,
    "no_equity": ["Ags", "Bond", "FX", "Metals", "OilGas", "Sector", "Vol"],
    "fin_only": ["Bond", "Equity", "FX", "Vol"],
    "com_only": ["Ags", "Metals", "OilGas"],
}


def make_weights(tset, bset, cw, sw):
    tsigs, bsigs = TREND_SETS[tset], BRK_SETS[bset]
    w: dict[str, float] = {}
    trend_brk_w = 1.0 - cw - sw
    if not tsigs and not bsigs:
        if trend_brk_w > 1e-9:
            return None                    # weight with nothing to give it to
    n = len(tsigs) + len(bsigs)
    for s in tsigs + bsigs:
        w[s] = trend_brk_w / n
    if cw > 0:
        w["carry"] = cw
    if sw > 0:
        w["skew"] = sw
    return w


def main():
    U = runner.load_futures()
    print(f"[sweep] {U.price.shape[1]} instruments, "
          f"{U.price.shape[0]} days, train cut {config.TRAIN_END}")

    grid = [(t, b, c, s, k)
            for t, b, c, s, k in itertools.product(
                TREND_SETS, BRK_SETS, CARRY_W, SKEW_W, CLASS_SETS)
            if make_weights(t, b, c, s) is not None]
    print(f"[sweep] {len(grid)} configurations")

    os.makedirs(os.path.join(config.RESULTS_DIR, "rets"), exist_ok=True)
    rows = []
    t0 = time.time()
    for i, (t, b, c, s, k) in enumerate(grid):
        w = make_weights(t, b, c, s)
        res = U.evaluate(w, classes=CLASS_SETS[k])
        key = f"{t}|{b}|c{c}|s{s}|{k}"
        res["ret"].rename("ret").to_frame().to_parquet(
            os.path.join(config.RESULTS_DIR, "rets", f"{i:04d}.parquet"))
        m = portfolio.metrics(res["ret"], None, config.TRAIN_END)
        cost_ann = float((res["gross_ret"] - res["ret"])
                         .loc[:config.TRAIN_END].mean() * 252)
        rows.append({"cfg": i, "key": key, "trend": t, "brk": b, "carry": c,
                     "skew": s, "classes": k, **{f"tr_{kk}": vv
                                                 for kk, vv in m.items()},
                     "tr_cost_ann": cost_ann})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[sweep]  {i + 1}/{len(grid)}  {el / 60:.1f}m  "
                  f"eta {el / (i + 1) * (len(grid) - i - 1) / 60:.0f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(config.RESULTS_DIR, "sweep_train.parquet"),
                  index=False)
    print(f"[sweep] done in {(time.time() - t0) / 60:.1f}m")
    cols = ["key", "tr_cagr", "tr_sharpe", "tr_vol", "tr_max_dd", "tr_cost_ann"]
    print("\n[sweep] top 20 by TRAIN sharpe:")
    print(df.sort_values("tr_sharpe", ascending=False).head(20)[cols]
          .to_string(index=False))
    print(f"\n[sweep] train sharpe distribution: "
          f"{df.tr_sharpe.describe().round(2).to_dict()}")


if __name__ == "__main__":
    main()

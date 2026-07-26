"""The full 10-minute study: sweep on train, select on validation, test once.

The grid is deliberately modest (~1.3k configurations x 21 symbols).  The
daily-bar study demonstrated what a 1.8M-combination search does on a noisy
target: it finds something regardless of whether anything is there.  With a
five-year train window and ~20 symbols, a small grid over *structurally
different* choices (which level, how deep a pullback, which reward shape) is
as far as the data can honestly support.

Selection rule, fixed before looking: highest validation Calmar among configs
with >=150 validation trades, positive train expectancy, positive validation
expectancy, and train max drawdown inside 25%.  The test window is run once,
for the selected configuration only.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

sys.path.insert(0, "/home/user/Analytics-DB/04-Intraday-Alpha")

import numpy as np
import pandas as pd

from intraday import config, ingest, study
from intraday.config import IntradayParams

pd.set_option("display.width", 240)

GRID = dict(
    side=[1, -1],
    level_kind=["or", "pdh"],
    or_bars=[3, 6],
    confirm_atr=[0.0, 0.25],
    pullback_atr=[0.35, 0.7, 1.2],
    max_wait_bars=[6, 12],
    stop_atr=[1.0, 2.0],
    target_atr=[1.5, 3.0, 6.0],
    max_hold_bars=[12, 39],
    trend_filter=["none", "above_pdc"],
)


def combos(grid):
    keys = list(grid)
    for v in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, v))


def main():
    panel = ingest.load_panel()
    print(f"[study] panel {len(panel):,} bars, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    book = study.Book(panel)

    tr_lo, tr_hi = config.TRAIN
    va_lo, va_hi = config.VALID
    te_lo, te_hi = config.TEST

    n_cfg = len(list(combos(GRID)))
    print(f"[study] sweeping {n_cfg} configurations on train {tr_lo}..{tr_hi}")

    rows = []
    t0 = time.time()
    for i, g in enumerate(combos(GRID)):
        p = IntradayParams(**g)
        t = book.trades(p, tr_lo, tr_hi)
        if len(t) < 300:
            continue
        st = study.trade_stats(t)
        dr = study.daily_returns(t)
        m = study.metrics(dr)
        if not m:
            continue
        rows.append({**g, **{f"tr_{k}": v for k, v in st.items()},
                     **{f"trm_{k}": v for k, v in m.items()}})
        if (i + 1) % 100 == 0:
            print(f"[study]   {i + 1}/{n_cfg}  kept {len(rows)}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(config.RESULTS_DIR, "sweep10.parquet"), index=False)
    print(f"[study] train sweep: {len(res)} viable configs "
          f"({time.time() - t0:.0f}s)")
    if len(res) == 0:
        print("[study] nothing viable on train; stopping")
        return

    print("\n[study] top of train sweep by expectancy:")
    cols = list(GRID) + ["tr_n", "tr_bps_net", "tr_win", "trm_cagr",
                         "trm_sharpe", "trm_max_dd"]
    print(res.sort_values("tr_bps_net", ascending=False).head(12)[cols]
          .to_string(index=False))

    # ---------------- validation of the train survivors ----------------
    ok = res[(res.tr_bps_net > 0) & (res.trm_max_dd > -0.25) & (res.tr_n >= 300)]
    ok = ok.sort_values("tr_bps_net", ascending=False).head(60)
    print(f"\n[study] {len(ok)} train survivors -> validation {va_lo}..{va_hi}")

    vrows = []
    for _, r in ok.iterrows():
        g = {k: r[k] for k in GRID}
        g["side"] = int(g["side"])
        for k in ("or_bars", "max_wait_bars", "max_hold_bars"):
            g[k] = int(g[k])
        p = IntradayParams(**g)
        t = book.trades(p, va_lo, va_hi)
        if len(t) < 150:
            continue
        st = study.trade_stats(t)
        m = study.metrics(study.daily_returns(t))
        if not m:
            continue
        calmar = m["cagr"] / abs(m["max_dd"]) if m["max_dd"] < 0 else np.nan
        vrows.append({**g, "tr_bps": r["tr_bps_net"], "tr_cagr": r["trm_cagr"],
                      "tr_dd": r["trm_max_dd"],
                      "va_n": st["n"], "va_bps": st["bps_net"],
                      "va_win": st["win"], "va_cagr": m["cagr"],
                      "va_sharpe": m["sharpe"], "va_dd": m["max_dd"],
                      "va_calmar": calmar})
    vres = pd.DataFrame(vrows)
    vres.to_parquet(os.path.join(config.RESULTS_DIR, "valid10.parquet"), index=False)
    if len(vres) == 0:
        print("[study] no config carries >=150 trades into validation; stopping")
        return
    print("\n[study] validation results (sorted by calmar):")
    print(vres.sort_values("va_calmar", ascending=False).head(15).to_string(index=False))

    sel = vres[(vres.va_bps > 0) & (vres.va_calmar > 0)]
    if len(sel) == 0:
        print("\n[study] NOTHING SURVIVES VALIDATION. "
              "Train->valid rank correlation of expectancy: "
              f"{vres['tr_bps'].corr(vres['va_bps'], method='spearman'):.2f}")
        with open(os.path.join(config.RESULTS_DIR, "frozen10.json"), "w") as fh:
            json.dump({"survived": False}, fh)
        return

    pick = sel.sort_values("va_calmar", ascending=False).iloc[0]
    frozen = {k: (int(pick[k]) if k in ("side", "or_bars", "max_wait_bars",
                                        "max_hold_bars") else
                  (pick[k] if isinstance(pick[k], str) else float(pick[k])))
              for k in GRID}
    print(f"\n[study] SELECTED on validation: {frozen}")
    print(f"[study]   valid: {pick.va_n:.0f} trades, {pick.va_bps:.1f} bps, "
          f"cagr {pick.va_cagr:.1%}, sharpe {pick.va_sharpe:.2f}, dd {pick.va_dd:.1%}")
    with open(os.path.join(config.RESULTS_DIR, "frozen10.json"), "w") as fh:
        json.dump({"survived": True, "params": frozen,
                   "validation": {"bps": float(pick.va_bps),
                                  "cagr": float(pick.va_cagr),
                                  "sharpe": float(pick.va_sharpe),
                                  "dd": float(pick.va_dd)}}, fh, indent=2)

    # ---------------- the single test run ----------------
    print("\n" + "=" * 78)
    print(f"OUT-OF-SAMPLE TEST  {te_lo} .. {te_hi}  (single frozen configuration)")
    print("=" * 78)
    p = IntradayParams(**frozen)
    t = book.trades(p, te_lo, te_hi)
    st = study.trade_stats(t)
    dr = study.daily_returns(t)
    m = study.metrics(dr)
    print("trades :", st)
    print("daily  :", {k: round(v, 4) for k, v in m.items()})
    t.to_parquet(os.path.join(config.RESULTS_DIR, "test_trades10.parquet"), index=False)
    dr.to_frame("ret").to_parquet(os.path.join(config.RESULTS_DIR, "test_daily10.parquet"))

    if m:
        print("\nby symbol (test):")
        print(t.groupby("symbol").agg(n=("ret", "size"),
                                      bps=("ret", lambda x: x.mean() * 1e4),
                                      win=("ret", lambda x: (x > 0).mean()))
              .sort_values("bps").to_string())
        print("\nby month (test):")
        mm = dr.groupby(pd.Grouper(freq="ME")).sum()
        print((mm * 100).round(2).to_string())


if __name__ == "__main__":
    main()

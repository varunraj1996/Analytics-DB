"""Crypto study: sweep on train, select on validation, one frozen test run.

Train <=2021-12-31 | validate 2022-2023 | test 2024-01-01 .. 2026-05-22.
The test window is deliberately the spot-ETF era.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/06-Crypto-Alpha")

import numpy as np
import pandas as pd

from crypto import book, config, ingest, signals as sg

pd.set_option("display.width", 250)

TR, VA, TE = (None, config.TRAIN_END), ("2022-01-01", config.VALID_END), \
             ("2024-01-01", config.TEST_END)


def build_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    W = lambda f: ingest.wide(panel, f)          # noqa: E731
    px = W("price")
    s: dict[str, pd.DataFrame] = {}
    # price, time series
    s["ewmac16"] = sg.ewmac(px, 16, 64)
    s["ewmac32"] = sg.ewmac(px, 32, 128)
    s["brk80"] = sg.breakout(px, 80)
    s["brk160"] = sg.breakout(px, 160)
    # on-chain, time series
    s["mvrv"] = sg.mvrv_value(W("mvrv"))
    s["nvt"] = sg.nvt(W("mktcap"), W("tx_cnt"))
    s["addr"] = sg.address_momentum(W("addr_act"))
    s["hash"] = sg.hash_ribbon(W("hashrate"))
    s["netflow"] = sg.exchange_netflow(W("flow_in"), W("flow_out"), W("mktcap"))
    s["exsply"] = sg.exchange_supply_trend(W("sply_ex"))
    # cross-sectional
    s["xsmom"] = sg.xs_momentum(px, 60)
    s["xsrev"] = sg.xs_reversal(px, 7)
    s["xsmvrv"] = sg.xs_from(s["mvrv"])
    s["xsaddr"] = sg.xs_from(s["addr"])
    return s


def report(name, r, extra=""):
    out = []
    for tag, (lo, hi) in (("train", TR), ("valid", VA), ("test", TE)):
        m = book.metrics(r, lo, hi)
        out.append(f"{tag} {m['cagr']*100:7.1f}%/{m['sharpe']:5.2f}" if m else f"{tag}    n/a")
    print(f"{name:<26} " + "  ".join(out) + extra)


def main():
    panel = ingest.load_panel()
    px = ingest.wide(panel, "price")
    S = build_signals(panel)
    print(f"[study] {px.shape[1]} assets, {px.shape[0]} days, "
          f"{px.index.min().date()} -> {px.index.max().date()}")
    print("[study] signal coverage (fraction of asset-days with a value):")
    print("   " + "  ".join(f"{k}:{v.notna().mean().mean():.2f}" for k, v in S.items()))

    print("\n" + "=" * 96)
    print("SINGLE SIGNALS, dollar-neutral where cross-sectional  (CAGR/Sharpe)")
    print("=" * 96)
    for k in S:
        r = book.build(px, S[k])["ret"]
        report(k, r)

    # benchmark
    btc = px["BTC"].pct_change()
    report("--- BTC buy & hold", btc)
    eq = px.pct_change().mean(axis=1)
    report("--- equal-weight basket", eq)

    # ---------------- sweep blends on TRAIN only ----------------
    FAMS = {
        "trend": ["ewmac16", "ewmac32", "brk80", "brk160"],
        "onchain": ["mvrv", "nvt", "addr", "hash"],
        "flow": ["netflow", "exsply"],
        "xs": ["xsmom", "xsrev", "xsmvrv", "xsaddr"],
    }
    grid = []
    for wt in itertools.product([0, .25, .5], repeat=4):
        if abs(sum(wt) - 1.0) > 1e-9:
            continue
        for lo_only in (False, True):
            grid.append((dict(zip(FAMS, wt)), lo_only))
    print(f"\n[study] {len(grid)} blends on train")

    rows = []
    for fam_w, lo_only in grid:
        w = {}
        for fam, fw in fam_w.items():
            if fw <= 0:
                continue
            for s in FAMS[fam]:
                w[s] = fw / len(FAMS[fam])
        fc = sg.combine(S, w)
        spec = config.PortfolioSpec(long_only=lo_only)
        r = book.build(px, fc, spec=spec)["ret"]
        mt = book.metrics(r, *TR)
        if not mt:
            continue
        rows.append({**{f"w_{k}": v for k, v in fam_w.items()}, "long_only": lo_only,
                     "tr_cagr": mt["cagr"], "tr_sharpe": mt["sharpe"],
                     "tr_dd": mt["max_dd"]})
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(config.RESULTS_DIR, "sweep.parquet"), index=False)
    print("\n[study] top 12 on TRAIN:")
    print(df.sort_values("tr_sharpe", ascending=False).head(12).round(3).to_string(index=False))

    # ---------------- validate ----------------
    vr = []
    for _, row in df.sort_values("tr_sharpe", ascending=False).head(30).iterrows():
        fam_w = {k[2:]: row[k] for k in row.index if k.startswith("w_")}
        w = {}
        for fam, fw in fam_w.items():
            if fw <= 0:
                continue
            for s in FAMS[fam]:
                w[s] = fw / len(FAMS[fam])
        spec = config.PortfolioSpec(long_only=bool(row.long_only))
        r = book.build(px, sg.combine(S, w), spec=spec)["ret"]
        mv = book.metrics(r, *VA)
        if not mv:
            continue
        vr.append({**{f"w_{k}": v for k, v in fam_w.items()},
                   "long_only": bool(row.long_only),
                   "tr_sharpe": row.tr_sharpe, "tr_dd": row.tr_dd,
                   "va_cagr": mv["cagr"], "va_sharpe": mv["sharpe"],
                   "va_dd": mv["max_dd"]})
    v = pd.DataFrame(vr)
    print("\n[study] validation of the train top-30:")
    print(v.sort_values("va_sharpe", ascending=False).head(12).round(3).to_string(index=False))

    ok = v[(v.tr_sharpe >= 0.4) & (v.va_sharpe > 0)]
    if len(ok) == 0:
        print("\n[study] NOTHING passes train>=0.4 and positive validation Sharpe.")
        json.dump({"survived": False}, open(os.path.join(config.RESULTS_DIR, "frozen.json"), "w"))
        return
    pick = ok.sort_values("va_sharpe", ascending=False).iloc[0]
    fam_w = {k[2:]: float(pick[k]) for k in pick.index if k.startswith("w_")}
    frozen = {"survived": True, "families": fam_w, "long_only": bool(pick.long_only),
              "train": {"sharpe": float(pick.tr_sharpe)},
              "valid": {"sharpe": float(pick.va_sharpe), "cagr": float(pick.va_cagr)}}
    json.dump(frozen, open(os.path.join(config.RESULTS_DIR, "frozen.json"), "w"), indent=2)
    print(f"\n[study] FROZEN: {fam_w}  long_only={bool(pick.long_only)}")

    # ---------------- the one test run ----------------
    w = {}
    for fam, fw in fam_w.items():
        if fw <= 0:
            continue
        for s in FAMS[fam]:
            w[s] = fw / len(FAMS[fam])
    spec = config.PortfolioSpec(long_only=bool(pick.long_only))
    res = book.build(px, sg.combine(S, w), spec=spec)
    r = res["ret"]
    print("\n" + "=" * 96)
    print(f"OUT-OF-SAMPLE TEST  {TE[0]} .. {TE[1]}   (single frozen run)")
    print("=" * 96)
    for tag, (lo, hi) in (("train", TR), ("valid", VA), ("test", TE)):
        m = book.metrics(r, lo, hi)
        print(f"  {tag:5s}: CAGR {m['cagr']*100:7.2f}%  Sharpe {m['sharpe']:5.2f}  "
              f"vol {m['vol']*100:5.1f}%  DD {m['max_dd']*100:6.1f}%  skew {m['skew']:+.2f}")
    for tag, (lo, hi) in (("train", TR), ("valid", VA), ("test", TE)):
        m = book.metrics(btc, lo, hi)
        print(f"  BTC hodl {tag:5s}: CAGR {m['cagr']*100:7.2f}%  Sharpe {m['sharpe']:5.2f}  "
              f"DD {m['max_dd']*100:6.1f}%")

    rt = r.loc[TE[0]:TE[1]]
    print("\n  calendar years (test):")
    print(((1 + rt).groupby(rt.index.year).prod() - 1).mul(100).round(1).to_string())
    print("\n  cost stress (test):")
    for bps in (10, 20, 40):
        rr = book.build(px, sg.combine(S, w), spec=spec, half_spread_bps=bps)["ret"]
        m = book.metrics(rr, *TE)
        print(f"    {bps:>2} bps/side: CAGR {m['cagr']*100:7.2f}%  Sharpe {m['sharpe']:5.2f}")
    r.to_frame("ret").to_parquet(os.path.join(config.RESULTS_DIR, "frozen_returns.parquet"))


if __name__ == "__main__":
    main()

"""Stage 3: the frozen strategy meets the test window, once.

Everything this script uses - the rule ensemble, the meta-model, the score
floor, the portfolio capacity - was fixed by ``04_validate.py`` before any of
2015-2017 was looked at.  Nothing here is tuned; the only job is to run it and
report what happened, including the robustness checks that would embarrass it
if the edge were fragile.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, data, meta, pipeline, search, strategy, workspace

pd.set_option("display.width", 220)

ALL_KEYS = ["side", "base_len", "breakout_margin", "vol_mult", "trend_ma",
            "regime_ma", "min_close_loc", "max_ext_atr", "min_base_tightness",
            "pullback_atr", "entry_window", "stop_atr", "target_atr",
            "max_hold", "trail_atr"]
INT_KEYS = {"side", "base_len", "trend_ma", "regime_ma", "entry_window", "max_hold"}


def to_params(d: dict) -> config.StrategyParams:
    return config.StrategyParams(**{k: (int(d[k]) if k in INT_KEYS else float(d[k]))
                                    for k in ALL_KEYS})


def yearly(equity: np.ndarray, calendar: np.ndarray, lo: int, hi: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(calendar[lo:hi + 1])
    s = pd.Series(equity[lo:hi + 1], index=idx)
    out = s.resample("YE").last() / s.resample("YE").first() - 1.0
    return out.rename("return").to_frame()


def main():
    ws = workspace.load()
    tr_lo, tr_hi = ws.train
    va_lo, va_hi = ws.valid
    te_lo, te_hi = ws.test
    uni = config.DEFAULT_UNIVERSE

    with open(os.path.join(config.RESULTS_DIR, "frozen_config.json")) as fh:
        frozen = json.load(fh)
    spec = config.PortfolioSpec(**frozen["portfolio"])
    thr = frozen["score_cut"]
    sides = frozen["sides"]
    rank_policy = frozen.get("rank", "model")
    print(f"[test] frozen: {len(frozen['configs'])} rules, sides {sides}, "
          f"keep top {frozen['keep_q']:.0%} by score (cut {thr:+.4f}), "
          f"{spec.max_positions} slots, risk {spec.risk_per_trade:.3%}/trade, "
          f"gross target {spec.gross_target:.1f}x, rank by {rank_policy}")
    print(f"[test] validation said: cagr {frozen['validation']['cagr']:.1%} "
          f"sharpe {frozen['validation']['sharpe']:.2f} "
          f"dd {frozen['validation']['max_dd']:.1%}")

    E = {w: search.Eligible(ws.P, ws.F, uni, ws.member, lo, hi)
         for w, (lo, hi) in (("train", ws.train), ("valid", ws.valid),
                             ("test", ws.test))}

    pools = {}
    for w, (lo, hi) in (("train", ws.train), ("valid", ws.valid), ("test", ws.test)):
        frames = [pipeline.build_trades(ws, to_params(c), uni, lo, hi, E[w])
                  for c in frozen["configs"]]
        pools[w] = pipeline.combine(frames)
        print(f"[test] {w:>5} pooled candidates: {len(pools[w]):,}")

    P, F = ws.P, ws.F
    X = {w: meta.build_matrix(p, P, F) for w, p in pools.items()}

    # ---- headline: model sees the train window only ----------------------
    fit = X["train"][X["train"]["exit_day"] <= tr_hi]
    models = meta.bagged_train(fit, n_models=5)
    print(f"[test] meta-model fitted on {len(fit):,} completed train trades")

    def frozen_filter(w, s):
        return (s >= thr) & X[w]["side"].isin(sides).to_numpy()

    def ranking(w, s, keep):
        """The frozen allocation policy: which candidate wins a scarce slot."""
        if rank_policy == "liquidity":
            return X[w]["addv"].to_numpy()[keep]
        if rank_policy == "random":
            return np.random.default_rng(0).random(int(keep.sum()))
        return s[keep]

    results = {}
    for w, (lo, hi) in (("train", ws.train), ("valid", ws.valid), ("test", ws.test)):
        s = meta.bagged_score(models, X[w])
        keep = frozen_filter(w, s)
        sub = X[w][keep].reset_index(drop=True)
        r = pipeline.run_portfolio_full(ws, sub, score=ranking(w, s, keep),
                                        spec=spec, day_lo=lo, day_hi=hi)
        results[w] = r
        m = r["metrics"]
        print(f"[test] {w:>5}: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"sortino {m['sortino']:5.2f}  dd {m['max_dd']:7.2%}  "
              f"calmar {m['calmar']:5.2f}  taken {m['n_taken']:,}  "
              f"expo {m['avg_exposure']:.2f}  open {m['avg_open']:.1f}")

    te = results["test"]["metrics"]
    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE TEST  " + f"{ws.label(te_lo)} -> {ws.label(te_hi)}")
    print("=" * 78)
    for k in ("cagr", "vol", "sharpe", "sortino", "max_dd", "calmar",
              "total_return", "years", "n_candidates", "n_taken", "take_rate",
              "avg_exposure", "avg_open", "win_rate_taken", "avg_bars",
              "trades_per_year", "final_equity"):
        v = te.get(k)
        print(f"  {k:>16}: {v:,.4f}" if isinstance(v, float) else f"  {k:>16}: {v}")

    print("\n[test] calendar-year returns")
    print(yearly(results["test"]["equity"], ws.calendar, te_lo, te_hi)
          .assign(**{"return": lambda d: (d["return"] * 100).round(2)}).to_string())

    # ---- benchmark -------------------------------------------------------
    spy = data.load_benchmark("spy").set_index("date")["close"]
    spy = spy.reindex(pd.DatetimeIndex(ws.calendar[te_lo:te_hi + 1]), method="ffill")
    b = spy.to_numpy()
    yrs = len(b) / 252
    print(f"\n[test] SPY buy & hold over the same window: "
          f"cagr {(b[-1] / b[0]) ** (1 / yrs) - 1:.2%}, "
          f"max dd {(b / np.maximum.accumulate(b) - 1).min():.2%}")

    # ---- robustness ------------------------------------------------------
    print("\n" + "=" * 78)
    print("ROBUSTNESS (all on the test window, frozen strategy)")
    print("=" * 78)
    s_te = meta.bagged_score(models, X["test"])
    keep = frozen_filter("test", s_te)
    sub = X["test"][keep].reset_index(drop=True)
    sub_s = ranking("test", s_te, keep)

    print("\n-- does the meta-model ranking actually earn its place?")
    rng = np.random.default_rng(0)
    for label, sc in ((f"frozen policy ({rank_policy})", sub_s),
                      ("meta-model score", s_te[keep]),
                      ("random order", rng.random(len(sub))),
                      ("most liquid first", sub["addv"].to_numpy()),
                      ("alphabetical (no score)", None)):
        r = pipeline.run_portfolio_full(ws, sub, score=sc, spec=spec,
                                        day_lo=te_lo, day_hi=te_hi)
        m = r["metrics"]
        tk = r["trades"][r["taken"]]
        print(f"   {label:<24} cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"dd {m['max_dd']:7.2%}  bps/trade {tk['ret'].mean() * 1e4:6.1f}")

    print("\n-- gross exposure target (the levered variant is a disclosed choice)")
    for gross in (1.0, 1.5, 2.0):
        sp = config.PortfolioSpec(
            max_positions=spec.max_positions, risk_per_trade=spec.risk_per_trade,
            max_new_per_day=spec.max_new_per_day, gross_target=gross,
            max_weight=min(0.20, 4.0 * gross / spec.max_positions))
        r = pipeline.run_portfolio_full(ws, sub, score=sub_s, spec=sp,
                                        day_lo=te_lo, day_hi=te_hi)
        m = r["metrics"]
        print(f"   gross {gross:.1f}x: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"dd {m['max_dd']:7.2%}  expo {m['avg_exposure']:.2f}")

    print("\n-- transaction cost sensitivity")
    for mult in (1, 2, 4, 8):
        c = config.CostModel(base_bps=5.0 * mult, impact_coef=5.0 * mult,
                             min_cents_per_share=0.005 * mult)
        r = pipeline.run_portfolio_full(ws, sub, score=sub_s, spec=spec, costs=c,
                                        day_lo=te_lo, day_hi=te_hi)
        m = r["metrics"]
        print(f"   costs x{mult}: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"dd {m['max_dd']:7.2%}")

    print("\n-- participation cap (fraction of a name's ADDV we allow ourselves)")
    for part in (0.02, 0.01, 0.005, 0.002):
        r = pipeline.run_portfolio_full(ws, sub, score=sub_s, spec=spec,
                                        participation=part, day_lo=te_lo, day_hi=te_hi)
        m = r["metrics"]
        print(f"   {part:.3%} of ADDV: cagr {m['cagr']:7.2%}  sharpe {m['sharpe']:5.2f}  "
              f"final ${m['final_equity']:,.0f}")

    print("\n-- capacity: the same strategy run at larger AUM")
    for aum in (1e6, 1e7, 5e7, 2e8, 1e9):
        sp = config.PortfolioSpec(
            starting_equity=aum, max_positions=spec.max_positions,
            risk_per_trade=spec.risk_per_trade,
            max_new_per_day=spec.max_new_per_day, max_weight=spec.max_weight)
        r = pipeline.run_portfolio_full(ws, sub, score=sub_s, spec=sp,
                                        day_lo=te_lo, day_hi=te_hi)
        m = r["metrics"]
        print(f"   ${aum / 1e6:>7,.0f}m: cagr {m['cagr']:7.2%}  "
              f"sharpe {m['sharpe']:5.2f}  expo {m['avg_exposure']:.2f}  "
              f"taken {m['n_taken']:,}")

    print("\n-- how much rides on the best trades")
    pnl = results["test"]["pnl"][results["test"]["taken"]]
    pnl = np.sort(pnl)[::-1]
    tot = pnl.sum()
    for n in (1, 5, 10, 25):
        print(f"   top {n:>2} trades = {pnl[:n].sum() / tot:6.1%} of test P&L")

    print("\n-- liquidity tier of the trades actually taken")
    tk = results["test"]["trades"][results["test"]["taken"]]
    q = pd.cut(tk["addv"] / 1e6, [0, 25, 100, 500, 1e9],
               labels=["$5-25m", "$25-100m", "$100-500m", ">$500m"])
    print(tk.groupby(q, observed=False)
          .agg(n=("ret", "size"), mean_bps=("ret", lambda x: x.mean() * 1e4),
               win=("ret", lambda x: (x > 0).mean())).to_string())

    print("\n-- long vs short contribution")
    tkx = results["test"]["trades"].copy()
    tkx["pnl"] = results["test"]["pnl"]
    tkx = tkx[results["test"]["taken"]]
    print(tkx.groupby("side").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                  win=("ret", lambda x: (x > 0).mean()),
                                  bps=("ret", lambda x: x.mean() * 1e4)).to_string())

    # ---- refit variant ---------------------------------------------------
    print("\n-- variant: meta-model refitted on train+validation (test still clean)")
    fit2 = pd.concat([X["train"][X["train"]["exit_day"] <= tr_hi],
                      X["valid"][X["valid"]["exit_day"] <= va_hi]], ignore_index=True)
    models2 = meta.bagged_train(fit2, n_models=5)
    s2 = meta.bagged_score(models2, X["test"])
    k2 = s2 >= thr
    r2 = pipeline.run_portfolio_full(ws, X["test"][k2].reset_index(drop=True),
                                     score=s2[k2], spec=spec,
                                     day_lo=te_lo, day_hi=te_hi)
    m2 = r2["metrics"]
    print(f"   cagr {m2['cagr']:7.2%}  sharpe {m2['sharpe']:5.2f}  "
          f"dd {m2['max_dd']:7.2%}  taken {m2['n_taken']:,}")

    # ---- save ------------------------------------------------------------
    out = os.path.join(config.RESULTS_DIR, "test_results.json")
    with open(out, "w") as fh:
        json.dump({w: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in results[w]["metrics"].items()}
                   for w in results}, fh, indent=2)
    for w in results:
        np.save(os.path.join(config.RESULTS_DIR, f"equity_{w}.npy"),
                results[w]["equity"])
    for w in results:
        results[w]["trades"].to_parquet(
            os.path.join(config.RESULTS_DIR, f"trades_{w}.parquet"), index=False)
    print(f"\n[test] results -> {out}")


if __name__ == "__main__":
    main()

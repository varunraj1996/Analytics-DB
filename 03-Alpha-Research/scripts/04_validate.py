"""Stage 2: shortlist on TRAIN, then measure on VALIDATION.

Nothing here looks at the test window.  Three questions are answered:

1. Which configurations from the sweep are *stable* rather than lucky?
   Stability is judged by the family a configuration sits in - a rule whose
   neighbours in parameter space all work is worth more than an isolated spike.
2. Does pooling several diverse rules (long and short, different base lengths)
   beat the single best one?
3. Does the meta-model add anything once it is asked to rank candidates it has
   never seen?
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, meta, pipeline, search, workspace

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

MASK_KEYS = ["side", "base_len", "breakout_margin", "vol_mult", "trend_ma",
             "regime_ma", "min_close_loc", "max_ext_atr", "min_base_tightness"]
FILL_KEYS = ["pullback_atr", "entry_window"]
EXIT_KEYS = ["stop_atr", "target_atr", "max_hold", "trail_atr"]
ALL_KEYS = MASK_KEYS + FILL_KEYS + EXIT_KEYS


def shortlist(df: pd.DataFrame, n: int = 24) -> pd.DataFrame:
    """Prefer configurations whose whole family works."""
    d = df[(df.n_trades >= 800) & (df.t_stat >= 4) & (df.cagr > 0.02)].copy()
    if len(d) == 0:
        d = df.sort_values("t_stat", ascending=False).head(n).copy()
        d["family_t"] = d["t_stat"]
        return d

    fam = d.groupby(MASK_KEYS)["t_stat"].median().rename("family_t")
    fam_n = d.groupby(MASK_KEYS)["t_stat"].size().rename("family_n")
    d = d.merge(fam, on=MASK_KEYS).merge(fam_n, on=MASK_KEYS)
    d = d[d.family_n >= 5]

    d["rank_blend"] = (d["t_stat"].rank(pct=True) * 0.4
                       + d["family_t"].rank(pct=True) * 0.35
                       + d["cagr"].rank(pct=True) * 0.25)
    d = d.sort_values("rank_blend", ascending=False)

    # keep the list diverse: at most three variants per mask family, and make
    # sure both sides of the market are represented
    out, seen = [], {}
    for _, row in d.iterrows():
        key = tuple(row[k] for k in MASK_KEYS)
        if seen.get(key, 0) >= 3:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(row)
        if len(out) >= n:
            break
    res = pd.DataFrame(out)
    if (res["side"] == -1).sum() == 0:
        shorts = d[d.side == -1].head(4)
        res = pd.concat([res, shorts], ignore_index=True)
    return res.reset_index(drop=True)


def params_from(row) -> config.StrategyParams:
    return config.StrategyParams(**{k: (int(row[k]) if k in
                                        ("side", "base_len", "trend_ma", "regime_ma",
                                         "entry_window", "max_hold")
                                        else float(row[k])) for k in ALL_KEYS})


def main():
    ws = workspace.load()
    tr_lo, tr_hi = ws.train
    va_lo, va_hi = ws.valid
    uni = config.DEFAULT_UNIVERSE

    src = os.path.join(config.RESULTS_DIR, "train_search.parquet")
    df = pd.read_parquet(src)
    print(f"[val] {len(df):,} sweep results loaded")

    cand = shortlist(df)
    print(f"[val] shortlist of {len(cand)} configurations "
          f"({(cand.side == 1).sum()} long / {(cand.side == -1).sum()} short)")

    # one Eligible per window, reused across configs
    E_tr = search.Eligible(ws.P, ws.F, uni, ws.member, tr_lo, tr_hi)
    E_va = search.Eligible(ws.P, ws.F, uni, ws.member, va_lo, va_hi)

    rows = []
    trades_tr, trades_va, used = [], [], []
    for i, row in cand.iterrows():
        p = params_from(row)
        t_tr = pipeline.build_trades(ws, p, uni, tr_lo, tr_hi, E_tr)
        t_va = pipeline.build_trades(ws, p, uni, va_lo, va_hi, E_va)
        if len(t_va) < 100 or len(t_tr) < 300:
            continue
        trades_tr.append(t_tr)
        trades_va.append(t_va)
        used.append({k: (int(getattr(p, k)) if k in
                         ("side", "base_len", "trend_ma", "regime_ma",
                          "entry_window", "max_hold") else float(getattr(p, k)))
                     for k in ALL_KEYS})

        r_tr = pipeline.run_portfolio_full(ws, t_tr, day_lo=tr_lo, day_hi=tr_hi)
        r_va = pipeline.run_portfolio_full(ws, t_va, day_lo=va_lo, day_hi=va_hi)
        if r_tr is None or r_va is None:
            continue
        rows.append({
            "cfg": i, "side": p.side, "base_len": p.base_len,
            "pull": p.pullback_atr, "win_d": p.entry_window,
            "stop": p.stop_atr, "tgt": p.target_atr, "hold": p.max_hold,
            "trail": p.trail_atr,
            "tr_n": len(t_tr), "tr_bps": t_tr["ret"].mean() * 1e4,
            "tr_cagr": r_tr["metrics"]["cagr"], "tr_sh": r_tr["metrics"]["sharpe"],
            "va_n": len(t_va), "va_bps": t_va["ret"].mean() * 1e4,
            "va_cagr": r_va["metrics"]["cagr"], "va_sh": r_va["metrics"]["sharpe"],
            "va_dd": r_va["metrics"]["max_dd"],
        })
        print(f"[val]  cfg {i:>3} side={p.side:+d} base={p.base_len:>3} "
              f"pull={p.pullback_atr} stop={p.stop_atr} tgt={p.target_atr} "
              f"hold={p.max_hold:>2} | train {r_tr['metrics']['cagr']:6.1%} "
              f"sh {r_tr['metrics']['sharpe']:4.2f} | valid "
              f"{r_va['metrics']['cagr']:6.1%} sh {r_va['metrics']['sharpe']:4.2f}",
              flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(config.RESULTS_DIR, "validation.parquet"), index=False)
    print("\n[val] individual configurations, train vs validation")
    print(res.to_string(index=False))
    if len(res):
        print(f"\n[val] rank correlation train->valid CAGR: "
              f"{res['tr_cagr'].corr(res['va_cagr'], method='spearman'):.2f}")

    # ---------------------------------------------------------------- ensemble
    print("\n" + "=" * 78)
    print("ENSEMBLE of the shortlisted rules, pooled into one candidate stream")
    print("=" * 78)
    pool_tr = pipeline.combine(trades_tr)
    pool_va = pipeline.combine(trades_va)
    print(f"[val] pooled candidates: train {len(pool_tr):,}  valid {len(pool_va):,}")

    for tag, pool, lo, hi in (("train", pool_tr, tr_lo, tr_hi),
                              ("valid", pool_va, va_lo, va_hi)):
        r = pipeline.run_portfolio_full(ws, pool, day_lo=lo, day_hi=hi)
        m = r["metrics"]
        print(f"[val] {tag} ensemble (no model): cagr {m['cagr']:6.1%} "
              f"sharpe {m['sharpe']:4.2f} dd {m['max_dd']:6.1%} "
              f"taken {m['n_taken']:,}/{m['n_candidates']:,} "
              f"expo {m['avg_exposure']:.2f}")

    # ------------------------------------------------------------- meta-model
    print("\n" + "=" * 78)
    print("META-MODEL: rank the pooled candidates")
    print("=" * 78)
    P, F = ws.P, ws.F
    Xtr = meta.build_matrix(pool_tr, P, F)
    Xva = meta.build_matrix(pool_va, P, F)
    # only learn from trades that finished inside the train window
    fit_rows = Xtr[Xtr["exit_day"] <= tr_hi]
    print(f"[val] fitting on {len(fit_rows):,} completed train trades")
    models = meta.bagged_train(fit_rows, n_models=5)

    s_tr = meta.bagged_score(models, Xtr)
    s_va = meta.bagged_score(models, Xva)
    print("\n[val] validation deciles by model score (out of sample):")
    print(meta.decile_report(Xva, s_va).to_string(index=False))
    print("\n[val] top feature gains:")
    print(meta.importance(models[0]).head(12).to_string(index=False))

    # ------------------------------------------------------------------
    # Joint selection on validation: score floor x portfolio capacity.
    # The single biggest constraint in the baseline is idle capital - a dozen
    # slots but only two positions open on an average day - so the number of
    # slots and the risk per trade are selected here alongside the threshold.
    # ------------------------------------------------------------------
    print("\n[val] joint sweep: score floor x portfolio capacity")
    grid = []
    for thr in (0.0, 0.40, 0.45, 0.50, 0.55):
        keep_va = s_va >= thr
        if keep_va.sum() < 300:
            continue
        sub = Xva[keep_va].reset_index(drop=True)
        sub_s = s_va[keep_va]
        for max_pos in (12, 20, 30, 40):
            for risk in (0.0075, 0.015, 0.025, 0.04):
                for max_new in (6, 12, 20):
                    spec = config.PortfolioSpec(
                        max_positions=max_pos, risk_per_trade=risk,
                        max_new_per_day=max_new,
                        max_weight=min(0.20, 4.0 / max_pos))
                    r = pipeline.run_portfolio_full(
                        ws, sub, score=sub_s, spec=spec,
                        day_lo=va_lo, day_hi=va_hi)
                    if r is None:
                        continue
                    m = r["metrics"]
                    grid.append({
                        "thr": thr, "max_pos": max_pos, "risk": risk,
                        "max_new": max_new, "cagr": m["cagr"],
                        "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                        "calmar": m["calmar"], "expo": m["avg_exposure"],
                        "open": m["avg_open"], "taken": m["n_taken"],
                    })
    g = pd.DataFrame(grid)
    g.to_parquet(os.path.join(config.RESULTS_DIR, "valid_portfolio.parquet"),
                 index=False)
    print(g.sort_values("calmar", ascending=False).head(15).to_string(index=False))

    # Selection rule, fixed in advance: highest validation CAGR among the
    # configurations whose validation drawdown stays inside 25%.
    ok = g[g.max_dd > -0.25]
    if len(ok) == 0:
        ok = g
    pick = ok.sort_values("cagr", ascending=False).iloc[0]
    print(f"\n[val] SELECTED: thr {pick.thr:.2f} max_pos {int(pick.max_pos)} "
          f"risk {pick.risk} max_new {int(pick.max_new)} -> "
          f"valid cagr {pick.cagr:.1%} sharpe {pick.sharpe:.2f} "
          f"dd {pick.max_dd:.1%} expo {pick.expo:.2f}")

    frozen = {
        "threshold": float(pick.thr),
        "portfolio": {"max_positions": int(pick.max_pos),
                      "risk_per_trade": float(pick.risk),
                      "max_new_per_day": int(pick.max_new),
                      "max_weight": float(min(0.20, 4.0 / pick.max_pos))},
        "configs": used,
        "validation": {"cagr": float(pick.cagr), "sharpe": float(pick.sharpe),
                       "max_dd": float(pick.max_dd)},
    }
    with open(os.path.join(config.RESULTS_DIR, "frozen_config.json"), "w") as fh:
        json.dump(frozen, fh, indent=2)
    print(f"[val] frozen config written to "
          f"{os.path.join(config.RESULTS_DIR, 'frozen_config.json')}")


if __name__ == "__main__":
    main()

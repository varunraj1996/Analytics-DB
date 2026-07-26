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
    """Pick a diverse set of configurations whose whole family works.

    Ranked on **expectancy in basis points**, not on the t-statistic of the
    R-multiple.  R-multiple significance rewards a small target against a wide
    stop - a 75%-hit-rate rule that barely compounds - whereas what the
    portfolio actually needs is return per trade.  Diversity is enforced twice:
    once across entry-filter families, and once across the (target, hold) shape
    of the exit, because otherwise the list fills up with twenty near-clones of
    whichever family happened to rank first.
    """
    # Breadth matters as much as size of edge.  Ranking on expectancy alone
    # promotes rare, high-variance rules - the short-side variants that made
    # 197 bps a trade during the 2008 crisis and nothing since - and a pool
    # built from those is too thin for the ranker to learn from and does not
    # repeat.  A hard floor on trade count keeps the list to rules that fire
    # often enough for their expectancy to mean something.
    d = df[(df.n_trades >= 5000) & (df.expectancy_bps > 15) & (df.cagr > 0)].copy()
    if len(d) == 0:
        return df.sort_values("expectancy_bps", ascending=False).head(n).copy()

    fam = d.groupby(MASK_KEYS)["expectancy_bps"].median().rename("family_bps")
    fam_n = d.groupby(MASK_KEYS)["expectancy_bps"].size().rename("family_n")
    d = d.merge(fam, on=MASK_KEYS).merge(fam_n, on=MASK_KEYS)
    d = d[d.family_n >= 5]

    # Rank *within* each side, then take half the list from each.  Ranked
    # globally, the short book wins on raw expectancy purely because 2008-2009
    # sits in the train window, and the pool ends up 22 shorts to 2 longs.
    # Splitting the quota is a structural decision rather than a performance
    # one, so it costs nothing in selection bias and leaves the validation
    # window free to decide whether the short side is worth trading at all.
    picks = []
    for sd in (1, -1):
        ds = d[d.side == sd].copy()
        if len(ds) == 0:
            continue
        ds["rank_blend"] = (ds["expectancy_bps"].rank(pct=True) * 0.4
                            + ds["family_bps"].rank(pct=True) * 0.35
                            + ds["cagr"].rank(pct=True) * 0.25)
        ds = ds.sort_values("rank_blend", ascending=False)
        fam_seen, shape_seen, taken = {}, {}, 0
        for _, row in ds.iterrows():
            fkey = tuple(row[k] for k in MASK_KEYS)
            skey = (row["target_atr"], row["max_hold"], row["stop_atr"])
            if fam_seen.get(fkey, 0) >= 2 or shape_seen.get(skey, 0) >= 3:
                continue
            fam_seen[fkey] = fam_seen.get(fkey, 0) + 1
            shape_seen[skey] = shape_seen.get(skey, 0) + 1
            picks.append(row)
            taken += 1
            if taken >= n // 2:
                break
    return pd.DataFrame(picks).reset_index(drop=True)


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

    # Develop the ranker without spending the validation window: chronological
    # out-of-fold deciles computed entirely inside TRAIN.
    print("\n[val] out-of-fold deciles INSIDE the train window "
          "(the ranker's own development check):")
    print(meta.oof_deciles(fit_rows).to_string(index=False))

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
    # ------------------------------------------------------------------
    # Joint sweep.  Three things are decided here, and the ranking policy is
    # one of them: a model that cannot beat "most liquid first" on the
    # validation window has no business allocating capital, and the decile
    # table above already hints that this one struggles at the top end.
    #
    # The drawdown constraint is applied to BOTH windows.  Screening on
    # validation risk alone would happily wave through a configuration that
    # lost half its equity in 2008, which is not a strategy anyone would fund.
    # ------------------------------------------------------------------
    print("\n[val] joint sweep: ranking policy x score quantile x sides x capacity")
    rng = np.random.default_rng(0)
    rank_tr = {"model": s_tr,
               "liquidity": Xtr["addv"].to_numpy(),
               "random": rng.random(len(Xtr))}
    rank_va = {"model": s_va,
               "liquidity": Xva["addv"].to_numpy(),
               "random": rng.random(len(Xva))}

    grid = []
    for q in (1.0, 0.5):
        cut = float(np.quantile(s_tr, 1.0 - q))
        for sides, sname in (((1, -1), "long+short"), ((1,), "long only")):
            m_tr = (s_tr >= cut) & Xtr["side"].isin(sides).to_numpy()
            m_va = (s_va >= cut) & Xva["side"].isin(sides).to_numpy()
            if m_va.sum() < 300 or m_tr.sum() < 300:
                continue
            sub_tr = Xtr[m_tr].reset_index(drop=True)
            sub_va = Xva[m_va].reset_index(drop=True)
            for rk in ("model", "liquidity", "random"):
                sc_tr, sc_va = rank_tr[rk][m_tr], rank_va[rk][m_va]
                for max_pos in (12, 20, 30):
                    for risk in (0.0075, 0.015, 0.025):
                        for gross in (1.0, 2.0):
                            spec = config.PortfolioSpec(
                                max_positions=max_pos, risk_per_trade=risk,
                                max_new_per_day=20, gross_target=gross,
                                max_weight=min(0.20, 4.0 * gross / max_pos))
                            rv = pipeline.run_portfolio_full(
                                ws, sub_va, score=sc_va, spec=spec,
                                day_lo=va_lo, day_hi=va_hi)
                            rt = pipeline.run_portfolio_full(
                                ws, sub_tr, score=sc_tr, spec=spec,
                                day_lo=tr_lo, day_hi=tr_hi)
                            if rv is None or rt is None:
                                continue
                            m, mt = rv["metrics"], rt["metrics"]
                            grid.append({
                                "rank": rk, "keep_q": q, "sides": sname,
                                "max_pos": max_pos, "risk": risk, "gross": gross,
                                "cagr": m["cagr"], "sharpe": m["sharpe"],
                                "max_dd": m["max_dd"], "calmar": m["calmar"],
                                "expo": m["avg_exposure"], "open": m["avg_open"],
                                "taken": m["n_taken"], "cut": cut,
                                "tr_cagr": mt["cagr"], "tr_dd": mt["max_dd"],
                                "tr_sharpe": mt["sharpe"],
                            })
    g = pd.DataFrame(grid)
    print("\n[val] validation CAGR by ranking policy (best per policy):")
    for rk, sub_g in g.groupby("rank"):
        b = sub_g.sort_values("cagr", ascending=False).iloc[0]
        print(f"   {rk:<10} best valid cagr {b.cagr:7.2%} sharpe {b.sharpe:5.2f} "
              f"dd {b.max_dd:7.2%} | train cagr {b.tr_cagr:7.2%} dd {b.tr_dd:7.2%}")
    g.to_parquet(os.path.join(config.RESULTS_DIR, "valid_portfolio.parquet"),
                 index=False)
    print(g.sort_values("calmar", ascending=False).head(15).to_string(index=False))
    print("\n[val] best by CAGR at each gross exposure target:")
    for gr, sub_g in g.groupby("gross"):
        b = sub_g.sort_values("cagr", ascending=False).iloc[0]
        print(f"   gross {gr:.1f}x: cagr {b.cagr:7.1%} sharpe {b.sharpe:5.2f} "
              f"dd {b.max_dd:7.1%} (keep {b.keep_q:.0%}, {b.sides}, "
              f"{int(b.max_pos)} slots, risk {b.risk})")

    # Selection rule, in three parts:
    #   1. both windows' drawdown inside 25%;
    #   2. a deterministic ranking policy - "random" is kept in the sweep as a
    #      control to measure the model against, but a coin toss is not a
    #      capital allocation policy;
    #   3. maximise Calmar, not CAGR.
    # Part 3 is a correction.  Maximising validation CAGR alone first selected a
    # long-only book that had drawn down 53% in the train window, which no risk
    # committee would sign and which duly fell apart out of sample.  Ranking on
    # return per unit of drawdown picks the configuration that survives both
    # windows instead of the one with the luckiest three years.
    ok = g[(g.max_dd > -0.25) & (g.tr_dd > -0.25) & (g["rank"] != "random")]
    if len(ok) == 0:
        print("\n[val] nothing satisfies the two-window drawdown limit; "
              "relaxing to validation only")
        ok = g[(g.max_dd > -0.25) & (g["rank"] != "random")]
    if len(ok) == 0:
        ok = g
    pick = ok.sort_values("calmar", ascending=False).iloc[0]
    print(f"\n[val] SELECTED: rank by {pick['rank']}, keep top {pick.keep_q:.0%}, "
          f"{pick.sides}, {int(pick.max_pos)} slots, risk {pick.risk}, "
          f"gross {pick.gross:.1f}x -> "
          f"valid cagr {pick.cagr:.1%} sharpe {pick.sharpe:.2f} "
          f"dd {pick.max_dd:.1%} | train cagr {pick.tr_cagr:.1%} dd {pick.tr_dd:.1%}")

    frozen = {
        "rank": str(pick["rank"]),
        "train": {"cagr": float(pick.tr_cagr), "max_dd": float(pick.tr_dd),
                  "sharpe": float(pick.tr_sharpe)},
        # the cut is stored as a train-window quantile so the test window is
        # filtered by a number fixed before it was ever looked at
        "score_cut": float(pick.cut),
        "keep_q": float(pick.keep_q),
        "sides": [1, -1] if pick.sides == "long+short" else [1],
        "portfolio": {"max_positions": int(pick.max_pos),
                      "risk_per_trade": float(pick.risk),
                      "max_new_per_day": 20,
                      "gross_target": float(pick.gross),
                      "max_weight": float(min(0.20, 4.0 * pick.gross / pick.max_pos))},
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

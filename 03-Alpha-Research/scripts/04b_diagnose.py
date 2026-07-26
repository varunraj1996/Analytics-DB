"""Why is the validation result so good?  Decompose it before trusting it.

A validation Sharpe above 5 is a claim about the world that almost never
survives contact with reality, so this script takes the result apart:

* capacity vs ranking - how much comes from simply taking more trades, and how
  much from the model choosing which ones;
* outliers - is the P&L a handful of buyouts and gap-ups;
* the ``f_age`` feature - it dominates the model by 5x, and "how long has this
  symbol existed in a panel of companies that all survived to 2017" is exactly
  the shape a survivorship artefact would take;
* the clean universe - the same machinery restricted to point-in-time S&P 500
  members, where bankruptcies are rare and the survivorship channel is weakest.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, meta, pipeline, search, workspace
from alpha.strategy import FEATURE_COLS

pd.set_option("display.width", 220)

ws = workspace.load()
tr_lo, tr_hi = ws.train
va_lo, va_hi = ws.valid
uni = config.DEFAULT_UNIVERSE

frozen = json.load(open(os.path.join(config.RESULTS_DIR, "frozen_config.json")))
cfgs = frozen["configs"]
INT_KEYS = {"side", "base_len", "trend_ma", "regime_ma", "entry_window", "max_hold"}
KEYS = list(cfgs[0].keys())


def to_p(d):
    return config.StrategyParams(**{k: (int(d[k]) if k in INT_KEYS else float(d[k]))
                                    for k in KEYS})


E_tr = search.Eligible(ws.P, ws.F, uni, ws.member, tr_lo, tr_hi)
E_va = search.Eligible(ws.P, ws.F, uni, ws.member, va_lo, va_hi)
pool_tr = pipeline.combine([pipeline.build_trades(ws, to_p(c), uni, tr_lo, tr_hi, E_tr)
                            for c in cfgs])
pool_va = pipeline.combine([pipeline.build_trades(ws, to_p(c), uni, va_lo, va_hi, E_va)
                            for c in cfgs])
X_tr = meta.build_matrix(pool_tr, ws.P, ws.F)
X_va = meta.build_matrix(pool_va, ws.P, ws.F)
fit = X_tr[X_tr["exit_day"] <= tr_hi]

SPEC = config.PortfolioSpec(**frozen["portfolio"])


def run(sub, score, spec=SPEC, lo=va_lo, hi=va_hi, **kw):
    r = pipeline.run_portfolio_full(ws, sub, score=score, spec=spec,
                                    day_lo=lo, day_hi=hi, **kw)
    return r


print("=" * 78)
print("1. CAPACITY vs RANKING (validation window)")
print("=" * 78)
models = meta.bagged_train(fit, n_models=5)
s_va = meta.bagged_score(models, X_va)

for label, spec, sc in [
    ("12 slots,  6/day, arbitrary pick", config.PortfolioSpec(), None),
    ("20 slots, 20/day, arbitrary pick",
     config.PortfolioSpec(max_positions=20, risk_per_trade=0.025,
                          max_new_per_day=20, max_weight=0.2), None),
    ("20 slots, 20/day, random pick",
     config.PortfolioSpec(max_positions=20, risk_per_trade=0.025,
                          max_new_per_day=20, max_weight=0.2),
     np.random.default_rng(0).random(len(X_va))),
    ("20 slots, 20/day, MODEL pick", SPEC, s_va),
]:
    r = run(X_va, sc, spec)
    m = r["metrics"]
    tk = r["trades"][r["taken"]]
    print(f"  {label:<36} cagr {m['cagr']:8.1%}  sharpe {m['sharpe']:5.2f}  "
          f"dd {m['max_dd']:7.1%}  taken {m['n_taken']:>6,}  "
          f"bps/trade {tk['ret'].mean() * 1e4:6.1f}")

print()
print("=" * 78)
print("2. OUTLIERS: is the P&L a few lottery tickets?")
print("=" * 78)
r = run(X_va, s_va)
tk = r["trades"][r["taken"]].copy()
tk["pnl"] = r["pnl"][r["taken"]]
print(f"  taken {len(tk):,}   mean {tk['ret'].mean() * 1e4:.1f} bps   "
      f"median {tk['ret'].median() * 1e4:.1f} bps")
print("  return percentiles:",
      {f"p{p}": f"{np.percentile(tk['ret'], p) * 100:.2f}%"
       for p in (1, 5, 25, 50, 75, 95, 99, 99.9)})
print(f"  max single-trade return: {tk['ret'].max():.1%}  "
      f"min: {tk['ret'].min():.1%}")
srt = np.sort(tk["pnl"].to_numpy())[::-1]
for n in (1, 5, 10, 25, 100):
    print(f"    top {n:>3} trades = {srt[:n].sum() / srt.sum():6.1%} of P&L")
print(f"  P&L with the best 1% of trades removed: "
      f"{srt[int(len(srt) * 0.01):].sum() / srt.sum():.1%} of total")
print("\n  most-traded symbols:")
print(tk.groupby("ticker").agg(n=("pnl", "size"), pnl=("pnl", "sum"))
      .sort_values("pnl", ascending=False).head(8).to_string())

print()
print("=" * 78)
print("3. THE f_age FEATURE")
print("=" * 78)
imp = meta.importance(models[0])
print(imp.head(5).to_string(index=False))
no_age = [c for c in FEATURE_COLS if c != "f_age"]
models_na = meta.bagged_train(fit, n_models=5, feature_cols=no_age)
s_na = meta.bagged_score(models_na, X_va, no_age)
r_na = run(X_va, s_na)
m = r_na["metrics"]
tk_na = r_na["trades"][r_na["taken"]]
print(f"\n  with f_age   : cagr {r['metrics']['cagr']:8.1%}  "
      f"sharpe {r['metrics']['sharpe']:5.2f}  "
      f"bps/trade {tk['ret'].mean() * 1e4:6.1f}")
print(f"  without f_age: cagr {m['cagr']:8.1%}  sharpe {m['sharpe']:5.2f}  "
      f"bps/trade {tk_na['ret'].mean() * 1e4:6.1f}")

print("\n  raw relationship between age and trade return (validation pool):")
q = pd.qcut(X_va["f_age"], 5, labels=False, duplicates="drop")
print(X_va.groupby(q).agg(n=("ret", "size"),
                          bps=("ret", lambda x: x.mean() * 1e4),
                          win=("ret", lambda x: (x > 0).mean())).to_string())

print()
print("=" * 78)
print("4. CLEAN UNIVERSE: point-in-time S&P 500 members only")
print("=" * 78)
uni_sp = config.UniverseSpec(min_addv_usd=5e6, sp500_only=True)
Esp_tr = search.Eligible(ws.P, ws.F, uni_sp, ws.member, tr_lo, tr_hi)
Esp_va = search.Eligible(ws.P, ws.F, uni_sp, ws.member, va_lo, va_hi)
sp_tr = pipeline.combine([pipeline.build_trades(ws, to_p(c), uni_sp, tr_lo, tr_hi, Esp_tr)
                          for c in cfgs])
sp_va = pipeline.combine([pipeline.build_trades(ws, to_p(c), uni_sp, va_lo, va_hi, Esp_va)
                          for c in cfgs])
Xsp_tr = meta.build_matrix(sp_tr, ws.P, ws.F)
Xsp_va = meta.build_matrix(sp_va, ws.P, ws.F)
print(f"  pooled candidates: train {len(Xsp_tr):,}  valid {len(Xsp_va):,}")
msp = meta.bagged_train(Xsp_tr[Xsp_tr["exit_day"] <= tr_hi], n_models=5,
                        feature_cols=no_age)
ssp = meta.bagged_score(msp, Xsp_va, no_age)
rsp = run(Xsp_va, ssp)
m = rsp["metrics"]
tksp = rsp["trades"][rsp["taken"]]
print(f"  S&P500 PIT, no f_age: cagr {m['cagr']:8.1%}  sharpe {m['sharpe']:5.2f}  "
      f"dd {m['max_dd']:7.1%}  taken {m['n_taken']:,}  "
      f"bps/trade {tksp['ret'].mean() * 1e4:6.1f}  expo {m['avg_exposure']:.2f}")

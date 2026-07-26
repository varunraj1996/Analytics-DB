"""How much of the edge is real and how much is an artefact of the data?

Three known weaknesses of the vendor panel are quantified here rather than
hand-waved:

1. SURVIVORSHIP - the file set is "every US stock trading in Nov 2017", so
   companies that went to zero before then are simply absent.  A dip-buying
   strategy is exactly the kind that this flatters.  Measured by checking how
   many point-in-time S&P 500 members we actually hold prices for, year by
   year: the shortfall is the delisted population we are blind to.

2. BACK-ADJUSTMENT GHOSTS - prices are anchored at the last print, so a name
   that later did a 1-for-100 reverse split shows a hundredfold inflated price
   and dollar volume in its early history, and can sail through a liquidity
   screen it never deserved.

3. UNIVERSE SENSITIVITY - the same rule is run on progressively cleaner
   universes.  If the edge only lives in the dirtiest tier, it is not an edge.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import numpy as np
import pandas as pd

from alpha import config, search, strategy, universe, workspace

pd.set_option("display.width", 200)

ws = workspace.load()
P, F = ws.P, ws.F
lo, hi = ws.train


# --------------------------------------------------------------------------
print("=" * 78)
print("1. SURVIVORSHIP: point-in-time S&P 500 members we actually have prices for")
print("=" * 78)
mem = universe.load_membership()
mem["year"] = mem["date"].dt.year
listed = {s for s in P.symbols}
rows = []
for yr, grp in mem.groupby("year"):
    tickers = set()
    for t in grp["tickers"]:
        tickers.update(str(t).split(","))
    have = sum(1 for t in tickers
               if t in listed or t.replace(".", "-") in listed)
    rows.append({"year": yr, "index_names_seen": len(tickers),
                 "with_prices": have, "coverage": have / max(len(tickers), 1)})
cov = pd.DataFrame(rows)
cov = cov[(cov.year >= 2001) & (cov.year <= 2017)]
print(cov.to_string(index=False))
print(f"\nmean coverage 2001-2017: {cov['coverage'].mean():.1%}")
print("The missing fraction is the survivorship hole: names that were in the "
      "index at the time but had disappeared by Nov 2017.")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("2. BACK-ADJUSTMENT GHOSTS: how many 'liquid' rows are inflated micro-caps?")
print("=" * 78)
c = P.close
addv = F["addv21"]
liq = np.isfinite(addv) & (addv >= 5e6) & (c >= 5) & (F["avgvol50"] >= 2e5)
print(f"rows passing the liquidity screen: {liq.sum():,}")
for band, lo_p, hi_p in [("$5-25", 5, 25), ("$25-100", 25, 100),
                         ("$100-250", 100, 250), ("$250-1000", 250, 1000),
                         (">$1000", 1000, 1e9)]:
    sel = liq & (c >= lo_p) & (c < hi_p)
    print(f"  adjusted price {band:>10}: {sel.sum():>10,} rows "
          f"({sel.mean() / max(liq.mean(), 1e-12):6.2%} of screened)")

# A ghost betrays itself by an implausible total decline over the sample.
last_close = {}
for s in range(len(P.starts)):
    last_close[s] = P.close[P.ends[s] - 1]
first_close = {s: P.close[P.starts[s]] for s in range(len(P.starts))}
ratio = np.array([first_close[s] / max(last_close[s], 1e-9)
                  for s in range(len(P.starts))])
print(f"\nsymbols whose adjusted price fell >99% over the sample: "
      f"{(ratio > 100).sum():,} of {len(ratio):,}")
print(f"symbols that fell >99.9%: {(ratio > 1000).sum():,}  "
      "(near-certain reverse-split ghosts)")


# --------------------------------------------------------------------------
print()
print("=" * 78)
print("3. UNIVERSE SENSITIVITY of a representative configuration (train only)")
print("=" * 78)
base = dict(side=1, base_len=20, breakout_margin=0.0, vol_mult=1.5,
            trend_ma=200, regime_ma=200, min_close_loc=0.0, max_ext_atr=99.0,
            min_base_tightness=0.0, pullback_atr=1.0, entry_window=5,
            stop_atr=1.0, target_atr=0.75, max_hold=10, trail_atr=0.0)
p = config.StrategyParams(**base)

TIERS = [
    ("wide      addv>$5m",   config.UniverseSpec()),
    ("mid       addv>$25m",  config.UniverseSpec(min_addv_usd=25e6)),
    ("large     addv>$100m", config.UniverseSpec(min_addv_usd=100e6)),
    ("price>=$10 addv>$25m", config.UniverseSpec(min_addv_usd=25e6, min_price=10.0)),
    ("S&P500 point-in-time", config.UniverseSpec(min_addv_usd=5e6, sp500_only=True)),
]

block_end = P.ends[P.sym_id]
out = []
for name, uni in TIERS:
    E = search.Eligible(P, F, uni, ws.member, lo, hi)
    rows_ = search.mask_rows(E, p, ws.regime[p.regime_ma])
    if len(rows_) < 50:
        out.append({"universe": name, "n_signals": len(rows_)})
        continue
    f_sig, f_row, f_px, f_atr = search.find_fills(
        rows_, block_end[rows_], P.open, P.high, P.low, P.close, F["atr14"],
        np.int64(p.side), float(p.pullback_atr), np.int64(p.entry_window))
    x_row, x_px, _, _, _, keep = search.eval_exits(
        f_sig, f_row, f_px, f_atr, block_end[f_sig], P.open, P.high, P.low,
        P.close, np.int64(p.side), float(p.stop_atr), float(p.target_atr),
        np.int64(p.max_hold), float(p.trail_atr))
    k = keep.astype(bool)
    epx, xpx, atr_k = f_px[k], x_px[k], f_atr[k]
    stop_px = epx - p.side * p.stop_atr * atr_k
    r = p.side * (xpx - epx) / np.abs(epx - stop_px)
    ret = p.side * (xpx / epx - 1.0)
    n = len(r)
    ed = P.day[f_row[k]].astype(np.int64)
    xd = P.day[x_row[k]].astype(np.int64)
    addv_k = F["addv21"][f_sig[k]]
    order = np.lexsort((-addv_k, ed)).astype(np.int64)
    eq, n_taken = search.fast_portfolio(
        order, ed, xd, epx, xpx, stop_px, np.full(n, float(p.side)), addv_k,
        ws.n_days, 1e6, 0.0075, 12, 0.20, 6, 5.0, 5.0, 0.005, 0.01)
    seg = eq[lo:hi + 1]
    yrs = len(seg) / 252
    cagr = (seg[-1] / seg[0]) ** (1 / yrs) - 1
    dr = np.diff(seg) / seg[:-1]
    out.append({
        "universe": name, "n_signals": len(rows_), "n_trades": n,
        "uniq_syms": len(np.unique(P.sym_id[f_sig[k]])),
        "win_rate": (r > 0).mean(), "avg_r": r.mean(),
        "exp_bps": ret.mean() * 1e4,
        "t_stat": r.mean() / r.std(ddof=1) * np.sqrt(n),
        "cagr": cagr, "n_taken": n_taken,
        "sharpe": dr.mean() * 252 / (dr.std(ddof=1) * np.sqrt(252)),
    })

print(pd.DataFrame(out).to_string(index=False))

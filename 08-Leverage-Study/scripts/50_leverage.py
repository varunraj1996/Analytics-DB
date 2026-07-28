"""Can trend-gated leveraged ETFs reach 25% CAGR at a survivable drawdown?

The last untested route to the user's 25% target on a small cash account.
Sharpe on reachable data has a measured ceiling (~0.73 crypto, swing dead),
so the remaining dial is volatility per unit of Sharpe — and leveraged ETFs
are how a cash account buys volatility without margin. These are REAL
instruments from 2009-2010: their prices already contain fee drag, financing
cost and daily-reset decay, so nothing here is synthetic.

Declared BEFORE any result was inspected:
  * windows: train 2010-07-01 → 2015-12-31, validate 2016 → 2019,
    test 2020 → present. (TQQQ lists 2010-02; six months of burn-in for the
    200dma.) Test carries the COVID crash, the 2022 bear and 2024-26.
  * families: QQQ/QLD/TQQQ (1x/2x/3x Nasdaq), SPY/SSO/UPRO (S&P).
  * mechanisms: buy-and-hold per tier, and a trend gate — long when the 1x
    index closes above its N-day MA (N ∈ {100, 200}), else cash at 0%.
    Signal on the 1x close (less noisy than the levered fund), executed at
    the NEXT close. Cash earning 0 understates results in 2022-26; noted.
  * costs: 2 bps per side on gate flips. Expense ratios are already in the
    fund prices.
  * selection: worst-regime rule — maximise min(train, validation) Sharpe,
    train ≥ 0.5 — then ONE reading of the test window for the winner and,
    for context, the full grid's test CAGR/DD ladder (labelled as such).

Run: python3 08-Leverage-Study/scripts/50_leverage.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

EQ = "/home/user/Analytics-DB/market-data/equities"
TR = ("2010-07-01", "2015-12-31")
VA = ("2016-01-01", "2019-12-31")
TE = ("2020-01-01", None)
COST = 2e-4


def load(t: str) -> pd.Series:
    df = pd.read_csv(os.path.join(EQ, f"{t}.csv"), parse_dates=["Date"])
    s = df.set_index("Date")["Adj Close"].astype(float).dropna()
    s.name = t
    return s


def metrics(r: pd.Series, lo, hi) -> dict:
    x = r.loc[lo:hi].dropna()
    if len(x) < 200:
        return {}
    eq = (1 + x).cumprod()
    yrs = len(x) / 252
    vol = x.std(ddof=1) * np.sqrt(252)
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(252),
            "vol": vol,
            "dd": (eq / eq.cummax() - 1).min()}


def gated(fund: pd.Series, index: pd.Series, n: int | None) -> pd.Series:
    r = fund.pct_change()
    if n is None:
        return r
    sig = (index > index.rolling(n).mean()).astype(float)
    pos = sig.reindex(r.index).ffill().shift(1).fillna(0.0)   # next close
    return pos * r - pos.diff().abs().fillna(0.0) * COST


def main():
    fam = {"Nasdaq": [("QQQ", 1, "QQQ"), ("QLD", 2, "QQQ"), ("TQQQ", 3, "QQQ")],
           "S&P": [("SPY", 1, "SPY"), ("SSO", 2, "SPY"), ("UPRO", 3, "SPY")]}
    px = {t: load(t) for t in ("QQQ", "SPY", "TQQQ", "QLD", "SSO", "UPRO")}
    for t, s in px.items():
        print(f"[data] {t}: {len(s):,} days {s.index.min().date()} -> {s.index.max().date()}")

    rows, best = [], None
    for family, members in fam.items():
        for fund, lev, idx in members:
            for n, gname in ((None, "none"), (100, "100dma"), (200, "200dma")):
                r = gated(px[fund], px[idx], n)
                a, b = metrics(r, *TR), metrics(r, *VA)
                if not a or not b:
                    continue
                worst = min(a["sharpe"], b["sharpe"])
                rows.append({"family": family, "fund": fund, "lev": lev,
                             "gate": gname, "tr_sh": a["sharpe"],
                             "tr_cagr": a["cagr"], "tr_dd": a["dd"],
                             "va_sh": b["sharpe"], "va_cagr": b["cagr"],
                             "va_dd": b["dd"], "worst": worst, "_r": r})
                if a["sharpe"] >= 0.5 and (best is None or worst > best["worst"]):
                    best = rows[-1]

    df = pd.DataFrame([{k: v for k, v in row.items() if k != "_r"} for row in rows])
    print("\nGRID — train / validation only (test not consulted):")
    print(df.round(2).to_string(index=False))

    print(f"\nworst-regime rule selects: {best['fund']} ({best['lev']}x) "
          f"gate={best['gate']}  (worst Sharpe {best['worst']:.2f})")

    print("\n" + "=" * 78)
    print("ONE FROZEN TEST READ (2020-01 -> present) for the selected config:")
    m = metrics(best["_r"], *TE)
    print(f"  {best['fund']} {best['gate']}: CAGR {m['cagr']:7.2%}  "
          f"Sharpe {m['sharpe']:5.2f}  vol {m['vol']:5.1%}  DD {m['dd']:7.2%}")

    print("\nCONTEXT LADDER on the same window (grid, labelled as context not selection):")
    print(f"{'fund':>6} {'gate':>7} {'CAGR':>8} {'Sharpe':>7} {'vol':>7} {'maxDD':>8}")
    for row in rows:
        m = metrics(row["_r"], *TE)
        if m:
            print(f"{row['fund']:>6} {row['gate']:>7} {m['cagr']:>8.2%} "
                  f"{m['sharpe']:>7.2f} {m['vol']:>7.1%} {m['dd']:>8.2%}")

    print("\nfull-history check for the selected config (2010-07 -> present):")
    m = metrics(best["_r"], "2010-07-01", None)
    print(f"  CAGR {m['cagr']:7.2%}  Sharpe {m['sharpe']:5.2f}  DD {m['dd']:7.2%}")


if __name__ == "__main__":
    main()

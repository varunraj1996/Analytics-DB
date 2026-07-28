"""Qullamaggie v2, re-tested on the modern panel — the study's missing decade.

Every conclusion in RESULTS.md was drawn from a survivorship-biased panel
that ends 1999-11 → 2017-11. This runs the same code, unchanged, on FNSPID:
2,897 liquid tickers, 2010 → 2023, split-repaired. It contains the 2018
volatility shock, the COVID crash, the 2021 melt-up and the 2022 bear — none
of which the original study could see.

Splits are declared in ``panel.py`` before any result here was looked at:
train ≤2015, validate 2016-2019, test 2020-2023.

Two comparisons matter and both are reported:
  * does the *entry style* still beat chasing? (the one large effect found)
  * does the stack survive validation this time, where before it did not?

Run: python3 scripts/43_modern.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/user/Analytics-DB/07-Swing-Momentum")

import numpy as np
import pandas as pd

from swing import panel as PN
from swing import qulla2 as Q2
from swing import setups as ST

pd.set_option("display.width", 220)


def regime_from_panel(ws, bench: str = "SPY"):
    """Kullamägi's trend/chop tiers, with the benchmark taken from the panel.

    ``qulla2.regime`` reads SPY from the 1999-2017 dump, which does not reach
    this window. Same rule, same causality: tier 2 needs the index above a
    rising 20dma *and* positive breadth; tier 1 needs it above the 50dma.
    """
    P = ws.P
    idx = np.flatnonzero(P.symbols == bench)
    if len(idx) == 0:
        raise SystemExit(f"{bench} not in panel")
    rows = np.flatnonzero(P.sym_id == idx[0])
    s = pd.Series(P.close[rows], index=pd.DatetimeIndex(ws.cal)[P.day[rows]])
    s = s.reindex(pd.DatetimeIndex(ws.cal)).ffill()
    ma20, ma50 = s.rolling(20).mean(), s.rolling(50).mean()
    r10 = pd.Series(ws.breadth["ratio10"], index=s.index)
    trend = (s > ma20) & (ma20 > ma20.shift(5)) & (r10 > 1.2)
    tier = np.where(trend.fillna(False), 2, np.where((s > ma50).fillna(False), 1, 0))
    return tier.astype(np.int8)


def fwd_stats(ws, mask, lo, hi, h=10):
    """Excess over the same-day universe — the survivorship control."""
    P = ws.P
    be = P.ends[P.sym_id]
    ia = np.arange(P.n)
    nxt = np.minimum(ia + 1, be - 1)
    tgt = np.minimum(ia + 1 + h, be - 1)
    fret = np.where((P.open[nxt] > 0) & (nxt < tgt),
                    P.close[tgt] / np.maximum(P.open[nxt], 1e-9) - 1.0, np.nan)
    elig = PN.base_liquidity(ws) & np.isfinite(fret)
    umean = pd.DataFrame({"day": P.day[elig], "r": fret[elig]}).groupby("day")["r"].mean()
    m = mask & np.isfinite(fret) & (P.day >= lo) & (P.day <= hi)
    d = pd.DataFrame({"day": P.day[m], "r": fret[m]})
    e = (d["r"] - d["day"].map(umean)).dropna()
    if len(e) < 30:
        return {"n": len(e)}
    return {"n": len(e), "raw_bps": d["r"].mean() * 1e4, "excess_bps": e.mean() * 1e4,
            "t": e.mean() / e.std(ddof=1) * np.sqrt(len(e))}


def trades(ws, mask, lo, hi, stop_adr=1.5):
    P, F = ws.P, ws.F
    rows = np.flatnonzero(mask & (P.day >= lo) & (P.day <= hi)).astype(np.int64)
    if len(rows) < 30:
        return None
    out = ST.simulate(rows, P.ends[P.sym_id][rows], P.open, P.high, P.low, P.close,
                      F["ma10"], F["adr20"], float(stop_adr), np.int64(3), 0.5,
                      3.0, np.int64(40), True)
    sig, ent, ext, epx, risk, pnl, reason, bars, turn = out
    t = pd.DataFrame({"sig": sig, "entry_day": P.day[ent], "exit_day": P.day[ext],
                      "epx": epx, "risk": risk, "pnl": pnl, "turn": turn, "bars": bars})
    t["r"] = t.pnl / t.risk
    return t


def portfolio(ws, t, tier, lo, hi, risk_frac=0.01, max_pos=10, cost_bps=15.0):
    """Simulate only the trades that ENTER inside [lo, hi].

    Restricting the trade set to the window is what makes the three windows
    independent. Running the whole trade list and slicing the equity curve
    instead lets every window inherit the others' positions, and extending
    the segment to the last exit of *any* trade runs each window to the end
    of the panel — which is how train, validate and test came back with the
    identical -63.19% drawdown and the identical 1,296 fills.
    """
    t = t[(t.entry_day >= lo) & (t.entry_day <= hi)].reset_index(drop=True)
    if len(t) < 30:
        return None
    mult = np.array([0.0, 1.0, 2.0])[tier[t.entry_day.to_numpy()]]
    order = np.argsort(t.entry_day.to_numpy(), kind="stable").astype(np.int64)
    eq, taken = ST.portfolio(order, t.entry_day.to_numpy().astype(np.int64),
                             t.exit_day.to_numpy().astype(np.int64), t.epx.to_numpy(),
                             t.risk.to_numpy(), t.pnl.to_numpy(), t.turn.to_numpy(),
                             ws.n_days, 1e6, float(risk_frac), int(max_pos), 0.25,
                             4, float(cost_bps), mult)
    # run only to the last exit of trades that entered in this window
    hi2 = min(ws.n_days - 1, max(int(hi), int(t.exit_day.max())))
    seg = eq[lo:hi2 + 1]
    if seg[0] <= 0 or len(seg) < 60:
        return None
    r = np.diff(seg) / np.maximum(seg[:-1], 1e-9)
    dd = (seg / np.maximum.accumulate(seg) - 1).min()
    return {"cagr": (max(seg[-1], 1e-9) / seg[0]) ** (252 / len(seg)) - 1,
            "sharpe": r.mean() / max(r.std(ddof=1), 1e-12) * np.sqrt(252),
            "max_dd": dd, "taken": int(taken.sum())}


def main():
    ws = PN.build_modern()
    Q2.extend(ws)
    print(f"[modern] {ws.P.n:,} rows, {len(ws.P.symbols):,} tickers, "
          f"{pd.Timestamp(ws.cal[0]).date()} -> {pd.Timestamp(ws.cal[-1]).date()}")

    tr = ws.span(None, PN.M_TRAIN_END)
    va = ws.span("2016-01-01", PN.M_VALID_END)
    te = ws.span("2020-01-01", None)

    chase = PN.qulla_breakout(ws)              # the version that lost -85 bps
    m0 = Q2.pullback_bounce(ws)                # his actual entry
    rs = Q2.rs_top(ws)
    lab = Q2.clusters(ws)
    lead = Q2.sector_leadership(ws, lab)
    tier = regime_from_panel(ws)
    print(f"[modern] chase {chase.sum():,} | pullback {m0.sum():,} | "
          f"regime trend/neutral/chop {(tier==2).mean():.0%}/{(tier==1).mean():.0%}/"
          f"{(tier==0).mean():.0%}")

    layers = {"chase (first pass)": chase,
              "L0 pullback entry": m0,
              "L1 + RS top decile": m0 & rs,
              "L2 + sector leadership": m0 & rs & lead}

    for name, (lo, hi) in (("TRAIN <=2015", tr), ("VALIDATE 2016-2019", va),
                           ("TEST 2020-2023", te)):
        print("\n" + "=" * 96)
        print(f"{name}: 10-day excess over the same-day universe, and the trade sim")
        print("=" * 96)
        rows = []
        for lbl, m in layers.items():
            fs = fwd_stats(ws, m, lo, hi)
            row = {"layer": lbl, **{f"fwd_{k}": v for k, v in fs.items()}}
            t = trades(ws, m, lo, hi)
            if t is not None:
                row.update({"trades": len(t), "win%": (t.r > 0).mean() * 100,
                            "avg_R": t.r.mean(),
                            "t_R": t.r.mean() / t.r.std(ddof=1) * np.sqrt(len(t))})
            rows.append(row)
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n" + "=" * 96)
    print("PORTFOLIO (1% risk x regime tier, 10 slots, 15 bps/side)")
    print("=" * 96)
    t_all = trades(ws, layers["L2 + sector leadership"], 0, ws.n_days - 1)
    for lbl, (lo, hi) in (("train", tr), ("valid", va), ("test", te)):
        pm = portfolio(ws, t_all, tier, lo, hi)
        if pm:
            print(f"  {lbl:6s} cagr {pm['cagr']:7.2%}  sharpe {pm['sharpe']:5.2f}  "
                  f"dd {pm['max_dd']:7.2%}  taken {pm['taken']:,}")
    print("\n  control — regime gate off (always 1x):")
    flat = np.ones_like(tier)
    for lbl, (lo, hi) in (("train", tr), ("valid", va), ("test", te)):
        pm = portfolio(ws, t_all, flat, lo, hi)
        if pm:
            print(f"  {lbl:6s} cagr {pm['cagr']:7.2%}  sharpe {pm['sharpe']:5.2f}  "
                  f"dd {pm['max_dd']:7.2%}")


if __name__ == "__main__":
    main()

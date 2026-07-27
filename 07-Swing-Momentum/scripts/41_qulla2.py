"""Qullamaggie v2: ablation of every ingredient, then the gated portfolio.

Layer order mirrors how he describes his own process:
    L0  pullback-bounce entry alone
    L1  + relative strength (top decile 3-month RS)
    L2  + sector rotation (top-quartile correlation-cluster momentum)
    L3  + regime gate (entries only when SPY trends; sized up in strong tape)

For each layer: raw 10d forward drift, excess over the same-day universe
(the survivorship control), full trade simulation, and the portfolio.
Train <=2009 for the ablation; 2010-2013 validation; 2014-2017 test reported
once at the end for the chosen layer stack.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/user/Analytics-DB/07-Swing-Momentum")

import numpy as np
import pandas as pd

from swing import panel as PN
from swing import qulla2 as Q2
from swing import setups as ST

pd.set_option("display.width", 250)


def fwd_stats(ws, mask, lo, hi, h=10):
    P = ws.P
    be = P.ends[P.sym_id]
    ia = np.arange(P.n)
    nxt = np.minimum(ia + 1, be - 1)
    tgt = np.minimum(ia + 1 + h, be - 1)
    fret = np.where((P.open[nxt] > 0) & (nxt < tgt),
                    P.close[tgt] / np.maximum(P.open[nxt], 1e-9) - 1.0, np.nan)
    elig = PN.base_liquidity(ws) & np.isfinite(fret)
    uni = pd.DataFrame({"day": P.day[elig], "r": fret[elig]})
    umean = uni.groupby("day")["r"].mean()

    m = mask & np.isfinite(fret) & (P.day >= lo) & (P.day <= hi)
    d = pd.DataFrame({"day": P.day[m], "r": fret[m]})
    d["ex"] = d["r"] - d["day"].map(umean)
    e = d["ex"].dropna()
    if len(e) < 30:
        return {"n": len(e)}
    return {"n": len(e), "raw_bps": d["r"].mean() * 1e4,
            "excess_bps": e.mean() * 1e4,
            "t": e.mean() / e.std(ddof=1) * np.sqrt(len(e))}


def trades(ws, mask, lo, hi, trail="ma10", partial_days=3, target_r=3.0,
           max_hold=40, stop_adr=1.5):
    P, F = ws.P, ws.F
    m = mask & (P.day >= lo) & (P.day <= hi)
    rows = np.flatnonzero(m).astype(np.int64)
    if len(rows) < 30:
        return None
    be = P.ends[P.sym_id][rows]
    out = ST.simulate(rows, be, P.open, P.high, P.low, P.close, F[trail],
                      F["adr20"], float(stop_adr), np.int64(partial_days),
                      0.5, float(target_r), np.int64(max_hold), True)
    sig, ent, ext, epx, risk, pnl, reason, bars, turn = out
    t = pd.DataFrame({"sig": sig, "entry_day": P.day[ent], "exit_day": P.day[ext],
                      "epx": epx, "risk": risk, "pnl": pnl, "turn": turn,
                      "reason": reason, "bars": bars,
                      "ticker": P.symbols[P.sym_id[sig]]})
    t["r"] = t.pnl / t.risk
    return t


def tstats(t):
    return {"trades": len(t), "win%": (t.r > 0).mean() * 100,
            "avg_R": t.r.mean(), "t": t.r.mean() / t.r.std(ddof=1) * np.sqrt(len(t)),
            "bars": t.bars.mean()}


def run_portfolio(ws, t, tier, risk_frac=0.01, max_pos=10, max_new=4,
                  cost_bps=15.0, lo=0, hi=None):
    hi = ws.n_days - 1 if hi is None else hi
    t = t[(t.entry_day >= lo) & (t.entry_day <= hi)].reset_index(drop=True)
    if len(t) == 0:
        return None
    # regime tier at entry -> risk multiplier: 0 in chop, 1x neutral, 2x trend
    mult = np.array([0.0, 1.0, 2.0])[tier[t.entry_day.to_numpy()]]
    order = np.lexsort((-t.r.to_numpy() * 0, t.entry_day.to_numpy())).astype(np.int64)
    eq, taken = ST.portfolio(order, t.entry_day.to_numpy().astype(np.int64),
                             t.exit_day.to_numpy().astype(np.int64),
                             t.epx.to_numpy(), t.risk.to_numpy(),
                             t.pnl.to_numpy(), t.turn.to_numpy(),
                             ws.n_days, 1e6, float(risk_frac), int(max_pos),
                             0.25, int(max_new), float(cost_bps), mult)
    seg = eq[lo:hi + 1]
    r = np.diff(seg) / seg[:-1]
    yrs = len(seg) / 252
    dd = (seg / np.maximum.accumulate(seg) - 1).min()
    return {"cagr": (seg[-1] / seg[0]) ** (1 / yrs) - 1,
            "sharpe": r.mean() / max(r.std(ddof=1), 1e-12) * np.sqrt(252),
            "max_dd": dd, "taken": int(taken.sum()), "n_cand": len(t)}


def main():
    ws = PN.build()
    Q2.extend(ws)
    tr = ws.span(None, PN.TRAIN_END)
    va = ws.span("2010-01-01", "2013-12-31")
    te = ws.span("2014-01-01", None)

    print("[q2] building layers...")
    m0 = Q2.pullback_bounce(ws)
    rs = Q2.rs_top(ws)
    lab = Q2.clusters(ws)
    lead = Q2.sector_leadership(ws, lab)
    tier = Q2.regime(ws)
    print(f"[q2] signals: entry {m0.sum():,}  rs {rs.mean():.1%} of rows  "
          f"leadership {lead.mean():.1%}  regime days trend/neutral/chop: "
          f"{(tier == 2).mean():.0%}/{(tier == 1).mean():.0%}/{(tier == 0).mean():.0%}")

    layers = {
        "L0 entry only": m0,
        "L1 + RS top decile": m0 & rs,
        "L2 + sector leadership": m0 & rs & lead,
        "L3 + regime!=chop": m0 & rs & lead & (tier[ws.P.day] > 0),
    }

    print("\n" + "=" * 108)
    print("ABLATION - TRAIN (<=2009): 10d forward drift and full trade sim")
    print("=" * 108)
    rows = []
    for name, m in layers.items():
        fs = fwd_stats(ws, m, *tr)
        t = trades(ws, m, *tr)
        row = {"layer": name, **{f"fwd_{k}": v for k, v in fs.items()}}
        if t is not None:
            row.update(tstats(t))
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n" + "=" * 108)
    print("SAME ABLATION - VALIDATION (2010-2013)")
    print("=" * 108)
    rows = []
    for name, m in layers.items():
        fs = fwd_stats(ws, m, *va)
        t = trades(ws, m, *va)
        row = {"layer": name, **{f"fwd_{k}": v for k, v in fs.items()}}
        if t is not None:
            row.update(tstats(t))
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ---- portfolio with the full stack, regime-tiered sizing --------------
    print("\n" + "=" * 108)
    print("PORTFOLIO (1% risk x regime multiplier 0/1/2, 10 slots, 15bps/side)")
    print("=" * 108)
    full = layers["L2 + sector leadership"]      # gate applied via sizing
    t_all = trades(ws, full, 0, ws.n_days - 1)
    for name, (lo, hi) in (("train", tr), ("valid", va), ("test", te)):
        pm = run_portfolio(ws, t_all, tier, lo=lo, hi=hi)
        if pm:
            print(f"  {name:5s}: cagr {pm['cagr']:7.2%}  sharpe {pm['sharpe']:5.2f}  "
                  f"dd {pm['max_dd']:7.2%}  taken {pm['taken']:,}/{pm['n_cand']:,}")
    print("\n  control - same portfolio, regime gate OFF (always 1x):")
    flat = np.ones_like(tier)
    for name, (lo, hi) in (("train", tr), ("valid", va), ("test", te)):
        pm = run_portfolio(ws, t_all, flat, lo=lo, hi=hi)
        if pm:
            print(f"  {name:5s}: cagr {pm['cagr']:7.2%}  sharpe {pm['sharpe']:5.2f}  "
                  f"dd {pm['max_dd']:7.2%}  taken {pm['taken']:,}")

    print("\n[q2] TEST-window drift for the full stack (reported once):")
    for name, m in layers.items():
        fs = fwd_stats(ws, m, *te)
        print(f"  {name:26s} n={fs.get('n', 0):>6,}  raw {fs.get('raw_bps', float('nan')):7.1f}  "
              f"excess {fs.get('excess_bps', float('nan')):7.1f} bps  t={fs.get('t', float('nan')):5.2f}")


if __name__ == "__main__":
    main()

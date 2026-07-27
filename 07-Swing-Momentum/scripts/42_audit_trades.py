"""Turn the backtest's own trades into trade cards and let the agent grade them.

The agent was built to police a human's discretionary trades. Pointing it at
the mechanical strategy instead answers a sharper question: **where does my
own implementation stop being Kullamägi's method?** Every trade the v2 stack
takes is rendered as a trade card — real geometry, real stop, real risk, the
regime tier the backtest actually used, and the equity curve the backtest
actually had at that moment — and run through the same guardrails a live
trade would face.

What it cannot check, it says so: the deterministic layer emits INFO for
fields the backtest has no notion of, and those absences are themselves the
finding — they are the parts of the framework the strategy never modelled.

Run: python3 scripts/42_audit_trades.py
"""
from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "/home/user/Analytics-DB/07-Swing-Momentum")

import numpy as np
import pandas as pd

from agent import Policy, TradeCard, review_card
from agent.judge import HeuristicJudge
from swing import panel as PN
from swing import qulla2 as Q2
from swing import setups as ST

RISK_FRAC = 0.01          # the backtest's risk unit
MAX_WEIGHT = 0.25         # …and its per-position notional cap
TIER_MULT = (0.0, 1.0, 2.0)
# the throttle setting the backtest is implicitly running at each tier
TIER_LEVEL = (0, 2, 3)
TIER_REGIME = ("STAND_DOWN", "PROBE", "PRESS")


def pct_rank_by_day(ws, col, elig):
    """Cross-sectional percentile of a feature within the eligible universe."""
    P, F = ws.P, ws.F
    idx = np.flatnonzero(elig & np.isfinite(F[col]))
    d = pd.DataFrame({"day": P.day[idx], "v": F[col][idx]})
    out = np.full(P.n, np.nan)
    out[idx] = d.groupby("day")["v"].rank(pct=True).to_numpy()
    return out


def cluster_rank_by_day(ws, lab, col="mom65"):
    """Percentile of each row's cluster among that day's cluster momentum."""
    P, F = ws.P, ws.F
    ok = (lab >= 0) & np.isfinite(F[col]) & PN.base_liquidity(ws)
    idx = np.flatnonzero(ok)
    d = pd.DataFrame({"day": P.day[idx], "lab": lab[idx], "v": F[col][idx]})
    g = d.groupby(["day", "lab"])["v"].median().rename("gm").reset_index()
    g["rk"] = g.groupby("day")["gm"].rank(pct=True)
    merged = d.merge(g[["day", "lab", "rk"]], on=["day", "lab"], how="left")
    out = np.full(P.n, np.nan)
    out[idx] = merged["rk"].to_numpy()
    return out


def run_book(ws, t, tier):
    """The regime-tiered portfolio: its equity curve, and which candidates it
    actually filled. Auditing signals the book never took would grade trades
    that were never made."""
    mult = np.array(TIER_MULT)[tier[t.entry_day.to_numpy()]]
    order = np.argsort(t.entry_day.to_numpy(), kind="stable").astype(np.int64)
    eq, taken = ST.portfolio(order, t.entry_day.to_numpy().astype(np.int64),
                             t.exit_day.to_numpy().astype(np.int64),
                             t.epx.to_numpy(), t.risk.to_numpy(),
                             t.pnl.to_numpy(), t.turn.to_numpy(),
                             ws.n_days, 1e6, RISK_FRAC, 10,
                             MAX_WEIGHT, 4, 15.0, mult)
    return eq, taken.astype(bool)


def build_cards(ws, t, tier, rs_pct, sec_pct, eq):
    P, F = ws.P, ws.F
    sig = t.sig.to_numpy()
    ed = t.entry_day.to_numpy()
    cal = pd.DatetimeIndex(ws.cal)

    with np.errstate(invalid="ignore", divide="ignore"):
        leg = F["leghigh15"] / np.maximum(F["close75ago"], 1e-9) - 1.0
        depth = 1.0 - P.close / np.maximum(F["leghigh15"], 1e-9)
        dry = F["vol3"] / np.maximum(F["avgvol50"], 1e-9)

    eq20 = np.full(len(eq), np.nan)
    eq20[20:] = eq[20:] / np.maximum(eq[:-20], 1e-9) - 1.0

    cards = []
    for k in range(len(t)):
        i, j = int(sig[k]), int(ed[k])
        tr = int(tier[j])
        mult = TIER_MULT[tr]
        risk_ps = float(t.risk.iloc[k])
        epx = float(t.epx.iloc[k])
        # the backtest's own sizing rule, including its notional cap — the
        # cap binds often, so sizing off risk alone would overstate exposure
        shares = RISK_FRAC * mult * 1e6 / max(risk_ps, 1e-9)
        shares = min(shares, MAX_WEIGHT * 1e6 / max(epx, 1e-9))
        acct_risk = shares * risk_ps / 1e6
        cards.append(TradeCard(
            ticker=str(t.ticker.iloc[k]), action="OPEN",
            date=str(cal[j].date()), setup="pullback_bounce",
            prior_leg_pct=float(leg[i]), pullback_days=int(F["pull_days"][i]),
            base_depth_pct=float(depth[i]),
            ma10=float(F["ma10"][i]), ma20=float(F["ma20"][i]),
            ma50=float(F["ma50"][i]),
            ma10_rising=bool(F["ma10"][i] > F["ma10_5ago"][i]),
            touched_ma=bool(F["ll5"][i] <= F["ma10"][i] * 1.03),
            volume_dryup_ratio=float(dry[i]),
            trigger_close_above_prior_high=True,
            trigger_volume_expansion=float(P.volume[i] /
                                           max(F["prev_vol"][i], 1.0)),
            entry_price=epx, stop_price=float(epx - risk_ps),
            # his rule is the consolidation low; the backtest uses the signal
            # bar's low, so pass the real base low and let the check speak
            consolidation_low=float(F["ll10"][i]),
            adr_pct=float(F["adr20"][i]),
            shares=shares, equity=1e6, baseline_risk_pct=acct_risk or None,
            exposure_level=TIER_LEVEL[tr], regime_call=TIER_REGIME[tr],
            index_above_20dma=bool(tr >= 1),
            breadth_ratio10=float(ws.breadth["ratio10"][j])
            if np.isfinite(ws.breadth["ratio10"][j]) else None,
            equity_curve_20d_pct=float(eq20[j]) if np.isfinite(eq20[j]) else None,
            rs_percentile=float(rs_pct[i]) if np.isfinite(rs_pct[i]) else None,
            sector_rank_pct=float(sec_pct[i]) if np.isfinite(sec_pct[i]) else None,
            partial_days=3, move_to_breakeven_after_partial=True,
            trail_ma="ma10", addv_usd=float(F["addv21"][i]),
            rationale="mechanical signal: pullback-bounce, RS and sector "
                      "filtered, sized by regime tier",
        ))
    return cards


def main():
    ws = PN.build()
    Q2.extend(ws)
    print("[audit] building layers...")
    m0 = Q2.pullback_bounce(ws)
    rs = Q2.rs_top(ws)
    lab = Q2.clusters(ws)
    lead = Q2.sector_leadership(ws, lab)
    tier = Q2.regime(ws)
    elig = PN.base_liquidity(ws)
    rs_pct = pct_rank_by_day(ws, "mom65", elig)
    sec_pct = cluster_rank_by_day(ws, lab)

    mask = m0 & rs & lead
    P, F = ws.P, ws.F
    rows = np.flatnonzero(mask).astype(np.int64)
    be = P.ends[P.sym_id][rows]
    out = ST.simulate(rows, be, P.open, P.high, P.low, P.close, F["ma10"],
                      F["adr20"], 1.5, np.int64(3), 0.5, 3.0, np.int64(40), True)
    sig, ent, ext, epx, risk, pnl, reason, bars, turn = out
    t = pd.DataFrame({"sig": sig, "entry_day": P.day[ent], "exit_day": P.day[ext],
                      "epx": epx, "risk": risk, "pnl": pnl, "turn": turn,
                      "bars": bars, "ticker": P.symbols[P.sym_id[sig]]})
    print(f"[audit] {len(t):,} candidates from the L2 stack")

    eq, taken = run_book(ws, t, tier)
    t = t[taken].reset_index(drop=True)
    print(f"[audit] {len(t):,} of them actually filled by the book — "
          f"grading those")
    cards = build_cards(ws, t, tier, rs_pct, sec_pct, eq)

    judge = HeuristicJudge()
    policy = Policy()
    reviews = [review_card(c, policy, judge) for c in cards]

    years = pd.DatetimeIndex([c.date for c in cards]).year
    win = np.where(years <= 2009, "train", np.where(years <= 2013, "valid", "test"))

    print("\n" + "=" * 78)
    print("VERDICT DISTRIBUTION — the strategy's own trades vs the rulebook")
    print("=" * 78)
    vd = pd.crosstab(win, [r.verdict for r in reviews], normalize="index")
    print((vd * 100).round(1).to_string())

    print("\n" + "=" * 78)
    print("WHICH RULES IT BREAKS (% of trades, blocking findings)")
    print("=" * 78)
    cnt = Counter(f.code for r in reviews for f in r.findings
                  if f.severity == "VETO")
    fl = Counter(f.code for r in reviews for f in r.findings
                 if f.severity == "FLAG")
    n = len(reviews)
    tab = pd.DataFrame(
        [{"code": c, "veto_%": 100 * cnt.get(c, 0) / n,
          "flag_%": 100 * fl.get(c, 0) / n}
         for c in sorted(set(cnt) | set(fl))]).sort_values("veto_%",
                                                           ascending=False)
    print(tab.round(1).to_string(index=False))

    print("\n" + "=" * 78)
    print("WHAT THE BACKTEST CANNOT EVEN BE ASKED (unverifiable fields)")
    print("=" * 78)
    unk = Counter(f.code for r in reviews for f in r.findings
                  if f.severity == "INFO")
    for c, k in unk.most_common():
        print(f"  {c:28s} {100 * k / n:5.1f}% of trades")
    if not unk:
        print("  (none — every field the guardrails need was present)")

    print("\n" + "=" * 78)
    print("THREE EXAMPLES")
    print("=" * 78)
    shown = 0
    for r in reviews:
        if r.verdict == "VETO" and shown < 2:
            print(r.render() + "\n")
            shown += 1
    for r in reviews:
        if r.verdict == "PASS":
            print(r.render())
            break


if __name__ == "__main__":
    main()

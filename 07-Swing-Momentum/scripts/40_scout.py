"""Do these setups have an edge at all?  Raw per-trade stats, train window only."""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/user/Analytics-DB/07-Swing-Momentum")

import numpy as np
import pandas as pd

from swing import panel as PN
from swing import setups as ST

pd.set_option("display.width", 240)


def run(ws, mask, day_lo, day_hi, trail_ma="ma20", stop_adr_mult=1.0,
        partial_days=4, partial_frac=0.5, target_r=2.0, max_hold=40):
    P, F = ws.P, ws.F
    m = mask & (P.day >= day_lo) & (P.day <= day_hi)
    rows = np.flatnonzero(m).astype(np.int64)
    if len(rows) < 50:
        return None, None
    be = P.ends[P.sym_id][rows]
    out = ST.simulate(rows, be, P.open, P.high, P.low, P.close, F[trail_ma],
                      F["adr20"], float(stop_adr_mult), np.int64(partial_days),
                      float(partial_frac), float(target_r), np.int64(max_hold), True)
    sig, ent, ext, epx, risk, pnl, reason, bars, turn = out
    if len(sig) == 0:
        return None, None
    t = pd.DataFrame({
        "sig": sig, "entry": ent, "exit": ext, "epx": epx, "risk": risk,
        "pnl": pnl, "reason": reason, "bars": bars, "turn": turn,
        "entry_day": P.day[ent], "exit_day": P.day[ext],
        "sym": P.sym_id[sig], "ticker": P.symbols[P.sym_id[sig]],
    })
    t["r"] = t["pnl"] / t["risk"]
    t["ret"] = t["pnl"] / t["epx"]
    return t, len(rows)


def stats(t, n_sig):
    return {
        "signals": n_sig, "trades": len(t), "fill%": len(t) / max(n_sig, 1),
        "win%": (t.r > 0).mean(), "avg_R": t.r.mean(), "med_R": t.r.median(),
        "ret_bps": t.ret.mean() * 1e4, "bars": t.bars.mean(),
        "stop%": (t.reason == 1).mean(), "trail%": (t.reason == 2).mean(),
        "t_stat": t.r.mean() / t.r.std(ddof=1) * np.sqrt(len(t)),
    }


def main():
    ws = PN.build(rebuild="--rebuild" in sys.argv)
    lo, hi = ws.span(None, PN.TRAIN_END)
    print(f"[scout] panel {ws.P.n:,} rows, {len(ws.P.symbols):,} symbols, "
          f"train days {lo}..{hi} ({ws.cal[lo]} -> {ws.cal[hi]})")

    print("\n" + "=" * 104)
    print("RAW SETUP EDGE ON TRAIN (entry next open, ADR-capped stop, "
          "half off at day 4, remainder trailed on the 20dma)")
    print("=" * 104)
    rows = []
    for name, fn in ST.__dict__.items():
        pass
    for name, fn in PN.SETUPS.items():
        mask = fn(ws)
        t, n_sig = run(ws, mask, lo, hi)
        if t is None:
            print(f"{name}: no trades")
            continue
        rows.append({"setup": name, **stats(t, n_sig)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # sensitivity of the flagship setup to its two defining filters
    print("\n" + "=" * 104)
    print("QULLAMAGGIE SETUP: does the prior-leg requirement actually matter?")
    print("=" * 104)
    rows = []
    for leg in (0.0, 0.15, 0.30, 0.50, 0.80):
        mask = PN.qulla_breakout(ws, leg_min=leg)
        t, n_sig = run(ws, mask, lo, hi)
        if t is None:
            continue
        rows.append({"leg_min": leg, **stats(t, n_sig)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nADR STOP CONSTRAINT (skip if the day's low is > k ADRs away):")
    rows = []
    for k in (0.5, 1.0, 1.5, 3.0, 99.0):
        mask = PN.qulla_breakout(ws)
        t, n_sig = run(ws, mask, lo, hi, stop_adr_mult=k)
        if t is None:
            continue
        rows.append({"stop_adr_mult": k, **stats(t, n_sig)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nEXIT STYLE (trailing MA and scale-out timing):")
    rows = []
    for ma in ("ma10", "ma20", "ma50"):
        for pd_ in (3, 5, 10):
            mask = PN.qulla_breakout(ws)
            t, n_sig = run(ws, mask, lo, hi, trail_ma=ma, partial_days=pd_)
            if t is None:
                continue
            rows.append({"trail": ma, "partial_day": pd_, **stats(t, n_sig)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Panel + features + the setup masks for each trader's screen."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")
from alpha import features as A  # noqa: E402  (tested rolling primitives)

SCRATCH = os.environ.get(
    "ALPHA_SCRATCH",
    "/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad",
)
PANEL = os.path.join(SCRATCH, "data", "panel.parquet")
CACHE = os.path.join(SCRATCH, "data", "swing_ws.pkl")
RESULTS = os.path.join(SCRATCH, "results_swing")
os.makedirs(RESULTS, exist_ok=True)

TRAIN_END, VALID_END = "2009-12-31", "2013-12-31"   # test = 2014-01 .. 2017-11


class WS:
    def __init__(self, P, F, breadth):
        self.P, self.F, self.breadth = P, F, breadth
        self.cal = P.calendar
        self.n_days = len(P.calendar)

    def span(self, lo, hi):
        a = 0 if lo is None else int(np.searchsorted(self.cal, np.datetime64(lo), "left"))
        b = self.n_days - 1 if hi is None else int(
            np.searchsorted(self.cal, np.datetime64(hi), "right")) - 1
        return a, b


def _emp(n):
    return np.empty(n, np.float64)


def build(rebuild: bool = False) -> WS:
    if os.path.exists(CACHE) and not rebuild:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    df = pd.read_parquet(PANEL)
    P = A.Panel(df)
    del df
    s, e, n = P.starts, P.ends, P.n
    F: dict[str, np.ndarray] = {}

    for m in (10, 20, 50, 200):
        F[f"ma{m}"] = A._roll_mean(P.close, s, e, m, _emp(n))
    F["atr14"] = A._wilder_atr(P.high, P.low, P.close, s, e, 14, _emp(n))

    # Kullamägi's ADR: mean of (high/low - 1) over 20 sessions, a fraction
    hl = P.high / np.maximum(P.low, 1e-9) - 1.0
    F["adr20"] = A._roll_mean(hl, s, e, 20, _emp(n))

    # consolidation extremes, excluding today (shifted by one bar)
    for w in (5, 10, 15, 20, 30, 40, 60):
        F[f"hh{w}"] = A._shift1(A._roll_max(P.high, s, e, w, _emp(n)), s, e)
        F[f"ll{w}"] = A._shift1(A._roll_min(P.low, s, e, w, _emp(n)), s, e)
    F["ll252"] = A._shift1(A._roll_min(P.low, s, e, 252, _emp(n)), s, e)
    F["hh252"] = A._shift1(A._roll_max(P.high, s, e, 252, _emp(n)), s, e)

    for w in (20, 40, 65, 126):
        F[f"mom{w}"] = A._lag_ratio(P.close, s, e, w)

    F["avgvol50"] = A._roll_mean(P.volume, s, e, 50, _emp(n))
    F["addv21"] = A._roll_mean(P.close * P.volume, s, e, 21, _emp(n))
    F["avgc7"] = A._roll_mean(P.close, s, e, 7, _emp(n))
    F["avgc65"] = A._roll_mean(P.close, s, e, 65, _emp(n))
    F["avgc126"] = A._roll_mean(P.close, s, e, 126, _emp(n))
    F["minv3"] = A._roll_min(P.volume, s, e, 3, _emp(n))
    F["age"] = A._bars_since_start(s, e, _emp(n))

    ret = np.zeros(n)
    A._pct_change(P.close, s, e, ret)
    F["ret1"] = ret
    F["prev_close"] = A._shift1(P.close, s, e)

    # consecutive up days, for Bonde's "not up 3+ in a row" filter
    up = (ret > 0).astype(np.float64)
    F["up_streak"] = _streak(up, s, e)

    breadth = _breadth(P, F, ret)

    ws = WS(P, F, breadth)
    with open(CACHE, "wb") as fh:
        pickle.dump(ws, fh, protocol=4)
    return ws


def _streak(up, starts, ends):
    out = np.zeros(len(up))
    for k in range(len(starts)):
        a, b = starts[k], ends[k]
        c = 0.0
        for i in range(a, b):
            c = c + 1.0 if up[i] > 0 else 0.0
            out[i] = c
    return out


def _breadth(P, F, ret) -> dict[str, np.ndarray]:
    """Bonde's Market Monitor, computed from the whole panel.

    Counts are over every symbol with a print that day, so the denominator
    moves with the panel; ratios and percentages are therefore the meaningful
    series, not the raw counts.
    """
    day = P.day
    nd = len(P.calendar)
    live = np.bincount(day, minlength=nd).astype(float)
    up4 = np.bincount(day, weights=(ret >= 0.04).astype(float), minlength=nd)
    dn4 = np.bincount(day, weights=(ret <= -0.04).astype(float), minlength=nd)
    with np.errstate(invalid="ignore", divide="ignore"):
        above40 = np.bincount(
            day, weights=(P.close > F["ma50"]).astype(float), minlength=nd) / np.maximum(live, 1)
        q_up25 = np.bincount(
            day, weights=(F["mom65"] >= 0.25).astype(float), minlength=nd)

    def roll(x, w):
        c = np.cumsum(np.r_[0.0, x])
        out = np.full(nd, np.nan)
        out[w - 1:] = c[w:] - c[:-w]
        return out

    r5 = roll(up4, 5) / np.maximum(roll(dn4, 5), 1.0)
    r10 = roll(up4, 10) / np.maximum(roll(dn4, 10), 1.0)
    return {"up4": up4, "dn4": dn4, "live": live, "t2108": above40,
            "ratio5": r5, "ratio10": r10, "q_up25": q_up25}


# ---------------------------------------------------------------------------
# setup masks
# ---------------------------------------------------------------------------
def base_liquidity(ws, min_price=5.0, min_addv=3e6, min_age=260):
    P, F = ws.P, ws.F
    return ((P.close >= min_price) & (F["addv21"] >= min_addv)
            & (F["age"] >= min_age) & np.isfinite(F["adr20"]) & (F["adr20"] > 0))


def qulla_breakout(ws, leg_min=0.30, leg_max=5.0, leg_win=65, cons=15,
                   tight_max=0.25, adr_min=0.04, need_ma=True):
    """Big prior move, then a tight base, then a closing breakout of it."""
    P, F = ws.P, ws.F
    with np.errstate(invalid="ignore"):
        leg = np.fmax.reduce([F["mom20"], F["mom40"], F["mom65"]]) if leg_win >= 65 \
            else np.fmax(F["mom20"], F["mom40"])
        hh, ll = F[f"hh{cons}"], F[f"ll{cons}"]
        base_range = (hh - ll) / np.maximum(P.close, 1e-9)
        m = (leg >= leg_min) & (leg <= leg_max)
        m &= P.close > hh                      # closing breakout of the base
        m &= base_range <= tight_max
        m &= F["adr20"] >= adr_min
        if need_ma:
            m &= (P.close > F["ma10"]) & (P.close > F["ma20"]) & (F["ma10"] > F["ma20"])
        m &= base_liquidity(ws)
    return np.nan_to_num(m, nan=False).astype(bool)


def momentum_burst(ws, jump=0.04, cons_lo=3, cons_hi=20, max_streak=2,
                   close_loc=0.6, adr_min=0.0):
    """Bonde's day-one range expansion: +4% on rising volume, closing strong,
    out of a quiet stretch, and not already extended by consecutive up days."""
    P, F = ws.P, ws.F
    with np.errstate(invalid="ignore"):
        rng = np.maximum(P.high - P.low, 1e-9)
        m = (F["ret1"] >= jump)
        m &= P.volume > A._shift1(P.volume, P.starts, P.ends)
        m &= ((P.close - P.low) / rng) >= close_loc
        m &= F["up_streak"] <= max_streak
        base = (F[f"hh{cons_hi}"] - F[f"ll{cons_hi}"]) / np.maximum(P.close, 1e-9)
        m &= base <= 0.35
        if adr_min > 0:
            m &= F["adr20"] >= adr_min
        m &= base_liquidity(ws)
    return np.nan_to_num(m, nan=False).astype(bool)


def episodic_pivot(ws, gap=0.10, vol_mult=3.0, quiet_max=0.20):
    """Gap up on a volume explosion out of a neglected, quiet name - the
    price-only proxy for post-earnings drift (we have no earnings dates)."""
    P, F = ws.P, ws.F
    with np.errstate(invalid="ignore"):
        gap_up = P.open / np.maximum(F["prev_close"], 1e-9) - 1.0
        m = (gap_up >= gap)
        m &= P.volume >= vol_mult * F["avgvol50"]
        m &= np.abs(F["mom65"]) <= quiet_max        # neglected beforehand
        m &= P.close > P.open                        # held the gap
        m &= base_liquidity(ws)
    return np.nan_to_num(m, nan=False).astype(bool)


def double_trouble(ws, ratio=1.8, quiet=0.01):
    """Bonde's anticipation scan: up 80%+ off the 252-day low, sitting still
    today - the calm before the range expansion."""
    P, F = ws.P, ws.F
    with np.errstate(invalid="ignore"):
        m = (P.close / np.maximum(F["ll252"], 1e-9)) >= ratio
        m &= np.abs(F["ret1"]) <= quiet
        m &= F["minv3"] >= 100_000
        m &= base_liquidity(ws)
    return np.nan_to_num(m, nan=False).astype(bool)


SETUPS = {"qulla": qulla_breakout, "burst": momentum_burst,
          "ep": episodic_pivot, "dt": double_trouble}

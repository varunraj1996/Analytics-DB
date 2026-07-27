"""Qullamaggie v2 — the pullback-continuation entry with the layers the first
pass omitted.

The first test bought the *breakout close* at the next open and lost -85 bps
per 10 days to short-term reversal.  That is not his entry.  Per his own
writeups and the followers who document him (Anand Raghavan, Vinod on X), the
bread-and-butter entry is:

    big leg up  ->  orderly pullback to a RISING 10/20dma on drying volume
    ->  higher lows, tightening  ->  buy the first strength day (his ORH),
    stop under the pullback low  ->  scale out, trail the 10/20dma.

Plus two portfolio layers he is explicit about:

    * relative strength / sector rotation - only the strongest names in the
      strongest groups get his money;
    * regime - he "waits during SPY/QQQ chop but unleashes in trend".

Every ingredient here is toggleable so an ablation can price each one.

Data caveat carried from the first pass: the panel is survivorship-biased and
ends 2017-11, so *levels* are optimistic; the controls (excess over the
same-day universe) and the ablation *differences* are the trustworthy part.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")
sys.path.insert(0, "/home/user/Analytics-DB/07-Swing-Momentum")

from numba import njit

from alpha import features as A  # noqa: E402
from swing import panel as PN  # noqa: E402


# ---------------------------------------------------------------------------
# extra features for the pullback structure
# ---------------------------------------------------------------------------
def extend(ws) -> None:
    P, F = ws.P, ws.F
    s, e, n = P.starts, P.ends, P.n
    if "ma10_5ago" in F:
        return
    F["ma10_5ago"] = _lag(F["ma10"], s, e, 5)
    F["ma20_5ago"] = _lag(F["ma20"], s, e, 5)
    F["prev_high"] = A._shift1(P.high, s, e)
    F["prev_vol"] = A._shift1(P.volume, s, e)
    F["avgvol10"] = A._roll_mean(P.volume, s, e, 10, np.empty(n))
    F["vol3"] = A._roll_mean(P.volume, s, e, 3, np.empty(n))
    # leg high: highest close of the prior 15 sessions (excluding today)
    F["leghigh15"] = A._shift1(A._roll_max(P.close, s, e, 15, np.empty(n)), s, e)
    F["close75ago"] = _lagv(P.close, s, e, 75)
    # sessions spent below 99% of the leg high (pullback duration proxy)
    below = np.zeros(n)
    _count_below(P.close, F["leghigh15"], s, e, below)
    F["pull_days"] = below


@njit(cache=True)
def _lag(x, starts, ends, k):
    out = np.full(len(x), np.nan)
    for i in range(len(starts)):
        a, b = starts[i], ends[i]
        if b - a > k:
            out[a + k:b] = x[a:b - k]
    return out


def _lagv(x, starts, ends, k):
    return _lag(x, starts, ends, k)


@njit(cache=True)
def _count_below(close, ref, starts, ends, out):
    for i in range(len(starts)):
        a, b = starts[i], ends[i]
        c = 0.0
        for j in range(a, b):
            r = ref[j]
            if r == r and close[j] < 0.99 * r:
                c += 1.0
            else:
                c = 0.0
            out[j] = c


# ---------------------------------------------------------------------------
# the v2 signal, ingredient by ingredient
# ---------------------------------------------------------------------------
def pullback_bounce(ws, leg_min=0.30, pull_min=3, pull_max=15,
                    depth_max=0.25, need_dryup=True, need_ma=True):
    """Bounce day out of an orderly pullback to the rising 10/20dma."""
    P, F = ws.P, ws.F
    with np.errstate(invalid="ignore", divide="ignore"):
        leg = F["leghigh15"] / np.maximum(F["close75ago"], 1e-9) - 1.0
        depth = 1.0 - P.close / np.maximum(F["leghigh15"], 1e-9)

        m = leg >= leg_min                                  # the prior leg
        m &= (F["pull_days"] >= pull_min) & (F["pull_days"] <= pull_max)
        m &= depth <= depth_max                             # orderly, not broken
        if need_ma:
            m &= F["ma10"] > F["ma10_5ago"]                 # rising 10dma
            m &= P.close > F["ma20"]                        # structure intact
            m &= F["ll5"] <= F["ma10"] * 1.03               # actually touched it
        if need_dryup:
            m &= F["vol3"] < F["avgvol50"]                  # volume dried up
        # the bounce trigger: first strength day
        m &= P.close > F["prev_high"]
        m &= P.volume > F["prev_vol"]
        m &= PN.base_liquidity(ws)
    return np.nan_to_num(m, nan=False).astype(bool)


def rs_top(ws, q=0.90, col="mom65"):
    """Top-decile relative strength across the eligible universe, per day."""
    P, F = ws.P, ws.F
    elig = PN.base_liquidity(ws) & np.isfinite(F[col])
    idx = np.flatnonzero(elig)
    d = pd.DataFrame({"day": P.day[idx], "v": F[col][idx]})
    thr = d.groupby("day")["v"].quantile(q)
    cut = thr.reindex(np.arange(ws.n_days)).to_numpy()
    out = np.zeros(P.n, dtype=bool)
    out[idx] = F[col][idx] >= np.nan_to_num(cut[P.day[idx]], nan=np.inf)
    return out


def clusters(ws, n_clusters=20):
    """Data-driven 'sectors': correlation clusters of trailing-year returns,
    recomputed each January from the PRIOR year only (causal).  The panel has
    no GICS metadata, and Kullamägi trades themes rather than GICS anyway.
    Returns (labels per row, valid mask)."""
    from sklearn.cluster import MiniBatchKMeans
    P, F = ws.P, ws.F
    years = pd.DatetimeIndex(ws.cal).year
    ret = pd.DataFrame({"day": P.day, "sym": P.sym_id, "r": F["ret1"]})
    lab = np.full(P.n, -1, np.int32)

    for y in range(int(years.min()) + 1, int(years.max()) + 1):
        fit_days = np.flatnonzero(years == y - 1)
        use_days = np.flatnonzero(years == y)
        if len(fit_days) < 150 or len(use_days) == 0:
            continue
        sub = ret[(ret.day >= fit_days[0]) & (ret.day <= fit_days[-1])]
        wide = sub.pivot_table(index="day", columns="sym", values="r")
        wide = wide.dropna(axis=1, thresh=int(len(wide) * 0.85)).fillna(0.0)
        if wide.shape[1] < 100:
            continue
        X = wide.to_numpy().T
        X = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-9)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=7, n_init=5,
                             batch_size=1024).fit(X)
        sym2lab = dict(zip(wide.columns.to_numpy(), km.labels_))
        rows = np.flatnonzero((P.day >= use_days[0]) & (P.day <= use_days[-1]))
        ll = np.array([sym2lab.get(s, -1) for s in P.sym_id[rows]], np.int32)
        lab[rows] = ll
    return lab


def sector_leadership(ws, lab, top_frac=0.25, col="mom65"):
    """True where the row's cluster ranks in the top quartile of cluster
    momentum that day (cluster momentum = median member 3-month return)."""
    P, F = ws.P, ws.F
    ok = (lab >= 0) & np.isfinite(F[col]) & PN.base_liquidity(ws)
    idx = np.flatnonzero(ok)
    d = pd.DataFrame({"day": P.day[idx], "lab": lab[idx], "v": F[col][idx]})
    g = d.groupby(["day", "lab"])["v"].median().rename("gm").reset_index()
    g["rk"] = g.groupby("day")["gm"].rank(pct=True)
    good = g[g["rk"] >= 1 - top_frac][["day", "lab"]].assign(ok=True)
    d2 = d.merge(good, on=["day", "lab"], how="left")
    out = np.zeros(P.n, dtype=bool)
    out[idx] = d2["ok"].fillna(False).to_numpy()
    return out


def regime(ws):
    """Kullamägi's stated behaviour: wait in chop, unleash in trend.

    Tiers from index trend + Bonde breadth (both causal):
      2 = trend  : SPY > rising 20dma  AND 10d breakout/breakdown ratio > 1.2
      1 = neutral: SPY > 50dma
      0 = chop   : otherwise            (no new entries at all)
    """
    S = os.environ.get("ALPHA_SCRATCH",
                       "/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad")
    spy = pd.read_csv(f"{S}/kaggle_huge/data/ETFs/spy.us.txt",
                      parse_dates=["Date"]).set_index("Date")["Close"]
    spy = spy.reindex(pd.DatetimeIndex(ws.cal)).ffill()
    ma20 = spy.rolling(20).mean()
    ma50 = spy.rolling(50).mean()
    trend = (spy > ma20) & (ma20 > ma20.shift(5)) & \
            (pd.Series(ws.breadth["ratio10"], index=spy.index) > 1.2)
    neutral = spy > ma50
    tier = np.where(trend.fillna(False), 2, np.where(neutral.fillna(False), 1, 0))
    return tier.astype(np.int8)

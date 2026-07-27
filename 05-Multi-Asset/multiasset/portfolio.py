"""Positions, P&L and costs in currency space.

The accounting identity everything rests on:

    pnl_usd[t] = N[t-1] * point_size * (price[t] - price[t-1]) * fx[t]

with N in contracts (fractional allowed - the study measures the strategy, not
minimum-lot effects, and the robustness section quantifies what integer
rounding does at the chosen capital).  Because P&L is computed in points times
point value, back-adjusted prices that pass through zero (crude 2020, deep
back-adjustments) are handled with no special cases.

Position sizing (Carver's framework, lean):

    N_target = capital * vol_target_daily * w_inst * IDM * (fc / FC_CAP)
               ---------------------------------------------------------
                        point_size * vol_pts * fx

* ``w_inst``: equal weight across asset classes, equal within class, capped.
* ``IDM``: fixed diversification multiplier (no in-sample estimation).
* buffering: trade only when |N_target - N_held| > buffer * |N_scale|, then
  trade to the buffer edge - the standard turnover control.

Costs: |dN| * spread_points * point_size * fx  (spread_points is a half-spread,
i.e. the cost of one execution at market).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .signals import FC_CAP, MIN_WARMUP


def instrument_weights(meta: pd.DataFrame, live: pd.DataFrame,
                       cap: float) -> pd.DataFrame:
    """Date x instrument weights: equal across asset classes present that day,
    equal within class, per-instrument cap, renormalised to sum to 1."""
    klass = meta.set_index("instrument")["asset_class"]
    cols = live.columns
    kmap = klass.reindex(cols)
    out = pd.DataFrame(0.0, index=live.index, columns=cols)
    lv = live.to_numpy(dtype=bool)
    karr = kmap.to_numpy()
    classes = pd.unique(karr)
    class_masks = {c: (karr == c) for c in classes}
    for i in range(len(live.index)):
        row = lv[i]
        if not row.any():
            continue
        present = [c for c in classes if (row & class_masks[c]).any()]
        wk = 1.0 / len(present)
        w = np.zeros(len(cols))
        for c in present:
            m = row & class_masks[c]
            w[m] = wk / m.sum()
        w = np.minimum(w, cap)
        w = w / w.sum()
        out.iloc[i] = w
    return out


def run(price: pd.DataFrame, fx: pd.DataFrame, vol_pts: pd.DataFrame,
        forecast: pd.DataFrame, meta: pd.DataFrame,
        spec: config.PortfolioSpec | None = None,
        idm: float = 2.0, cost_mult: float = 1.0,
        weights: pd.DataFrame | None = None,
        vol_scale: bool = False) -> dict:
    """``vol_scale``: causal portfolio-level vol targeting.  The base system's
    average forecast uses about half its risk budget, so realised vol runs
    well under target; when enabled, positions are multiplied by
    ``clip(vol_target / trailing_realised_vol, 0.5, 3)`` where the trailing
    estimate (EWMA span 63, shifted one day) comes from the *unscaled* return
    stream - no information from the day being scaled."""
    spec = spec or config.DEFAULT_PORTFOLIO

    ps = meta.set_index("instrument")["point_size"].reindex(price.columns)
    sp = meta.set_index("instrument")["spread_points"].reindex(price.columns)

    # a cell is tradable once price, vol and forecast are all warm
    live = price.notna() & vol_pts.notna() & forecast.notna()
    warm = live.rolling(MIN_WARMUP, min_periods=1).sum() >= MIN_WARMUP * 0.8
    live &= warm

    if weights is None:
        weights = instrument_weights(meta, live, spec.instrument_weight_cap)

    vt_daily = spec.vol_target / np.sqrt(252)
    denom = (ps.to_numpy()[None, :] * vol_pts.to_numpy()
             * fx.to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        n_scale = (spec.capital * vt_daily * weights.to_numpy()
                   * min(idm, spec.idm_cap)) / denom
    n_target = n_scale * (forecast.to_numpy() / FC_CAP)
    n_target = np.where(live.to_numpy(), n_target, 0.0)
    n_scale = np.where(live.to_numpy(), np.abs(n_scale), 0.0)

    # ---- buffered positions, sequential over days -------------------------
    T, K = n_target.shape
    held = np.zeros(K)
    N = np.zeros((T, K))
    buf = spec.rebalance_buffer
    for t in range(T):
        tgt = n_target[t]
        width = buf * n_scale[t]
        up = tgt + width
        dn = tgt - width
        over = held > up
        under = held < dn
        held = np.where(over, up, np.where(under, dn, held))
        held = np.where(live.to_numpy()[t], held, 0.0)   # force flat when dead
        N[t] = held

    # ---- P&L --------------------------------------------------------------
    dp = price.diff().fillna(0.0).to_numpy()
    fxv = pd.DataFrame(fx).ffill().fillna(1.0).to_numpy()

    if vol_scale:
        # unscaled return stream, used only to estimate trailing vol
        pnl0 = np.zeros((T, K))
        pnl0[1:] = N[:-1] * ps.to_numpy()[None, :] * dp[1:] * fxv[1:]
        r0 = pd.Series(pnl0.sum(axis=1) / spec.capital, index=price.index)
        trail = (r0.ewm(span=63, min_periods=40).std() * np.sqrt(252)).shift(1)
        mult = (spec.vol_target / trail).clip(0.5, 3.0).fillna(1.0).to_numpy()
        N = N * mult[:, None]

    pnl = np.zeros((T, K))
    pnl[1:] = N[:-1] * ps.to_numpy()[None, :] * dp[1:] * fxv[1:]

    dN = np.abs(np.diff(N, axis=0, prepend=np.zeros((1, K))))
    cost = dN * sp.to_numpy()[None, :] * ps.to_numpy()[None, :] * fxv * cost_mult

    ret = (pnl - cost).sum(axis=1) / spec.capital
    gross_ret = pnl.sum(axis=1) / spec.capital

    out = {
        "ret": pd.Series(ret, index=price.index),
        "gross_ret": pd.Series(gross_ret, index=price.index),
        "cost": pd.Series(cost.sum(axis=1) / spec.capital, index=price.index),
        "positions": pd.DataFrame(N, index=price.index, columns=price.columns),
        "pnl_by_inst": pd.DataFrame(pnl - cost, index=price.index,
                                    columns=price.columns),
        "n_live": live.sum(axis=1),
    }
    return out


def metrics(ret: pd.Series, lo: str | None = None, hi: str | None = None) -> dict:
    r = ret.loc[lo:hi] if (lo or hi) else ret
    r = r.dropna()
    if len(r) < 60:
        return {}
    ann = r.mean() * 252
    vol = r.std(ddof=1) * np.sqrt(252)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "ann_mean": float(ann), "vol": float(vol),
            "sharpe": float(ann / vol) if vol > 0 else 0.0,
            "max_dd": float(dd), "years": float(yrs),
            "skew": float(r.skew()),
            "worst_day": float(r.min()), "best_day": float(r.max())}

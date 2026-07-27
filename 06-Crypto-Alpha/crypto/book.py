"""Portfolio simulation for a spot crypto book.

Simpler than the futures engine because everything is USD spot with no
contract multiplier: a position is a dollar weight, P&L is weight times the
next day's return, and costs are charged on the change in weight.

    w[t]        = clip(fc[t] / CAP, -1, 1) * (vol_target / vol[t]) * inst_w
    pnl[t+1]    = w[t] * ret[t+1]
    cost[t]     = |w[t] - w[t-1]| * half_spread

``fc`` is already lagged inside ``signals``, so ``w[t]`` uses only information
available at the close of day t and earns day t+1's return. Gross exposure is
capped, and an optional portfolio-level causal vol overlay scales the whole
book toward the target using a *trailing* estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build(price: pd.DataFrame, fc: pd.DataFrame,
          spec: config.PortfolioSpec | None = None,
          half_spread_bps: float | None = None,
          vol_scale: bool = True, gross_cap: float = 1.0,
          top_n: int | None = None, rebalance_freq: str | None = None) -> dict:
    """``top_n`` keeps only the N strongest forecasts each day and spreads the
    book across those; ``rebalance_freq`` (a pandas offset such as ``"2W"``)
    holds weights fixed between rebalance dates.

    Both exist for the small-account case. A book that rebalances 18 names
    daily turns over 21x its capital a year, which is free at institutional
    cost and ruinous at retail cost. Concentrating and trading fortnightly
    cuts that by roughly an order of magnitude — see RESULTS.md Addendum.
    """
    spec = spec or config.DEFAULT_PORTFOLIO
    hs = (config.DEFAULT_HALF_SPREAD_BPS if half_spread_bps is None
          else half_spread_bps) / 1e4

    if top_n is not None and top_n < fc.shape[1]:
        # long-only: the strongest forecasts are the largest ones
        fc = fc.where(fc.rank(axis=1, ascending=False) <= top_n)

    ret = price.pct_change()
    av = ret.ewm(span=spec.vol_lookback, min_periods=20).std() * np.sqrt(365)
    live = price.notna() & fc.notna() & av.notna() & (av > 0)

    n_live = live.sum(axis=1).clip(lower=1)
    inst_w = (1.0 / n_live).clip(upper=spec.max_weight)

    raw = (fc / config.FC_CAP).clip(-1, 1)
    w = raw.mul(spec.vol_target, axis=0).div(av).mul(inst_w, axis=0)
    if spec.long_only:
        w = w.clip(lower=0.0)
    w = w.where(live, 0.0).fillna(0.0)

    gross = w.abs().sum(axis=1)
    scale = (gross_cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    w = w.mul(scale, axis=0)

    if vol_scale:
        r0 = (w.shift(1) * ret).sum(axis=1)
        trail = r0.ewm(span=63, min_periods=40).std() * np.sqrt(365)
        mult = (spec.vol_target / trail).clip(0.3, 3.0).shift(1).fillna(1.0)
        w = w.mul(mult, axis=0)

    # rebalance buffer: only move when the target has drifted far enough
    if spec.rebalance_buffer > 0:
        held = np.zeros(w.shape[1])
        out = np.zeros(w.shape)
        tv = w.to_numpy()
        for t in range(len(w)):
            tgt = tv[t]
            width = spec.rebalance_buffer * np.abs(tgt).clip(min=1e-4)
            held = np.where(held > tgt + width, tgt + width,
                            np.where(held < tgt - width, tgt - width, held))
            out[t] = held
        w = pd.DataFrame(out, index=w.index, columns=w.columns)

    if rebalance_freq:
        # weights are only allowed to move on rebalance dates; between them
        # the book is held, which is what makes retail costs survivable
        marks = w.resample(rebalance_freq).last().index.intersection(w.index)
        w = w.where(pd.Series(True, index=marks).reindex(w.index, fill_value=False),
                    np.nan).ffill().fillna(0.0)

    pnl = (w.shift(1) * ret).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = turn * hs
    net = (pnl - cost).fillna(0.0)

    return {"ret": net, "gross_ret": pnl.fillna(0.0), "cost": cost,
            "weights": w, "gross": w.abs().sum(axis=1),
            "turnover": turn, "n_live": live.sum(axis=1)}


def metrics(r: pd.Series, lo: str | None = None, hi: str | None = None,
            ppy: int = 365) -> dict:
    x = r.loc[lo:hi].dropna() if (lo or hi) else r.dropna()
    if len(x) < 90:
        return {}
    ann = x.mean() * ppy
    v = x.std(ddof=1) * np.sqrt(ppy)
    eq = (1 + x).cumprod()
    yrs = len(x) / ppy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "sharpe": float(ann / v) if v > 0 else 0.0,
            "vol": float(v), "max_dd": float(dd), "years": float(yrs),
            "skew": float(x.skew()), "hit": float((x > 0).mean())}

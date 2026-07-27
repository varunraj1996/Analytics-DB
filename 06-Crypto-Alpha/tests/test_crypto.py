"""Leakage and accounting tests. Run: python3 -m pytest 06-Crypto-Alpha/tests -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto import book, config, signals  # noqa: E402


def frame(n=900, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    px = pd.DataFrame(
        {f"C{i}": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.04, n))) for i in range(k)},
        index=idx)
    return px


def test_signals_are_lagged_no_same_day_information():
    """A signal on day T must not move when day T's price changes."""
    px = frame()
    f1 = signals.ewmac(px, 8, 32)
    px2 = px.copy()
    px2.iloc[-1] *= 1.5                       # perturb only the final bar
    f2 = signals.ewmac(px2, 8, 32)
    assert f1.iloc[-1].equals(f2.iloc[-1]), "final forecast saw the final price"


def test_onchain_signal_is_lagged():
    px = frame()
    mv = pd.DataFrame(np.abs(np.random.default_rng(1).normal(2, .5, px.shape)),
                      index=px.index, columns=px.columns)
    f = signals.mvrv_value(mv)
    mv2 = mv.copy(); mv2.iloc[-1] *= 3
    assert f.iloc[-1].equals(signals.mvrv_value(mv2).iloc[-1])


def test_cross_sectional_is_centred():
    px = frame()
    xs = signals.xs_momentum(px, 30)
    m = xs.dropna(how="all").mean(axis=1).abs().max()
    assert m < 1e-9, "cross-sectional tilt should be dollar-neutral by construction"


def test_pnl_uses_next_day_return():
    """Constant full-long forecast: net P&L must equal weight*return shifted."""
    px = frame()
    fc = pd.DataFrame(config.FC_CAP, index=px.index, columns=px.columns)
    res = book.build(px, fc, vol_scale=False,
                     spec=config.PortfolioSpec(rebalance_buffer=0.0),
                     half_spread_bps=0.0)
    direct = (res["weights"].shift(1) * px.pct_change()).sum(axis=1).fillna(0.0)
    assert np.allclose(direct.to_numpy(), res["gross_ret"].to_numpy(), atol=1e-12)


def test_costs_scale_with_turnover():
    px = frame()
    sign = np.where(np.arange(len(px)) % 2 == 0, 2.0, -2.0)
    alt = pd.DataFrame(np.repeat(sign[:, None], px.shape[1], axis=1),
                       index=px.index, columns=px.columns)
    a = book.build(px, alt, half_spread_bps=0.0, vol_scale=False,
                   spec=config.PortfolioSpec(rebalance_buffer=0.0))
    b = book.build(px, alt, half_spread_bps=20.0, vol_scale=False,
                   spec=config.PortfolioSpec(rebalance_buffer=0.0))
    assert b["cost"].sum() > 0
    assert np.isclose(b["cost"].sum(), a["turnover"].sum() * 20e-4, rtol=1e-9)
    assert b["ret"].sum() < a["ret"].sum()


def test_gross_cap_respected():
    px = frame()
    fc = pd.DataFrame(config.FC_CAP, index=px.index, columns=px.columns)
    res = book.build(px, fc, gross_cap=0.5, vol_scale=False,
                     spec=config.PortfolioSpec(rebalance_buffer=0.0))
    assert res["gross"].max() <= 0.5 + 1e-9


def test_long_only_never_shorts():
    px = frame()
    fc = pd.DataFrame(-config.FC_CAP, index=px.index, columns=px.columns)
    res = book.build(px, fc, spec=config.PortfolioSpec(long_only=True,
                                                       rebalance_buffer=0.0))
    assert (res["weights"] >= -1e-12).all().all()

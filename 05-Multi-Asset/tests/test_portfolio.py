"""Accounting tests on synthetic instruments.

Run with:  python3 -m pytest 05-Multi-Asset/tests -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multiasset import config, portfolio, signals  # noqa: E402


def make_frames(prices: dict[str, np.ndarray], point=1.0, spread=0.01):
    idx = pd.bdate_range("2015-01-01", periods=len(next(iter(prices.values()))))
    price = pd.DataFrame({k: v for k, v in prices.items()}, index=idx)
    fx = pd.DataFrame(1.0, index=idx, columns=price.columns)
    meta = pd.DataFrame({
        "instrument": list(prices),
        "asset_class": ["Test"] * len(prices),
        "point_size": [point] * len(prices),
        "currency": ["USD"] * len(prices),
        "spread_points": [spread] * len(prices),
        "years": [10.0] * len(prices),
    })
    return price, fx, meta


def test_pnl_identity_constant_position():
    """One instrument, forecast pinned at +cap -> N is constant after warmup;
    P&L must equal N * point * (p_T - p_t0) - costs."""
    rng = np.random.default_rng(0)
    T = 900
    p = 100 + np.cumsum(rng.normal(0.05, 1.0, T))
    price, fx, meta = make_frames({"A": p}, point=5.0, spread=0.02)
    vol = signals.price_vol(price, 35)
    fc = pd.DataFrame(signals.FC_CAP, index=price.index, columns=price.columns)
    res = portfolio.run(price, fx, vol, fc, meta,
                        spec=config.PortfolioSpec(rebalance_buffer=0.0))
    N = res["positions"]["A"]
    # after warmup the position tracks 1/vol; verify P&L accounting exactly
    pnl_direct = (N.shift(1) * 5.0 * price["A"].diff()).fillna(0.0)
    total_from_ret = res["gross_ret"].sum() * config.DEFAULT_PORTFOLIO.capital
    assert np.isclose(pnl_direct.sum(), total_from_ret, rtol=1e-9)


def test_negative_adjusted_price_is_handled():
    """A back-adjusted series crossing zero must produce finite, sane P&L."""
    T = 800
    p = np.linspace(5.0, -5.0, T) + np.sin(np.arange(T) / 7.0)
    price, fx, meta = make_frames({"NEG": p}, point=1000.0, spread=0.001)
    vol = signals.price_vol(price, 35)
    fc = pd.DataFrame(-signals.FC_CAP, index=price.index, columns=price.columns)
    res = portfolio.run(price, fx, vol, fc, meta)
    assert np.isfinite(res["ret"].to_numpy()).all()
    # short a falling market -> positive gross P&L over the run
    assert res["gross_ret"].sum() > 0


def test_costs_charged_on_every_position_change():
    rng = np.random.default_rng(1)
    T = 700
    p = 50 + np.cumsum(rng.normal(0, 0.8, T))
    price, fx, meta = make_frames({"C": p}, point=2.0, spread=0.05)
    vol = signals.price_vol(price, 35)
    # alternating forecast forces heavy trading
    alt = np.where(np.arange(T) % 2 == 0, 2.0, -2.0)
    fc = pd.DataFrame({"C": alt}, index=price.index)
    res = portfolio.run(price, fx, vol, fc, meta,
                        spec=config.PortfolioSpec(rebalance_buffer=0.0))
    N = res["positions"]["C"].to_numpy()
    dN = np.abs(np.diff(N, prepend=0.0))
    expected = (dN * 0.05 * 2.0).sum()
    got = res["cost"].sum() * config.DEFAULT_PORTFOLIO.capital
    assert np.isclose(expected, got, rtol=1e-9)
    assert got > 0


def test_buffer_reduces_turnover_but_not_direction():
    rng = np.random.default_rng(2)
    T = 900
    p = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, T)))
    price, fx, meta = make_frames({"B": p})
    vol = signals.price_vol(price, 35)
    fc = signals.ewmac(price, vol, 16, 64)
    r0 = portfolio.run(price, fx, vol, fc, meta,
                       spec=config.PortfolioSpec(rebalance_buffer=0.0))
    r1 = portfolio.run(price, fx, vol, fc, meta,
                       spec=config.PortfolioSpec(rebalance_buffer=0.2))
    t0 = np.abs(np.diff(r0["positions"]["B"].to_numpy())).sum()
    t1 = np.abs(np.diff(r1["positions"]["B"].to_numpy())).sum()
    assert t1 < t0 * 0.7
    corr = r0["positions"]["B"].corr(r1["positions"]["B"])
    assert corr > 0.8


def test_vol_targeting_is_in_the_right_range():
    """20 independent random-walk instruments, full forecast: realised vol
    should land within a factor ~2 of target (IDM is fixed, not fitted)."""
    rng = np.random.default_rng(3)
    T = 1500
    prices = {f"I{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, T)))
              for i in range(20)}
    price, fx, meta = make_frames(prices)
    meta["asset_class"] = ["A", "B", "C", "D"] * 5
    vol = signals.price_vol(price, 35)
    fc = pd.DataFrame(signals.FC_CAP, index=price.index, columns=price.columns)
    res = portfolio.run(price, fx, vol, fc, meta, idm=2.0)
    realised = res["ret"].iloc[300:].std(ddof=1) * np.sqrt(252)
    target = config.DEFAULT_PORTFOLIO.vol_target
    assert 0.3 * target < realised < 2.0 * target


def test_forecast_sign_drives_position_sign():
    rng = np.random.default_rng(4)
    T = 600
    p = 100 + np.cumsum(rng.normal(0, 1, T))
    price, fx, meta = make_frames({"S": p})
    vol = signals.price_vol(price, 35)
    fc = pd.DataFrame(-1.0, index=price.index, columns=price.columns)
    res = portfolio.run(price, fx, vol, fc, meta)
    tail = res["positions"]["S"].iloc[-100:]
    assert (tail <= 0).all() and (tail < 0).any()

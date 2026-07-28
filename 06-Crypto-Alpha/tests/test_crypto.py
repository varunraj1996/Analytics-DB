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


# ---------------------------------------------------------------------------
# publication timing — the bias that cost half the reported test Sharpe
# ---------------------------------------------------------------------------
def test_signal_lag_respects_actual_publication_time():
    """Coin Metrics finalises day-T data 24.5-27.1h after the T stamp, i.e.
    always after T+1 00:00Z. A lag of 1 trades on unpublished numbers; it was
    doing exactly that, and correcting it moved the frozen test Sharpe from
    1.11 to 0.58. This asserts the lag can never silently go back."""
    from crypto import config
    assert config.SIGNAL_LAG >= 2, (
        "SIGNAL_LAG < 2 is look-ahead: day-T on-chain data is not published "
        "until after T+1 00:00Z on 100% of observed days"
    )


def test_a_feature_cannot_influence_a_position_before_it_is_published():
    """End-to-end version: perturbing the value stamped for day T must leave
    every weight up to and including T+1 untouched."""
    import numpy as np
    import pandas as pd
    from crypto import config, signals as sg

    idx = pd.bdate_range("2021-01-01", periods=300)
    raw = pd.DataFrame({"A": np.linspace(1.0, 2.0, len(idx))}, index=idx)
    base = sg.lag(sg._expanding_z(np.log(raw)))

    bumped = raw.copy()
    t = 250
    bumped.iloc[t] *= 5.0                      # a wild revision on day T
    after = sg.lag(sg._expanding_z(np.log(bumped)))

    upto = min(t + config.SIGNAL_LAG - 1, len(idx) - 1)
    pd.testing.assert_series_equal(base["A"].iloc[:upto + 1],
                                   after["A"].iloc[:upto + 1])
    assert not np.allclose(base["A"].iloc[t + config.SIGNAL_LAG],
                           after["A"].iloc[t + config.SIGNAL_LAG],
                           equal_nan=True)

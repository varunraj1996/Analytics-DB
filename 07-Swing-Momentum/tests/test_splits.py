"""Tests for the split repair, on hand-built series with known answers.

The point of these is that the repair must be *conservative*: a genuine
crash that stays down must survive untouched, because "fixing" it would
manufacture an edge out of a real loss.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swing.splits import detect, repair_one, repair_panel  # noqa: E402


def series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def frame(vals, tic="AAA", start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.DataFrame({"date": idx, "open": vals, "high": [v * 1.01 for v in vals],
                         "low": [v * 0.99 for v in vals], "close": vals,
                         "volume": [1e6] * len(vals), "ticker": tic})


def test_detects_a_four_for_one_split():
    s = series([100, 101, 102, 25.5, 25.7, 26.0])      # 4:1 on the fourth bar
    hits = detect(s)
    assert len(hits) == 1 and hits.iloc[0] == pytest.approx(0.25)


def test_detects_a_reverse_split():
    s = series([10, 10.1, 10.2, 51.0, 51.5, 52.0])     # 1:5 reverse
    hits = detect(s)
    assert len(hits) == 1 and hits.iloc[0] == pytest.approx(5.0)


def test_a_real_crash_that_stays_down_is_left_alone():
    """A biotech that loses 55% on a trial failure is not a split."""
    s = series([100, 101, 102, 45.0, 44.0, 46.0])      # ratio 0.44, no match
    assert detect(s).empty


def test_known_limitation_a_crash_landing_on_a_split_ratio_is_indistinguishable():
    """Documented, not fixed: without a corporate-actions feed, a genuine
    crash that happens to halve the price cannot be told from a 2:1 split.
    The repair will treat it as a split. This is the cost of the heuristic
    and the reason detection is kept conservative elsewhere."""
    s = series([100, 101, 102, 51.0, 50.0, 52.0])      # exactly ~2:1
    assert len(detect(s)) == 1


def test_a_crash_that_bounces_back_is_not_treated_as_a_split():
    """Down 50% then straight back up is a bad print, not a split."""
    s = series([100, 100, 100, 50.0, 100.0, 100.0])
    assert detect(s).empty


def test_repair_makes_returns_continuous():
    df = frame([100, 101, 102, 25.5, 25.7, 26.0])
    fixed, n = repair_one(df)
    assert n == 1
    r = fixed["close"].pct_change().dropna()
    assert r.abs().max() < 0.05                       # no phantom crash left
    # the post-split segment is untouched; history is scaled onto it
    assert fixed["close"].iloc[-1] == pytest.approx(26.0)
    assert fixed["close"].iloc[0] == pytest.approx(25.0)


def test_volume_moves_opposite_to_price():
    df = frame([100, 101, 102, 25.5, 25.7, 26.0])
    fixed, _ = repair_one(df)
    # pre-split share counts are multiplied by 4 as prices are quartered
    assert fixed["volume"].iloc[0] == pytest.approx(4e6)
    assert fixed["volume"].iloc[-1] == pytest.approx(1e6)


def test_all_ohlc_columns_are_scaled_together():
    df = frame([100, 101, 102, 25.5, 25.7, 26.0])
    fixed, _ = repair_one(df)
    row = fixed.iloc[0]
    assert row["high"] / row["close"] == pytest.approx(1.01)
    assert row["low"] / row["close"] == pytest.approx(0.99)


def test_two_splits_compound():
    s = [100, 100, 50.0, 50.0, 25.0, 25.0]            # 2:1 then 2:1
    fixed, n = repair_one(frame(s))
    assert n == 2
    assert fixed["close"].iloc[0] == pytest.approx(25.0)
    assert fixed["close"].pct_change().dropna().abs().max() < 0.05


def test_panel_repair_only_touches_affected_tickers():
    clean = frame([10, 10.1, 10.2, 10.3, 10.4, 10.5], tic="CLEAN")
    split = frame([100, 101, 102, 25.5, 25.7, 26.0], tic="SPLIT")
    out = repair_panel(pd.concat([clean, split], ignore_index=True), verbose=False)
    c = out[out.ticker == "CLEAN"].sort_values("date")["close"].to_numpy()
    assert np.allclose(c, [10, 10.1, 10.2, 10.3, 10.4, 10.5])
    s = out[out.ticker == "SPLIT"].sort_values("date")["close"]
    assert s.pct_change().dropna().abs().max() < 0.05

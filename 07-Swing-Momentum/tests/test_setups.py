"""Regression tests for the trade simulator on hand-built bars.

The partial-taking rule was dead code for the whole first pass: the inner
condition `r_now >= target_r or held >= partial_days` sat inside an outer
gate that already required `held >= partial_days`, so the target could never
fire early and "sell into strength" was never actually tested. Nothing
caught it because the simulator still produced plausible trades. These
tests pin the behaviour to specific bars so it cannot rot back.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swing import setups as ST  # noqa: E402


def bars():
    """Signal on bar 0; bar 1 opens at 100 and closes +3R; bar 3 collapses.

    A trader who sells half into the 3R spike and moves to breakeven ends
    positive. One who waits for a fixed day count gives it all back.
    """
    open_ = np.array([100., 100., 112., 90., 90., 90., 90., 90.])
    high = np.array([101., 113., 113., 95., 95., 95., 95., 95.])
    low = np.array([96., 99., 110., 88., 88., 88., 88., 88.])
    close = np.array([100., 112., 112., 90., 90., 90., 90., 90.])
    ma = np.full(8, 95.0)               # never triggers the trail here
    adr = np.full(8, 0.06)              # 6% ADR: a 4-point stop is allowed
    return open_, high, low, close, ma, adr


def run(target_r, partial_days=3):
    open_, high, low, close, ma, adr = bars()
    rows = np.array([0], np.int64)
    ends = np.array([len(open_)], np.int64)
    out = ST.simulate(rows, ends, open_, high, low, close, ma, adr,
                      1.5, np.int64(partial_days), 0.5, float(target_r),
                      np.int64(40), True)
    sig, ent, ext, epx, risk, pnl, reason, nbars, turn = out
    return {"pnl": float(pnl[0]), "risk": float(risk[0]),
            "entry": float(epx[0]), "reason": int(reason[0]),
            "exit": int(ext[0])}


def test_target_r_fires_before_the_day_count():
    """3R on the entry bar: half comes off there, stop goes to breakeven."""
    r = run(target_r=3.0)
    assert r["entry"] == pytest.approx(100.0)
    assert r["risk"] == pytest.approx(4.0)          # entry 100, stop = low[0]
    # 0.5 * (112-100) on the spike, then 0.5 * (90-100) stopped at breakeven
    assert r["pnl"] == pytest.approx(1.0)
    assert r["reason"] == ST.EXIT_STOP


def test_unreachable_target_gives_the_move_back():
    """The old behaviour, now reachable only by setting the target absurdly
    high: no early partial, so the whole position rides the collapse."""
    r = run(target_r=99.0)
    assert r["pnl"] == pytest.approx(-10.0)
    assert r["reason"] == ST.EXIT_STOP


def test_day_count_still_takes_the_partial_when_no_target_is_hit():
    open_ = np.array([100., 100., 101., 102., 103., 104., 90., 90.])
    high = np.array([101., 102., 102., 103., 104., 105., 95., 95.])
    low = np.array([96., 99., 100., 101., 102., 103., 88., 88.])
    close = np.array([100., 101., 102., 103., 104., 104., 90., 90.])
    ma = np.full(8, 95.0)
    adr = np.full(8, 0.06)
    out = ST.simulate(np.array([0], np.int64), np.array([8], np.int64),
                      open_, high, low, close, ma, adr, 1.5, np.int64(3),
                      0.5, 99.0, np.int64(40), True)
    pnl, reason = float(out[5][0]), int(out[6][0])
    # held reaches 3 on bar 4 (close 104): half off at +2.0, then the
    # remainder is stopped at breakeven for -5.0 when bar 6 gaps to 90
    assert pnl == pytest.approx(-3.0)
    # …which still beats the -10.0 the full position would have given back
    assert reason == ST.EXIT_STOP


def test_stop_wider_than_the_adr_limit_is_skipped():
    open_, high, low, close, ma, adr = bars()
    tight = np.full(8, 0.01)            # 1% ADR: a 4-point stop is 4 ADR
    out = ST.simulate(np.array([0], np.int64), np.array([8], np.int64),
                      open_, high, low, close, ma, tight, 1.5, np.int64(3),
                      0.5, 3.0, np.int64(40), True)
    assert len(out[0]) == 0             # no trade taken at all


def test_gap_through_the_stop_fills_at_the_open():
    open_ = np.array([100., 100., 80., 80., 80.])
    high = np.array([101., 101., 82., 82., 82.])
    low = np.array([96., 99., 79., 79., 79.])
    close = np.array([100., 100., 81., 81., 81.])
    out = ST.simulate(np.array([0], np.int64), np.array([5], np.int64),
                      open_, high, low, close, np.full(5, 70.0),
                      np.full(5, 0.06), 1.5, np.int64(3), 0.5, 3.0,
                      np.int64(40), True)
    # gapped below the 96 stop, so the fill is the 80 open, not the stop
    assert float(out[5][0]) == pytest.approx(-20.0)
    assert int(out[6][0]) == ST.EXIT_STOP

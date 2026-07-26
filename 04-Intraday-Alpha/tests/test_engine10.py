"""Synthetic-bar tests for the 10-minute state machine.

Run with:  python3 -m pytest 04-Intraday-Alpha/tests -q
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intraday import engine10  # noqa: E402


def run(bars, side=1, or_bars=2, confirm=0.0, pullback=1.0, max_wait=6,
        stop=2.0, target=3.0, max_hold=10, eod_flat=True, atr=1.0,
        level_kind="or", n_days=None):
    """bars: list of (open, high, low, close); one session unless day breaks given."""
    a = np.array([b[:4] for b in bars], dtype=np.float64)
    o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    m = len(o)
    day = np.array([b[4] if len(b) > 4 else 0 for b in bars], dtype=np.int64)
    bar = np.zeros(m, np.int64)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        bar[idx] = np.arange(len(idx))
    atr_arr = np.full(m, atr)

    # levels replicate study.Book._levels for the OR case
    level = np.full(m, np.nan)
    or_rng = np.full(m, np.nan)
    pdc = np.full(m, np.nan)
    prior_c = np.nan
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        k = min(or_bars, len(idx))
        oh = h[idx[:k]].max()
        ol = l[idx[:k]].min()
        for i in idx:
            if bar[i] >= or_bars:
                level[i] = oh if side > 0 else ol
                or_rng[i] = oh - ol
                pdc[i] = prior_c
        prior_c = c[idx[-1]]

    cap = 16
    te = np.empty(cap, np.int64); tx = np.empty(cap, np.int64)
    ep = np.empty(cap, np.float64); xp = np.empty(cap, np.float64)
    rs = np.empty(cap, np.int8); lv = np.empty(cap, np.float64)
    at = np.empty(cap, np.float64)
    nt = engine10.run_symbol(o, h, l, c, bar, day, atr_arr, level, pdc,
                             np.int64(side), float(confirm), float(pullback),
                             np.int64(max_wait), np.int64(or_bars),
                             float(stop), float(target), np.int64(max_hold),
                             eod_flat, 0.0, or_rng, False,
                             te, tx, ep, xp, rs, lv, at)
    return dict(n=nt, entry=te[:nt], exit=tx[:nt], epx=ep[:nt], xpx=xp[:nt],
                reason=rs[:nt])


OR = [(100, 101, 99, 100), (100, 101, 99, 100)]     # opening range: high 101


def test_no_breakout_no_trade():
    bars = OR + [(100, 100.9, 99.5, 100.5)] * 6
    assert run(bars)["n"] == 0


def test_breakout_without_pullback_is_skipped():
    # closes above 101 then runs away without ever retracing 1 ATR
    bars = OR + [(101, 102, 100.9, 101.8), (102, 103, 101.8, 102.9),
                 (103, 104, 102.8, 103.9), (104, 105, 103.9, 104.8)]
    assert run(bars)["n"] == 0


def test_pullback_entry_is_next_bar_open():
    bars = OR + [
        (101, 102, 100.9, 101.8),   # bar 2: breakout close 101.8, pull level 100.8->101 (floored at level)
        (101.5, 101.6, 100.9, 101.2),  # bar 3: low 100.9 <= 101 -> TRIGGERED
        (101.3, 101.9, 101.0, 101.5),  # bar 4: ENTER at open 101.3
        (101.5, 105.0, 101.4, 104.8),  # bar 5: target 101.3+3 = 104.3 hit
    ]
    r = run(bars)
    assert r["n"] == 1
    assert r["entry"][0] == 4
    assert np.isclose(r["epx"][0], 101.3)
    assert r["reason"][0] == engine10.X_TARGET
    assert np.isclose(r["xpx"][0], 104.3)


def test_pullback_level_never_below_breakout_level():
    # deep pullback config: the trigger level is floored at the OR high, so a
    # retrace through the level (failed breakout) voids rather than fills
    bars = OR + [
        (101, 102, 100.9, 101.2),      # breakout close only 0.2 above level
        (101.0, 101.1, 100.4, 100.6),  # closes back BELOW 101 -> setup dies
        (100.6, 103.0, 100.5, 102.9),
        (103, 104, 102.9, 103.8),
    ]
    r = run(bars, pullback=2.0)
    assert r["n"] == 0


def test_stop_on_entry_bar_uses_entry_open_reference():
    bars = OR + [
        (101, 102, 100.9, 101.8),
        (101.5, 101.6, 100.9, 101.2),   # trigger
        (101.0, 101.1, 98.0, 98.5),     # entry at 101.0, stop 99.0 hit intra-bar
    ]
    r = run(bars)
    assert r["n"] == 1
    assert r["reason"][0] == engine10.X_STOP
    assert np.isclose(r["xpx"][0], 99.0)


def test_gap_through_stop_fills_at_open_not_stop():
    bars = OR + [
        (101, 102, 100.9, 101.8),
        (101.5, 101.6, 100.9, 101.2),   # trigger
        (101.2, 101.6, 101.0, 101.4),   # entry 101.2, stop 99.2
        (98.0, 98.5, 97.5, 98.2),       # gaps to 98 -> fill at 98, not 99.2
    ]
    r = run(bars)
    assert r["reason"][0] == engine10.X_STOP
    assert np.isclose(r["xpx"][0], 98.0)


def test_ambiguous_bar_resolves_to_stop():
    bars = OR + [
        (101, 102, 100.9, 101.8),
        (101.5, 101.6, 100.9, 101.2),   # trigger
        (101.2, 104.4, 99.0, 100.0),    # both stop 99.2 and target 104.2 inside
    ]
    r = run(bars)
    assert r["reason"][0] == engine10.X_STOP


def test_eod_flat_closes_at_last_bar():
    bars = OR + [
        (101, 102, 100.9, 101.8),
        (101.5, 101.6, 100.9, 101.2),   # trigger
        (101.2, 101.6, 101.0, 101.4),   # entry; nothing hit
        (101.4, 101.7, 101.2, 101.6),   # last bar of session -> EOD close
    ]
    r = run(bars, stop=5.0, target=5.0)
    assert r["n"] == 1
    assert r["reason"][0] == engine10.X_EOD
    assert np.isclose(r["xpx"][0], 101.6)


def test_one_trade_per_symbol_day():
    day1 = OR + [
        (101, 102, 100.9, 101.8),
        (101.5, 101.6, 100.9, 101.2),   # trigger
        (101.2, 101.6, 98.0, 99.0),     # entry, stopped same bar
        (99, 103, 98.9, 102.9),         # second breakout would form...
        (103, 103.5, 100.0, 100.2),     # ...and pull back
        (100.3, 101.0, 100.1, 100.8),
    ]
    r = run(day1, stop=2.0, target=5.0)
    assert r["n"] == 1, "no re-entry after a completed round trip that day"


def test_short_side_mirrors():
    bars = [(100, 101, 99, 100), (100, 101, 99, 100),        # OR low = 99
            (99, 99.1, 97.9, 98.1),      # breakdown close 98.1 < 99
            (98.5, 99.05, 98.0, 98.6),   # rallies to 99.05 >= trigger 99 (floored at level)
            (98.9, 99.2, 96.0, 96.5)]    # short entry at 98.9; target 95.9? stop 100.9?
    r = run(bars, side=-1, pullback=2.0, stop=2.0, target=2.0)
    assert r["n"] == 1
    assert r["entry"][0] == 4
    assert np.isclose(r["epx"][0], 98.9)
    # target = 98.9 - 2 = 96.9, hit intra-bar (low 96.0)
    assert r["reason"][0] == engine10.X_TARGET
    assert np.isclose(r["xpx"][0], 96.9)


def test_sessions_are_independent():
    # position would be held into the next day if eod_flat were off; verify a
    # new session resets state and the OR is rebuilt
    day1 = OR + [
        (101, 102, 100.9, 101.8),
        (101.5, 101.6, 100.9, 101.2),
        (101.2, 101.6, 101.0, 101.4),   # entry, EOD close next bar
        (101.4, 101.7, 101.2, 101.6),
    ]
    day2 = [(x[0], x[1], x[2], x[3], 1) for x in [
        (102, 103, 101, 102), (102, 103, 101, 102),
        (103.2, 104, 103, 103.8),        # breakout over OR high 103
        (103.5, 103.6, 102.7, 103.0),    # pullback trigger (level 103)
        (103.1, 106.5, 103.0, 106.4),    # entry 103.1, target 106.1
    ]]
    r = run(day1 + day2, stop=5.0, target=3.0)
    assert r["n"] == 2
    assert r["reason"][0] == engine10.X_EOD
    assert r["reason"][1] == engine10.X_TARGET

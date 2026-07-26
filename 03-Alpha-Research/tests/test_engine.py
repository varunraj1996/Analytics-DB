"""Synthetic-bar tests for the fill and exit mechanics.

Two look-ahead bugs got through review of this engine during development, both
of the same family: crediting the strategy with a price that could not have
been available when the decision was taken.  These tests pin down the
mechanics on hand-built bars where the right answer is obvious, so that the
conservative conventions cannot quietly regress.

Run with:  python3 -m pytest 03-Alpha-Research/tests -q
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha import search  # noqa: E402

STARTS = np.array([0], dtype=np.int64)


def bars(rows):
    """rows: list of (open, high, low, close) -> arrays plus a flat ATR of 1."""
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def run(rows, sig_idx, side=1, pullback=1.0, window=3, stop=2.0, target=1.0,
        hold=5, trail=0.0, atr=1.0, mode=1):
    o, h, l, c = bars(rows)
    n = len(o)
    atr_arr = np.full(n, atr)
    sig_rows = np.array(sig_idx, dtype=np.int64)
    block_end = np.full(len(sig_rows), n, dtype=np.int64)
    f_sig, f_row, f_px, f_atr = search.find_fills(
        sig_rows, block_end, o, h, l, c, atr_arr,
        np.int64(side), float(pullback), np.int64(window))
    if len(f_sig) == 0:
        return None
    x_row, x_px, x_reason, x_mae, x_mfe, keep = search.eval_exits(
        f_sig, f_row, f_px, f_atr, np.full(len(f_sig), n, dtype=np.int64),
        o, h, l, c, np.int64(side), float(stop), float(target),
        np.int64(hold), float(trail), np.int64(mode))
    return dict(f_sig=f_sig, f_row=f_row, f_px=f_px, x_row=x_row, x_px=x_px,
                reason=x_reason, keep=keep, mfe=x_mfe, mae=x_mae)


# ---------------------------------------------------------------------------
def test_limit_never_touched_produces_no_trade():
    # signal closes at 100, limit sits at 99; price never trades below 99.5
    rows = [(100, 100, 100, 100),
            (100, 101, 99.5, 100.5),
            (100, 102, 99.6, 101.0),
            (101, 103, 100.0, 102.0)]
    assert run(rows, [0]) is None


def test_fill_at_limit_when_bar_trades_through():
    rows = [(100, 100, 100, 100),
            (100, 100.5, 98.0, 99.5),      # trades through 99
            (99, 100, 98.5, 99.0)]
    r = run(rows, [0], hold=1)
    assert r["f_row"][0] == 1
    assert np.isclose(r["f_px"][0], 99.0)


def test_gap_below_limit_fills_at_the_open_not_the_limit():
    # a gap down through the limit must not hand us the better price
    rows = [(100, 100, 100, 100),
            (95, 96, 94, 95.5),
            (95, 96, 94, 95.5)]
    r = run(rows, [0], hold=1)
    assert np.isclose(r["f_px"][0], 95.0), "fill should be the open, not 99"


def test_entry_bar_never_exits_at_the_open():
    """The open always precedes the fill, so it can never be an exit price.

    The bar opens at 100.8 - above the 100.0 target - then sells off to the 99
    limit.  Booking 100.8 would be selling before we had bought.  In capped
    mode the most we may claim is the target itself.
    """
    rows = [(100, 100, 100, 100),
            (100.8, 101.0, 98.5, 99.2),     # entry bar: open and high precede the dip
            (99.3, 99.8, 99.0, 99.5),
            (99.5, 101.5, 99.4, 101.0)]
    r = run(rows, [0], target=1.0, stop=2.0, hold=5, mode=1)
    assert r["f_row"][0] == 1
    assert r["x_px"][0] <= 100.0 + 1e-9, "must never book the entry bar's open"
    assert np.isclose(r["x_px"][0], 100.0)


def test_entry_bar_target_is_capped_not_uncapped():
    """Capped mode books the target; stop-only mode lets the winner run.

    The distinction is worth a lot of basis points, so it is pinned down: in
    mode 1 we get exactly the target on the entry bar, in mode 0 we hold and
    take whatever the next bars give.
    """
    rows = [(100, 100, 100, 100),
            (99.6, 100.4, 98.5, 100.2),     # fills at 99, then trades up past 100
            (100.5, 103.0, 100.4, 102.8)]
    capped = run(rows, [0], target=1.0, stop=2.0, hold=5, mode=1)
    uncapped = run(rows, [0], target=1.0, stop=2.0, hold=5, mode=0)
    assert np.isclose(capped["x_px"][0], 100.0)
    assert capped["x_row"][0] == 1
    assert uncapped["x_px"][0] > 100.0, "stop-only mode should run past the target"
    assert uncapped["x_row"][0] == 2


def test_entry_bar_low_can_still_trigger_the_stop():
    """The guard is one-directional: adverse moves on the entry bar do count."""
    rows = [(100, 100, 100, 100),
            (100, 100.2, 96.0, 96.5),       # fills at 99 then collapses to 96
            (96, 97, 95, 96)]
    r = run(rows, [0], stop=2.0, target=1.0, hold=5)
    assert r["x_row"][0] == 1
    assert r["reason"][0] == 1
    assert np.isclose(r["x_px"][0], 97.0)   # 99 - 2*ATR


def test_stop_wins_when_both_levels_are_inside_the_same_bar():
    rows = [(100, 100, 100, 100),
            (99.5, 99.6, 98.9, 99.0),       # fill at 99, target 100 not reached
            (99, 101.0, 96.5, 97.0)]        # both target 100 and stop 97 in range
    r = run(rows, [0], stop=2.0, target=1.0, hold=5)
    assert r["reason"][0] == 1, "ambiguous bars must resolve against us"
    assert np.isclose(r["x_px"][0], 97.0)


def test_time_stop_exits_at_the_close_of_the_last_held_bar():
    rows = [(100, 100, 100, 100),
            (100, 100.1, 98.9, 99.0)] + [(99, 99.4, 98.6, 99.1)] * 6
    r = run(rows, [0], stop=5.0, target=5.0, hold=3)
    assert r["reason"][0] == 3
    assert r["x_row"][0] == 1 + 3
    assert np.isclose(r["x_px"][0], 99.1)


def test_overlapping_signals_do_not_open_a_second_position():
    rows = [(100, 100, 100, 100),
            (100, 100.1, 98.9, 99.0),       # trade 1 fills at 99
            (99, 99.5, 98.5, 99.0),         # second signal, limit 98
            (99, 99.5, 97.5, 98.5),         # would fill trade 2 here
            (98.5, 99.0, 97.5, 98.5),
            (98.5, 99.0, 97.5, 98.5),
            (98.5, 99.0, 97.5, 98.5)]
    r = run(rows, [0, 2], stop=5.0, target=5.0, hold=4)
    assert len(r["keep"]) == 2, "both signals should have found a fill"
    assert r["keep"][0] == 1
    assert r["keep"][1] == 0, "second signal fires while the first is still open"


def test_short_side_mirrors_the_long_mechanics():
    # breakdown signal at 100; limit rests 1 ATR *above* at 101
    rows = [(100, 100, 100, 100),
            (100, 101.5, 100.6, 101.0),     # rallies through 101 -> short fill
            (101, 101.2, 99.0, 99.5),       # target 100 hit here
            (99.5, 99.8, 97.5, 98.0)]
    r = run(rows, [0], side=-1, pullback=1.0, stop=2.0, target=1.0, hold=5)
    assert np.isclose(r["f_px"][0], 101.0)
    assert r["x_row"][0] == 2
    assert r["reason"][0] == 2
    assert np.isclose(r["x_px"][0], 100.0)


def test_mode2_executes_at_the_trigger_close_and_never_exits_on_that_bar():
    """Mode 2 is what the reported results use, so it gets its own guard.

    The trigger bar is deliberately given a range that contains both the stop
    and the target measured from its own close.  Neither may fire: in mode 2 the
    position does not exist until that bar has closed.
    """
    rows = [(100, 100, 100, 100),
            (100, 103.0, 96.0, 99.5),       # pullback through 99 triggers; wild range
            (99.5, 99.7, 99.3, 99.6),
            (99.6, 101.0, 99.5, 100.8)]     # target 100.5 legitimately hit here
    o, h, l, c = bars(rows)
    r = run(rows, [0], pullback=1.0, stop=2.0, target=1.0, hold=5, mode=2)
    assert r["f_row"][0] == 1
    entry = c[1]                             # executed on the trigger bar's close
    assert np.isclose(entry, 99.5)
    assert r["x_row"][0] == 3, "no exit may be booked on the trigger bar"
    assert np.isclose(r["x_px"][0], entry + 1.0)


def test_mode2_time_stop_counts_from_the_bar_after_the_trigger():
    rows = [(100, 100, 100, 100),
            (100, 100.2, 98.9, 99.5)] + [(99.5, 99.7, 99.3, 99.5)] * 6
    r = run(rows, [0], pullback=1.0, stop=9.0, target=9.0, hold=2, mode=2)
    # entered on the close of bar 1, held two further sessions -> exit on bar 3
    assert r["x_row"][0] == 3
    assert r["reason"][0] == 3


def test_mfe_excludes_the_entry_bar():
    rows = [(100, 100, 100, 100),
            (100.8, 103.0, 98.5, 99.2),     # huge high before the fill
            (99.2, 99.4, 99.0, 99.2)]
    r = run(rows, [0], stop=5.0, target=5.0, hold=2)
    assert r["mfe"][0] < 0.5, "entry-bar high must not inflate the excursion"

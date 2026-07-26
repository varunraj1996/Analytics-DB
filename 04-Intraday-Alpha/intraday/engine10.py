"""Bar-level intraday engine.

The daily-bar study died of intra-bar ambiguity, so this engine is built on
one rule: **nothing is ever decided or priced inside the bar that decides it**.

* Signals (breakout, pullback trigger) are evaluated on *completed* bars.
* Every entry executes at the **next bar's open**.
* Stops and targets are evaluated bar by bar from the entry bar onward; if the
  bar opens through the level, the fill is the open (gaps hurt us); if both
  stop and target sit inside one bar's range, the stop is assumed first.
* On the entry bar itself the same conventions apply against the entry price -
  the entry IS the open, so the rest of that bar genuinely lies in the future.
* ``eod_flat`` closes everything at the last bar's close of the session.

State machine per (symbol, session):

    IDLE -- breakout close confirmed --> ARMED (pullback level fixed)
    ARMED -- close (or low) reaches pullback level --> TRIGGERED
    TRIGGERED -- next bar open --> IN_POSITION
    IN_POSITION -- stop / target / time / EOD --> IDLE (no re-entry same day)

The pullback trigger uses the bar's LOW touching the level (long side), but the
entry is still the next bar's open - touching intra-bar is observable in
retrospect from a completed bar, and we never claim to have traded *at* it.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# exit reasons
X_STOP, X_TARGET, X_TIME, X_EOD = 1, 2, 3, 4


@njit(cache=True)
def wilder_atr_bars(high, low, close, day_start, n):
    """ATR over 10-min bars, continuous across sessions but gap-aware:
    the true range at a session's first bar uses the prior session's close."""
    m = len(high)
    out = np.empty(m)
    prev = np.nan
    acc = 0.0
    for i in range(m):
        if i == 0:
            tr = high[i] - low[i]
        else:
            d1 = high[i] - low[i]
            d2 = abs(high[i] - close[i - 1])
            d3 = abs(low[i] - close[i - 1])
            tr = max(d1, d2, d3)
        if i < n:
            acc += tr
            out[i] = np.nan
            if i == n - 1:
                prev = acc / n
                out[i] = prev
        else:
            prev = (prev * (n - 1) + tr) / n
            out[i] = prev
    return out


@njit(cache=True)
def run_symbol(open_, high, low, close, bar, day_id, atr,
               level_long, prior_close,
               side, confirm_atr, pullback_atr, max_wait, or_bars,
               stop_atr, target_atr, max_hold, eod_flat,
               min_or_range_atr, or_range, use_pdc_filter,
               t_entry, t_exit, t_epx, t_xpx, t_reason, t_level, t_atr):
    """One pass over a symbol's bars.  Returns number of trades written.

    ``level_long`` holds, per bar, the breakout level in force for that session
    (opening-range high or prior-day high, already side-adjusted by caller:
    for shorts pass the mirrored level as ``level_long`` and side=-1 with
    prices unchanged - comparisons below are side-aware).
    """
    m = len(open_)
    nt = 0

    IDLE, ARMED, TRIGGERED, INPOS = 0, 1, 2, 3
    state = IDLE
    pull_level = 0.0
    armed_at = -1
    entry_i = -1
    epx = 0.0
    stop = 0.0
    tgt = 0.0
    a_entry = 0.0
    traded_today = False
    cur_day = -1

    for i in range(m):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            state = IDLE
            traded_today = False

        # last bar of the session?
        eod = (i + 1 >= m) or (day_id[i + 1] != cur_day)

        if state == INPOS:
            o = open_[i]
            exit_px = np.nan
            reason = 0
            if side > 0:
                if o <= stop:
                    exit_px = o; reason = X_STOP
                elif o >= tgt:
                    exit_px = o; reason = X_TARGET
                elif low[i] <= stop:
                    exit_px = stop; reason = X_STOP
                elif high[i] >= tgt:
                    exit_px = tgt; reason = X_TARGET
            else:
                if o >= stop:
                    exit_px = o; reason = X_STOP
                elif o <= tgt:
                    exit_px = o; reason = X_TARGET
                elif high[i] >= stop:
                    exit_px = stop; reason = X_STOP
                elif low[i] <= tgt:
                    exit_px = tgt; reason = X_TARGET
            if reason == 0 and i - entry_i >= max_hold:
                exit_px = close[i]; reason = X_TIME
            if reason == 0 and eod and eod_flat:
                exit_px = close[i]; reason = X_EOD

            if reason != 0:
                t_entry[nt] = entry_i
                t_exit[nt] = i
                t_epx[nt] = epx
                t_xpx[nt] = exit_px
                t_reason[nt] = reason
                t_level[nt] = pull_level
                t_atr[nt] = a_entry
                nt += 1
                state = IDLE
                traded_today = True
            continue

        if state == TRIGGERED:
            # the trigger fired on the previous bar; we buy this bar's open,
            # unless the session ended underneath us
            entry_i = i
            epx = open_[i]
            a = atr[i - 1]
            stop = epx - side * stop_atr * a
            tgt = epx + side * target_atr * a
            a_entry = a
            state = INPOS
            # the entry bar itself is now processed as IN_POSITION next loop -
            # but we must not skip its stop/target scan, so re-run this bar:
            # (duplicate of the INPOS block, entry-bar edition)
            o = epx
            exit_px = np.nan
            reason = 0
            if side > 0:
                if low[i] <= stop:
                    exit_px = stop; reason = X_STOP
                elif high[i] >= tgt:
                    exit_px = tgt; reason = X_TARGET
            else:
                if high[i] >= stop:
                    exit_px = stop; reason = X_STOP
                elif low[i] <= tgt:
                    exit_px = tgt; reason = X_TARGET
            if reason == 0 and eod and eod_flat:
                exit_px = close[i]; reason = X_EOD
            if reason != 0:
                t_entry[nt] = entry_i
                t_exit[nt] = i
                t_epx[nt] = epx
                t_xpx[nt] = exit_px
                t_reason[nt] = reason
                t_level[nt] = pull_level
                t_atr[nt] = a_entry
                nt += 1
                state = IDLE
                traded_today = True
            continue

        if eod:
            state = IDLE
            continue

        if traded_today:            # one round-trip per symbol-day
            continue

        lvl = level_long[i]
        if not np.isfinite(lvl) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if bar[i] < or_bars:        # opening range still forming
            continue

        if state == IDLE:
            if min_or_range_atr > 0.0 and or_range[i] < min_or_range_atr * atr[i]:
                continue
            if use_pdc_filter and np.isfinite(prior_close[i]):
                if side > 0 and close[i] <= prior_close[i]:
                    continue
                if side < 0 and close[i] >= prior_close[i]:
                    continue
            brk = (close[i] - lvl) * side
            if brk > confirm_atr * atr[i]:
                # breakout confirmed on this completed bar
                pull_level = close[i] - side * pullback_atr * atr[i]
                # the pullback must stay on the right side of the level:
                # a retrace through the breakout level itself voids the setup
                if (pull_level - lvl) * side < 0.0:
                    pull_level = lvl
                state = ARMED
                armed_at = i
            continue

        if state == ARMED:
            if i - armed_at > max_wait:
                state = IDLE
                continue
            # failed breakout: close back through the level kills the setup
            if (close[i] - level_long[i]) * side < 0.0:
                state = IDLE
                continue
            touched = (low[i] <= pull_level) if side > 0 else (high[i] >= pull_level)
            if touched:
                state = TRIGGERED
            continue

    return nt

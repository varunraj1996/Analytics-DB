"""Brute-force machinery.

The naive way to search this space is to re-run the whole panel for every
parameter combination, which costs ~2 seconds a shot and caps you at a few
thousand evaluations.  That is not enough to say anything about a fifteen
dimensional grid, so the work is factorised instead:

    entry filters  ->  mask          (numpy, over eligible rows only)
    pullback/limit ->  fills         (numba, over signal rows only)
    stop/target/…  ->  exits         (numba, over fill rows only)

Each stage only touches the rows the previous stage kept, so a full sweep of
~200 exit variants on top of an existing fill set costs milliseconds.  That
buys roughly five orders of magnitude more evaluations than the naive loop,
which is what makes an honest train/validate/test split affordable: the search
can be greedy and exhaustive on the train window and still leave the other two
windows untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from . import config


# ---------------------------------------------------------------------------
# Stage 0: rows that could ever be traded, regardless of parameters
# ---------------------------------------------------------------------------
class Eligible:
    """Compressed view of the panel: only rows passing the fixed screens."""

    def __init__(self, P, F, uni: config.UniverseSpec,
                 member: np.ndarray | None = None,
                 day_lo: int = 0, day_hi: int = 10 ** 9):
        c = P.close
        ok = (
            (F["addv21"] >= uni.min_addv_usd)
            & (F["avgvol50"] >= uni.min_share_vol)
            & (c >= uni.min_price)
            & (c <= uni.max_price)
            & (F["age"] >= uni.min_history_days)
            & np.isfinite(F["atr14"]) & (F["atr14"] > 0)
            & np.isfinite(F["ma200"])
            & (P.day >= day_lo) & (P.day <= day_hi)
        )
        if uni.sp500_only:
            ok &= member
        self.rows = np.flatnonzero(ok).astype(np.int64)
        self.P = P
        self.F = F

        # compressed copies of everything the mask stage needs
        r = self.rows
        self.close = P.close[r]
        self.high = P.high[r]
        self.low = P.low[r]
        self.volume = P.volume[r]
        self.day = P.day[r]
        self.atr = F["atr14"][r]
        self.avgvol50 = F["avgvol50"][r]
        self.ma = {n: F[f"ma{n}"][r] for n in (20, 50, 100, 200)}
        self.hh = {n: F[f"hh{n}"][r] for n in (15, 20, 30, 40, 60, 80, 120)}
        self.ll = {n: F[f"ll{n}"][r] for n in (15, 20, 30, 40, 60, 80, 120)}
        self.block_end = P.ends[P.sym_id[r]]

    def __len__(self):
        return len(self.rows)


def mask_rows(E: Eligible, p: config.StrategyParams,
              regime_ok: np.ndarray | None) -> np.ndarray:
    """Absolute panel row indices of every signal produced by `p`."""
    c, atr = E.close, E.atr
    hh, ll = E.hh[p.base_len], E.ll[p.base_len]

    with np.errstate(invalid="ignore"):
        m = (c > hh + p.breakout_margin * atr) if p.side > 0 \
            else (c < ll - p.breakout_margin * atr)
        m &= np.isfinite(hh) & np.isfinite(ll)
        if p.vol_mult > 0:
            m &= E.volume > p.vol_mult * E.avgvol50
        if p.trend_ma:
            ma = E.ma[p.trend_ma]
            m &= (c > ma) if p.side > 0 else (c < ma)
        if p.min_base_tightness > 0:
            m &= (hh - ll) <= p.min_base_tightness * atr
        if p.min_close_loc > 0:
            rng = np.maximum(E.high - E.low, 1e-9)
            loc = (c - E.low) / rng if p.side > 0 else (E.high - c) / rng
            m &= loc >= p.min_close_loc
        if p.max_ext_atr < 99:
            ext = (c - E.ma[50]) / atr
            m &= (ext <= p.max_ext_atr) if p.side > 0 else (-ext <= p.max_ext_atr)
        if regime_ok is not None and p.regime_ma:
            m &= regime_ok[E.day]

    return E.rows[np.nan_to_num(m, nan=False)]


# ---------------------------------------------------------------------------
# Stage 1: resting limit order
# ---------------------------------------------------------------------------
@njit(cache=True)
def find_fills(sig_rows, block_end, open_, high, low, close, atr,
               side, pullback_atr, entry_window):
    n = len(sig_rows)
    f_sig = np.empty(n, np.int64)
    f_row = np.empty(n, np.int64)
    f_px = np.empty(n, np.float64)
    f_atr = np.empty(n, np.float64)
    k = 0
    for q in range(n):
        i = sig_rows[q]
        A = atr[i]
        if not (A > 0.0):
            continue
        limit = close[i] - side * pullback_atr * A
        b = block_end[q]
        jmax = i + 1 + entry_window
        if jmax > b:
            jmax = b
        for j in range(i + 1, jmax):
            if side > 0:
                if low[j] <= limit:
                    f_px[k] = open_[j] if open_[j] < limit else limit
                    f_sig[k] = i
                    f_row[k] = j
                    f_atr[k] = A
                    k += 1
                    break
            else:
                if high[j] >= limit:
                    f_px[k] = open_[j] if open_[j] > limit else limit
                    f_sig[k] = i
                    f_row[k] = j
                    f_atr[k] = A
                    k += 1
                    break
    return f_sig[:k], f_row[:k], f_px[:k], f_atr[:k]


# ---------------------------------------------------------------------------
# Stage 2: exits (and the one-position-per-symbol rule)
# ---------------------------------------------------------------------------
@njit(cache=True)
def eval_exits(f_sig, f_row, f_px, f_atr, f_blockend, open_, high, low, close,
               side, stop_atr, target_atr, max_hold, trail_atr,
               entry_bar_mode=1):
    """Resolve every fill.  ``keep`` marks the ones an actually-sequential
    trader could have taken (no overlapping position in the same symbol).

    ``entry_bar_mode`` decides how the entry bar is treated, which matters more
    than it looks.  The bar's open always precedes our fill, so it is never a
    valid exit price.  The bar's *high* is ambiguous - it may have printed
    before the pullback reached our limit:

    ``0`` (stop only)
        Assume the high came first, so no target exit on the entry bar.  Bar-wise
        this is the adverse assumption, but it lets a winner run past the target
        and exit at the next bar's open, which flatters the result.
    ``1`` (capped)
        If the target is inside the entry bar's range, book it at the target.
        Measured against mode 0 this is *wildly* optimistic - the entry bar's
        high is usually what printed before the pullback reached the limit, so
        this hands the strategy a winner it never earned.
    ``2`` (execute at the close of the trigger day - the default)
        The intraday pullback is only used as the *trigger*.  Execution happens
        at that day's close, an observable price, and exits are evaluated from
        the next bar onward.  Nothing in the trade depends on the unknowable
        ordering of prices inside a bar.  This is what the reported numbers use.
    """
    n = len(f_sig)
    x_row = np.empty(n, np.int64)
    x_px = np.empty(n, np.float64)
    x_reason = np.empty(n, np.int8)
    x_mae = np.empty(n, np.float64)
    x_mfe = np.empty(n, np.float64)
    keep = np.zeros(n, np.uint8)

    last_exit = -1
    last_sym_end = -1

    for q in range(n):
        i = f_sig[q]
        b = f_blockend[q]
        if b != last_sym_end:        # new symbol block
            last_sym_end = b
            last_exit = -1

        A = f_atr[q]
        j = f_row[q]
        # mode 2 executes on the close of the trigger bar, so the position only
        # exists from the next session onward
        epx = close[j] if entry_bar_mode == 2 else f_px[q]
        k0 = j + 1 if entry_bar_mode == 2 else j

        stop = epx - side * stop_atr * A
        tgt = epx + side * target_atr * A
        peak = epx
        exit_k = -1
        xpx = 0.0
        reason = 3
        mae = 0.0
        mfe = 0.0

        # the holding window is always measured from the trigger bar, so that
        # max_hold means the same number of sessions in every execution mode
        kmax = j + max_hold + 1
        if kmax > b:
            kmax = b
        for k in range(k0, kmax):
            if trail_atr > 0.0 and k > j:
                c1 = close[k - 1]
                if side > 0:
                    if c1 > peak:
                        peak = c1
                    ts = peak - trail_atr * A
                    if ts > stop:
                        stop = ts
                else:
                    if c1 < peak:
                        peak = c1
                    ts = peak + trail_atr * A
                    if ts < stop:
                        stop = ts

            first_bar = (k == j) and (entry_bar_mode != 2)
            # the open always precedes the fill, so it is never an exit price
            # on the entry bar; the target is allowed only in capped mode
            allow_open = not first_bar
            allow_tgt = (not first_bar) or (entry_bar_mode == 1)

            if side > 0:
                eb = (low[k] - epx) / A
                eg = (high[k] - epx) / A
            else:
                eb = (epx - high[k]) / A
                eg = (epx - low[k]) / A
            if eb < mae:
                mae = eb
            if eg > mfe and not first_bar:
                mfe = eg

            o = open_[k]
            if side > 0:
                if allow_open and o <= stop:
                    xpx = o; exit_k = k; reason = 1; break
                if allow_open and o >= tgt:
                    xpx = o; exit_k = k; reason = 2; break
                if low[k] <= stop:
                    xpx = stop; exit_k = k; reason = 1; break
                if allow_tgt and high[k] >= tgt:
                    xpx = tgt; exit_k = k; reason = 2; break
            else:
                if allow_open and o >= stop:
                    xpx = o; exit_k = k; reason = 1; break
                if allow_open and o <= tgt:
                    xpx = o; exit_k = k; reason = 2; break
                if high[k] >= stop:
                    xpx = stop; exit_k = k; reason = 1; break
                if allow_tgt and low[k] <= tgt:
                    xpx = tgt; exit_k = k; reason = 2; break

        if exit_k < 0:
            kk = j + max_hold
            if kk > b - 1:
                kk = b - 1
                reason = 4
            xpx = close[kk]
            exit_k = kk
        if exit_k < k0:          # mode 2 with no room left in the symbol block
            exit_k = k0 if k0 < b else b - 1
            xpx = close[exit_k]
            reason = 4

        x_row[q] = exit_k
        x_px[q] = xpx
        x_reason[q] = reason
        x_mae[q] = mae
        x_mfe[q] = mfe

        if i > last_exit:
            keep[q] = 1
            last_exit = exit_k

    return x_row, x_px, x_reason, x_mae, x_mfe, keep


# ---------------------------------------------------------------------------
# Stage 3: a cheap but honest portfolio proxy for ranking configurations
# ---------------------------------------------------------------------------
@njit(cache=True)
def fast_portfolio(order, entry_day, exit_day, entry_px, exit_px, stop_px,
                   side, addv, n_days, start_equity, risk_frac, max_pos,
                   max_w, max_new, base_bps, impact_coef, min_cps,
                   participation):
    """Slot-constrained compounding on realised P&L.

    No intraday mark-to-market, so the drawdown it reports is measured on
    closed trades only - good enough to *rank* configurations, not to quote.
    The full daily-marked simulation in ``engine.run_portfolio`` is what gets
    reported for anything that survives.
    """
    equity = start_equity
    eq = np.empty(n_days, np.float64)
    slot_free_at = np.zeros(max_pos, np.int64)
    pending = np.zeros(n_days + 8, np.float64)       # P&L landing on each day
    n_taken = 0
    p = 0
    n = len(order)
    new_today = 0
    cur_day = -1

    for d in range(n_days):
        equity += pending[d]
        eq[d] = equity
        if equity <= 0:
            for dd in range(d, n_days):
                eq[dd] = 0.0
            break

        if d != cur_day:
            cur_day = d
            new_today = 0

        while p < n:
            t = order[p]
            if entry_day[t] > d:
                break
            if entry_day[t] < d:
                p += 1
                continue
            if new_today >= max_new:
                break
            slot = -1
            for sl in range(max_pos):
                if slot_free_at[sl] <= d:
                    slot = sl
                    break
            if slot < 0:
                break

            risk_ps = abs(entry_px[t] - stop_px[t])
            if risk_ps <= 0.0:
                p += 1
                continue
            notional = risk_frac * equity / risk_ps * entry_px[t]
            cap = max_w * equity
            if notional > cap:
                notional = cap
            cap = participation * addv[t]
            if notional > cap:
                notional = cap
            cap = equity / max_pos * 2.0
            if notional > cap:
                notional = cap
            if notional < 500.0:
                p += 1
                continue

            shares = notional / entry_px[t]
            bps = base_bps + impact_coef / np.sqrt(max(addv[t], 1.0) / 1e6)
            cst = 2.0 * notional * bps / 1e4
            fl = 2.0 * shares * min_cps
            if cst < fl:
                cst = fl
            pnl = side[t] * shares * (exit_px[t] - entry_px[t]) - cst

            xd = exit_day[t]
            if xd <= d:
                # opened and closed on the same bar: today's pending bucket has
                # already been swept, so realise it straight away
                equity += pnl
            elif xd < n_days:
                pending[xd] += pnl
            slot_free_at[slot] = xd + 1
            n_taken += 1
            new_today += 1
            p += 1

    return eq, n_taken


# ---------------------------------------------------------------------------
@dataclass
class Evaluation:
    n_signals: int
    n_fills: int
    n_trades: int
    win_rate: float
    avg_r: float
    t_stat: float
    expectancy_bps: float
    cagr: float
    sharpe: float
    max_dd: float
    n_taken: int
    avg_bars: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)

"""Silver -> gold: per-symbol rolling features, computed once and reused.

Everything is stored as flat float32 arrays over a panel that is sorted by
(ticker, date), together with ``starts``/``ends`` offsets so that numba kernels
can walk one symbol at a time without any pandas overhead.

The parameter search varies ``base_len`` over a small grid, so the rolling
extremes are pre-computed for every candidate length rather than recomputed
inside the search loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

BASE_LENS = (15, 20, 30, 40, 60, 80, 120)


# ---------------------------------------------------------------------------
# numba rolling primitives (per contiguous symbol block)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _roll_max(x, starts, ends, n, out):
    """Monotonic-deque rolling maximum: O(rows), not O(rows * window)."""
    dq = np.empty(len(x), np.int64)
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        head = 0
        tail = 0                      # deque occupies dq[head:tail]
        for i in range(a, b):
            while tail > head and x[dq[tail - 1]] <= x[i]:
                tail -= 1
            dq[tail] = i
            tail += 1
            if dq[head] <= i - n:
                head += 1
            out[i] = x[dq[head]] if i - a >= n - 1 else np.nan
    return out


@njit(cache=True)
def _roll_min(x, starts, ends, n, out):
    dq = np.empty(len(x), np.int64)
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        head = 0
        tail = 0
        for i in range(a, b):
            while tail > head and x[dq[tail - 1]] >= x[i]:
                tail -= 1
            dq[tail] = i
            tail += 1
            if dq[head] <= i - n:
                head += 1
            out[i] = x[dq[head]] if i - a >= n - 1 else np.nan
    return out


@njit(cache=True)
def _roll_mean(x, starts, ends, n, out):
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        acc = 0.0
        for i in range(a, b):
            acc += x[i]
            if i - a >= n:
                acc -= x[i - n]
            if i - a >= n - 1:
                out[i] = acc / n
            else:
                out[i] = np.nan
    return out


@njit(cache=True)
def _roll_std(x, starts, ends, n, out):
    """Rolling sample stdev via running sums - O(rows)."""
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        s1 = 0.0
        s2 = 0.0
        for i in range(a, b):
            s1 += x[i]
            s2 += x[i] * x[i]
            if i - a >= n:
                old = x[i - n]
                s1 -= old
                s2 -= old * old
            if i - a >= n - 1:
                v = (s2 - s1 * s1 / n) / (n - 1)
                out[i] = np.sqrt(v) if v > 0.0 else 0.0
            else:
                out[i] = np.nan
    return out


@njit(cache=True)
def _wilder_atr(high, low, close, starts, ends, n, out):
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        prev = np.nan
        acc = 0.0
        for i in range(a, b):
            if i == a:
                tr = high[i] - low[i]
            else:
                d1 = high[i] - low[i]
                d2 = abs(high[i] - close[i - 1])
                d3 = abs(low[i] - close[i - 1])
                tr = d1
                if d2 > tr:
                    tr = d2
                if d3 > tr:
                    tr = d3
            k = i - a
            if k < n:
                acc += tr
                out[i] = np.nan
                if k == n - 1:
                    prev = acc / n
                    out[i] = prev
            else:
                prev = (prev * (n - 1) + tr) / n
                out[i] = prev
    return out


@njit(cache=True)
def _bars_since_start(starts, ends, out):
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        for i in range(a, b):
            out[i] = i - a
    return out


# ---------------------------------------------------------------------------
class Panel:
    """Flat arrays + symbol offsets + a global integer calendar."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values(["ticker", "date"], ignore_index=True)
        self.tickers = df["ticker"].to_numpy()
        self.dates = df["date"].to_numpy()

        codes, uniq = pd.factorize(df["ticker"], sort=True)
        self.symbols = np.asarray(uniq)
        self.sym_id = codes.astype(np.int32)

        # symbol block boundaries (df is sorted by ticker)
        change = np.flatnonzero(np.r_[True, self.sym_id[1:] != self.sym_id[:-1]])
        self.starts = change.astype(np.int64)
        self.ends = np.r_[change[1:], len(df)].astype(np.int64)

        # global trading calendar -> integer day index
        self.calendar = np.array(sorted(pd.unique(df["date"])))
        self.day = np.searchsorted(self.calendar, self.dates).astype(np.int32)

        self.open = df["open"].to_numpy(np.float64)
        self.high = df["high"].to_numpy(np.float64)
        self.low = df["low"].to_numpy(np.float64)
        self.close = df["close"].to_numpy(np.float64)
        self.volume = df["volume"].to_numpy(np.float64)
        self.n = len(df)

    # ------------------------------------------------------------------
    def _empty(self):
        return np.empty(self.n, dtype=np.float64)

    def build(self) -> dict:
        s, e = self.starts, self.ends
        f: dict[str, np.ndarray] = {}

        f["atr14"] = _wilder_atr(self.high, self.low, self.close, s, e, 14, self._empty())
        f["atr20"] = _wilder_atr(self.high, self.low, self.close, s, e, 20, self._empty())

        for n in (20, 50, 100, 200):
            f[f"ma{n}"] = _roll_mean(self.close, s, e, n, self._empty())

        for n in BASE_LENS:
            # Pivot = highest high of the n bars ENDING YESTERDAY.
            hi = _roll_max(self.high, s, e, n, self._empty())
            lo = _roll_min(self.low, s, e, n, self._empty())
            f[f"hh{n}"] = _shift1(hi, s, e)
            f[f"ll{n}"] = _shift1(lo, s, e)

        f["hh252"] = _shift1(_roll_max(self.high, s, e, 252, self._empty()), s, e)

        dv = self.close * self.volume
        f["addv21"] = _roll_mean(dv, s, e, 21, self._empty())
        f["avgvol50"] = _roll_mean(self.volume, s, e, 50, self._empty())

        ret = np.zeros(self.n)
        _pct_change(self.close, s, e, ret)
        f["ret1"] = ret
        f["vol20"] = _roll_std(ret, s, e, 20, self._empty())
        f["vol60"] = _roll_std(ret, s, e, 60, self._empty())

        f["mom21"] = _lag_ratio(self.close, s, e, 21)
        f["mom63"] = _lag_ratio(self.close, s, e, 63)
        f["mom126"] = _lag_ratio(self.close, s, e, 126)
        f["mom252"] = _lag_ratio(self.close, s, e, 252)

        age = np.empty(self.n)
        f["age"] = _bars_since_start(s, e, age)

        return f


@njit(cache=True)
def _shift1(x, starts, ends):
    out = np.empty_like(x)
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        out[a] = np.nan
        for i in range(a + 1, b):
            out[i] = x[i - 1]
    return out


@njit(cache=True)
def _pct_change(x, starts, ends, out):
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        out[a] = 0.0
        for i in range(a + 1, b):
            out[i] = x[i] / x[i - 1] - 1.0
    return out


@njit(cache=True)
def _lag_ratio(x, starts, ends, n):
    out = np.empty_like(x)
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        for i in range(a, b):
            if i - n < a:
                out[i] = np.nan
            else:
                out[i] = x[i] / x[i - n] - 1.0
    return out

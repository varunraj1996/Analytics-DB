"""Repair unadjusted stock splits in a raw price panel.

The FNSPID panel fetched in `market-data/equity-panel` is not consistently
split-adjusted: AAPL carries a single −74.3% overnight move on 2020-07-06,
which is a 4:1 split expressed as a price gap, and 1,239 such rows exist
across 296 of its 2,897 tickers.

Left alone this is fatal to every strategy in this directory. A phantom −75%
gap is a breakout system's worst input: it fires stops, manufactures new
252-day lows, and poisons every moving average for a year afterwards. It
would also *flatter* a long/short or mean-reversion test, since the price
"recovers" the next day.

The repair back-adjusts prices before a detected split by the split ratio, so
returns become continuous. Detection is deliberately conservative — a move
must sit close to a recognised split ratio, and must not reverse the next
day, because a genuine crash that stays down is not a split and a genuine
crash that bounces is not either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# forward splits (price falls) and reverse splits (price rises)
RATIOS = (1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2 / 3, 3 / 4, 2 / 5,
          2.0, 3.0, 4.0, 5.0, 10.0, 3 / 2, 4 / 3, 5 / 2)
# Tolerance is set from the data rather than assumed. Across the 1,239
# overnight drops worse than -45% in the FNSPID panel, distance-to-nearest-
# clean-ratio is bimodal: the 10th percentile sits at 0.4% (true splits) while
# the median is 10% and the 75th percentile 54% (genuine crashes). 3.5% sits
# in the gap, and catches AAPL's seam at 2.7% — its ratio is 0.2567 rather
# than exactly 0.25 because the source switch carries a day of real return.
TOL = 0.035
REVERSAL = 0.5      # next-day move undoing >50% of it means it was not a split


def detect(close: pd.Series, tol: float = TOL) -> pd.Series:
    """Split ratios indexed by the date the split takes effect.

    A move qualifies only if it is large, close to a recognised ratio, and
    *isolated* — neither the following day nor the preceding day undoes it.
    Checking both sides matters: without the backward check, the recovery leg
    of a one-day round trip reads as a reverse split.
    """
    r = close / close.shift(1)
    nxt = close.shift(-1) / close
    prv = r.shift(1)
    hits = {}
    for i, ratio in r.items():
        if not np.isfinite(ratio) or abs(ratio - 1.0) < 0.30:
            continue
        best = min(RATIOS, key=lambda x: abs(ratio / x - 1.0))
        if abs(ratio / best - 1.0) > tol:
            continue
        inv = 1.0 / best
        # a real split does not un-happen tomorrow, and is not itself the
        # undoing of yesterday
        for other in (nxt.get(i, np.nan), prv.get(i, np.nan)):
            if np.isfinite(other) and abs(other - inv) < REVERSAL * abs(inv - 1.0):
                break
        else:
            hits[i] = best
    return pd.Series(hits, dtype=float).sort_index()


def repair_one(df: pd.DataFrame, price_cols=("open", "high", "low", "close"),
               vol_col: str = "volume") -> tuple:
    """Back-adjust one ticker's OHLCV. Returns (frame, splits_found)."""
    df = df.sort_values("date").copy()
    splits = detect(df.set_index("date")["close"])
    if splits.empty:
        return df, 0
    # every bar strictly before a split is scaled by that split's ratio, so
    # cumulative adjustment is the product of all later ratios
    factor = pd.Series(1.0, index=df["date"].to_numpy())
    for dt, ratio in splits.items():
        factor[factor.index < dt] *= ratio
    f = factor.to_numpy()
    for c in price_cols:
        if c in df:
            df[c] = df[c].to_numpy() * f
    if vol_col in df:                      # shares move the other way
        df[vol_col] = df[vol_col].to_numpy() / f
    return df, len(splits)


def repair_panel(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Back-adjust every ticker in a long-format panel."""
    out, n_split, n_tick = [], 0, 0
    for tic, g in panel.groupby("ticker", observed=True):
        fixed, k = repair_one(g)
        out.append(fixed)
        n_split += k
        n_tick += bool(k)
    res = pd.concat(out, ignore_index=True)
    if verbose:
        print(f"[splits] repaired {n_split} splits across {n_tick} tickers "
              f"of {panel['ticker'].nunique()}")
    return res

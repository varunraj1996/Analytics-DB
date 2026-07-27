"""Signal construction on wide (dates x instruments) matrices.

Every signal is emitted in comparable "forecast" units: an expected-value z
capped at ±FC_CAP, so that combining across signals and instruments is a
weighted average rather than a units negotiation.

Causality: every normalisation uses expanding or trailing windows only.  The
forecast scalar (the divisor that maps a raw signal onto forecast units) is an
*expanding median of absolute values* per instrument - by the time it matters
it is stable, and early noisy values are handled by the warm-up mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FC_CAP = config.DEFAULT_PORTFOLIO.fc_cap
MIN_WARMUP = 130          # bars before an instrument's forecasts are usable


def _scale(raw: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
    """Map a raw signal to forecast units: divide by its own expanding
    mean-absolute value, cap at ±FC_CAP."""
    scal = raw.abs().expanding(min_periods=min_periods).median()
    out = raw / scal.replace(0.0, np.nan)
    return out.clip(-FC_CAP, FC_CAP)


def price_vol(price: pd.DataFrame, span: int) -> pd.DataFrame:
    """EWMA daily vol of price *points* (not returns - adjusted prices can be
    negative, so point vol is the only well-defined risk unit)."""
    dp = price.diff()
    v = dp.ewm(span=span, min_periods=20).std()
    # floor at a tiny fraction of the trailing absolute price to avoid
    # dividing by ~0 in dead markets
    floor = price.abs().rolling(252, min_periods=20).median() * 1e-4
    return v.clip(lower=floor)


def ewmac(price: pd.DataFrame, vol_pts: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    raw = (price.ewm(span=fast, min_periods=fast).mean()
           - price.ewm(span=slow, min_periods=slow).mean()) / vol_pts
    return _scale(raw)


def breakout(price: pd.DataFrame, n: int) -> pd.DataFrame:
    hi = price.rolling(n, min_periods=n // 2).max()
    lo = price.rolling(n, min_periods=n // 2).min()
    mid = (hi + lo) / 2.0
    rng = (hi - lo).replace(0.0, np.nan)
    raw = (price - mid) / rng            # in [-0.5, 0.5]
    sm = raw.ewm(span=max(n // 4, 2), min_periods=2).mean()
    return _scale(sm)


def carry_forecast(carry_ann: pd.DataFrame, price: pd.DataFrame,
                   vol_pts: pd.DataFrame) -> pd.DataFrame:
    """Annualised carry return / annualised return vol == carry Sharpe."""
    ann_vol_ret = vol_pts * np.sqrt(252) / price.abs()
    raw = carry_ann / ann_vol_ret.replace(0.0, np.nan)
    return _scale(raw, min_periods=120)


def skew_signal(price: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Negative skew premium: prefer being long markets with recent negative
    skew (Carver's 'skew abs' family, simplified)."""
    dp = price.diff()
    sk = dp.rolling(window, min_periods=window // 2).skew()
    return _scale(-sk)


def combine(parts: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted average of forecasts, renormalised by the weights actually
    available per cell (an instrument without carry data still trades trend)."""
    num = None
    den = None
    for k, w in weights.items():
        if w == 0 or k not in parts:
            continue
        f = parts[k]
        m = f.notna().astype(float) * w
        fv = f.fillna(0.0) * w
        num = fv if num is None else num + fv
        den = m if den is None else den + m
    if num is None:
        raise ValueError("no signals with positive weight")
    out = num / den.replace(0.0, np.nan)
    return out.clip(-FC_CAP, FC_CAP)

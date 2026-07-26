"""The breakout -> pullback rule, plus everything needed to score a run.

The rule, stated once in plain language:

    A name that has been building a base for `base_len` sessions closes above
    the top of that base on expanding volume, in an uptrend, in a market that
    is itself in an uptrend.  We do **not** chase the breakout.  We wait for
    price to trade back down through a level `pullback_atr` ATRs below the
    breakout close, at any point in the next `entry_window` sessions.  On the
    session where that pullback happens we buy at the close.  Risk is a fixed
    ATR distance, reward a fixed ATR distance, with a time stop.

The short side is the exact mirror (breakdown, rally into resistance).

The pullback level is a *trigger*, not a fill price.  Filling at the level
itself would mean claiming a price whose position within the bar is unknowable,
and doing so inflated the measured edge enough to make the validation window
outperform the train window.  ``scripts/04c_decompose.py`` shows the arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyParams, UniverseSpec

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
def market_regime(bench: pd.DataFrame, calendar: np.ndarray, ma: int) -> np.ndarray:
    """Boolean per calendar day: is the market above its own moving average?

    Uses only information available at the close of the day in question, and
    is forward-filled onto the panel calendar (the benchmark occasionally has
    a session the panel does not, and vice versa).
    """
    b = bench.copy()
    b["ma"] = b["close"].rolling(ma).mean()
    b["ok"] = (b["close"] > b["ma"]).astype(np.int8)
    b = b.set_index("date")["ok"].reindex(pd.DatetimeIndex(calendar), method="ffill")
    return b.fillna(0).to_numpy().astype(bool)


def setup_mask(P, F, p: StrategyParams, uni: UniverseSpec,
               regime_ok: np.ndarray | None,
               member: np.ndarray | None = None) -> np.ndarray:
    """Vectorised breakout (or breakdown) detection over the whole panel."""
    atr = F["atr14"] if p.atr_len == 14 else F["atr20"]
    hh = F[f"hh{p.base_len}"]
    ll = F[f"ll{p.base_len}"]
    c, h, l, v = P.close, P.high, P.low, P.volume

    with np.errstate(invalid="ignore"):
        if p.side > 0:
            m = c > hh + p.breakout_margin * atr
        else:
            m = c < ll - p.breakout_margin * atr

        m &= v > p.vol_mult * F["avgvol50"]

        if p.trend_ma:
            ma = F[f"ma{p.trend_ma}"]
            m &= (c > ma) if p.side > 0 else (c < ma)

        if p.min_base_tightness > 0:
            m &= (hh - ll) <= p.min_base_tightness * atr

        if p.min_close_loc > 0:
            rng = np.maximum(h - l, 1e-9)
            loc = (c - l) / rng if p.side > 0 else (h - c) / rng
            m &= loc >= p.min_close_loc

        if p.max_ext_atr < 99:
            ext = (c - F["ma50"]) / atr
            m &= (ext <= p.max_ext_atr) if p.side > 0 else (-ext <= p.max_ext_atr)

        # tradability
        m &= F["addv21"] >= uni.min_addv_usd
        m &= F["avgvol50"] >= uni.min_share_vol
        m &= c >= uni.min_price
        m &= c <= uni.max_price
        m &= F["age"] >= uni.min_history_days
        m &= np.isfinite(atr) & (atr > 0)
        m &= np.isfinite(hh) & np.isfinite(ll)

        if regime_ok is not None and p.regime_ma:
            m &= regime_ok[P.day]

        if uni.sp500_only:
            if member is None:
                raise ValueError("sp500_only requires the membership mask")
            m &= member

    return np.nan_to_num(m, nan=False).astype(np.bool_)


# ---------------------------------------------------------------------------
def attach_features(t: pd.DataFrame, P, F) -> pd.DataFrame:
    """Snapshot of everything known at the close of the signal day."""
    i = t["sig_row"].to_numpy()
    atr = F["atr14"]
    c = P.close
    out = t.copy()
    out["f_atr_pct"] = atr[i] / c[i]
    out["f_vol20"] = F["vol20"][i]
    out["f_vol_ratio"] = F["vol20"][i] / np.maximum(F["vol60"][i], 1e-9)
    out["f_volsurge"] = P.volume[i] / np.maximum(F["avgvol50"][i], 1.0)
    out["f_mom21"] = F["mom21"][i]
    out["f_mom63"] = F["mom63"][i]
    out["f_mom126"] = F["mom126"][i]
    out["f_mom252"] = F["mom252"][i]
    out["f_ext50"] = (c[i] - F["ma50"][i]) / atr[i]
    out["f_ext200"] = (c[i] - F["ma200"][i]) / atr[i]
    out["f_from52w"] = c[i] / np.maximum(F["hh252"][i], 1e-9) - 1.0
    rng = np.maximum(P.high[i] - P.low[i], 1e-9)
    out["f_closeloc"] = (c[i] - P.low[i]) / rng
    out["f_daygain"] = F["ret1"][i]
    out["f_barrange"] = rng / atr[i]
    for n in (20, 40, 60):
        out[f"f_base{n}"] = (F[f"hh{n}"][i] - F[f"ll{n}"][i]) / atr[i]
    out["f_logaddv"] = np.log10(np.maximum(F["addv21"][i], 1.0))
    out["f_logpx"] = np.log10(np.maximum(c[i], 0.01))
    out["f_age"] = np.log10(np.maximum(F["age"][i], 1.0))
    out["f_wait"] = out["wait"]
    out["f_gap"] = P.open[out["entry_row"].to_numpy()] / c[i] - 1.0
    out["f_pull_depth"] = (c[i] - out["entry_px"]) / atr[i] * out["side"]
    return out


FEATURE_COLS = [
    "f_atr_pct", "f_vol20", "f_vol_ratio", "f_volsurge", "f_mom21", "f_mom63",
    "f_mom126", "f_mom252", "f_ext50", "f_ext200", "f_from52w", "f_closeloc",
    "f_daygain", "f_barrange", "f_base20", "f_base40", "f_base60",
    "f_logaddv", "f_logpx", "f_age", "f_wait", "f_gap", "f_pull_depth",
]


# ---------------------------------------------------------------------------
def metrics(equity: np.ndarray, calendar: np.ndarray,
            day_lo: int, day_hi: int, rf: float = 0.0) -> dict:
    eq = equity[day_lo:day_hi + 1]
    eq = eq[np.isfinite(eq)]
    if len(eq) < 20 or eq[0] <= 0:
        return {}
    r = np.diff(eq) / eq[:-1]
    yrs = len(eq) / TRADING_DAYS
    total = eq[-1] / eq[0]
    cagr = total ** (1 / yrs) - 1 if total > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - rf) / vol if vol > 0 else 0.0
    dn = r[r < 0]
    sortino = (r.mean() * TRADING_DAYS) / (dn.std(ddof=1) * np.sqrt(TRADING_DAYS)) \
        if len(dn) > 2 and dn.std(ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = dd.min()
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_dd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "total_return": total - 1.0, "years": yrs,
        "final_equity": eq[-1],
    }


def trade_stats(t: pd.DataFrame, taken: np.ndarray | None = None) -> dict:
    d = t if taken is None else t[taken]
    if len(d) == 0:
        return {}
    r = d["r_mult"].to_numpy()
    ret = d["ret"].to_numpy()
    win = r > 0
    return {
        "n_trades": len(d),
        "win_rate": win.mean(),
        "avg_r": np.nanmean(r),
        "median_r": np.nanmedian(r),
        "avg_ret": np.nanmean(ret),
        "payoff": (np.nanmean(r[win]) / abs(np.nanmean(r[~win]))) if (~win).any() and win.any() else np.nan,
        "avg_bars": d["bars_held"].mean(),
        "fill_wait": d["wait"].mean(),
        "expectancy_bps": np.nanmean(ret) * 1e4,
    }

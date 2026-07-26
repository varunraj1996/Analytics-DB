"""Walk-forward ML sizing layer for the 10-minute study.

Position, stated up front: **a model on top of a signal cannot create alpha
that the signal does not have.**  The daily study measured this directly - its
meta-labeller scored -38 bps/trade out of sample against +23 bps for random
ordering.  This module exists to answer the question honestly a second time,
with the failure modes from that attempt designed out:

* **Return labels, not win/lose labels.**  Classifying wins teaches the model
  to love small-target/wide-stop shapes; regressing (winsorised) returns with
  a Huber loss ranks by what the portfolio actually eats.
* **Walk-forward only.**  The model that scores month M was fitted on trades
  that *exited* before M began.  There is no single grand fit, so there is no
  window where the model has seen its own future.
* **Sizing, not selection.**  The output multiplies the trade's risk weight in
  [0.5x .. 1.5x].  A model that is pure noise therefore converges to ~1x and
  does no damage; only a model with real ranking power moves the needle.
  (A selection threshold, by contrast, lets a noisy model silently discard
  the trades that would have saved it.)
* **A built-in control.**  Every evaluation reports the same portfolio under
  shuffled scores.  If the model cannot beat its own shuffle, the answer is
  "no model", and that is a reportable result.

Features are all observable at the *trigger* bar (the entry is the next open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

FEATURES = [
    "f_or_range_atr",    # opening range width / ATR (day character)
    "f_trig_dist",       # trigger close vs pullback level, ATR units
    "f_bar_of_day",      # when in the session the entry happens
    "f_atr_pct",         # ATR as a fraction of price (regime vol)
    "f_gap_open",        # session open vs prior close, ATR units
    "f_day_mom",         # session-to-date move at the trigger, ATR units
    "f_vol_surge",       # trigger-bar volume vs session-to-date median
    "f_prev_day_ret",    # prior session's close-to-close return
]


def attach_features(t: pd.DataFrame, book) -> pd.DataFrame:
    """Everything known at the close of the trigger bar (= entry_i - 1)."""
    out = t.copy()
    cols = {f: np.full(len(t), np.nan) for f in FEATURES}

    for sym, grp in t.groupby("symbol"):
        a = book.per_sym[sym]
        idx = grp.index.to_numpy()
        ei = grp["entry_i"].to_numpy().astype(int)
        trig = ei - 1                       # trigger bar
        atr = np.maximum(a["atr"][trig], 1e-9)
        close = a["close"]
        open_ = a["open"]
        high = a["high"]
        low = a["low"]
        day_id = a["day_id"]
        bar = a["bar"]
        side = grp["side"].to_numpy()

        # per-bar session start index
        n = len(close)
        starts = np.flatnonzero(np.r_[True, day_id[1:] != day_id[:-1]])
        sess_start_of = np.zeros(n, np.int64)
        for s, e in zip(starts, np.r_[starts[1:], n]):
            sess_start_of[s:e] = s
        s0 = sess_start_of[trig]
        prior_end = s0 - 1                  # last bar of prior session (or -1)
        pc = np.where(prior_end >= 0, close[np.maximum(prior_end, 0)], np.nan)
        p2 = prior_end - 39                 # ~one session earlier
        pc2 = np.where(p2 >= 0, close[np.maximum(p2, 0)], np.nan)

        # opening range width of the entry session (first 3 bars, fixed)
        or_hi = np.array([high[s:s + 3].max() for s in s0])
        or_lo = np.array([low[s:s + 3].min() for s in s0])
        cols["f_or_range_atr"][idx] = (or_hi - or_lo) / atr
        cols["f_trig_dist"][idx] = (close[trig] - grp["level"].to_numpy()) * side / atr
        cols["f_bar_of_day"][idx] = bar[ei]
        cols["f_atr_pct"][idx] = atr / close[trig]
        cols["f_gap_open"][idx] = (open_[s0] - pc) / atr
        cols["f_day_mom"][idx] = (close[trig] - open_[s0]) / atr * side
        v = a["volume"]
        med = np.array([np.median(v[s:i + 1]) if i >= s else 1.0
                        for s, i in zip(s0, trig)])
        cols["f_vol_surge"][idx] = v[trig] / np.maximum(med, 1.0)
        cols["f_prev_day_ret"][idx] = (pc - pc2) / np.maximum(pc2, 1e-9)

    for f, arr in cols.items():
        out[f] = arr
    return out


def _fit(train: pd.DataFrame, seed: int):
    y = train["ret"].to_numpy(np.float64)
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    m = lgb.LGBMRegressor(
        objective="huber", alpha=0.005, n_estimators=200, learning_rate=0.05,
        num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, max_depth=4, n_jobs=4,
        random_state=seed, verbosity=-1)
    m.fit(train[FEATURES].to_numpy(np.float32), y)
    return m


def walk_forward_scores(t: pd.DataFrame, refit: str = "QE",
                        min_train: int = 800, n_seeds: int = 3) -> np.ndarray:
    """Score every trade using only models fitted on earlier, *exited* trades.

    ``refit`` is a pandas offset alias - models are refitted at each period
    boundary.  Trades in the first period(s), before ``min_train`` completed
    trades exist, stay NaN (unsized -> neutral 1x).
    """
    if lgb is None:
        raise ImportError("lightgbm required")
    t = t.sort_values("entry_ts").reset_index(drop=True)
    scores = np.full(len(t), np.nan)
    # exit timestamp proxy: entry date (intraday strategy -> same session)
    periods = t["date"].dt.to_period("Q" if refit == "QE" else "M")
    for per in periods.unique():
        mask_now = (periods == per).to_numpy()
        mask_past = (t["date"] < per.start_time).to_numpy()
        if mask_past.sum() < min_train:
            continue
        fit_rows = t[mask_past]
        preds = np.mean([
            _fit(fit_rows, seed=11 + s).predict(
                t.loc[mask_now, FEATURES].to_numpy(np.float32))
            for s in range(n_seeds)], axis=0)
        scores[mask_now] = preds
    return scores


def size_multiplier(scores: np.ndarray, lo: float = 0.5, hi: float = 1.5) -> np.ndarray:
    """Map raw scores to risk multipliers by cross-sectional rank.

    NaN (unscored) -> 1.0.  The mapping is rank-based per rolling batch of 250
    trades so that the multiplier distribution is stationary even if the raw
    score scale drifts between refits.
    """
    out = np.ones(len(scores))
    idx = np.flatnonzero(np.isfinite(scores))
    if len(idx) < 50:
        return out
    s = pd.Series(scores[idx])
    rk = s.rolling(250, min_periods=50).apply(
        lambda w: (w.iloc[-1] > w).mean(), raw=False)
    rk = rk.fillna(0.5).to_numpy()
    out[idx] = lo + (hi - lo) * rk
    return out

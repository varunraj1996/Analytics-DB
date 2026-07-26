"""Meta-labelling: let the rule decide *whether* to trade, let a model decide
*which* of the resulting candidates to spend capital on.

The portfolio can hold a dozen positions and the rule fires several thousand
times a year, so on most days the binding decision is not "is this a setup?"
but "which of today's eleven setups do I want?".  That ranking problem is
where a model earns its keep, and it is a much easier learning problem than
predicting returns from scratch: the base rate is already close to even, the
features are all observable at the signal close, and a bad score costs nothing
except a skipped trade.

Training discipline:

* the model only ever sees trades whose **exit** falls inside the train window,
  so nothing it learns depends on a bar the split boundary should have hidden;
* the score threshold and the ranking policy are picked on the validation
  window;
* the test window is scored by a frozen model and never feeds back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from .strategy import FEATURE_COLS, attach_features


def make_labels(t: pd.DataFrame, cost_bps: float = 20.0) -> np.ndarray:
    """1 when the trade cleared a round-trip cost hurdle, else 0.

    Kept for reference, and as a cautionary tale.  Ranking candidates by the
    probability of *winning* actively destroys value here: the configurations
    that win most often are the ones with a small target and a wide stop, so a
    hit-rate model sorts the candidate pool almost exactly backwards.  On the
    validation window its top decile returned 36 bps against 84 bps for its
    bottom decile, with win rate rising monotonically the whole way.  What the
    portfolio needs ranked is expected *return*, which is what ``train_return``
    below fits.
    """
    return (t["ret"].to_numpy() > cost_bps / 1e4).astype(np.int8)


def clipped_returns(t: pd.DataFrame, q: float = 0.005) -> np.ndarray:
    """Trade returns with the extreme tails winsorised.

    A handful of buyouts and gap-downs would otherwise dominate the squared
    error and turn the ranker into an outlier detector.
    """
    r = t["ret"].to_numpy(np.float64)
    lo, hi = np.quantile(r, [q, 1.0 - q])
    return np.clip(r, lo, hi)


def build_matrix(t: pd.DataFrame, P, F) -> pd.DataFrame:
    return attach_features(t, P, F)


def train_return(t_train: pd.DataFrame, feature_cols=None, seed: int = 7,
                 n_estimators: int = 400, **kw):
    """Fit an expected-return ranker on train-window trades only.

    Huber loss rather than L2: trade returns are heavy-tailed enough that
    squared error spends most of its capacity on a few extreme observations.
    """
    if lgb is None:
        raise ImportError("lightgbm is required for the meta-model")
    cols = feature_cols or FEATURE_COLS
    X = t_train[cols].to_numpy(np.float32)
    y = clipped_returns(t_train)

    params = dict(
        objective="huber", alpha=0.01, n_estimators=n_estimators,
        learning_rate=0.03, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=5.0, max_depth=6, n_jobs=4, random_state=seed,
        verbosity=-1,
    )
    params.update(kw)
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def score(model, t: pd.DataFrame, feature_cols=None) -> np.ndarray:
    cols = feature_cols or FEATURE_COLS
    X = t[cols].to_numpy(np.float32)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def bagged_train(t_train: pd.DataFrame, n_models: int = 5, **kw) -> list:
    """A small seed ensemble - single LightGBM fits on noisy financial labels
    are unstable enough that the seed alone can move the validation result."""
    return [train_return(t_train, seed=100 + i, **kw) for i in range(n_models)]


def bagged_score(models: list, t: pd.DataFrame, feature_cols=None) -> np.ndarray:
    return np.mean([score(m, t, feature_cols) for m in models], axis=0)


def oof_deciles(t_train: pd.DataFrame, feature_cols=None, n_folds: int = 4,
                n_models: int = 2) -> pd.DataFrame:
    """Chronological out-of-fold decile report, computed inside TRAIN only.

    This is how the ranker is developed without consuming the validation
    window: fold k trains on everything before it and scores only forward.
    """
    t = t_train.sort_values("entry_day").reset_index(drop=True)
    edges = np.linspace(0, len(t), n_folds + 1).astype(int)
    parts = []
    for k in range(1, n_folds):
        fit = t.iloc[:edges[k]]
        hold = t.iloc[edges[k]:edges[k + 1]]
        if len(fit) < 2000 or len(hold) < 500:
            continue
        ms = bagged_train(fit, n_models=n_models, feature_cols=feature_cols)
        s = bagged_score(ms, hold, feature_cols)
        parts.append(pd.DataFrame({"score": s, "ret": hold["ret"].to_numpy(),
                                   "fold": k}))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["bucket"] = d.groupby("fold")["score"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    return (d.groupby("bucket")
            .agg(n=("ret", "size"), mean_bps=("ret", lambda x: x.mean() * 1e4),
                 win=("ret", lambda x: (x > 0).mean()))
            .reset_index())


def importance(model, feature_cols=None) -> pd.DataFrame:
    cols = feature_cols or FEATURE_COLS
    return (pd.DataFrame({"feature": cols, "gain": model.booster_.feature_importance("gain")})
            .sort_values("gain", ascending=False, ignore_index=True))


def decile_report(t: pd.DataFrame, s: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Does the score actually separate winners from losers out of sample?"""
    d = pd.DataFrame({"score": s, "ret": t["ret"].to_numpy(),
                      "r": t["r_mult"].to_numpy()})
    d["bucket"] = pd.qcut(d["score"], n, labels=False, duplicates="drop")
    g = d.groupby("bucket").agg(
        n=("ret", "size"), mean_bps=("ret", lambda x: x.mean() * 1e4),
        win=("ret", lambda x: (x > 0).mean()), avg_r=("r", "mean"))
    return g.reset_index()

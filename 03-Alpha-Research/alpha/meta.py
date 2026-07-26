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
    """1 when the trade cleared a round-trip cost hurdle, else 0."""
    return (t["ret"].to_numpy() > cost_bps / 1e4).astype(np.int8)


def build_matrix(t: pd.DataFrame, P, F) -> pd.DataFrame:
    return attach_features(t, P, F)


def train(t_train: pd.DataFrame, feature_cols=None, cost_bps: float = 20.0,
          seed: int = 7, n_estimators: int = 400, **kw):
    """Fit the candidate ranker on train-window trades only."""
    if lgb is None:
        raise ImportError("lightgbm is required for the meta-model")
    cols = feature_cols or FEATURE_COLS
    X = t_train[cols].to_numpy(np.float32)
    y = make_labels(t_train, cost_bps)

    params = dict(
        objective="binary", n_estimators=n_estimators, learning_rate=0.03,
        num_leaves=31, min_child_samples=200, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        max_depth=6, n_jobs=4, random_state=seed, verbosity=-1,
    )
    params.update(kw)
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def score(model, t: pd.DataFrame, feature_cols=None) -> np.ndarray:
    cols = feature_cols or FEATURE_COLS
    X = t[cols].to_numpy(np.float32)
    return model.predict_proba(X)[:, 1]


def bagged_train(t_train: pd.DataFrame, n_models: int = 5, **kw) -> list:
    """A small seed ensemble - single LightGBM fits on noisy financial labels
    are unstable enough that the seed alone can move the validation result."""
    return [train(t_train, seed=100 + i, **kw) for i in range(n_models)]


def bagged_score(models: list, t: pd.DataFrame, feature_cols=None) -> np.ndarray:
    return np.mean([score(m, t, feature_cols) for m in models], axis=0)


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

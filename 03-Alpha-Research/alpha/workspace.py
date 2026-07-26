"""Loads the panel once, caches features, and exposes the split boundaries.

Every script in ``scripts/`` starts by calling :func:`load` so that the train /
validate / test day indices are derived in exactly one place.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

from . import config, data, features, strategy, universe

CACHE = os.path.join(config.DATA_DIR, "workspace.pkl")


class Workspace:
    def __init__(self, P, F, regime: dict, calendar: np.ndarray,
                 member: np.ndarray | None = None):
        self.P = P
        self.F = F
        self.regime = regime
        self.calendar = calendar
        self.n_days = len(calendar)
        self.member = member

    def span(self, split: tuple[str, str]) -> tuple[int, int]:
        lo = int(np.searchsorted(self.calendar, np.datetime64(split[0]), "left"))
        hi = int(np.searchsorted(self.calendar, np.datetime64(split[1]), "right")) - 1
        return lo, hi

    @property
    def train(self):
        return self.span(config.TRAIN)

    @property
    def valid(self):
        return self.span(config.VALID)

    @property
    def test(self):
        return self.span(config.TEST)

    def label(self, day: int) -> str:
        return str(pd.Timestamp(self.calendar[day]).date())


def load(rebuild: bool = False) -> Workspace:
    if os.path.exists(CACHE) and not rebuild:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    t0 = time.time()
    panel = data.load_panel()
    print(f"[ws] panel {len(panel):,} rows in {time.time() - t0:.1f}s")

    P = features.Panel(panel)
    del panel
    t0 = time.time()
    F = P.build()
    print(f"[ws] features {len(F)} arrays in {time.time() - t0:.1f}s")

    bench = data.load_benchmark("spy")
    regime = {ma: strategy.market_regime(bench, P.calendar, ma)
              for ma in (50, 100, 200)}

    member = universe.membership_mask(P)
    print(f"[ws] S&P500 point-in-time rows: {member.sum():,} "
          f"({member.mean():.1%} of panel)")

    ws = Workspace(P, F, regime, P.calendar, member)
    with open(CACHE, "wb") as fh:
        pickle.dump(ws, fh, protocol=4)
    return ws

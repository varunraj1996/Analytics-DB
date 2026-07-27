"""Shared plumbing: load panels into wide matrices, evaluate one config."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, portfolio, signals


class Universe:
    """Wide matrices plus per-config forecast assembly, computed lazily and
    cached so a sweep over weightings re-uses every signal matrix."""

    def __init__(self, panel: pd.DataFrame, meta: pd.DataFrame):
        self.price = panel.pivot(index="date", columns="instrument", values="price")
        self.fx = (panel.pivot(index="date", columns="instrument", values="fx")
                   .ffill().fillna(1.0)) if "fx" in panel else \
            pd.DataFrame(1.0, index=self.price.index, columns=self.price.columns)
        self.carry = panel.pivot(index="date", columns="instrument",
                                 values="carry_ann") if "carry_ann" in panel else None
        self.meta = meta
        self.vol = signals.price_vol(self.price, config.DEFAULT_PORTFOLIO.vol_lookback)
        self._cache: dict[str, pd.DataFrame] = {}
        self._weights_cache: pd.DataFrame | None = None

    def sig(self, name: str) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name]
        if name.startswith("ewmac"):
            f, s = name[5:].split("_")
            m = signals.ewmac(self.price, self.vol, int(f), int(s))
        elif name.startswith("brk"):
            m = signals.breakout(self.price, int(name[3:]))
        elif name == "carry":
            if self.carry is None:
                m = pd.DataFrame(np.nan, index=self.price.index,
                                 columns=self.price.columns)
            else:
                m = signals.carry_forecast(self.carry, self.price, self.vol)
        elif name == "skew":
            m = signals.skew_signal(self.price)
        else:
            raise KeyError(name)
        self._cache[name] = m
        return m

    def forecast(self, weights: dict[str, float]) -> pd.DataFrame:
        parts = {k: self.sig(k) for k, w in weights.items() if w > 0}
        return signals.combine(parts, weights)

    def evaluate(self, weights: dict[str, float],
                 spec: config.PortfolioSpec | None = None,
                 classes: list[str] | None = None,
                 cost_mult: float = 1.0, idm: float = 2.0,
                 vol_scale: bool = True) -> dict:
        fc = self.forecast(weights)
        meta = self.meta
        price, fx, vol = self.price, self.fx, self.vol
        if classes is not None:
            keep = meta[meta["asset_class"].isin(classes)]["instrument"]
            cols = [c for c in price.columns if c in set(keep)]
            price, fx, vol, fc = (price[cols], fx[cols], vol[cols], fc[cols])
            meta = meta[meta["instrument"].isin(cols)]
        return portfolio.run(price, fx, vol, fc, meta, spec=spec,
                             idm=idm, cost_mult=cost_mult, vol_scale=vol_scale)


def load_futures() -> Universe:
    panel = pd.read_parquet(config.FUT_PANEL)
    meta = pd.read_parquet(config.FUT_META)
    return Universe(panel, meta)


def load_sg() -> Universe:
    """Cross-check universe.  No per-instrument spread data, so conservative
    class-level half-spreads are synthesised in price points via typical
    spread fractions: FX 1.5 bps, indices 2 bps, metals 3 bps, crypto 8 bps,
    stock CFDs 3 bps of price."""
    panel = pd.read_parquet(config.SG_PANEL)
    frac = {"forex": 1.5e-4, "indices": 2e-4, "commodities": 3e-4,
            "crypto": 8e-4, "stock": 3e-4}
    last = panel.groupby("instrument").agg(price=("price", "last"),
                                           asset_class=("asset_class", "last"))
    meta = pd.DataFrame({
        "instrument": last.index,
        "asset_class": last["asset_class"].to_numpy(),
        "point_size": 1.0,
        "currency": "USD",
        "spread_points": (last["price"]
                          * last["asset_class"].map(frac).fillna(5e-4)).to_numpy(),
        "years": 10.0,
    })
    return Universe(panel, meta)


def wmetrics(res: dict) -> dict[str, dict]:
    r = res["ret"]
    return {
        "train": portfolio.metrics(r, None, config.TRAIN_END),
        "valid": portfolio.metrics(r, "2010-01-01", config.VALID_END),
        "test": portfolio.metrics(r, "2017-01-01", config.TEST_END),
        "full": portfolio.metrics(r),
    }

"""Central configuration for the breakout-pullback alpha research pipeline.

Everything that a downstream stage needs to know about *where* data lives and
*how* the sample is split lives here, so that no stage can silently disagree
with another about the train / validate / test boundaries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Raw vendor files and generated artefacts are deliberately kept OUT of the git
# repository - only code and results are versioned.
SCRATCH = os.environ.get(
    "ALPHA_SCRATCH",
    "/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad",
)
RAW_DIR = os.path.join(SCRATCH, "kaggle_huge", "data")
SP500_DIR = os.path.join(SCRATCH, "testclone")
DATA_DIR = os.path.join(SCRATCH, "data")
RESULTS_DIR = os.path.join(SCRATCH, "results")

PANEL_PARQUET = os.path.join(DATA_DIR, "panel.parquet")
FEATURES_NPZ = os.path.join(DATA_DIR, "features.npz")

for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Sample splits  (walk-forward, strictly chronological, no overlap)
# --------------------------------------------------------------------------
# The vendor panel ends 2017-11-10.  We discard the pre-decimalisation era
# (tick sizes of 1/16 make intraday limit-fill modelling unrealistic) and keep
# a clean 2001+ sample.
SAMPLE_START = "2001-01-01"

TRAIN = ("2001-01-01", "2011-12-31")   # parameter search happens ONLY here
VALID = ("2012-01-01", "2014-12-31")   # model / threshold selection
TEST = ("2015-01-01", "2017-11-10")    # touched exactly once, at the very end

# --------------------------------------------------------------------------
# Trading / cost assumptions
# --------------------------------------------------------------------------


@dataclass
class CostModel:
    """Conservative round-trip cost assumptions.

    ``base_bps`` is charged on every fill (entry and exit).  ``impact_coef``
    adds a size-dependent term that grows as a name becomes less liquid:
    ``impact_coef / sqrt(ADDV in $m)`` basis points.  A hard floor of half a
    cent per share is applied so that low-priced names are not free to trade.
    """

    base_bps: float = 5.0
    impact_coef: float = 5.0
    min_cents_per_share: float = 0.005


@dataclass
class PortfolioSpec:
    starting_equity: float = 1_000_000.0
    risk_per_trade: float = 0.0075      # fraction of equity risked to the stop
    max_positions: int = 12
    max_weight: float = 0.20            # cap on any single position
    max_new_per_day: int = 6
    gross_target: float = 1.0           # 1.0 == fully invested, no margin


@dataclass
class UniverseSpec:
    """Point-in-time tradability screens.

    ``max_price`` and ``min_share_vol`` exist for a specific data reason: the
    vendor series are back-adjusted and anchored at the last print, so a name
    that later did a 1-for-100 reverse split shows a *hundredfold inflated*
    price (and dollar volume) in its early history.  A share-count floor is
    immune to that distortion, and a price ceiling removes the worst ghosts.
    See ``universe.py`` for the S&P-500 point-in-time cross-check that pins
    down how much this matters.
    """

    min_price: float = 5.0
    max_price: float = 1_000.0
    min_addv_usd: float = 5_000_000.0   # 21-day average daily dollar volume
    min_share_vol: float = 200_000.0    # 50-day average share volume
    min_history_days: int = 260         # at least ~1y of prints before trading
    sp500_only: bool = False            # clean-universe cross-check


DEFAULT_COSTS = CostModel()
DEFAULT_PORTFOLIO = PortfolioSpec()
DEFAULT_UNIVERSE = UniverseSpec()


@dataclass
class StrategyParams:
    """Breakout -> pullback -> intraday limit entry.

    All levels are expressed in ATR units so that the rule is scale-free and
    transfers across the whole cross-section.
    """

    side: int = 1                 # +1 breakout/long, -1 breakdown/short
    base_len: int = 40            # lookback that defines the breakout pivot
    breakout_margin: float = 0.0  # required excess over the pivot, in ATR
    min_base_tightness: float = 0.0   # max (range / ATR) of the base; 0 = off
    vol_mult: float = 1.5         # breakout-day volume vs 50d average
    trend_ma: int = 200           # long-term trend filter (0 = off)
    regime_ma: int = 200          # market regime filter on SPY (0 = off)
    min_close_loc: float = 0.0    # where the breakout bar closed in its range
    max_ext_atr: float = 99.0     # reject names already extended from the 50dma

    pullback_atr: float = 0.5     # limit sits this many ATR below breakout close
    entry_window: int = 3         # trading days the limit order stays live

    stop_atr: float = 1.5
    target_atr: float = 3.0
    max_hold: int = 10            # time stop, trading days
    trail_atr: float = 0.0        # 0 = off, else chandelier trail in ATR

    atr_len: int = 14

    def key(self) -> tuple:
        return tuple(sorted(self.__dict__.items()))

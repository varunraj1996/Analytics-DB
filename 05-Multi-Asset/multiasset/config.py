"""Configuration for the multi-asset trend/carry program."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

SCRATCH = os.environ.get(
    "ALPHA_SCRATCH",
    "/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad",
)
PST = os.path.join(SCRATCH, "pst", "data", "futures")
SG = os.path.join(SCRATCH, "intraday_sg")
DATA_DIR = os.path.join(SCRATCH, "data_ma")
RESULTS_DIR = os.path.join(SCRATCH, "results_ma")
for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

FUT_PANEL = os.path.join(DATA_DIR, "futures_panel.parquet")
FUT_META = os.path.join(DATA_DIR, "futures_meta.parquet")
SG_PANEL = os.path.join(DATA_DIR, "sg_panel.parquet")

# Splits (see STATE.md; fixed before any results were looked at)
TRAIN_END = "2009-12-31"
VALID_END = "2016-12-31"
TEST_END = "2024-03-28"

# ---------------------------------------------------------------------------
# Universe hygiene
MIN_YEARS = 8.0          # instrument must have at least this much history
MAX_FFILL_DAYS = 5       # forward-fill limit when aligning to the calendar

# Instruments that duplicate another contract on the list (micro/mini variants
# of the same underlying).  The full-size contract is kept.
DUPLICATE_SUFFIXES = ("_micro", "_mini", "_micro_OLD", "_mini_OLD")

# Asset-class default spread costs in PRICE POINTS where spreadcosts.csv has
# no entry - deliberately punitive so unknown-cost instruments do not sneak in
# cheaply.  (SpreadCost in pysystemtrade is half the bid-ask in points.)
DEFAULT_SPREAD_BY_CLASS = {
    "Equity": None,      # None = drop if unknown
    "Bond": None,
    "FX": None,
    "Metals": None,
    "OilGas": None,
    "Ags": None,
    "STIR": None,
    "Vol": None,
    "Crypto": None,
    "Other": None,
}


@dataclass
class PortfolioSpec:
    capital: float = 1_000_000.0
    vol_target: float = 0.25          # annualised portfolio vol target
    idm_cap: float = 2.5              # cap on the diversification multiplier
    instrument_weight_cap: float = 0.10
    vol_lookback: int = 35            # EWMA span for instrument price vol
    rebalance_buffer: float = 0.10    # fraction of position; trade only outside
    fc_cap: float = 2.0               # signal cap in z-units


DEFAULT_PORTFOLIO = PortfolioSpec()

EWMAC_PAIRS = ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256))
BREAKOUT_WINDOWS = (20, 40, 80, 160, 320)

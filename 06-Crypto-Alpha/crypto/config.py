"""Configuration for the crypto / on-chain study."""
from __future__ import annotations

import os
from dataclasses import dataclass

SCRATCH = os.environ.get(
    "ALPHA_SCRATCH",
    "/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad",
)
CM_DIR = os.path.join(SCRATCH, "cm")                 # coinmetrics csv dump
DATA_DIR = os.path.join(SCRATCH, "data_crypto")
RESULTS_DIR = os.path.join(SCRATCH, "results_crypto")
for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

PANEL = os.path.join(DATA_DIR, "crypto_panel.parquet")
COVERAGE = os.path.join(DATA_DIR, "coverage.parquet")

# Splits chosen before any result was inspected.  The test window deliberately
# spans the spot-ETF era (IBIT launched 2024-01-11) and the 2024-25 cycle.
TRAIN_END = "2021-12-31"
VALID_END = "2023-12-31"
TEST_END = "2026-05-23"

# Costs: crypto spot is dearer than futures.  10 bps per side is roughly a
# taker fee plus slippage on a major, and is punitive for BTC/ETH where a
# patient maker order costs far less.
DEFAULT_HALF_SPREAD_BPS = 10.0

# Signal lag: an on-chain metric stamped for day T is published after T closes,
# and Coin Metrics revises it.  Every feature is shifted by this many days
# before it can influence a position.
#
# This was 1 ("act on the next day's open") and that was look-ahead. The
# `AssetEODCompletionTime` column records when each day's data was actually
# finalised, and across BTC/ETH/LTC/ADA/XRP/DOGE the median lag is 24.5-27.1
# hours after the T stamp — later than T+1 00:00Z on **100%** of recent days.
# A position opened at T+1 on a signal stamped T was therefore trading on a
# number Coin Metrics had not yet published. 2 is the smallest honest value.
SIGNAL_LAG = 2

FC_CAP = 2.0


@dataclass
class PortfolioSpec:
    capital: float = 1_000_000.0
    vol_target: float = 0.30        # crypto: higher target than the futures book
    vol_lookback: int = 35
    max_weight: float = 0.35
    rebalance_buffer: float = 0.15
    long_only: bool = False


DEFAULT_PORTFOLIO = PortfolioSpec()

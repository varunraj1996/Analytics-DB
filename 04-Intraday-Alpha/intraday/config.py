"""Configuration for the 10-minute intraday study.

This is the successor to ``03-Alpha-Research``, which established two things:
the daily-bar version of this strategy has no persistent edge, and every
attempt to price an intraday entry off daily OHLC manufactures one.  The fix
is real intraday bars, which is what this study uses - at the cost of a much
smaller universe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

SCRATCH = os.environ.get(
    "ALPHA_SCRATCH",
    "/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad",
)
PK_DIR = os.path.join(SCRATCH, "intraday_pk", "data")       # 5-min parquet, UTC
SG_DIR = os.path.join(SCRATCH, "intraday_sg")               # 5-min csv, UTC
DATA_DIR = os.path.join(SCRATCH, "data10")
RESULTS_DIR = os.path.join(SCRATCH, "results10")
PANEL_PARQUET = os.path.join(DATA_DIR, "panel10.parquet")

for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Bars: regular session New York, 09:30-16:00 -> 39 ten-minute bars/day.
BAR_MINUTES = 10
SESSION_BARS = 39

# --------------------------------------------------------------------------
# Splits.  The two sources overlap 2020-2023; only piekstra covers 2024+.
# Everything before 2022 is kept but flagged: the Alpaca feed behind the ETF
# data is visibly sparse in 2020-21 (~55-68 five-minute rows/day vs 78+).
TRAIN = ("2018-08-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-02-13")

# --------------------------------------------------------------------------


@dataclass
class CostModel:
    """Per-side costs in basis points.

    These symbols (TQQQ, SOXL, SPXL, ...) are among the most liquid ETFs in
    the world with 1-2 cent spreads on $20-100 prices; AAPL/TSLA/NFLX likewise.
    5 bps per side is deliberately harsher than reality to leave margin for
    the fills we model at bar opens.
    """

    per_side_bps: float = 5.0


@dataclass
class PortfolioSpec:
    starting_equity: float = 1_000_000.0
    max_positions: int = 6
    risk_per_trade: float = 0.01     # equity fraction risked to the stop
    max_weight: float = 0.35         # per-name cap
    gross_target: float = 1.0


@dataclass
class IntradayParams:
    """Opening-range / rolling-high breakout with a pullback re-entry.

    All in 10-minute bars and ATR units, where ATR is computed over the last
    ``atr_bars`` 10-minute bars (cross-session).
    """

    side: int = 1
    level_kind: str = "or"      # "or" = opening range high, "pdh" = prior day high
    or_bars: int = 3            # opening range length (3 = first 30 minutes)
    confirm_atr: float = 0.0    # close must clear the level by this much
    pullback_atr: float = 0.5   # retrace this far below the breakout close
    max_wait_bars: int = 12     # pullback must trigger within this many bars
    entry: str = "next_open"    # execution at the next bar's open

    stop_atr: float = 1.5
    target_atr: float = 3.0
    max_hold_bars: int = 24     # 4 hours
    eod_flat: bool = True       # always flat by the session close

    atr_bars: int = 39          # one session of trailing ATR
    min_or_range_atr: float = 0.0   # optional: skip dead opens
    trend_filter: str = "none"  # "none" | "above_pdc" (close above prior day close)


DEFAULT_COSTS = CostModel()
DEFAULT_PORTFOLIO = PortfolioSpec()

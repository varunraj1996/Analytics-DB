"""The trade card: what the agent is asked to pass judgement on.

One card describes one intended action (open, add, trim, exit, or a stop
move) with the state a Qullamaggie-style decision actually depends on — the
setup geometry, the risk arithmetic, the portfolio's current exposure, and
the trader's own recent feedback (equity curve, whether their breakouts are
working).  Everything is optional except ticker/action, because a real
front-end will not always have every field; the guardrails distinguish
"violates the rule" from "cannot tell", and only the first can veto.

The ``rationale`` field is free text written by a human (or another model).
It is treated as untrusted data everywhere downstream — see ``judge.py``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Optional

ACTIONS = ("OPEN", "ADD", "TRIM", "EXIT", "MOVE_STOP")
REGIMES = ("PRESS", "PROBE", "STAND_DOWN")


@dataclass
class TradeCard:
    # --- identity ---------------------------------------------------------
    ticker: str
    action: str = "OPEN"
    trade_id: str = ""
    date: str = ""

    # --- setup geometry ---------------------------------------------------
    setup: str = ""                          # e.g. "pullback_bounce", "ep"
    prior_leg_pct: Optional[float] = None    # 0.45 = +45% leg into the base
    prior_leg_months: Optional[float] = None
    pullback_days: Optional[int] = None
    base_depth_pct: Optional[float] = None   # 0.18 = 18% off the leg high
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    ma10_rising: Optional[bool] = None
    touched_ma: Optional[bool] = None        # pullback actually reached 10/20
    volume_dryup_ratio: Optional[float] = None   # 3d vol / 50d avg vol, <1 dry
    trigger_close_above_prior_high: Optional[bool] = None
    trigger_volume_expansion: Optional[float] = None   # today / prior day vol
    base_count: Optional[int] = None         # 1st base off lows … 4th base

    # --- price and risk ---------------------------------------------------
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    original_stop: Optional[float] = None    # for MOVE_STOP checks
    consolidation_low: Optional[float] = None
    breakout_level: Optional[float] = None
    adr_pct: Optional[float] = None          # ADR20 as a fraction of price
    shares: Optional[float] = None
    equity: Optional[float] = None
    cost_basis: Optional[float] = None       # for ADD

    # --- portfolio --------------------------------------------------------
    exposure_level: Optional[int] = None     # 0..4 throttle setting
    gross_exposure_pct: Optional[float] = None   # 0.80 = 80% of equity
    open_positions: Optional[int] = None
    baseline_risk_pct: Optional[float] = None    # the trader's normal risk unit

    # --- regime read (P1) -------------------------------------------------
    regime_call: str = ""                    # PRESS | PROBE | STAND_DOWN
    breakout_follow_through: Optional[float] = None  # own recent breakouts ok
    equity_curve_20d_pct: Optional[float] = None     # own equity, last 20d
    equity_at_new_high: Optional[bool] = None
    consecutive_losses: Optional[int] = None
    index_above_20dma: Optional[bool] = None
    index_20dma_rising: Optional[bool] = None
    breadth_ratio10: Optional[float] = None
    leaders_clustering: Optional[bool] = None

    # --- leadership (P10) -------------------------------------------------
    rs_percentile: Optional[float] = None    # 0.95 = top 5% of the universe
    sector: str = ""
    sector_rank_pct: Optional[float] = None  # 1.0 = strongest group that day

    # --- exit plan (P8/P9) ------------------------------------------------
    partial_target_r: Optional[float] = None
    partial_days: Optional[int] = None
    move_to_breakeven_after_partial: Optional[bool] = None
    trail_ma: str = ""                       # "ma10" | "ma20" | "ma50" | ""
    hard_target_price: Optional[float] = None   # a cap on winners = P9 breach

    # --- liquidity --------------------------------------------------------
    addv_usd: Optional[float] = None

    # --- free text (UNTRUSTED) -------------------------------------------
    rationale: str = ""

    # ------------------------------------------------------------------
    # derived quantities — all return None when inputs are missing
    # ------------------------------------------------------------------
    @property
    def risk_per_share(self) -> Optional[float]:
        if self.entry_price is None or self.stop_price is None:
            return None
        return self.entry_price - self.stop_price

    @property
    def risk_in_adr(self) -> Optional[float]:
        r = self.risk_per_share
        if r is None or not self.adr_pct or not self.entry_price:
            return None
        return r / (self.adr_pct * self.entry_price)

    @property
    def account_risk_pct(self) -> Optional[float]:
        """Fraction of equity at risk to the stop."""
        r = self.risk_per_share
        if r is None or not self.shares or not self.equity:
            return None
        return r * self.shares / self.equity

    @property
    def position_weight_pct(self) -> Optional[float]:
        if not (self.shares and self.entry_price and self.equity):
            return None
        return self.shares * self.entry_price / self.equity

    @property
    def extension_adr(self) -> Optional[float]:
        """How far above the 10dma the entry is, in average daily ranges.

        His own framing: extended above the moving average is where you sell
        into strength, so a large positive number on an OPEN is a chase.
        """
        if not (self.entry_price and self.ma10 and self.adr_pct):
            return None
        return (self.entry_price - self.ma10) / (self.adr_pct * self.entry_price)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "TradeCard":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    def facts_for_judge(self) -> dict:
        """Card plus derived numbers, with the untrusted text pulled out."""
        d = self.to_dict()
        d.pop("rationale", None)
        for k in ("risk_per_share", "risk_in_adr", "account_risk_pct",
                  "position_weight_pct", "extension_adr"):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(float(v), 4)
        return d


def load_jsonl(path) -> list:
    """Read trade cards from JSONL; ignores blank lines and ``#`` comments."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out

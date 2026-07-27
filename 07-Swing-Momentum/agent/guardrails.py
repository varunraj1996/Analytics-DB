"""Deterministic guardrails — the arithmetic half of the compliance agent.

Design rules, which are the reason this layer exists at all:

1. **No model in the loop.** Every check here is a comparison between
   numbers on the card and a policy threshold. Same card in, same findings
   out, forever. The LLM judge (``judge.py``) handles what cannot be
   reduced to arithmetic — is the base *orderly*, is the regime read honest,
   does the rationale smell like revenge trading.
2. **Guardrails outrank the judge.** A ``VETO`` here cannot be argued away
   by the model; the judge may only add severity. Prompt injection in the
   free-text ``rationale`` therefore cannot unblock a trade.
3. **Missing is not passing, and missing is not failing.** A check whose
   inputs are absent emits ``INFO`` (``*_UNKNOWN``) rather than silently
   passing. What you cannot measure, you flag.

Thresholds live in ``Policy`` so a desk can tighten them without touching
rule logic, and so the evals can prove the defaults behave as documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .principles import CODE_TO_PRINCIPLE
from .trade_card import TradeCard

VETO, FLAG, INFO = "VETO", "FLAG", "INFO"
_SEV_RANK = {INFO: 0, FLAG: 1, VETO: 2}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    evidence: str = ""

    @property
    def principle(self) -> str:
        return CODE_TO_PRINCIPLE.get(self.code, "")

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "principle": self.principle, "message": self.message,
                "evidence": self.evidence}


@dataclass
class Policy:
    """Thresholds. Defaults are his stated numbers, not tuned ones."""
    # P7 — risk
    max_risk_adr_flag: float = 1.0        # stop wider than 1 ADR: warn
    max_risk_adr_veto: float = 1.5        # …wider than 1.5 ADR: refuse
    max_account_risk_flag: float = 0.010  # 1.0% of equity
    max_account_risk_veto: float = 0.020  # 2.0%
    max_position_weight: float = 0.35     # single name vs equity
    size_tolerance: float = 0.25          # |actual/intended - 1| allowed
    # P4 — setup
    min_prior_leg: float = 0.30
    max_base_depth: float = 0.25
    min_pullback_days: int = 3
    max_pullback_days: int = 15
    dryup_ratio_max: float = 1.0          # 3d volume must be under 50d avg
    late_base_count: int = 4
    # P5 — extension
    max_extension_adr_flag: float = 3.0
    max_extension_adr_veto: float = 6.0
    # P10 — leadership
    min_rs_percentile: float = 0.90
    min_sector_rank_pct: float = 0.75
    # P8/P9 — exits
    max_partial_days: int = 5
    allowed_trail_ma: tuple = ("ma10", "ma20", "ma50")
    # P2 — exposure ladder: level -> (min gross, max gross) as % of equity
    exposure_bands: dict = field(default_factory=lambda: {
        0: (0.00, 0.05), 1: (0.00, 0.25), 2: (0.20, 0.60),
        3: (0.50, 1.00), 4: (0.90, 2.00)})
    regime_levels: dict = field(default_factory=lambda: {
        "STAND_DOWN": (0, 1), "PROBE": (1, 2), "PRESS": (2, 4)})
    # P3 — feedback loop
    drawdown_cut_threshold: float = -0.05   # equity -5% over 20d => cut, not press
    loss_streak_cut: int = 3
    # liquidity
    min_addv_usd: float = 3e6


# ---------------------------------------------------------------------------
# individual checks. each appends Findings; none of them raise.
# ---------------------------------------------------------------------------
def _unknown(out, code, what):
    out.append(Finding(code, INFO, f"cannot verify: {what} missing",
                       "field absent from trade card"))


def _check_stop(c: TradeCard, p: Policy, out: list) -> None:
    # MOVE_STOP is exempt: raising a stop to breakeven or onto the moving
    # average puts it above entry on purpose. _check_stop_move owns that case.
    if c.action in ("TRIM", "EXIT", "MOVE_STOP"):
        return
    if c.entry_price is None or c.stop_price is None:
        _unknown(out, "STOP_NOT_STRUCTURAL", "entry_price/stop_price")
        return
    if c.stop_price >= c.entry_price:
        out.append(Finding("STOP_ABOVE_ENTRY", VETO,
                           "stop is at or above the entry — no defined risk",
                           f"entry {c.entry_price} stop {c.stop_price}"))
        return
    ref, name = None, ""
    if c.consolidation_low is not None:
        ref, name = c.consolidation_low, "consolidation_low"
    elif c.breakout_level is not None:
        ref, name = c.breakout_level, "breakout_level"
    if ref is None:
        _unknown(out, "STOP_NOT_STRUCTURAL", "consolidation_low/breakout_level")
    elif c.stop_price > ref * 1.001:
        out.append(Finding("STOP_NOT_STRUCTURAL", FLAG,
                           "stop sits above the level the setup defines; a "
                           "normal wiggle takes it out",
                           f"stop {c.stop_price} > {name} {ref}"))


def _check_stop_move(c: TradeCard, p: Policy, out: list) -> None:
    if c.original_stop is None or c.stop_price is None:
        return
    if c.stop_price < c.original_stop * 0.999:
        out.append(Finding("STOP_WIDENED", VETO,
                           "stop was moved lower after entry — stops only "
                           "ever tighten",
                           f"{c.original_stop} -> {c.stop_price}"))


def _check_risk(c: TradeCard, p: Policy, out: list) -> None:
    if c.action in ("TRIM", "EXIT", "MOVE_STOP"):
        return
    radr = c.risk_in_adr
    if radr is None:
        _unknown(out, "RISK_TOO_WIDE_ADR", "adr_pct or stop distance")
    elif radr > p.max_risk_adr_veto:
        out.append(Finding("RISK_TOO_WIDE_ADR", VETO,
                           "stop is further than the hard ADR limit — this is "
                           "the filter he says he will not bend; skip it",
                           f"risk {radr:.2f} ADR > {p.max_risk_adr_veto}"))
    elif radr > p.max_risk_adr_flag:
        out.append(Finding("RISK_TOO_WIDE_ADR", FLAG,
                           "stop wider than one ADR — noise will hit it",
                           f"risk {radr:.2f} ADR"))

    ar = c.account_risk_pct
    if ar is None:
        _unknown(out, "ACCOUNT_RISK_EXCEEDED", "shares or equity")
    elif ar > p.max_account_risk_veto:
        out.append(Finding("ACCOUNT_RISK_EXCEEDED", VETO,
                           "risk to the stop exceeds the hard per-trade cap",
                           f"{ar:.2%} of equity > {p.max_account_risk_veto:.2%}"))
    elif ar > p.max_account_risk_flag:
        out.append(Finding("ACCOUNT_RISK_EXCEEDED", FLAG,
                           "risk to the stop is above the normal unit",
                           f"{ar:.2%} of equity"))

    w = c.position_weight_pct
    if w is not None and w > p.max_position_weight:
        out.append(Finding("CONCENTRATION", FLAG,
                           "single position is a large share of the account",
                           f"{w:.0%} of equity"))


def _check_size_from_stop(c: TradeCard, p: Policy, out: list) -> None:
    """Size must be the output of the stop, not an opinion about the name."""
    if c.action in ("TRIM", "EXIT", "MOVE_STOP"):
        return
    intended = c.baseline_risk_pct
    actual = c.account_risk_pct
    if intended is None or actual is None:
        return
    if intended <= 0:
        return
    ratio = actual / intended
    if abs(ratio - 1.0) > p.size_tolerance:
        sev = VETO if ratio > 1.0 + 2 * p.size_tolerance else FLAG
        out.append(Finding("SIZE_NOT_FROM_STOP", sev,
                           "position size does not follow from the stop "
                           "distance and the declared risk unit",
                           f"risking {actual:.2%} vs unit {intended:.2%} "
                           f"({ratio:.2f}x)"))


def _check_setup(c: TradeCard, p: Policy, out: list) -> None:
    if c.action != "OPEN":
        return
    if c.prior_leg_pct is None:
        _unknown(out, "NO_PRIOR_LEG", "prior_leg_pct")
    elif c.prior_leg_pct < p.min_prior_leg:
        out.append(Finding("NO_PRIOR_LEG", VETO,
                           "no big prior leg — this is a base with nothing "
                           "behind it, not a continuation setup",
                           f"leg {c.prior_leg_pct:.0%} < {p.min_prior_leg:.0%}"))
    if c.base_depth_pct is not None and c.base_depth_pct > p.max_base_depth:
        out.append(Finding("BASE_TOO_DEEP", VETO,
                           "pullback is too deep to be orderly — the trend "
                           "structure is broken",
                           f"depth {c.base_depth_pct:.0%} > {p.max_base_depth:.0%}"))
    if c.pullback_days is not None:
        if c.pullback_days < p.min_pullback_days:
            out.append(Finding("PULLBACK_LENGTH", FLAG,
                               "barely any consolidation — nothing has reset",
                               f"{c.pullback_days} days"))
        elif c.pullback_days > p.max_pullback_days:
            out.append(Finding("PULLBACK_LENGTH", FLAG,
                               "pullback has dragged on; momentum has decayed",
                               f"{c.pullback_days} days"))
    if c.ma10_rising is False:
        out.append(Finding("MA_NOT_RISING", VETO,
                           "the 10dma is not rising — the pullback is a "
                           "downtrend",
                           "ma10_rising=false"))
    if c.volume_dryup_ratio is not None and c.volume_dryup_ratio > p.dryup_ratio_max:
        out.append(Finding("NO_VOLUME_DRYUP", FLAG,
                           "volume did not dry up into the pullback — supply "
                           "is still coming out",
                           f"3d/50d volume {c.volume_dryup_ratio:.2f}"))
    if c.trigger_close_above_prior_high is False:
        out.append(Finding("NO_TRIGGER", VETO,
                           "no strength day yet — the name has not resumed",
                           "trigger_close_above_prior_high=false"))
    if c.base_count is not None and c.base_count >= p.late_base_count:
        out.append(Finding("LATE_STAGE_BASE", FLAG,
                           "late-stage base; the easy part of the move is "
                           "behind it",
                           f"base #{c.base_count}"))


def _check_extension(c: TradeCard, p: Policy, out: list) -> None:
    if c.action not in ("OPEN", "ADD"):
        return
    ext = c.extension_adr
    if ext is None:
        _unknown(out, "EXTENDED_ENTRY", "ma10 or adr_pct")
    elif ext > p.max_extension_adr_veto:
        out.append(Finding("EXTENDED_ENTRY", VETO,
                           "far extended above the 10dma — this is where he "
                           "sells into strength, not where he buys",
                           f"{ext:.1f} ADR above ma10"))
    elif ext > p.max_extension_adr_flag:
        out.append(Finding("EXTENDED_ENTRY", FLAG,
                           "entry is chasing an extended move",
                           f"{ext:.1f} ADR above ma10"))


def _check_leadership(c: TradeCard, p: Policy, out: list) -> None:
    if c.action != "OPEN":
        return
    if c.rs_percentile is None:
        _unknown(out, "WEAK_RS", "rs_percentile")
    elif c.rs_percentile < p.min_rs_percentile:
        sev = VETO if c.rs_percentile < 0.70 else FLAG
        out.append(Finding("WEAK_RS", sev,
                           "not a leader — his money only goes to top "
                           "relative strength",
                           f"RS {c.rs_percentile:.2f} < {p.min_rs_percentile}"))
    if c.sector_rank_pct is None:
        _unknown(out, "WEAK_SECTOR", "sector_rank_pct")
    elif c.sector_rank_pct < p.min_sector_rank_pct:
        out.append(Finding("WEAK_SECTOR", FLAG,
                           "the group is not leading; leaders cluster, and "
                           "this one is not in the cluster",
                           f"sector {c.sector or '?'} rank "
                           f"{c.sector_rank_pct:.2f}"))


def _check_exit_plan(c: TradeCard, p: Policy, out: list) -> None:
    if c.action not in ("OPEN", "ADD"):
        return
    if c.partial_days is None and c.partial_target_r is None:
        out.append(Finding("NO_PARTIAL_PLAN", FLAG,
                           "no plan to sell into strength",
                           "partial_days and partial_target_r both unset"))
    elif c.partial_days is not None and c.partial_days > p.max_partial_days:
        out.append(Finding("PARTIAL_TOO_LATE", FLAG,
                           "first partial is scheduled well past the window "
                           "where these moves pay",
                           f"day {c.partial_days}"))
    if c.move_to_breakeven_after_partial is False:
        out.append(Finding("NO_BREAKEVEN", FLAG,
                           "remainder is not moved to breakeven after the "
                           "partial", "move_to_breakeven_after_partial=false"))
    if not c.trail_ma:
        out.append(Finding("NO_TRAIL_PLAN", FLAG,
                           "no moving-average trail for the remainder",
                           "trail_ma unset"))
    elif c.trail_ma not in p.allowed_trail_ma:
        out.append(Finding("NO_TRAIL_PLAN", FLAG,
                           "trail is not one of the moving averages he uses",
                           f"trail_ma={c.trail_ma}"))
    if c.hard_target_price is not None:
        out.append(Finding("FIXED_TARGET_CAP", FLAG,
                           "a hard price target caps the few large winners "
                           "the distribution depends on",
                           f"target {c.hard_target_price}"))


def _check_adds(c: TradeCard, p: Policy, out: list) -> None:
    if c.action != "ADD" or c.cost_basis is None or c.entry_price is None:
        return
    if c.entry_price < c.cost_basis:
        out.append(Finding("ADD_TO_LOSER", VETO,
                           "adding below cost — averaging down is the one "
                           "thing the distribution cannot survive",
                           f"add at {c.entry_price} vs cost {c.cost_basis}"))


def _check_regime(c: TradeCard, p: Policy, out: list) -> None:
    """P1: the regime call must be supported by the evidence on the card."""
    if not c.regime_call:
        _unknown(out, "REGIME_EVIDENCE", "regime_call")
        return
    ev = {"breakout_follow_through": c.breakout_follow_through,
          "equity_curve_20d_pct": c.equity_curve_20d_pct,
          "index_above_20dma": c.index_above_20dma,
          "breadth_ratio10": c.breadth_ratio10}
    if all(v is None for v in ev.values()):
        out.append(Finding("REGIME_EVIDENCE", FLAG,
                           "regime call asserted with no supporting evidence "
                           "— the read is supposed to come from your own "
                           "results, not from a feeling", f"call={c.regime_call}"))
        return
    bad = []
    if c.regime_call == "PRESS":
        if c.breakout_follow_through is not None and c.breakout_follow_through < 0.4:
            bad.append(f"only {c.breakout_follow_through:.0%} of recent "
                       "breakouts followed through")
        if c.index_above_20dma is False:
            bad.append("index below its 20dma")
        if c.breadth_ratio10 is not None and c.breadth_ratio10 < 1.0:
            bad.append(f"breadth ratio {c.breadth_ratio10:.2f} negative")
        if c.equity_curve_20d_pct is not None and c.equity_curve_20d_pct < p.drawdown_cut_threshold:
            bad.append(f"own equity {c.equity_curve_20d_pct:.1%} over 20d")
    if bad:
        out.append(Finding("REGIME_CONTRADICTED", VETO,
                           "PRESS is called while the trader's own feedback "
                           "says otherwise", "; ".join(bad)))


def _check_exposure(c: TradeCard, p: Policy, out: list) -> None:
    """P2: throttle setting vs regime, and gross exposure vs the setting."""
    lvl = c.exposure_level
    if lvl is None:
        _unknown(out, "EXPOSURE_BAND", "exposure_level")
    elif lvl not in p.exposure_bands:
        out.append(Finding("EXPOSURE_BAND", FLAG,
                           "exposure level is off the 0-4 throttle",
                           f"level={lvl}"))
        lvl = None

    if lvl is not None and c.regime_call in p.regime_levels:
        lo, hi = p.regime_levels[c.regime_call]
        if lvl > hi:
            out.append(Finding("EXPOSURE_REGIME_MISMATCH", VETO,
                               "exposure setting is above what this regime "
                               "read allows",
                               f"{c.regime_call} permits {lo}-{hi}, card says {lvl}"))
        elif lvl < lo:
            out.append(Finding("EXPOSURE_REGIME_MISMATCH", FLAG,
                               "under-exposed for the regime read — in trend "
                               "the throttle is supposed to open up",
                               f"{c.regime_call} expects {lo}-{hi}, card says {lvl}"))

    if lvl is not None and c.gross_exposure_pct is not None:
        lo_g, hi_g = p.exposure_bands[lvl]
        g = c.gross_exposure_pct
        if g > hi_g * 1.05:
            out.append(Finding("EXPOSURE_BAND", VETO,
                               "gross exposure exceeds the band for this "
                               "throttle setting",
                               f"level {lvl} allows <= {hi_g:.0%}, gross {g:.0%}"))
        elif g < lo_g * 0.95:
            out.append(Finding("EXPOSURE_BAND", FLAG,
                               "gross exposure is below the band for this "
                               "throttle setting",
                               f"level {lvl} expects >= {lo_g:.0%}, gross {g:.0%}"))
        if lvl < 4 and g > 1.0:
            out.append(Finding("EXPOSURE_BAND", VETO,
                               "leverage is in use below max-press; level 4 "
                               "is the only setting where it belongs",
                               f"gross {g:.0%} at level {lvl}"))


def _check_feedback_loop(c: TradeCard, p: Policy, out: list) -> None:
    """P3: the anti-martingale rule, which is the framework's spine."""
    if c.action in ("TRIM", "EXIT", "MOVE_STOP"):
        return
    losing = ((c.equity_curve_20d_pct is not None
               and c.equity_curve_20d_pct < p.drawdown_cut_threshold)
              or (c.consecutive_losses is not None
                  and c.consecutive_losses >= p.loss_streak_cut))
    if not losing:
        return
    why = []
    if c.equity_curve_20d_pct is not None and c.equity_curve_20d_pct < p.drawdown_cut_threshold:
        why.append(f"equity {c.equity_curve_20d_pct:.1%} over 20d")
    if c.consecutive_losses is not None and c.consecutive_losses >= p.loss_streak_cut:
        why.append(f"{c.consecutive_losses} losses in a row")
    ctx = "; ".join(why)

    ar, base = c.account_risk_pct, c.baseline_risk_pct
    if ar is not None and base and ar > base * 1.05:
        out.append(Finding("REVENGE_SIZING", VETO,
                           "size is being increased while the equity curve is "
                           "falling — never trade bigger to make it back",
                           f"{ar:.2%} vs unit {base:.2%} ({ctx})"))
    if c.exposure_level is not None and c.exposure_level >= 3:
        out.append(Finding("PRESS_INTO_DRAWDOWN", VETO,
                           "pressing the throttle while your own feedback "
                           "says cut back toward cash",
                           f"level {c.exposure_level} ({ctx})"))


def _check_liquidity(c: TradeCard, p: Policy, out: list) -> None:
    if c.addv_usd is not None and c.addv_usd < p.min_addv_usd:
        out.append(Finding("LIQUIDITY", FLAG,
                           "too thin to get in and out at these sizes",
                           f"ADDV ${c.addv_usd:,.0f}"))


_CHECKS = (_check_stop, _check_stop_move, _check_risk, _check_size_from_stop,
           _check_setup, _check_extension, _check_leadership, _check_exit_plan,
           _check_adds, _check_regime, _check_exposure, _check_feedback_loop,
           _check_liquidity)


def run_guardrails(card: TradeCard, policy: Optional[Policy] = None) -> list:
    """All deterministic checks, most severe first."""
    policy = policy or Policy()
    out: list = []
    for fn in _CHECKS:
        try:
            fn(card, policy, out)
        except Exception as exc:                       # never fail closed-open
            out.append(Finding("GUARDRAIL_ERROR", FLAG,
                               f"check {fn.__name__} raised", repr(exc)))
    return sorted(out, key=lambda f: -_SEV_RANK[f.severity])


def guardrail_verdict(findings) -> str:
    if any(f.severity == VETO for f in findings):
        return "VETO"
    if any(f.severity == FLAG for f in findings):
        return "FLAG"
    return "PASS"


IMPLEMENTED_CODES = {
    "STOP_ABOVE_ENTRY", "STOP_NOT_STRUCTURAL", "STOP_WIDENED",
    "RISK_TOO_WIDE_ADR", "ACCOUNT_RISK_EXCEEDED", "CONCENTRATION",
    "SIZE_NOT_FROM_STOP", "NO_PRIOR_LEG", "BASE_TOO_DEEP", "PULLBACK_LENGTH",
    "MA_NOT_RISING", "NO_VOLUME_DRYUP", "NO_TRIGGER", "LATE_STAGE_BASE",
    "EXTENDED_ENTRY", "WEAK_RS", "WEAK_SECTOR", "NO_PARTIAL_PLAN",
    "PARTIAL_TOO_LATE", "NO_BREAKEVEN", "NO_TRAIL_PLAN", "FIXED_TARGET_CAP",
    "ADD_TO_LOSER", "REGIME_EVIDENCE", "REGIME_CONTRADICTED",
    "EXPOSURE_REGIME_MISMATCH", "EXPOSURE_BAND", "REVENGE_SIZING",
    "PRESS_INTO_DRAWDOWN", "LIQUIDITY",
}

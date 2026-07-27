"""The Qullamaggie rulebook, in one place.

Both layers of the compliance agent read from here: the deterministic
guardrails in ``guardrails.py`` implement the arithmetic subset, and the
LLM judge in ``judge.py`` gets these texts verbatim as its rubric.  Keeping
one registry is what makes the two layers auditable against each other — a
guardrail that cites no principle, or a principle no layer covers, is a bug
this module makes visible (see ``coverage_report``).

Sources: Kristjan Kullamägi's own writeups and interviews, his decision
framework (PRESS / PROBE / STAND DOWN and the 0-4 exposure throttle), and
the followers who document his process trade by trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principle:
    pid: str
    title: str
    text: str
    checkable: bool          # can a deterministic guardrail decide it?
    codes: tuple = field(default_factory=tuple)   # guardrail codes that test it


PRINCIPLES: tuple[Principle, ...] = (
    Principle(
        "P1", "Read the regime from your own results",
        "The market call is not a forecast, it is a read of feedback you "
        "already have: are your breakouts following through or failing, is "
        "your equity curve making new highs or chopping, are leaders "
        "clustering in a few groups, are episodic-pivot gaps holding. That "
        "read resolves to PRESS, PROBE or STAND DOWN.",
        True, ("REGIME_EVIDENCE", "REGIME_CONTRADICTED")),
    Principle(
        "P2", "Exposure throttle 0-4",
        "Exposure is a throttle with five settings: 0 cash, 1 pilot, "
        "2 building, 3 aggressive, 4 max press (the only level where "
        "leverage belongs). The setting must match the regime read — you do "
        "not run level 3 in a STAND DOWN tape.",
        True, ("EXPOSURE_REGIME_MISMATCH", "EXPOSURE_BAND")),
    Principle(
        "P3", "Self-correcting feedback loop",
        "Trade works -> equity rises -> add size -> press. Trade fails -> "
        "equity falls -> cut size -> back toward cash. Never trade bigger to "
        "make it back.",
        True, ("REVENGE_SIZING", "PRESS_INTO_DRAWDOWN", "TILT_LANGUAGE")),
    Principle(
        "P4", "Setup quality: leg, then contraction",
        "A tradable name has already made a large leg (roughly 30-100%+ in "
        "one to three months) and is now contracting in an orderly way, "
        "pulling back to a rising 10 or 20 day moving average on drying "
        "volume. No prior leg, no trade.",
        True, ("NO_PRIOR_LEG", "BASE_TOO_DEEP", "PULLBACK_LENGTH",
               "MA_NOT_RISING", "NO_VOLUME_DRYUP", "LATE_STAGE_BASE")),
    Principle(
        "P5", "Entry on the first strength day, never chasing",
        "Entry is the first day the name resumes: it takes out the prior "
        "day's high on expanding volume, bought at the opening-range high. "
        "A name already extended far above its 10 day moving average is a "
        "place to sell into strength, not to buy.",
        True, ("EXTENDED_ENTRY", "NO_TRIGGER")),
    Principle(
        "P6", "Stop below the consolidation low",
        "The stop sits below the low of the consolidation, or below the "
        "breakout point — a level the setup itself defines. Stops are never "
        "widened after entry.",
        True, ("STOP_ABOVE_ENTRY", "STOP_NOT_STRUCTURAL", "STOP_WIDENED")),
    Principle(
        "P7", "Size from the stop, and cap the risk",
        "Position size is derived from the distance to the stop so that "
        "every trade risks the same small fraction of the account. If the "
        "stop is more than about one average daily range away, the trade is "
        "skipped rather than sized down into noise.",
        True, ("SIZE_NOT_FROM_STOP", "RISK_TOO_WIDE_ADR",
               "ACCOUNT_RISK_EXCEEDED", "CONCENTRATION", "LIQUIDITY")),
    Principle(
        "P8", "Sell into strength",
        "Take the first partial into strength — typically day three to five, "
        "or earlier when the move is already extended above the moving "
        "average — and move the remainder's stop to breakeven.",
        True, ("NO_PARTIAL_PLAN", "PARTIAL_TOO_LATE", "NO_BREAKEVEN")),
    Principle(
        "P9", "Trail the rest on the moving average",
        "What is left rides the 10 or 20 day moving average (a slower one "
        "for larger, longer-timeframe positions) and is sold when it closes "
        "below. Winners are not capped by a fixed target.",
        True, ("NO_TRAIL_PLAN", "FIXED_TARGET_CAP")),
    Principle(
        "P10", "Only leaders, only leading groups",
        "Money goes to the strongest names by relative strength, and the "
        "strongest names cluster — track which groups are leading and rotate "
        "with them. A weak name in a weak group is not a setup, whatever the "
        "chart looks like.",
        True, ("WEAK_RS", "WEAK_SECTOR")),
    Principle(
        "P11", "The waiting phase is work",
        "During corrections the job is building the watchlist of names "
        "holding up, not forcing trades. Cash is a position.",
        False, ()),
    Principle(
        "P12", "Many small losses, a few large wins",
        "The distribution is the strategy: lots of small stop-outs, "
        "occasional very large winners. Never average down, never add to a "
        "position below your cost, never let a small loss become a big one.",
        True, ("ADD_TO_LOSER",)),
)

# Codes emitted by the review layer rather than by a trading rule: they
# police the agent's own inputs, so they map to no principle and are exempt
# from the orphan check.
CONTROL_CODES = frozenset({"UNTRUSTED_TEXT_INJECTION", "GUARDRAIL_ERROR"})

BY_ID = {p.pid: p for p in PRINCIPLES}
ALL_CODES = {c for p in PRINCIPLES for c in p.codes}
CODE_TO_PRINCIPLE = {c: p.pid for p in PRINCIPLES for c in p.codes}


def rubric_text() -> str:
    """The rubric handed to the judge, grounded in the registry above."""
    lines = []
    for p in PRINCIPLES:
        lines.append(f"{p.pid} — {p.title}\n    {p.text}")
    return "\n".join(lines)


def coverage_report(implemented_codes) -> dict:
    """Which principles the deterministic layer actually tests.

    Honesty check on the agent itself: any checkable principle with no
    implemented code is a gap, and any code not in the registry is a rule
    the judge was never told about.
    """
    implemented = set(implemented_codes)
    gaps = [p.pid for p in PRINCIPLES
            if p.checkable and not (set(p.codes) & implemented)]
    orphans = sorted(implemented - ALL_CODES - CONTROL_CODES)
    unimplemented = sorted(ALL_CODES - implemented)
    return {"principles": len(PRINCIPLES),
            "deterministic_gaps": gaps,
            "orphan_codes": orphans,
            "declared_not_implemented": unimplemented,
            "judge_only": [p.pid for p in PRINCIPLES if not p.checkable]}

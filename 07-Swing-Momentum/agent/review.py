"""Orchestration: guardrails, then judge, then a verdict that cannot be
talked down.

Precedence, in one place so it is auditable:

    final = max(guardrail_verdict, judge_verdict)   by severity

The deterministic layer is authoritative for everything it can measure; the
judge can only make a verdict *worse*. That ordering is what makes the free
text safe to feed a model at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from .guardrails import (FLAG, VETO, Finding, Policy, guardrail_verdict,
                         run_guardrails)
from .judge import make_judge, merge_verdicts, scan_untrusted_text
from .trade_card import TradeCard


# findings this layer adds on top of the deterministic guardrails
REVIEW_CODES = frozenset({"UNTRUSTED_TEXT_INJECTION", "TILT_LANGUAGE"})


@dataclass
class Review:
    ticker: str
    action: str
    verdict: str
    guardrail_verdict: str
    judge_verdict: str
    findings: List[Finding] = field(default_factory=list)
    judge: Optional[object] = None
    judge_name: str = ""

    @property
    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == VETO]

    def to_dict(self) -> dict:
        j = self.judge
        return {
            "ticker": self.ticker, "action": self.action,
            "verdict": self.verdict,
            "guardrail_verdict": self.guardrail_verdict,
            "judge_verdict": self.judge_verdict,
            "judge": self.judge_name,
            "codes": [f.code for f in self.findings if f.severity != "INFO"],
            "findings": [f.to_dict() for f in self.findings],
            "judge_summary": getattr(j, "summary", ""),
            "violated_principles": list(getattr(j, "violated_principles", []) or []),
            "injection_detected": bool(getattr(j, "injection_detected", False)),
        }

    def render(self) -> str:
        mark = {"PASS": "PASS", "FLAG": "FLAG", "VETO": "VETO"}[self.verdict]
        lines = [f"{mark}  {self.ticker:<6s} {self.action:<9s}"
                 f"(guardrails {self.guardrail_verdict}, judge "
                 f"{self.judge_verdict} via {self.judge_name})"]
        for f in self.findings:
            if f.severity == "INFO":
                continue
            lines.append(f"    [{f.severity}] {f.code} ({f.principle}) — "
                         f"{f.message}" + (f"  ({f.evidence})" if f.evidence else ""))
        summary = getattr(self.judge, "summary", "")
        if summary:
            lines.append(f"    judge: {summary}")
        unknown = [f for f in self.findings if f.severity == "INFO"]
        if unknown:
            lines.append(f"    unverifiable: {', '.join(sorted({f.code for f in unknown}))}")
        return "\n".join(lines)


def review_card(card: TradeCard, policy: Optional[Policy] = None,
                judge=None) -> Review:
    findings = run_guardrails(card, policy)

    scan = scan_untrusted_text(card.rationale)
    if scan["injection"]:
        findings.insert(0, Finding(
            "UNTRUSTED_TEXT_INJECTION", FLAG,
            "the rationale contains text aimed at the reviewer rather than "
            "at the trade; it was ignored for grading",
            f"{scan['injection_hits']} pattern(s) matched"))
    if scan["tilt_markers"]:
        findings.append(Finding(
            "TILT_LANGUAGE", FLAG,
            "the write-up reads like tilt rather than a plan",
            ", ".join(scan["tilt_markers"])))

    findings.sort(key=lambda f: {"VETO": 0, "FLAG": 1, "INFO": 2}[f.severity])
    gv = guardrail_verdict(findings)
    judge = judge or make_judge()
    jv = judge.judge(card, findings)
    return Review(ticker=card.ticker, action=card.action,
                  verdict=merge_verdicts(gv, jv.verdict),
                  guardrail_verdict=gv, judge_verdict=jv.verdict,
                  findings=findings, judge=jv, judge_name=judge.name)


def review_many(cards, policy: Optional[Policy] = None, judge=None) -> List[Review]:
    judge = judge or make_judge()
    policy = policy or Policy()
    return [review_card(c if isinstance(c, TradeCard) else TradeCard.from_dict(c),
                        policy, judge) for c in cards]


def main(argv=None) -> int:
    import argparse
    from .trade_card import load_jsonl

    ap = argparse.ArgumentParser(
        description="Review trade cards against Qullamaggie's rules.")
    ap.add_argument("path", help="JSONL file of trade cards")
    ap.add_argument("--json", action="store_true", help="machine-readable out")
    ap.add_argument("--samples", type=int, default=1,
                    help="LLM judge self-consistency samples")
    ap.add_argument("--vetoes-only", action="store_true")
    args = ap.parse_args(argv)

    judge = make_judge(n_samples=args.samples)
    reviews = review_many(load_jsonl(args.path), judge=judge)
    if args.vetoes_only:
        reviews = [r for r in reviews if r.verdict == "VETO"]
    if args.json:
        print(json.dumps([r.to_dict() for r in reviews], indent=2))
    else:
        for r in reviews:
            print(r.render())
        n_veto = sum(r.verdict == "VETO" for r in reviews)
        n_flag = sum(r.verdict == "FLAG" for r in reviews)
        print(f"\n{len(reviews)} cards: {len(reviews) - n_veto - n_flag} pass, "
              f"{n_flag} flagged, {n_veto} vetoed  (judge: {judge.name})")
    return 1 if any(r.verdict == "VETO" for r in reviews) else 0


if __name__ == "__main__":
    raise SystemExit(main())

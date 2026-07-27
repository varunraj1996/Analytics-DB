"""Qullamaggie compliance agent: guardrails, an LLM judge, and evals.

    from agent import TradeCard, review_card
    print(review_card(TradeCard.from_dict(card)).render())

Layering is deliberate and is the whole design: deterministic checks decide
everything measurable and bound the model's authority; the LLM judge grades
what is left; the eval suite measures both.
"""
from .guardrails import Finding, Policy, guardrail_verdict, run_guardrails
from .judge import HeuristicJudge, LLMJudge, make_judge, merge_verdicts
from .principles import PRINCIPLES, coverage_report, rubric_text
from .review import Review, review_card, review_many
from .trade_card import TradeCard, load_jsonl

__all__ = ["TradeCard", "Policy", "Finding", "run_guardrails",
           "guardrail_verdict", "LLMJudge", "HeuristicJudge", "make_judge",
           "merge_verdicts", "Review", "review_card", "review_many",
           "PRINCIPLES", "rubric_text", "coverage_report", "load_jsonl"]

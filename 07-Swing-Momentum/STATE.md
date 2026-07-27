# 07-Swing-Momentum — task state (self-resumption source of truth)

Purpose: if the session is cut off by usage limits, resume from here.
Delete this section's Routine (list_triggers → delete_trigger) when DONE.

## Current request (user, 2026-07-27)

1. Audit: does the implementation account for everything in the uploaded
   Qullamaggie decision-framework PDF? Answer honestly, rule by rule.
2. Fix known script bugs and re-run:
   - [x] `swing/setups.py::simulate` — partial-taking inner condition made
     `target_r` dead code (always took partial at day N, never early into
     strength). FIXED: partial fires at `r_now >= target_r` OR
     `held >= partial_days`.
   - [x] `scripts/41_qulla2.py::run_portfolio` — trades exiting after the
     window edge dropped their P&L from segment CAGR; constant lexsort key.
     FIXED: segment extended to last exit; stable argsort by entry day.
   - [ ] Re-run `scripts/41_qulla2.py`, report metric deltas vs RESULTS.md
     addendum honestly.
3. Build the compliance agent in `07-Swing-Momentum/agent/`:
   - [ ] `trade_card.py` — TradeCard schema + deterministic guardrails coded
     from the PDF (stop below pullback low/breakout point, ADR-capped risk,
     size-from-stop, anti-martingale / no revenge sizing, no adding below
     cost, regime-tier vs exposure-level consistency, extension-above-MA
     checks, RS + sector-leadership checks, exposure ladder 0-4).
   - [ ] `judge.py` — LLM-as-Judge (python `anthropic` SDK, model
     `claude-opus-5`, adaptive thinking, structured outputs / messages.parse,
     PASS/FLAG/VETO + per-principle scores). Deterministic fallback when
     ANTHROPIC_API_KEY is absent so evals run in-sandbox.
   - [ ] `evals/golden.jsonl` — labeled conforming + violating trade cards.
   - [ ] `evals/run_evals.py` — precision/recall/F1 per principle, verdict
     accuracy, guardrail-vs-judge agreement.
   - [ ] Run evals, all green.
4. Docs: RESULTS.md addendum update (bug-fix deltas + PDF gap table),
   README updates.
5. Commit and push to `claude/rh-agent-alpha-strategy-nhlyth`.

## Standing user directives

- Self-resume on limit resets; do not wait to be restarted.
- Universe: expand if needed (already at full reachable universe — 6,009
  liquid US stocks of 7,195; post-2017 daily data blocked by org egress).
- Track multiple sectors: correlation-cluster rotation (20 clusters, causal
  yearly refit) is the sector layer; agent checks sector leadership per trade.

## Environment notes (do not re-derive)

- Blocked (org egress policy, do NOT retry): yfinance/Yahoo, Alpaca, Binance,
  FRED, stooq, figshare, kaggle, all commercial data APIs.
  Reachable: github clone/raw, gitlab.com, bitbucket.org, pypi.
- Panel data: scratchpad `kaggle_huge/` (daily, ends 2017-11).
- No ANTHROPIC_API_KEY in this sandbox — judge's LLM path activates on the
  user's machine; evals must pass via the deterministic fallback here.

# Swing Momentum — Qullamaggie & Stockbee setups

Mechanical implementations of the setups traded by **Kristjan Kullamägi
(Qullamaggie)** and **Pradeep Bonde (Stockbee)**, tested on 6,009 US stocks /
13.6M daily bars / 1999–2017.

Four setups: the Qullamaggie continuation breakout (prior 30–100% leg → tight
base → range expansion, ADR-capped stop, scale out at day 4, trail the 10/20
dma), Bonde's momentum burst (+4% on rising volume out of a quiet stretch),
his Double Trouble anticipation scan, and an episodic-pivot proxy.

**Result: negative.** The breakouts underperform the same-day universe by
−85 bps per 10 days (t = −11.5, negative in 13 of 15 years). Double Trouble
looks strong long-only (+63 bps/10d) but 85% of that is survivorship bias in
the universe; the market-neutral version is flat in every window, and what
remains does not clear costs.

**v2 changes the diagnosis, not the verdict.** The first pass tested a
breakout *chase* he does not trade. v2 implements the real entry — pullback
to a rising 10/20dma on drying volume, buy the first strength day — plus RS,
correlation-cluster sector rotation, and the trend/chop regime gate. Each
nuance is worth something real (the entry style alone moves 10-day drift by
+103 bps versus chasing) and the stack still does not survive validation or
clear costs.

Full account, including the controls that killed it and what the test cannot
reproduce (Kullamägi's intraday opening-range entry): **[`RESULTS.md`](RESULTS.md)**.

## The compliance agent

`agent/` reviews a proposed trade against his rules and returns
PASS / FLAG / VETO with a reason tied to a named principle: 30 deterministic
guardrails for everything measurable, an LLM judge (`claude-opus-5`) for the
judgement calls, and `final = max(guardrail, judge)` so the model can only
escalate. 42 labelled eval cases with a zero-false-clear safety gate.
See **[`agent/README.md`](agent/README.md)**.

Pointed at the backtest's own trades (`scripts/42_audit_trades.py`), it
grades the strategy against the same rulebook — which is how the gaps
between "implements the setup" and "implements the method" were found.

```
swing/panel.py       panel + features + four setup masks + Bonde's Market Monitor breadth
swing/setups.py      numba trade simulator (ADR-capped stop, scale-out, MA trail) + portfolio
swing/qulla2.py      pullback-bounce entry, RS, correlation-cluster sectors, regime tiers
agent/               guardrails + LLM-as-Judge + evals (see agent/README.md)
scripts/40_scout.py      raw edge, parameter sensitivity, survivorship controls
scripts/41_qulla2.py     ablation of every v2 ingredient, then the gated portfolio
scripts/42_audit_trades.py  the agent grading the strategy's own trades
```

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
   - [x] Re-ran `scripts/41_qulla2.py`: test DD -67.5% gated / -75.5%
     ungated (was -70/-77), drift unchanged, conclusions unchanged.
3. Build the compliance agent in `07-Swing-Momentum/agent/`:
   - [x] `trade_card.py` — TradeCard schema + deterministic guardrails coded
     from the PDF (stop below pullback low/breakout point, ADR-capped risk,
     size-from-stop, anti-martingale / no revenge sizing, no adding below
     cost, regime-tier vs exposure-level consistency, extension-above-MA
     checks, RS + sector-leadership checks, exposure ladder 0-4).
   - [x] `judge.py` — LLM-as-Judge (python `anthropic` SDK, model
     `claude-opus-5`, adaptive thinking, structured outputs / messages.parse,
     PASS/FLAG/VETO + per-principle scores). Deterministic fallback when
     ANTHROPIC_API_KEY is absent so evals run in-sandbox.
   - [x] `evals/golden.jsonl` (42 cases) — labeled conforming + violating trade cards.
   - [x] `evals/run_evals.py` — precision/recall/F1 per principle, verdict
     accuracy, guardrail-vs-judge agreement.
   - [x] Evals green: 100% verdict accuracy, 0 false clears, F1 100%,
     injection resistance 100%; 27 pytest tests pass.
4. [x] Docs: RESULTS.md Addendum 2 (gap table, bug deltas, trade audit),
   agent/README.md, 07 README, root README.
5. [x] Committed and pushed to `claude/rh-agent-alpha-strategy-nhlyth`.

**STATUS: COMPLETE.** Nothing pending; the hourly resumption Routine can be
deleted. Open follow-ups are research choices, not unfinished work:
stop at the consolidation low rather than the signal-bar low, and an
equity-curve-driven 0-4 exposure throttle wired into sizing.

## Standing user directives

- Self-resume on limit resets; do not wait to be restarted.
- Universe: expand if needed (already at full reachable universe — 6,009
  liquid US stocks of 7,195; post-2017 daily data blocked by org egress).
- Track multiple sectors: correlation-cluster rotation (20 clusters, causal
  yearly refit) is the sector layer; agent checks sector leadership per trade.

## Environment notes (do not re-derive)

- Blocked (org egress policy, do NOT retry): yfinance/Yahoo, Alpaca, Binance,
  FRED, stooq, figshare, kaggle, **huggingface.co**, **api.coingecko.com**,
  **data.binance.vision** (each verified 403-on-CONNECT 2026-07-27; the
  binance.vision HEAD response mimics an origin 403 — the GET shows the
  proxy denial). All commercial data APIs.
  Reachable: github clone/raw, gitlab.com, bitbucket.org, pypi.
- Unblock candidates for the environment allowlist, in value order:
  api.coingecko.com (crypto price/mktcap/volume, no key),
  huggingface.co (FNSPID equity panel), data.binance.vision (exchange klines).
  The user must add these in the claude.ai/code environment settings
  (environment env_01CJ9rCxriY2JmXLiAtbzJp9, network policy / allowed
  domains — see code.claude.com/docs/en/claude-code-on-the-web). No tool in
  this session can edit the policy.

## RUNBOOK — execute when api.coingecko.com stops returning 403 (hourly probe Routine active)

1. `cd 06-Crypto-Alpha && python3 scripts/31_fetch_local.py --out <scratch>/cm_new --sleep 8`
   (urllib honors HTTPS_PROXY; the script works in-session once allowed).
2. Copy `<scratch>/cm_new/*.csv` into `<scratch>/cm/`, EXCEPT sol.csv — keep
   the validated composite (NI3singh price + CM mktcap/volume) unless the
   fresh CoinGecko sol series agrees with it at >0.99 return corr on overlap,
   in which case prefer CoinGecko (longer + current).
3. `python3 -m crypto.ingest` — expect ~40+ assets.
4. Re-run the pre-declared selection grid (coverage gate 0/2/3/4 x volume
   weight 0/10/20%, majors sleeve 20%, worst-regime rule on train/valid
   ONLY) — see Addendum 5 code path. New assets have no on-chain fields, so
   consider the gate definition carefully and disclose any redefinition.
5. ONE test read of the picked config. Report train/valid/test + fee ladder.
   Disclose: universe includes cycle winners selected with hindsight; the
   defensible number is forward behaviour after this date.
6. If huggingface.co is also unblocked: fetch FNSPID full_history.zip,
   rebuild the 07 swing panel with post-2017 data, rerun 41_qulla2.py
   train/valid/test per its frozen procedure.
7. Update RESULTS.md files, commit, push, delete the probe Routine.
- Universe expansion re-tested 2026-07-27 against 7 git-hosted candidates,
  each verified by clone or raw fetch rather than by README claim. Nothing
  reachable carries a bulk daily US-equity panel past 2017:
  * `scienclick/stocks` — the same 7,195-ticker dump already in use, ends
    2017-11-10. Clonable, but adds nothing.
  * `Zdong104/FNSPID_Financial_News_Dataset` — **the one that would matter**:
    4,775 S&P 500 tickers, 1999-2023, daily OHLCV. The GitHub repo holds only
    two sample tickers; the bulk `full_history.zip` lives on Hugging Face,
    which is blocked. **If the user fetches this locally, it is the missing
    input**: a post-2017 window plus (via its news dates) a path to real
    episodic pivots. `swing/panel.py` ingests per-ticker CSV directly.
  * `com-480-data-visualization/StocksWise` (11 tickers), `vijinho/sp500`
    (index series only, ends 2018-12), `AlanWangyl/NASDAQ_...` and
    `eliangcs/pystock-data` (no data committed / ends 2017),
    `Finnworlds-.../Historical-Stock-Price-OHLC` (paid-API marketing repo).
- Panel data: scratchpad `kaggle_huge/` (daily, ends 2017-11).
- No ANTHROPIC_API_KEY in this sandbox — judge's LLM path activates on the
  user's machine; evals must pass via the deterministic fallback here.

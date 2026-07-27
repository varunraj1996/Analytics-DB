# Program state — multi-asset alpha research

**Read this first on every automated continuation.** Work the first unchecked
step; keep this file truthful; commit and push after each completed step.
An hourly Routine (`alpha-research-continuation`, trig_01Ephwk3JWdNX28Twpj5WKCi)
fires into the originating session so the program resumes automatically after
usage-limit outages. Delete that Routine when the program is COMPLETE.

## Why this direction

Two completed studies in this repo (03-Alpha-Research, 04-Intraday-Alpha)
established that the intraday breakout-pullback pattern on US equities/ETFs has
no edge net of costs, and that ML overlays cannot rescue a signal that does not
clear friction. The program therefore moved to the asset classes and signal
families with decades of documented, externally replicated persistence: **trend
and carry on a broad futures universe**, plus crypto/FX momentum as an
independent check.

## Data on disk (scratchpad; re-clone if the container was recycled)

| source | path | contents |
|---|---|---|
| pysystemtrade (github robcarver17/pysystemtrade, depth-1) | `pst/data/futures/` | 252 instruments: `adjusted_prices_csv` (daily back-adjusted, some from 1972, through 2024-03), `multiple_prices_csv` (price+carry contracts), `fx_prices_csv`, `csvconfig/instrumentconfig.csv` (asset class, point size, currency), `csvconfig/spreadcosts.csv` (per-instrument spread in price points) |
| TheSnowGuru dump (github, depth-1) | `intraday_sg/` | D1 (and intraday) for 18 FX pairs (2007+), gold/silver/brent, 5 index CFDs, 14 crypto (2017+), through 2023-09 |
| Scratch root | `/tmp/claude-0/-home-user-Analytics-DB/424c4d1d-0d1e-5add-87e3-f2d41232a901/scratchpad` | set `ALPHA_SCRATCH` env var to relocate |

Blocked (org egress policy, do NOT retry): yfinance/Yahoo, Alpaca, Binance,
FRED, stooq, figshare, all commercial data APIs. Reachable: github clone,
gitlab.com, bitbucket.org, pypi.

## Method (fixed)

Splits: **train ≤2009-12-31, validate 2010-01-01..2016-12-31, test
2017-01-01..2024-03-28** (futures panel; the snowguru cross-check uses its own
2007+ range with the same boundaries). All P&L in currency space (position ×
point size × price change) so back-adjusted prices that cross zero are handled
correctly. Per-instrument spread costs from `spreadcosts.csv` charged on every
position change; instruments without a cost entry get an asset-class default.
Portfolio: equal-risk across instruments (inverse price-vol in currency),
target 25% annualised vol, signals capped at ±2 z. Selection on validation by
pre-declared rule; ONE test run of the frozen configuration.

## Steps

- [x] 1. Probe data sources; clone pysystemtrade + inventory snowguru D1
- [x] 2. Create hourly continuation Routine; write this STATE.md
- [ ] 3. `multiasset/ingest.py`: futures panel (price, carry, USD conversion,
      asset class, point size, spread cost) + snowguru D1 panel. Quality
      report: instruments, spans, gaps.
- [ ] 4. `multiasset/signals.py` + `multiasset/portfolio.py`: EWMAC
      (2/8..64/256), Donchian breakout (20..320), annualised carry, skew;
      risk-parity sizing, vol targeting, cost model. Synthetic-data tests for
      the P&L accounting (negative adjusted prices, cost charging, vol target).
- [ ] 5. `scripts/20_sweep.py`: brute-force signal weightings/speeds on TRAIN
      only (grid over speed subsets x carry weight x breakout weight x vol
      target x asset-class inclusion). Rank by train Sharpe with a family-
      stability preference, shortlist ~30.
- [ ] 6. `scripts/21_validate.py`: run shortlist on VALIDATION; pre-declared
      selection = highest validation Sharpe among configs whose validation DD
      is within 1.5x train DD and whose train Sharpe >= 0.5. Freeze.
- [ ] 7. `scripts/22_test.py`: single frozen run on TEST (2017-2024-03).
      Robustness: cost x2/x4, drop-an-asset-class, sub-period table,
      snowguru cross-check panel run with the same frozen weights.
- [ ] 8. `RESULTS.md` + charts; commit, push. Mark program COMPLETE here and
      delete the continuation Routine (delete_trigger).

## Current status

Working step 3. Nothing else in flight.

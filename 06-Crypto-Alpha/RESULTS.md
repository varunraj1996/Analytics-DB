# Crypto on-chain alpha — results

**On-chain fundamentals work out of sample. Price-based trend does not.**

Frozen configuration, single test run on **2024-01-01 → 2026-05-22** (the spot-ETF era, untouched during selection):

| window | CAGR | Sharpe | vol | max DD | skew |
|---|---|---|---|---|---|
| train ≤2021 | +15.5% | 1.27 | 11.9% | −26.9% | +1.21 |
| validate 2022–23 | +9.8% | 0.71 | 14.5% | −10.9% | −0.01 |
| **test 2024–26** | **+7.8%** | **0.80** | 10.1% | **−11.7%** | **+0.68** |
| *BTC buy & hold, test* | *+27.6%* | *0.74* | *48.2%* | *−49.1%* | *+0.34* |

Same risk-adjusted return as holding Bitcoin, at **one fifth the volatility and one quarter the drawdown**, with positive skew. Sharpe is remarkably stable across the three windows (1.27 → 0.71 → 0.80) — no decay, which is the opposite of what the equity and futures-trend studies showed.

## The frozen strategy

Selected on validation from a 38-blend sweep over four signal families:

**50% on-chain valuation + 50% exchange flow, long-only, no trend, no cross-sectional.**

The sweep *rejected* price-based trend and cross-sectional momentum despite trend dominating the train window (train Sharpe 2.27 for trend-heavy blends) — validation demoted every one of them. That is the selection process doing its job: the 2017–21 train window is a bull market where any long-biased trend rule looks brilliant.

## Signal-by-signal, out of sample

Every single on-chain signal is positive in the test window — this is not one lucky factor:

| signal | what it measures | valid Sharpe | **test Sharpe** |
|---|---|---|---|
| `addr` | active-address growth | 0.27 | **0.68** |
| `hash` | hash-ribbon (miner capitulation) | −0.30 | **0.59** |
| `mvrv` | market cap / realised cap, inverted | 0.52 | **0.54** |
| `netflow` | net USD flowing onto exchanges, inverted | 0.45 | **0.44** |
| `exsply` | exchange balance trend, inverted | 0.71 | **0.29** |
| `nvt` | network value / transactions, inverted | 0.35 | **0.29** |

Individual signals are unstable window-to-window (`hash` flips −0.30 → +0.59, `exsply` fades 0.71 → 0.29). The *blend* is what is stable, which is the usual and correct reason to blend.

## The capital-efficiency finding

The book runs at **17% mean gross exposure and 10.1% realised vol against a 30% target**. The portfolio-level vol overlay is clipped at 3×, and that clip binds essentially all the time — so this is capital-starved, not signal-starved. Raising the gross cap does nothing (`gross_cap` 1×–4× gives identical results); the constraint is the overlay clip.

Scaling the *same return stream* to higher vol (Sharpe-invariant, so this is a leverage statement and not a new backtest):

| target vol | CAGR | Sharpe | max DD |
|---|---|---|---|
| 10% (as run) | +7.8% | 0.80 | −11.6% |
| 20% | +15.0% | 0.80 | −22.4% |
| **30%** | **+21.4%** | 0.80 | −32.4% |
| 40% | +27.0% | 0.80 | −41.6% |
| 50% | +31.6% | 0.80 | −49.9% |

**On the 30% CAGR question:** this stream reaches it at ~50% vol with a ~50% drawdown — the same drawdown as simply holding Bitcoin, but with a strategy that made money in 2024 *and* held up in the 2025 chop. At the more defensible 30% vol setting it delivers ~21% CAGR with a −32% drawdown. That is the honest trade-off; there is no configuration that produces 30% CAGR at a comfortable drawdown.

## Robustness and caveats

* **Costs:** 10 bps/side → Sharpe 0.80; 20 bps → 0.59; 40 bps → 0.17. Break-even around 45 bps/side. Fine for BTC/ETH, tight for small alts.
* **Return is front-loaded:** 2024 +21.9%, 2025 +1.6%, 2026 YTD −3.3%. Two and a half years is a short test and one good year carries much of it.
* **Diversification:** correlation with the futures trend/carry program is **+0.05** — genuinely independent. (A blended portfolio looks strong, but the overlap window includes the futures programme's own train period, so I am not quoting a combined Sharpe as a result.)
* **Leakage control:** every on-chain metric is lagged one day, because Coin Metrics publishes after the day closes and revises. `tests/test_crypto.py` asserts that perturbing the final bar cannot change the final forecast — the test that would catch the most common way these studies leak.
* **A real bug the tests caught:** the cross-sectional rank tilt was not dollar-neutral (rank-pct averages to 0.5 + 1/2k, not 0.5), leaving a small long bias. Fixed before any result was produced.

## What is still missing

* **MSTR / BMNR:** unavailable. My only MSTR data ends 2017-11, before MicroStrategy bought any Bitcoin — the wrong company.
* **IBIT / ETHA:** launched 2024/2025, absent from every reachable source. `ingest.us_session_proxy()` builds a defensible proxy (underlying sampled on US trading days, so weekend moves arrive as Monday gaps) but it ignores the sponsor fee and NAV premium/discount.
* **Options:** no data of any kind reachable. This remains the largest untested opportunity in the requested universe — the crypto vol risk premium and MSTR's rich implied vol are documented, persistent, and completely outside what I can measure here.

## Data

[`coinmetrics/data`](https://github.com/coinmetrics/data) community CSVs — 18 assets, 60,098 asset-days, 2010-07 → 2026-05, carrying MVRV, exchange in/outflows, exchange supply, active addresses, transaction counts and hash rate alongside price.

## Reproducing

```bash
python3 -m crypto.ingest         # coinmetrics -> panel + coverage report
python3 -m pytest tests -q       # 7 leakage / accounting tests
python3 scripts/30_study.py      # sweep, validate, single frozen test run
```

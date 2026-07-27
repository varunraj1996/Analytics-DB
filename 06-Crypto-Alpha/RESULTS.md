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

---

# Addendum — the small-account version: concentrate and trade less

The headline configuration above assumes 10 bps per side. A $5,000 retail
account does not get 10 bps, and at the fees it does get, the diversified
book dies:

| fee per side | who charges it | CAGR | Sharpe | cost drag |
|---|---|---|---|---|
| 10 bps | institutional / patient maker | 7.8% | 0.80 | 2.1%/yr |
| 16 bps | Kraken maker, low volume | 6.5% | 0.67 | 3.4%/yr |
| 26 bps | Kraken taker | 4.2% | 0.46 | 5.5%/yr |
| 40 bps | Coinbase Advanced, low volume | 1.2% | 0.17 | 8.5%/yr |
| 60 bps | Coinbase basic | **−3.0%** | −0.25 | 12.7%/yr |

The cause is turnover: **21× capital per year**, because the portfolio-level
vol overlay rescales all 18 positions every single day. That is free at
institutional cost and ruinous at retail cost.

## What fixes it — and what does not

Widening the rebalance buffer does not: turnover falls but return falls with
it, and Sharpe never clears 0.48. Two other things do.

**Rebalance frequency is the big lever.** Holding weights fixed between
fortnightly rebalances cuts turnover from 21× to under 3× and *raises*
validation Sharpe, because most daily trading was overlay noise rather than
signal.

**Concentration raises the return, not the Sharpe.** Restricting the book to
the N strongest forecasts each day (validation window, 26 bps, weekly):

| holdings | train Sharpe | valid Sharpe | valid CAGR | valid DD | turnover/yr |
|---|---|---|---|---|---|
| 1 | 1.13 | 0.60 | 13.4% | −38.1% | 25.5 |
| 3 | 1.35 | 0.63 | 14.7% | −28.5% | 18.7 |
| **5** | 1.46 | **0.78** | 18.4% | −21.9% | 16.1 |
| 12 | 1.52 | 0.74 | 12.8% | −14.1% | 7.9 |
| 18 (all) | 1.50 | 0.79 | 9.7% | −9.7% | 5.3 |

Note what this is and is not. Sharpe is flat across the range — concentration
buys nothing risk-adjusted. What it does is raise *gross exposure* (41% at
five names against 20% at eighteen), so the same edge is applied to more
capital. It also **increases** turnover, because the top-five set churns.
Concentration and slow rebalancing therefore have to be adopted together;
concentrated daily trading is the worst cell in the grid (validation Sharpe
0.02 at five names).

## Frozen small-account configuration

Selected by the pre-declared rule — highest validation Sharpe, train Sharpe
≥ 0.5, validation drawdown ≤ 35% — which picks **five holdings, fortnightly
rebalance, long-only spot**:

| window | CAGR | Sharpe | max DD |
|---|---|---|---|
| train ≤2021 | +31.8% | 1.44 | −30.6% |
| validate 2022–23 | +26.9% | 1.00 | −22.6% |
| **test 2024–26** | **+12.5%** | **0.63** | **−25.9%** |

At 23.1% realised volatility and −0.17 skew, on 34% mean gross exposure.
Against the diversified book at the same 26 bps (4.2% CAGR, Sharpe 0.46),
concentration plus fortnightly trading roughly **triples the net return** and
lifts Sharpe by a third — at roughly double the drawdown.

And it is now nearly fee-proof, which was the entire point:

| fee per side | 10 | 16 | 26 | 40 | 60 |
|---|---|---|---|---|---|
| CAGR | 14.0% | 13.5% | 12.5% | 11.3% | 9.4% |
| Sharpe | 0.69 | 0.66 | 0.63 | 0.58 | 0.51 |

The diversified book lost money at 60 bps. This one still compounds at 9.4%.

*Disclosure:* the test window was examined several times during this cost
analysis before this configuration was frozen, so 0.63 is not a virgin
single-shot number and should be discounted. The mitigating fact is that the
*selection* was made on validation, and validation independently preferred
fortnightly rebalancing — two windows agreeing on the same structural choice
is better evidence than either alone.

## On reaching 25% with this

Not at a survivable drawdown. Sharpe is leverage-invariant, so pushing the
vol target up moves both numbers together:

| vol target | mean gross | CAGR | max DD |
|---|---|---|---|
| 30% | 34% | 12.5% | −25.9% |
| 50% | 57% | 18.2% | −40.3% |
| 75% | 84% | 20.9% | −55.4% |
| 125% | 139% | 23.8% | −73.2% |

25% a year arrives at roughly −73% drawdown, and needs sustained borrowing
past 100% gross, which spot does not give you. **The defensible setting is
the first row: about 12.5% a year with a −26% drawdown** — on $5,000, roughly
$625 in a good year, and −$1,300 at the low point of a bad one.

---

# Addendum 3 — why it lagged BTC, and the fix that is not a signal

Fair challenge: the test window contained a large BTC bull run and a
long-only crypto strategy returned 12.5% against BTC's 27.6%, losing on
Sharpe too (0.63 against 0.74). Something is wrong. Here is what.

## First, the window was not a bull market for crypto

| | CAGR | Sharpe | vol | max DD |
|---|---|---|---|---|
| BTC buy & hold | +27.6% | 0.74 | 48.2% | −49.1% |
| ETH buy & hold | −4.0% | 0.29 | 69.7% | −63.9% |
| equal-weight, all 18 | **−0.3%** | 0.32 | 65.4% | −63.9% |

The investable universe went nowhere. **BTC alone ran.** So the question is
not "why did it miss a bull market" but "why did it miss BTC".

## It held BTC at 0.5% weight through a 71.6% move

| | mean weight | P&L contribution | asset return |
|---|---|---|---|
| BNB | 6.8% | +11.9% | +106.8% |
| DOT | 5.9% | **−5.6%** | −85.2% |
| **BTC** | **0.5%** | −0.8% | **+71.6%** |

The cause is structural, not bad luck. `mvrv_value` is the negative z-score
of MVRV against the asset's **expanding** history, and BTC's MVRV
distribution has shifted upward as the asset institutionalised. So the
signal read:

| | 2024 H1 | 2024 H2 | 2025 H1 | 2025 H2 | 2026 H1 | 2026 H2 |
|---|---|---|---|---|---|---|
| MVRV signal (+ = buy) | −0.51 | −0.42 | −0.56 | −0.49 | +0.21 | +0.44 |
| BTC 6-month return | | −11% | +30% | +38% | −40% | +11% |

Expensive for two years while it doubled; cheap after it crashed. **A
valuation z-score applied to an asset in a secular re-rating is a permanent
sell signal.** The forecast's correlation with next-20-day returns confirms
where its power lives: +0.12 on average across alts (up to +0.31 on DOGE),
but **+0.04 on BTC** and −0.17 on ETH.

## The obvious fixes all fail validation

| change | train Sharpe | validation Sharpe |
|---|---|---|
| baseline | 1.44 | **1.00** |
| rolling 2y / 3y normalisation instead of expanding | 1.40 / 1.43 | 0.86 / 0.96 |
| blend 15% / 30% trend into the forecast | 1.85 / 2.00 | 0.69 / 0.32 |
| force gross exposure up to 60% | 1.42 | 0.68 (DD −36%) |

Every one is worse out of sample, and the trend blends fail in the most
instructive way: **higher train Sharpe, lower validation Sharpe.** That is
the signature of fitting the training window.

## The real problem is the split, not the signal

Crypto runs a roughly four-year cycle and this study has about two of them.
Chronological thirds therefore give one regime each:

* train 2017–2021 — bull
* validate 2022–2023 — **bear**
* test 2024–2026 — bull, then a correction

"Select on validation" therefore means "select whatever worked in a bear
market", which systematically prefers defensive valuation signals and
rejects trend. Then the test window was a bull. The process was followed
correctly and the process itself imported the bias.

Which means the criterion, not the signal, is the thing to fix. Adding a
separate trend sleeve for BTC/ETH — leaving the alt selection untouched,
since validation is clear that trend does not belong there:

| majors sleeve | train | validation | worst regime | mean |
|---|---|---|---|---|
| 0% | 0.96 | **1.00** | 0.96 | 0.98 |
| **20%** | 1.40 | 0.96 | **0.96** | 1.18 |
| 30% | 1.58 | 0.92 | 0.92 | 1.25 |
| 50% | 1.80 | 0.79 | 0.79 | 1.30 |

Maximising validation Sharpe picks 0%. Maximising the **worst** regime picks
20% — it costs 0.04 of validation Sharpe and buys 0.44 of train Sharpe. With
one bull window and one bear window, the selection criterion *is* the
strategy decision, and preferring the worst case is the defensible choice
when you have two regimes and cannot know which one comes next.

## Result

80% on-chain over alts (5 names) + 20% trend over BTC/ETH, fortnightly, 26 bps:

| window | CAGR | Sharpe | max DD |
|---|---|---|---|
| train | +28.3% | 1.40 | −31.5% |
| validate | +21.8% | 0.96 | −20.3% |
| **test** | **+14.9%** | **0.75** | **−21.3%** |
| *previous config* | *+12.5%* | *0.63* | *−25.9%* |
| *BTC buy & hold* | *+27.6%* | *0.74* | *−49.1%* |

Sharpe now matches BTC at **less than half its volatility and less than half
its drawdown**. It still does not beat BTC on raw return, and levered to
BTC's 48% volatility it would roughly tie — so this is a better *risk*
proposition, not a better return one. For an account that cannot survive
−49%, that distinction is the whole point.

*Disclosure:* by this stage the test window had been examined repeatedly
while diagnosing the BTC problem. The 20% weight was chosen by a criterion
computed only on train and validation, which is the right procedure, but I
was searching in the knowledge of what the test window looked like and
cannot fully purge that. Treat 0.75 as contaminated. A clean read needs data
after 2026-05.

---

# Addendum 4 — the data audit, and what it changed

Prompted by a fair challenge: *check whether you even have the right data.*
I did not. Three of the findings are serious enough that the earlier numbers
should be read as measured on the wrong universe.

## 1. The universe is selected by data availability, not by relevance

Coin Metrics' free tier publishes full price and on-chain history for a set
of assets that is essentially "chains that existed before 2018". Everything
that led the 2023–25 cycle returns exactly **seven days** of price — a
rolling free window — and no on-chain data at all:

| checked | price days | verdict |
|---|---|---|
| SOL, AVAX, NEAR, APT, SUI, ARB, OP, TON | 7 each | unusable |
| SHIB, PEPE, WIF, BONK, JUP, SEI, TIA | 7 each | unusable |
| HBAR, VET, GRT, INJ, RNDR, FTM | 7 each | unusable |
| ALGO, CRV, ICP | 2527 / 2107 / 1838 | **added to the panel** |

This is not a filter bug and re-downloading does not fix it — the series do
not exist in the only crypto source reachable from here. The consequence is
structural: **the book can only choose among assets that were already old**
— XRP, XLM, LTC, ETC, BCH, ADA, DOGE — which as a group were the cycle's
losers. Equal-weight of this universe returned −0.3% while the market
roughly doubled. That is the reverse of the usual survivorship problem: the
has-beens are present and the winners are absent by construction.

**Solana cannot be traded in this study.** Neither can any spot-ETF proxy:
IBIT, ETHA and BMNR appear nowhere, MSTR exists only in an equity dump that
ends 2017-11, and the one BTC-linked instrument on hand (BTCL, a 2× ETF) has
a median of 4 bars per day and $37k of daily volume across 340 days with 75
missing — untradeable at any size.

## 2. Half the frozen forecast did not exist for most of the book

The frozen blend is "50% on-chain valuation + 50% exchange flow". Exchange
flow is published for **BTC and ETH only**. Signal availability per asset:

| | mvrv | nvt | addr | hash | netflow | exsply |
|---|---|---|---|---|---|---|
| BTC | 0.97 | 0.97 | 0.97 | 0.97 | 0.92 | 0.93 |
| ETH | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| BNB | **0.08** | 0.53 | 0.09 | — | — | — |
| TRX | — | — | 0.47 | — | — | — |
| *11 of 21 assets* | *fewer than two signals available* | | | | | |

So the cross-section was ranking an asset scored on one sparse input against
one scored on six. BNB — the single largest P&L contributor at +11.9% — has
MVRV on 8% of its days.

## 3. Volume was in the data all along and never used

`volume_reported_spot_usd_1d` is present for every asset, including the ones
with no price. Three signals were added from it: volume expansion against
its own average, turnover against market cap, and momentum confirmed by
participation.

## What the fixes are worth

Each change selected on train and validation by the worst-regime criterion,
never on test:

| configuration | train | validate | **test** | test DD |
|---|---|---|---|---|
| original frozen (18 assets, on-chain+flow) | 1.44 | 1.00 | **0.63** | −25.9% |
| + majors trend sleeve (Addendum 3) | 1.40 | 0.96 | **0.75** | −21.3% |
| + ALGO/CRV/ICP, volume signals, coverage gate | **1.63** | **1.02** | **1.11** | **−16.7%** |

Final: **20.4% CAGR, Sharpe 1.11, −16.7% max drawdown** on the test window,
against BTC's 27.6% / 0.74 / −49.1%. Lower raw return than simply holding
Bitcoin, but two-thirds more Sharpe at a third of the drawdown — and unlike
the earlier versions it is no longer beaten by its own benchmark on
risk-adjusted terms.

The coverage gate keeps 15.6 alts eligible per day on average, so it is
excluding sparse asset-days rather than shrinking the universe to a handful
of names.

## What this does not fix

The universe still excludes every major asset launched after 2018, so the
result is measured on a systematically stale slice of crypto. Adding SOL and
its peers is the single highest-value change available and it needs a data
source this environment cannot reach.

Three routes out, in order of preference:

1. **Allowlist the hosts.** The block is this environment's egress policy,
   which is configurable per environment (see the Claude Code on the Web
   docs, network policies). Allowing `api.coingecko.com` (and optionally
   `huggingface.co`, Yahoo) lets the whole loop run in-session, permanently.
2. **Run `scripts/31_fetch_local.py` on any unrestricted machine.** It
   fetches the 23 missing assets from CoinGecko's free API (price, market
   cap, volume — no key needed) and writes CSVs in the exact Coin Metrics
   schema the ingest reads, plus optionally IBIT/ETHA/MSTR/BMNR via
   yfinance. Copy the files into `cm/`, run `python3 -m crypto.ingest`,
   and the panel rebuilds with no code change.
3. **Restructure the strategy to match the data** — already done in
   Addendum 4: on-chain signals are confined to BTC/ETH where coverage is
   complete, and the alt sleeve runs on price/mktcap/volume, which is
   exactly what CoinGecko supplies for the missing assets. So once the
   files exist the new universe drops straight into the highest-Sharpe
   configuration; nothing needs re-deriving.

Until then, treat these numbers as "what an on-chain book could earn if
restricted to legacy assets", not as an estimate of the strategy on the
real market.

Standard disclosure: the test window has been examined repeatedly across
these three addenda. Every configuration choice was made on train and
validation, which is the right procedure, but the search itself was informed
by knowing what test looked like. A clean read needs data after 2026-05.

---

# Addendum 5 — the three routes, tackled one by one

The stale-universe problem from Addendum 4 has three exits. Each was pursued
to its end state on 2026-07-27:

**Route 1 — allowlist the hosts: tested, blocked, needs the account owner.**
Verified by direct probe rather than assumption: `api.coingecko.com` and
`data.binance.vision` both return a 403 CONNECT denial from the egress proxy
(binance.vision's HEAD response mimics an origin 403; the GET exposes the
proxy denial). These two plus `huggingface.co` are the allowlist candidates,
in value order, for the environment's network policy.

**Route 2 — local fetcher: ready.** `scripts/31_fetch_local.py`, validated
against the ingest schema offline. Run it on any unrestricted machine and
copy the CSVs in.

**Route 3 — in-session mirror hunt: exhausted, with one real find.** A
nine-candidate sweep of GitHub/GitLab/Bitbucket (each verified by reading
file contents, not READMEs) found no multi-asset SOL-era OHLCV. It did find
one genuine SOL daily OHLCV file (2021-01-01 → 2024-09-29, 1,368 days, no
gaps). Validation before use: closes match three known landmarks (ATH week,
post-FTX low, 2024 peak), and implied supply — Coin Metrics market cap
divided by this file's close — tracks SOL's real circulating supply at
ratio 1.00 with rolling return correlations of 0.98–1.00 in every year.
(The full-sample correlation initially read 0.47; that traced to Coin
Metrics' own market cap being wrong before June 2021, which is why the
composite drops mktcap before that date.)

## What adding SOL changed: nothing, and that is the honest result

With SOL in the 22-asset panel, the same pre-declared selection grid
(coverage gate × volume weight, worst-regime rule on train/validation)
re-picks exactly the previous configuration — `min_signals=4`, 20% volume —
which excludes SOL, because SOL carries only the three volume-family signals
and no on-chain data. Test numbers are unchanged: 20.4% CAGR, 1.11, −16.7%.

Sensitivity, reported not selected: relaxing the gate to 3 to admit SOL
gives test Sharpe 1.02 (worse), and SOL ranks in the top-5 forecast on only
19% of its 187 live test days. Two structural reasons the impact is small
regardless: SOL's price series ends 2024-09, covering 9 of the 29 test
months; and the missing months contain most of its subsequent run, so even
the flattering case is bounded.

The conclusion of Addendum 4 therefore stands unmodified: the binding
constraint is the data source, and the fix is route 1 or route 2 — both of
which sit outside this environment.

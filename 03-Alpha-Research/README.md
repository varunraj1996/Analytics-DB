# Intraday Breakout–Pullback Alpha Research

A systematic US equity strategy: wait for a stock to break out of a base, then
**refuse to chase it**. Rest a limit order below the breakout and only take the
trade if the market comes back to you intraday. Exit on an ATR stop, an ATR
target, or a time stop.

The point of this directory is not the rule — it is the apparatus around the
rule: a factorised brute-force sweep large enough to be meaningful, a strict
train / validate / test discipline, and an explicit audit of the ways the data
itself could manufacture a fake edge.

---

## 1. The rule

For each symbol, on each session:

| Stage | Condition |
|---|---|
| **Base** | highest high of the prior `base_len` sessions defines the pivot |
| **Breakout** | close > pivot + `breakout_margin` × ATR |
| **Confirmation** | volume > `vol_mult` × 50-day average volume |
| **Trend** | close > `trend_ma`-day moving average |
| **Regime** | SPY above its own `regime_ma`-day moving average |
| **Trigger** | price trades down through `close − pullback_atr × ATR` on any of the next `entry_window` sessions |
| **Entry** | market-on-close, on the session where the pullback triggered |
| **Stop** | entry − `stop_atr` × ATR |
| **Target** | entry + `target_atr` × ATR |
| **Time stop** | exit at the close of bar `max_hold` |

The short side is the exact mirror (breakdown, then a rally into resistance).

### Why the pullback is a trigger, not a fill

The obvious way to model this is to rest a limit order at the pullback level and
fill there. That is what the first version did, and it does not survive
scrutiny: **the panel is daily OHLC, so the ordering of prices inside a bar is
unknowable**, and a limit fill at an intraday extreme quietly claims several
prices that were never available in that order.

Two look-ahead bugs of exactly this shape got through review here, and both were
found by the results being too good rather than by reading the code:

1. Booking a **target exit on the entry bar** using that bar's high. The premise
   of the setup is that price *fell* to the limit, so that high is usually what
   printed before the fill. Alone, this produced a Sharpe of **8.7**; forcing
   the target to be taken at the level rather than the open still gave **15.4**.
2. Disabling the entry-bar target instead let winners run uncapped to the next
   open, which was better still. That variant made the **validation window
   out-perform the train window** — a thing real edges do not do.

The decomposition in `scripts/04c_decompose.py` settles it by measuring the setup
with closing prices only:

| leg | mean | t-stat |
|---|---|---|
| limit fill → close of the trigger day | +25.5 bps | 39 |
| **close(j) → close(j+1)** | **+29.6 bps** | **33** |
| close(j) → close(j+3) | +33.7 bps | 24 |

The close-to-close legs need no assumptions at all, and they carry a real
effect. So the reported strategy uses them: the intraday pullback is the
*trigger*, and execution is market-on-close on the trigger session. Everything
the decision depends on — the breakout, the intraday low touching the level — is
observable before that close.

Exits are then evaluated from the next session onward, with the surviving
conservative conventions: gaps through a level fill at the open, and when both
the stop and the target sit inside one bar the **stop** is assumed to have hit
first. `tests/test_engine.py` pins each of these down on hand-built bars,
including both discarded execution modes so the difference stays measurable.

---

## 2. Data

| Item | Source | Coverage |
|---|---|---|
| Daily OHLCV, split/dividend adjusted | Kaggle "Huge Stock Market Dataset" mirror ([`scienclick/stocks`](https://github.com/scienclick/stocks)) | 7,195 US stocks + 1,344 ETFs, 1962 → 2017-11-10 |
| Point-in-time S&P 500 membership | [`fja05680/sp500`](https://github.com/fja05680/sp500) | dated constituent lists, 1996 → 2026 |

After quality filtering the research panel is **13.6M rows across 6,009
symbols**. Bad prints, sub-300-row stubs and single-day spike-and-reverse
artefacts are dropped in `data.py`.

### Splits

| Window | Dates | Purpose |
|---|---|---|
| Train | 2001-01-02 → 2011-12-30 | the entire parameter sweep |
| Validate | 2012-01-03 → 2014-12-31 | rule shortlist, model threshold, portfolio sizing |
| Test | 2015-01-02 → 2017-11-10 | run once, frozen |

Pre-2001 data is used only as indicator warm-up: at 1/16 tick sizes the
intraday limit-fill model is not credible.

---

## 3. What the search actually searches

Re-running the panel per parameter combination costs ~2 seconds and caps you at
a few thousand evaluations — not enough to say anything about a fifteen
dimensional space. The sweep is factorised instead so each stage only touches
the rows the previous stage kept:

```
entry filters   ->  mask    (numpy, ~2.3M eligible rows)
pullback/limit  ->  fills   (numba, ~50k signal rows)
stop/target/…   ->  exits   (numba, ~10k fill rows)
```

A full 96-variant exit sweep on top of an existing fill set costs milliseconds.
Stage 1 runs **1,536 entry configurations × 12 limit variants × 96 exit
variants ≈ 1.77 million backtests** on the train window alone.

---

## 4. Bias audit

Run `scripts/03_bias_audit.py`. Three things could fake an edge here:

**Survivorship.** The vendor file set is "every US stock trading in November
2017", so companies that died earlier are absent. Measured against point-in-time
S&P 500 membership, price coverage averages **79.6%** over 2001–2017 — and rises
from 65% in 2001 to 96% in 2017, so the hole is *largest in the train window and
smallest in the test window*. A dip-buying strategy is exactly the kind this
flatters, so it is the single biggest caveat on these results.

**Back-adjustment ghosts.** Prices are anchored at the last print, so a name that
later did a 1-for-100 reverse split shows a hundredfold inflated price *and
dollar volume* in its early history, and can pass a liquidity screen it never
deserved. Only 20 of 6,009 symbols fell >99.9% over the sample; a share-count
floor (immune to the distortion) and a price ceiling handle the rest.

**Universe sensitivity.** The decisive check. Expectancy per trade across
liquidity tiers, same rule:

| Universe | trades | expectancy | t-stat |
|---|---|---|---|
| ADDV > $5m | 10,733 | 27.5 bps | 8.5 |
| ADDV > $25m | 5,551 | 26.7 bps | 6.5 |
| ADDV > $100m | 2,002 | 28.0 bps | 4.5 |
| S&P 500 point-in-time | 2,649 | 22.3 bps | 4.2 |

Expectancy is flat across tiers and survives in the clean large-cap universe.
Whatever this edge is, it is not a micro-cap or back-adjustment artefact.

---

## 5. Layout

```
alpha/
  config.py     splits, cost model, portfolio spec, strategy parameters
  data.py       vendor files -> one clean panel
  features.py   O(n) rolling primitives, per-symbol feature panel
  universe.py   point-in-time S&P 500 membership
  search.py     the factorised sweep + a fast portfolio proxy for ranking
  engine.py     intraday fills + daily marked, capital-constrained portfolio
  meta.py       LightGBM meta-labelling
  pipeline.py   the one canonical parameters -> equity curve path
  strategy.py   signal definition, feature snapshot, performance metrics
scripts/
  01_smoke.py       build caches, sanity-check the engine
  02_search.py      stage 1 brute force (train only)
  03_bias_audit.py  survivorship / adjustment / universe checks
  04_validate.py    shortlist, ensemble, meta-model, freeze the config
  05_test.py        the single out-of-sample run + robustness
  06_report.py      equity curve chart
tests/
  test_engine.py    synthetic-bar tests for the fill and exit mechanics
```

Run in order. Artefacts land in a scratch directory (`ALPHA_SCRATCH`), never in
the repository.

---

## 6. Costs and constraints

Every fill is charged `5 bps + 5/√(ADDV in $m)` bps, floored at half a cent per
share, on **both** sides. Positions are sized to risk a fixed fraction of equity
to the stop, capped by a per-name weight limit, by a **1%-of-ADDV participation
cap**, and by a hard no-leverage constraint on gross exposure. Shorts pay an
additional borrow charge.

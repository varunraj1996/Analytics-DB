# Intraday Breakout–Pullback Alpha Research

> **Result: the strategy does not work out of sample.** After removing three
> separate look-ahead biases, an un-tuned measurement of the underlying pattern
> shows no persistent edge: **+0.2 bps** per trigger on the long side in the
> train window and **+1.7 bps** in the test window, while the short side's
> apparent train edge (+32 bps) comes entirely from 2008 and 2011 and inverts
> afterwards (−35 bps in test). The frozen portfolio returned **−11.4% CAGR**
> out of sample against SPY's +10.1%. Section 7 has the full account, including
> the three intermediate versions that showed 30–400% CAGR and why each was
> wrong. This directory is a working research apparatus plus a well-evidenced
> negative result; it is not a deployable strategy.

A systematic US equity strategy: wait for a stock to break out of a base, then
**refuse to chase it**. Only take the trade if price comes back down through a
level below the breakout at some point intraday. Exit on an ATR stop, an ATR
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

The close-to-close legs need no intra-bar assumptions, so that is what the
reported strategy uses: the intraday pullback is the *trigger*, and execution is
market-on-close on the trigger session. Everything the decision depends on — the
breakout, the intraday low touching the level — is observable before that close.

> **Read this table with §7.2.** These legs were measured over the *pooled
> candidates of 24 shortlisted configurations*, and that shortlist was chosen on
> the train window. Measured instead on a single un-tuned configuration, the same
> +1-day leg is **+0.2 bps**, not +29.6. The difference between those two numbers
> is the parameter search, not the market — which is the whole reason a search
> this large has to be judged out of sample.

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
| Test | 2015-01-02 → 2017-11-10 | run once, frozen (in the event, twice — see §7.5) |

Pre-2001 data is used only as indicator warm-up: at 1/16 tick sizes neither the
intraday trigger nor the cost model is credible.

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

Expectancy is flat across tiers and survives in the clean large-cap universe, so
whatever this is, it is not a micro-cap or back-adjustment artefact.

> **Superseded in part.** These figures were produced before the execution model
> was fixed, on a hand-picked configuration, and they inherit bias #1 from §7.3.
> The conclusion that survives is the *comparative* one — the numbers do not
> depend on liquidity tier — which is still worth having, because it rules out
> the micro-cap explanation for the biases that were found later. The levels
> themselves are superseded by §7.2.

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

---

## 7. Results

### 7.1 The single locked out-of-sample run

Selection happened on train (the sweep) and validation (rule shortlist, ranking
policy, capacity, gross target). The test window was then run once.

| window | CAGR | Sharpe | max DD | trades | avg hold |
|---|---|---|---|---|---|
| train 2001-2011 | +17.0% | 1.03 | −19.2% | 4,430 | 10.2d |
| validate 2012-2014 | +9.1% | 0.91 | −10.8% | 1,095 | 10.2d |
| **test 2015-2017** | **−11.4%** | **−0.89** | **−39.1%** | 1,087 | 10.2d |

SPY over the same test window: **+10.1% CAGR, −13.0% max drawdown.** Calendar
years in the test window: 2015 −23.3%, 2016 −15.4%, 2017 +7.8%.

The failure is not a cost or capacity artefact — it survives every sensitivity:
doubling costs makes it −13.9%, halving the participation cap −12.2%, running at
$50m instead of $1m −5.5%. It is not concentration either: the worst 25 trades
account for only −64% of the loss, so the rest of the book is losing too.

### 7.2 Why: the pattern has no persistent edge

`scripts/07_decay.py` measures the raw close-to-close return after a pullback
trigger for one plain, **un-tuned** configuration per side, across every year.
No portfolio, no ranking, no parameter selection, so nothing in it can be an
artefact of how capital was allocated:

| side, +1 day | train 2001-2011 | validate 2012-2014 | test 2015-2017 |
|---|---|---|---|
| long (pullback after breakout) | **+0.2 bps** | +3.0 bps | +1.7 bps |
| short (rally after breakdown) | **+32.0 bps** | −23.8 bps | −35.2 bps |

The long side is flat *everywhere, including in the train window*. The short
side's train-window edge is concentrated in 2008 (+27.5 bps over 4,857 triggers)
and 2011 (+124 bps over 3,226) — crisis years — and is negative in every year
from 2012 on.

So the +17% train CAGR was not measuring an edge in the pattern. It was the best
of ~1.8 million parameter combinations, sitting on top of a short book that was
really a long-volatility position on two specific crises.

### 7.3 The three biases, and what each was worth

Each of these produced a result that looked like success. Each was caught by a
number being too good rather than by reading the code, which is the uncomfortable
part.

| # | bias | headline it produced | how it was caught |
|---|---|---|---|
| 1 | Target exit booked from the **entry bar's high**, which usually printed *before* the pullback filled | Sharpe **8.7**, later **15.4** | Sharpe implausible for US equities |
| 2 | Disabling the entry-bar target instead let winners run **uncapped to the next open** | 128% CAGR | **validation beat train** — real edges don't do that |
| 3 | Ranking the shortlist by expectancy over a train window containing 2008 | 30–42% CAGR on validation | short book was 22 of 24 rules; failed OOS |

Two mechanical bugs of the same family were also found and fixed: positions that
opened and closed on the same bar never released their portfolio slot (so twelve
slots silently clogged permanently), and their P&L was dropped from the equity
curve entirely — and because same-bar exits are disproportionately *winners*,
that one understated returns.

### 7.4 The ML overlay was actively harmful

Meta-labelling was supposed to pick which candidates deserve scarce slots. Out of
sample it was **worse than a coin toss**:

| ranking policy | test CAGR | bps/trade |
|---|---|---|
| meta-model score | −17.1% | −38.1 |
| random order | +5.9% | +23.3 |
| most liquid first | +6.2% | +16.1 |

Two things went wrong. The first was the label: a binary win/lose target teaches
the model to maximise *hit rate*, and the configurations that win most often are
the ones with a small target against a wide stop — precisely the ones that barely
compound. Its validation deciles ran monotonically *up* in win rate and *down* in
mean return. Refitting on returns with a Huber loss fixed the direction but not
the generalisation: the model remained the weakest of the three ranking policies
on validation (best 1.4% CAGR against 12.3% for a simple liquidity sort), which
was visible before the test window was touched and should have been enough to
drop it.

### 7.5 Process errors worth naming

* **Maximising validation CAGR was the wrong objective.** It first selected a
  long-only book that had drawn down **53% in the train window** — visible at
  selection time, ignored because the rule only looked at validation risk. The
  corrected rule (best Calmar subject to *both* windows staying inside 25%)
  admits only 30 of 216 configurations, and **all 30 are long+short**: long-only
  dip-buying loses 48–91% in 2008.
* **The test set was looked at twice.** The first locked run returned −17.1%.
  The two process errors above were then fixed and the test re-run, giving
  −11.4%. That second number is *not* a clean out-of-sample result and is
  reported as a second pass, not as a headline.

### 7.6 What would actually be needed

The honest read is that this pattern, on this data, is not where the alpha is.
To pursue the original >30% CAGR target seriously, the binding constraints are
data, not modelling:

1. **True intraday bars.** The whole premise is an intraday pullback entry, and
   daily OHLC cannot price it — every version that tried to fill inside the bar
   produced a fictitious edge. Minute bars would make the actual strategy
   testable rather than approximable.
2. **A survivorship-free universe with delistings.** Coverage of point-in-time
   S&P 500 members averages 79.6% here, and a dip-buying study is exactly what
   that flatters. Companies that went to zero are missing.
3. **Data past 2017.** The vendor panel ends 2017-11-10, so the test window is
   2.9 years — short enough that one bad regime dominates it.
4. **A different signal family.** Short-term reversal in liquid US equities is
   heavily arbitraged; the year-by-year table is consistent with the published
   account of that decay. Cross-sectional or event-driven signals are a better
   place to spend the search budget than a price-pattern rule.

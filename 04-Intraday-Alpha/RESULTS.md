# Results — 10-minute intraday breakout–pullback

**Headline: the pattern is real but smaller than the friction.** Gross
expectancy is a consistent **+5 to +10 bps per trade**; a round trip at the
study's 5 bps/side assumption costs 10 bps. The edge and the cost are the same
size, so net expectancy sits at roughly zero and the sign is decided by
execution quality rather than by the signal.

This is a materially different — and more useful — finding than the daily study,
which found no edge at all once look-ahead was removed. Here there *is* a
measurable effect; it just does not clear its own transaction costs at this
frequency on these instruments.

## 1. The un-tuned measurement

`scripts/12_decay.py`, all defaults, no parameter selection, 19 symbols:

| side | trades | gross | net @5bps/side | win rate |
|---|---|---|---|---|
| long (breakout → pullback) | 4,080 | **+5.1 bps** | −4.9 bps | 42.3% |
| short (breakdown → rally) | 4,029 | **+9.0 bps** | −1.0 bps | 41.0% |

Year by year the gross number is positive in 6 of 9 years on the long side and
6 of 9 on the short side — a weak but persistent effect, not a crisis artifact
like the daily study's short book.

## 2. Break-even is the whole story

`scripts/13_costs.py`:

| config | window | trades | gross | break-even /side | net @5bps | net @2bps | net @1bps |
|---|---|---|---|---|---|---|---|
| un-tuned long | full | 4,080 | +5.05 | **2.53 bps** | −4.95 | +1.05 | +3.05 |
| un-tuned short | full | 4,029 | +8.98 | **4.49 bps** | −1.02 | +4.98 | +6.98 |
| un-tuned short | test | 845 | +12.49 | 6.24 bps | +2.49 | +8.49 | +10.49 |
| frozen (selected on 2024) | full | 2,654 | +9.95 | 4.98 bps | −0.05 | +5.95 | +7.95 |
| frozen (selected on 2024) | test | 535 | +3.17 | 1.58 bps | −6.83 | −0.83 | +1.17 |

TQQQ trades around $35 with a one-cent spread, so the half-spread is ≈1.4 bps
and patient execution on these ETFs is realistically 1–2 bps per side. At that
level several variants turn positive. **But the number that has not been fitted
to anything — the frozen config on the test window — is only +3.17 bps gross,
i.e. break-even at 1.58 bps/side.** That is inside the noise of one's own fill
quality, which is not a business.

## 3. Train → validate → test

Sweep: 1,152 configurations, train 2018-08→2023-12. Selection rule fixed in
advance: best validation Calmar among configs with ≥150 validation trades,
positive expectancy in both windows, train drawdown inside 25%.

| window | trades | net bps | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|
| train 2018-2023 | — | best +3.7 | **negative for every config** | — | — |
| validate 2024 | 251 | +10.9 | +5.6% | 0.62 | −6.6% |
| **test 2025-2026** | 535 | **−6.8** | **−7.4%** | **−0.56** | −19.0% |

Note the first row: **not one of the 1,152 configurations produced a positive
portfolio CAGR in-sample.** The best train expectancy was +3.7 bps net. When the
in-sample optimum is already break-even, a positive validation year is a coin
landing heads, and the test window duly gave it back.

Selected config was short-side, opening-range breakdown with a deep (1.2 ATR)
pullback, 1.0 ATR stop / 6.0 ATR target, held to the close. On test: 31% win
rate, 61% stopped out, 35% closed at the bell.

## 4. Does ML help? — measured, not asserted

`scripts/11_ml_layer.py`. Walk-forward LightGBM (return labels, Huber loss,
refit quarterly, fitted only on trades that exited before the scored period),
mapped to a risk multiplier in 0.5×–1.5× by rolling rank. **Every run reports
the same portfolio under randomly shuffled multipliers as a control.**

| window | base 1× | model sized | shuffled control |
|---|---|---|---|
| train | −2.14% | −2.18% | −2.07% |
| validate | +5.56% | +5.46% | +5.62% |
| **test** | **−7.39%** | **−7.71%** | **−8.16%** |

Rank IC of score vs realised net return: **+0.062 train, −0.008 validate,
+0.027 test.**

The model is indistinguishable from its own shuffle in every window. On
validation it is *worse* than the shuffle. This is the second independent
measurement of the same conclusion — the daily study's meta-labeller scored
−38 bps/trade against +23 bps for random ordering.

**Why this is not surprising, and not a modelling failure.** A sizing or
selection model reallocates capital *among* the trades a signal generates. It
cannot change the average. When gross expectancy is +5 to +10 bps and the round
trip costs 10 bps, a model would have to isolate a subset with gross expectancy
several times the pool average, from ~10 features on 2,654 trades, and do it out
of sample. There is not enough signal-to-noise there, and the IC values say so
directly. ML is a lever on a signal that already clears costs; it is not a
substitute for one.

## 5. What this rules in and out

**Ruled out** — the intraday breakout–pullback pattern as a standalone strategy
at 10-minute frequency on liquid leveraged ETFs. The effect exists but is the
same size as the spread.

**Still open, in rough order of expected value:**

1. **Execution, not prediction.** The whole result turns on 1–2 bps per side.
   Passive/mid entry instead of taking the next bar's open is the single highest-
   leverage change available, and it is an execution problem, not an alpha one.
2. **A wider intraday universe.** 19 symbols dominated by 3× ETFs is a thin
   base, and 3× ETFs have a decay/rebalance signature that ordinary equities do
   not. Several hundred single names would let the cross-section do the work —
   pick the best 10 setups of 300 each day rather than trading all of them.
   This is the constraint I cannot lift from public GitHub data.
3. **Hold longer.** 35% of test trades closed at the bell rather than at a
   level, and the ATR-scaled target was rarely reached. Overnight holds change
   the cost/edge ratio because the edge scales with horizon while the round trip
   does not.
4. **A different signal family.** Nothing here suggests price-pattern breakouts
   are where the remaining intraday alpha lives.

## Reproducing

```bash
python3 -m intraday.ingest          # build the 10-min panel
python3 scripts/10_run_study.py     # sweep, select, single test run
python3 scripts/12_decay.py         # un-tuned expectancy by year
python3 scripts/13_costs.py         # gross vs break-even
python3 scripts/11_ml_layer.py      # walk-forward ML + shuffle control
python3 -m pytest tests -q          # 11 engine tests on synthetic bars
```

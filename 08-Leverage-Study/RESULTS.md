# Leverage study — can a $5,000 cash account reach 25% CAGR?

The last untested route to the 25% target. Sharpe on reachable data has a
measured ceiling (~0.73 crypto after honest accounting; swing dead; futures
excluded and infeasible at this account size), so the remaining dial is
volatility — and leveraged ETFs are how a cash account buys volatility
without margin. Real instruments from 2009-2010, so fee drag, financing cost
and daily-reset decay are inside the prices; nothing synthetic. Data:
yfinance adjusted closes via the Actions fetch; 2022-bear landmarks verified
(QQQ −34.8%, TQQQ −81.0%, SPY −24.5%) before anything was trusted.

Declared before results: train 2010-07→2015, validate 2016→2019, test
2020→present; QQQ/QLD/TQQQ and SPY/SSO/UPRO; buy-and-hold vs a 100/200dma
trend gate signalled on the 1x index, executed next close, cash at 0%;
worst-regime selection, one frozen test read.

## What the honest procedure selects: no leverage at all

Validation Sharpe across the Nasdaq tiers: 1x 1.09, 2x 1.02, 3x 1.01. **The
daily-reset decay eats the Sharpe as fast as the leverage adds volatility**,
so the worst-regime rule picks *unlevered QQQ, ungated* (worst Sharpe 1.09)
— and both gates hurt in every train/validation cell, because 2010-2019 was
a bull decade where every gate exit was a whipsaw.

The one frozen test read of the selected configuration:

| | CAGR | Sharpe | vol | max DD |
|---|---|---|---|---|
| **QQQ, no gate, 2020 → 2026-07** | **20.2%** | 0.86 | 25.0% | −35.1% |
| same, full 2010 → 2026 | 19.9% | 0.98 | — | −35.1% |

**The disciplined answer to the 25% question is therefore: no.** What
survives selection delivers ~20% in an unusually good Nasdaq era, with a
−35% drawdown along the way.

## The 25%+ configurations exist — and here is their price

Context ladder on the test window (grid readout, explicitly *not* selected —
validation ranked every one of these below plain QQQ):

| | CAGR | Sharpe | max DD |
|---|---|---|---|
| QLD (2x) + 200dma gate | **31.2%** | 0.94 | **−40.2%** |
| TQQQ (3x) + 200dma gate | **41.7%** | 0.93 | **−54.8%** |
| TQQQ (3x) ungated | 32.0% | 0.75 | **−81.7%** |

Three facts to hold together:

1. **Leverage buys CAGR, never Sharpe.** Measured, not asserted: Sharpe is
   flat-to-falling across every tier in every window. 25%+ is available only
   by accepting proportionally deeper drawdowns.
2. **The gate's value is regime-dependent and asymmetric.** It *cost* Sharpe
   through the 2010s bull (validation 1.01 → 0.75 on TQQQ) and *paid*
   through 2020-2022 (test 0.75 → 0.93, drawdown −81.7% → −54.8%). Same
   conclusion as every regime gate in this repository: it reduces damage, it
   does not create edge.
3. **The test era contains no lost decade.** QQQ's own dot-com drawdown was
   −83.0%; a daily-3x product held through it loses **−99.9%** before fees,
   and even the 200dma gate only softens that to −89.9%. Every number above
   is conditional on the post-2010 regime.

## Bottom line for the account in question

On $5,000, all of these are cash-account instruments. The menu, with real
out-of-sample numbers attached:

* what honest selection endorses: **QQQ, ~20% recent-era CAGR, −35% path**;
* the cheapest 25%+ ticket: **QLD + 200dma, 31% test CAGR at a −40% path**,
  knowingly overriding validation to buy return with drawdown;
* TQQQ variants: 32-42% with −55% to −82% paths and a wipeout mode in any
  dot-com-class bear.

None of this is alpha. It is equity beta at multiple, priced honestly — the
25% exists, and its currency is drawdown, not skill.

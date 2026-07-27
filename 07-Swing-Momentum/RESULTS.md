# Swing-momentum setups: Qullamaggie & Stockbee, tested

Faithful mechanical implementations of the setups used by **Kristjan
Kullamägi (Qullamaggie)** and **Pradeep Bonde (Stockbee)** — Bonde mentored
Kullamägi, and the two systems share one spine: *a big prior move → a quiet
contracting base → range expansion.*

**Verdict: the breakout setups lose money on this data, significantly and
persistently. The anticipation setup's apparent edge is almost entirely
survivorship bias in the universe, and what survives the control is too small
to trade.**

## The rules as implemented

| | Qullamaggie breakout | Stockbee momentum burst | Episodic pivot | Double Trouble |
|---|---|---|---|---|
| Prior leg | +30–100% in 1–3 mo | — | quiet ≤20% in 3 mo | ≥ 1.8× the 252-day low |
| Base | ≤25% range over 15d | 3–20 quiet days | — | — |
| Trigger | close breaks base high | +4% on rising volume, closes in top 40% of range, ≤2 up-days prior | gap ≥10% on ≥3× volume, holds the gap | today's move within ±1% |
| Filter | ADR ≥ 4%, price > 10 & 20 dma | — | — | 3-day min volume ≥ 100k |
| Stop | signal-bar low, **skipped if > 1 ADR away** | | | |
| Exit | half off at day 4 → stop to breakeven → trail remainder on the 10/20/50 dma | | | |

Execution: signal on the close of day *t*, **entry at the open of day t+1**.
Stops are intraday (gaps fill at the open); moving-average exits are
market-on-close. Nothing is decided and priced inside the same bar.

Data: 6,009 US stocks, 13.6M daily bars, 1999-11 → 2017-11.

## Result 1 — traded as designed, they lose

Train window (≤2009), full trade simulation with ADR-capped stops and the
scale-out/trail management:

| setup | signals | trades | win% | avg R | ret/trade | stopped out |
|---|---|---|---|---|---|---|
| Qullamaggie breakout | 10,893 | 4,633 | 25.1% | **−0.24** | −92 bps | 91% |
| Stockbee burst | 43,012 | 12,765 | 32.4% | +0.02 | −44 bps | 81% |
| Episodic pivot | 212 | 71 | 28.2% | +0.27 | −27 bps | 85% |
| Double Trouble | 103,406 | 36,538 | 24.2% | **+0.24** | **+15 bps** | 83% |

Varying the defining parameters does not rescue the breakout: requiring a
larger prior leg (0% → 80%) leaves avg R between −0.22 and −0.28; loosening or
tightening the ADR stop constraint (0.5× → unconstrained) leaves it between
−0.17 and −0.24; every trailing-MA / scale-out combination is negative.

## Result 2 — the raw drift confirms it

Buy at the next open, hold, no stops. Train window:

| setup | 3d | 5d | 10d | 20d |
|---|---|---|---|---|
| Qullamaggie breakout | −41 bps (t −6.2) | −79 (t −9.6) | **−128 (t −12.0)** | −74 (t −5.1) |
| Stockbee burst | −21 (t −6.7) | −32 (t −8.5) | −28 (t −5.5) | −53 (t −7.5) |
| Double Trouble | +16 (t 8.9) | +30 (t 13.7) | +63 (t 21.8) | +114 (t 28.7) |
| *all liquid stocks (baseline)* | *+5* | *+7* | *+16* | *+19* |

Double Trouble also shows a clean dose–response — ratio 1.3 → 1.5 → 1.8 → 2.2
gives 21 → 26 → 30 → 36 bps at 5 days — which normally marks a real effect.

## Result 3 — the control that kills it

The dataset contains only stocks that were still listed in November 2017, so
the whole universe drifts up and "stock is 80% above its 252-day low" selects
hardest on exactly the survivors. Two controls:

**A. Excess over the same-day universe mean** (10-day forward):

| setup | n | excess | t |
|---|---|---|---|
| Qullamaggie breakout | 22,769 | **−85.5 bps** | **−11.5** |
| Stockbee burst | 97,908 | −25.8 bps | −8.9 |
| Episodic pivot | 742 | −91.7 bps | −2.6 |
| Double Trouble | 305,889 | **+9.6 bps** | +6.5 |

Double Trouble's +63 bps collapses to **+9.6 bps** once the universe drift is
removed — 85% of the apparent edge was survivorship. The breakout is negative
in **13 of 15 calendar years**, so it is not a regime artifact.

**B. Long top-decile / short bottom-decile, market-neutral:**

| score | train | validate | test |
|---|---|---|---|
| DT ratio | −6.4 bps (t −0.2) | +2.4 (t 0.1) | −22.3 (t −0.7) |
| 6-month momentum | −22.7 (t −0.5) | −2.1 (t −0.1) | +19.3 (t 0.5) |
| 52-week-high proximity | −63.0 (t −1.2) | +5.4 (t 0.2) | +53.3 (t 1.1) |

Nothing. Not one score is significant in any window. The cross-section carries
no information; the long-only version was riding the biased universe.

At +9.6 bps per 10-day hold, Double Trouble grosses roughly 2.4%/yr rolled
continuously, against a 20–40 bps round trip on the kind of names it selects.
It does not clear its own costs.

## Why the breakouts are *negative*, and what this does not test

−85 bps of underperformance per 10 days with t = −11.5 is not noise — it is
short-term reversal. Buying at the **next open** after a sharp breakout close
means paying for the breakout day's move *and* the overnight gap, and that is
systematically punished.

That is precisely the part of Kullamägi's method this test cannot reproduce.
He does not buy the next open — he buys the **opening-range high** intraday
with a stop at that same day's low, which is both a far earlier entry and a
much tighter risk unit. My 10-minute study (`04-Intraday-Alpha`) measured the
intraday version of this family directly and found the pattern is real but
worth **+5 to +10 bps gross against a ~10 bps round trip** — an edge the same
size as the friction, where the sign is decided by execution quality.

So the honest reading across both studies: **whatever edge these setups have
lives in the intraday entry, not in the daily pattern, and it is approximately
the size of the spread.** That is consistent with them working for a skilled
discretionary trader with excellent fills and position concentration, and not
working as a mechanical daily system.

## Caveats

* Survivorship bias is severe here and is the study's main limitation — it is
  also, ironically, what this study measures most clearly.
* Data ends 2017-11, so nothing after that is tested.
* No earnings dates, so the Episodic Pivot is a price-only proxy (gap + volume
  + prior quiet) rather than a true post-earnings-drift trade. With only 742
  observations it is underpowered either way; PEAD itself is well documented
  and this is *not* evidence against it.
* Kullamägi trades a concentrated book with discretionary selection among
  signals. A mechanical test of the screen is not a test of the trader.

## Sources

- [Qullamaggie — 3 timeless setups](https://qullamaggie.com/my-3-timeless-setups-that-have-made-me-tens-of-millions/)
- [How to trade like Qullamaggie: setups, strategy, screener](https://breakoutshappen.com/stock-news/how-to-trade-like-qullamaggie-setups-strategy-and-screener)
- [Stockbee methodology writeup](https://github.com/dcimring/stockbee-dashboard) (`docs/METHODOLOGY.md`)
- [Pradeep Bonde on Episodic Pivots — TraderLion](https://traderlion.com/podcast/discover-episodic-pivots/)
- [Deepvue: Pradeep Bonde's four screens](https://deepvue.com/screener/pradeep-bonde-screens/)

## Reproducing

```bash
python3 scripts/40_scout.py     # raw setup edge, parameter sensitivity
```

---

# Addendum — Qullamaggie v2: the nuances, priced one by one

Pushback taken seriously: the first pass tested a breakout-chase he does not
trade. v2 implements his actual bread-and-butter — **pullback to a rising
10/20dma on drying volume after a 30%+ leg, buy the first strength day, stop
at its low** — plus the three layers he is explicit about: top-decile relative
strength, sector rotation (correlation-cluster leadership, recomputed causally
each year), and the trend/chop regime gate sized 0×/1×/2×.

**Each nuance bought something real.** Excess return over the same-day
universe, 10-day horizon, train window:

| layer | excess/10d | t |
|---|---|---|
| naive breakout-chase (first pass) | **−85.5 bps** | −11.5 |
| pullback-bounce entry alone | **+17.9 bps** | 1.0 |
| + RS top decile | **+76.6 bps** | 2.8 |
| + sector leadership | +90.6 bps | 2.0 |

The entry style alone is worth ~103 bps/10d of drift versus chasing — the
single biggest effect found in this entire directory. RS adds ~59 more.

**And yet it does not survive.** Validation (2010–13): the sector layer flips
to −126 bps (t=−2.3); RS adds nothing; test-window excess for every layer is
−32 to +24 bps, all |t| < 1.2. The full portfolio (1% risk × regime tier, 10
slots, 15 bps/side) loses in every window, gate on or off — with ~2.5×
turnover per trade, costs of 40–75 bps/trade sit on top of a gross edge of
+10–30 bps. The regime gate did exactly what he says it does — halved the
drawdown versus no gate in test (−70% vs −77%) — but a smaller loss is still
a loss.

**Where this leaves the claim (v2).** Across three independent implementations
(daily chase, intraday ORH at 10-min, daily pullback with RS/sector/regime),
the same number keeps appearing: the pattern's gross edge is 0–30 bps against
20–75 bps of realistic friction. The nuances are real — they moved the drift
by over 100 bps — but on THIS data (survivorship-biased, ending 2017, daily
bars) they close most of the gap and not all of it. What would genuinely
change the verdict, in order: a fresh universe through 2026 including
delistings; true intraday entries (his ORH is a 1/5/60-minute decision);
earnings dates for real episodic pivots. The machinery is built and tested —
it is the data that is binding.

---

# Addendum 2 — audited against his decision framework, and two bugs found

Direct question: *have you accounted for everything, and does the script
execute properly?* Honest answer: **no on the first, and no on the second —
two real bugs, both now fixed.**

## 1. What the framework says and what the code did

The decision framework is not a setup document. It is a **feedback loop**:
the market call comes from the trader's own results, and the exposure
throttle follows the equity curve.

> Trade works → equity up → add size → press.
> Trade fails → equity down → cut size → cash.
> Never trade bigger to make it back.

Mapping every rule in it against what `swing/` actually implemented:

| Framework rule | Status before this audit |
|---|---|
| Big prior leg → orderly contraction → pullback to rising 10/20 MA on drying volume | **implemented** (`qulla2.pullback_bounce`) |
| Buy the first strength day out of the pullback | **implemented** (close > prior high on expanding volume) |
| Stop below the consolidation low / breakout point | **partial** — uses the *signal bar's* low, not the base low |
| Size from the stop; skip if the stop is more than ~1 ADR away | **implemented**, but at 1.5 ADR rather than 1.0 |
| Sell into strength; partial day 3–5; then breakeven | **implemented** — but see bug 1: the "into strength" half never fired |
| Trail the remainder on the 10/20 MA | **implemented** |
| Only top-RS names, only leading groups; track rotation | **implemented** (top-decile RS, 20 causal correlation clusters) |
| Wait in chop, unleash in trend | **implemented as an index/breadth gate** — not as the framework defines it |
| **Regime read from your own breakouts firing or failing** | **missing** |
| **Regime read from your own equity curve making highs or not** | **missing** |
| **PRESS / PROBE / STAND DOWN as an explicit state** | **missing** |
| **Exposure throttle 0–4, leverage only at 4** | **missing** — sizing was 0×/1×/2× off the index gate alone |
| **Cut size after losses; never size up to recover** | **missing** — risk per trade was constant regardless of drawdown |
| **Trim because the move is extended above the MA** | **missing** — partials were taken on a day count only |
| Watchlist-building during corrections | **not applicable** to a mechanical scan |

The five bolded rows are one idea: **the strategy never looked at its own
results.** Its regime input was SPY and breadth — external, and available to
everybody — while the framework's regime input is the trader's own
feedback. That is the substantive thing I had not accounted for, and it is
the part of his process that most plausibly does work, because it is the
only part that adapts to the operator rather than to the tape.

## 2. Two bugs in the script

**Bug 1 — the profit-taking rule was dead code.** In `swing/setups.py`:

```python
if (not took_partial) and held >= partial_days:      # outer gate
    if r_now >= target_r or held >= partial_days:    # …inner is always true
```

The inner condition can only be evaluated when `held >= partial_days` is
already true, so `target_r` never decided anything. Every trade took its
partial on a fixed day count, and "sell into strength" — the rule that gets
you paid when a name runs 3R in two days — was never tested. Fixed by
lifting the outer gate so the target can fire early.

**Bug 2 — the window-edge P&L leak.** In `scripts/41_qulla2.py`, segment
returns were sliced `eq[lo:hi+1]`, so any trade entered inside a window but
exiting after it contributed nothing to that window's CAGR. With a ~5-day
average hold this mostly affects the boundary, but it is a silent
understatement. Fixed by extending the segment to the last exit. A constant
sort key (`-t.r * 0`) in the same function was also replaced with a stable
sort by entry day — it was harmless, but it read as if it did something.

**What the fixes changed:** essentially nothing about the conclusion, which
is the point of reporting them.

| | before | after |
|---|---|---|
| test portfolio, gate on | −70% max DD | −67.5% max DD, −23.4% CAGR, Sharpe −2.07 |
| test portfolio, gate off | −77% max DD | −75.5% max DD, −29.1% CAGR, Sharpe −2.60 |
| train / valid / test drift by layer | unchanged | unchanged |

The regime gate still halves nothing that matters: it makes a loss smaller.
Every ablation number is within rounding of the previously reported values.

## 3. The agent, pointed at my own backtest

`agent/` implements the framework as an executable rulebook — 30
deterministic guardrails plus an LLM judge, `final = max(guardrail, judge)`
so the model can only escalate (see `agent/README.md`). Running it over the
**strategy's own filled trades** turns the table above into measurements:

| finding | share of filled trades | what it means |
|---|---|---|
| `STOP_NOT_STRUCTURAL` | ~98% flag | the stop is the signal bar's low, which sits above the base low almost always — a materially tighter, more easily-shaken stop than the one he describes |
| `RISK_TOO_WIDE_ADR` | ~19% flag | trades between 1.0 and 1.5 ADR of stop distance, which his stated rule would skip |
| `REGIME_CONTRADICTED` + `PRESS_INTO_DRAWDOWN` | ~7% veto | the book sized *up* to its top tier while its own equity was more than 5% off its high — precisely the behaviour the framework exists to prevent |
| `ACCOUNT_RISK_EXCEEDED` | flag on tier-2 trades | 2% of equity per trade at the top tier, against his ~1% norm |

Under 1% of the strategy's trades are clean passes. That is not a bug —
most findings are FLAG-level and a mechanical scan will always look sloppy
next to a discretionary trader's own account of his rules — but the two
VETO categories are exactly the missing feedback loop, now with a number
attached: **7% of the trades this system took, it took while pressing into
its own drawdown.**

## 4. Does this rescue the strategy?

No, and it is worth being clear about why not. The gaps are real and
fixable, but they are *risk-management* gaps, and the measured problem is a
*gross edge* problem: +10 to +30 bps per trade against 40–75 bps of
round-trip friction on this universe. An exposure throttle changes the path
and the drawdown; it cannot turn a negative expectancy into a positive one.
What would change the verdict is still what the v2 addendum said — a
survivorship-free universe through 2026, and true intraday entries — plus
one item this audit adds: **a stop at the consolidation low rather than the
signal bar's low**, which is the single largest divergence between the code
and the method, and the one most likely to matter, since it changes both
the stop-out rate and the risk unit every position is sized from.

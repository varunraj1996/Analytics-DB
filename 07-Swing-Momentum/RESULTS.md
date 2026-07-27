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

**Where this leaves the claim.** Across three independent implementations
(daily chase, intraday ORH at 10-min, daily pullback with RS/sector/regime),
the same number keeps appearing: the pattern's gross edge is 0–30 bps against
20–75 bps of realistic friction. The nuances are real — they moved the drift
by over 100 bps — but on THIS data (survivorship-biased, ending 2017, daily
bars) they close most of the gap and not all of it. What would genuinely
change the verdict, in order: a fresh universe through 2026 including
delistings; true intraday entries (his ORH is a 1/5/60-minute decision);
earnings dates for real episodic pivots. The machinery is built and tested —
it is the data that is binding.

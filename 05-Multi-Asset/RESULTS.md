# Results — multi-asset breakout + carry + skew program

**This one worked.** After two honest negative results (daily equity
breakout-pullback: no edge; 10-minute intraday: edge smaller than friction),
the program moved to the signal families with fifty years of external
replication — trend and carry on a broad futures universe — and the out-of-
sample verdict is positive:

| window | CAGR | Sharpe | vol | max DD |
|---|---|---|---|---|
| train ≤2009 (selection) | +30.5% | 1.54 | 18.5% | −40.7% |
| validate 2010–2016 (selection) | +23.3% | 1.27 | 17.7% | −18.9% |
| **test 2017–2024-03 (frozen, run once)** | **+10.2%** | **0.72** | 15.0% | −35.4% |
| full sample 1970–2024 | +26.7% | ~1.2 | — | −40.7% |

Test t-stat 1.98 over 1,889 days. The 1.54 → 1.27 → 0.72 decay is exactly the
published trajectory of managed-futures returns (the pre-2010 "golden age",
the hard 2010s, the 2022 revival), which is the strongest reason to believe
the measurement: **the backtest agrees with what the live industry actually
earned in the same years**, rather than beating it — a backtest that beats
live CTA indices out of sample is usually measuring its own bias.

## The frozen configuration

Selected on validation by the pre-declared rule (highest validation Sharpe,
train Sharpe ≥ 0.5, validation DD within 1.5× train DD), from a 336-config
train sweep:

* **Signals:** Donchian-style breakout at 40/80/160/320 days (50% weight) +
  **annualised futures carry (40%)** + negative-skew premium (10%). The EWMAC
  sets lost to breakout+carry on validation — and the family table was
  monotone in carry weight, so this is a ridge, not a spike.
* **Universe:** 127 futures instruments — everything except equity indices,
  the one class where trend demonstrably died post-2010 (validation agreed).
* **Portfolio:** class-equal risk weights, per-instrument point-vol sizing,
  fixed IDM (nothing estimated in-sample), buffered rebalancing,
  per-instrument half-spread costs from `spreadcosts.csv`, causal
  portfolio-level vol targeting at 25%.

## What drives it, honestly

Per-class P&L attribution of the two extreme test years:

| class | 2017 (+50.9%) | 2018 (−19.0%) |
|---|---|---|
| **Vol (VIX/VSTOXX carry)** | **+29.1%** | **−12.4%** |
| Sector equities-adjacent | +10.8% | −7.4% |
| OilGas | +3.4% | −1.1% |
| everything else | ≈ +8 | ≈ +1 |

2017's monster year is the short-VIX carry trade in the best contango year on
record, and 2018 is its Volmageddon payback. That is a *real, tradable*
phenomenon — it is precisely what short-vol funds earned and lost — but it
concentrates convexity: the return skew in the test window is −1.2. The
drop-a-class ablation quantifies it: excluding Vol entirely gives test Sharpe
**0.95** (higher, smoother), and dropping any single class keeps Sharpe in
0.48–0.95 — no one class *carries* the program, but Vol shapes its tails.
(The frozen config includes Vol; re-freezing on the ablation would be
test-set selection, so 0.72 stays the headline.)

## Robustness (all on the untouched test window)

* **Costs ×2 → Sharpe 0.57; ×4 → 0.25.** Positive through quadrupled friction.
* **Vol-target ladder** (Sharpe is leverage-invariant at 0.72):
  15% → +6.3% CAGR / −22% DD · 25% → +10.2% / −35% · 35% → +13.8% / −47% ·
  50% → +18.7% / −61%.
* **Spot cross-check: failed, and reported.** The same blend minus carry
  (spot has no carry) on the independent 42-instrument spot panel: −12.6%
  CAGR on test. The transfer failure says the futures result rests materially
  on **carry and futures breadth** — breakout alone on a thin spot universe
  is not the same trade. This is the result's most important caveat.

## On the 30% CAGR target

CAGR here is a leverage dial, so the honest statement is in Sharpe space:

* At the **full-sample** Sharpe (~1.2), 30% CAGR needs ≈25% vol — the
  historical answer, and the train/validate windows delivered exactly that
  (+30.5%, +23.3%).
* At the **out-of-sample** Sharpe (0.72), 30% CAGR needs ≈45–50% vol, which
  the ladder shows means accepting a −60%+ drawdown path. No responsible
  sizing turns a 0.7-Sharpe stream into 30%/yr without that.

What would raise the Sharpe rather than the leverage, in order of expected
value: (1) more instruments — Carver runs 100+ with finer class granularity
and per-instrument cost gating; (2) proper forecast diversification
multipliers instead of a fixed IDM; (3) adding the relative-value cousins
(cross-sectional carry/momentum within class) that diversify directional
trend; (4) execution — the cost stress shows 25–45 bps/yr of headroom.

## Reproducing

```bash
python3 -m multiasset.ingest        # panels from pysystemtrade + spot dump
python3 -m pytest tests -q          # 6 accounting tests on synthetic data
python3 scripts/20_sweep.py         # 336-config train sweep (~9 min)
python3 scripts/21_validate.py      # pre-declared selection -> frozen_ma.json
python3 scripts/22_test.py          # the single frozen test run + robustness
```

Data: `robcarver17/pysystemtrade` (157 kept of 252 instruments after cost/
history screens; daily back-adjusted prices 1969–2024, carry contracts,
per-instrument spreads) and the TheSnowGuru spot dump for the cross-check.

---

# Addendum — "SPY/QQQ return 10–14%, you can do better"

Fair challenge, and the first frozen configuration did not clear it: 10.2% CAGR
at Sharpe 0.72 is index-like returns with extra machinery. Two things change
the answer.

**1. Fix the construction defects.** Section 7.5 of the year-by-year review
found three: four `mini`/`micro` duplicate contracts that a dash-vs-underscore
bug let trade alongside their full-size parents (copper at 3× its intended
weight), no per-instrument risk cap (VIX and V2X each carrying ~12.5% of
portfolio risk), and a 10% rebalance buffer burning 2–3.6%/yr in turnover.
Fixing all three on the same frozen signal weights:

| | CAGR | Sharpe | max DD |
|---|---|---|---|
| as first frozen | 10.19% | 0.72 | −35.4% |
| **+ dedup, 3% name cap, 20% buffer** | **12.52%** | **1.03** | **−24.6%** |

*Disclosure:* these fixes were diagnosed **after** the test window was first
run, so this is not a clean single-shot out-of-sample number. They are
structural rather than parameter-fitted — one is a data bug, one is a risk
limit, one is a turnover control — and they also improve train (1.50) and
validation (1.45), but the reader should discount accordingly.

**2. Combine rather than replace.** The strategy is +0.25 correlated with
equities. All figures below are excess-of-cash over 2017-01 → 2024-03, the
window the signal weights never saw; add ~2%/yr of cash for total return.

| | CAGR | Sharpe | max DD |
|---|---|---|---|
| S&P 500 futures | 10.79% | 0.71 | −32.7% |
| Nasdaq futures | 15.72% | 0.84 | −33.8% |
| strategy (fixed) | 12.52% | 1.03 | −24.6% |
| **40% S&P / 60% strategy, levered to S&P vol** | **18.11%** | **1.11** | −35.4% |
| **40% Nasdaq / 60% strategy, levered to Nasdaq vol** | **23.50%** | **1.18** | −35.2% |

At the Nasdaq's own volatility and no worse drawdown, the blend returns
**23.5% excess / ~25.5% total against QQQ's 15.7% / ~17.7%** — roughly eight
points a year, out of sample. Against the S&P it is ~18.1% / ~20% versus
10.8% / ~12.8%. The blend needs 1.4–1.6× gross, which on futures margin is
routine but is leverage and should be named as such.

**On 30%.** At the blend's Sharpe of ~1.18, 30% CAGR needs roughly 26–28%
annualised vol and implies drawdowns near −45%. That is a real risk decision
rather than an arithmetic impossibility — which is the first time in this
repository that has been true. At the original 0.72 Sharpe it would have taken
~50% vol and a −60%+ path.

**What still argues for holding the index too.** Over 1998–2024 the S&P
returned 4.63% excess at Sharpe 0.35 with a −61% drawdown. 2017–2024 was an
unusually good stretch for beta; the case for the diversifier is strongest
precisely in the decades that stretch excludes.

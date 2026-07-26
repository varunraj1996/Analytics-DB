# 10-Minute Intraday Breakout–Pullback Study

Successor to `03-Alpha-Research`, which ended in a documented negative result
and one hard lesson: **an intraday entry cannot be priced off daily OHLC** —
every attempt manufactured a fictitious edge out of intra-bar ambiguity. This
study repeats the question on real intraday bars, at the 10-minute frequency,
which makes the fills observable instead of assumed.

## Data

Network egress here is restricted to GitHub, so the universe is what is
actually committed to public repositories as intraday history:

| Source | Symbols | Bars | Coverage |
|---|---|---|---|
| [`piekstra/market-data`](https://github.com/piekstra/market-data) | 16 leveraged ETFs (TQQQ, SOXL, SPXL, LABU, DPST, …) | 5-min parquet, real volume | 2020-07 → **2026-02** |
| [`TheSnowGuru/…Tick-Bar-Data`](https://github.com/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data) | AAPL, NFLX, TSLA | 5-min CSV, vendor-lot volume | 2018-07 → 2023-09 |

Ingestion (`intraday/ingest.py`) converts both to America/New_York, keeps the
09:30–16:00 regular session only, resamples 5→10 minutes (39 bars/session),
and **drops any symbol-day with fewer than 30 of the 39 bars** — a sparse day
means the feed missed prints, and a breakout study on missing prints invents
gaps that never traded. Result: **363,535 bars, 19 symbols, 2018-07 → 2026-02**,
dominated by seven deep series (TQQQ 1,335 days, TSLA/NFLX/AAPL ~1,277, SOXL
1,138, SPXL 967, LABU 603).

This universe is small and peculiar — mostly 3× leveraged ETFs — and that is a
disclosed limitation, not a hidden one. It is what exists as free intraday
history; the compensation is that these are among the most liquid instruments
in the US market, so the cost model is trustworthy.

## The strategy

Opening-range (or prior-day-high) breakout with a pullback re-entry, per
symbol, per session — the intraday version of the original brief:

```
IDLE  --close clears level (+confirm·ATR)-->  ARMED   (pullback level fixed)
ARMED --bar's low touches pullback level -->  TRIGGERED
TRIGGERED --------- next bar's open ------->  IN POSITION
IN POSITION -- stop / target / time / EOD ->  done (one round trip per day)
```

Engine conventions (`intraday/engine10.py`), all pinned by
`tests/test_engine10.py` on hand-built bars:

* signals evaluate on **completed** bars only; every entry is the **next bar's
  open** — nothing is ever decided and priced inside the same bar;
* a close back through the breakout level while armed **voids the setup**
  (failed breakout) rather than filling it;
* the pullback level is floored at the breakout level itself;
* gaps through a stop fill at the open; ambiguous stop+target bars resolve to
  the stop; positions are flat by the session close (`eod_flat`);
* costs: 5 bps per side, deliberately above realistic spreads for these names.

## Method

Train 2018-08 → 2023-12, validate 2024, test 2025-01 → 2026-02 (single run of
the single selected configuration). The grid is deliberately modest — 1,152
combinations over structurally different choices (level kind, OR length,
pullback depth, reward shape, trend filter) — because the daily study
demonstrated what an oversized search does on a noisy target: it finds
something whether or not anything is there.

Selection, fixed in advance: highest validation Calmar among configurations
with ≥150 validation trades, positive expectancy in *both* windows, and train
drawdown inside 25%.

## Results

See `RESULTS.md` (written after the run completes — the numbers land there,
whichever way they point).

# Swing Momentum — Qullamaggie & Stockbee setups

Mechanical implementations of the setups traded by **Kristjan Kullamägi
(Qullamaggie)** and **Pradeep Bonde (Stockbee)**, tested on 6,009 US stocks /
13.6M daily bars / 1999–2017.

Four setups: the Qullamaggie continuation breakout (prior 30–100% leg → tight
base → range expansion, ADR-capped stop, scale out at day 4, trail the 10/20
dma), Bonde's momentum burst (+4% on rising volume out of a quiet stretch),
his Double Trouble anticipation scan, and an episodic-pivot proxy.

**Result: negative.** The breakouts underperform the same-day universe by
−85 bps per 10 days (t = −11.5, negative in 13 of 15 years). Double Trouble
looks strong long-only (+63 bps/10d) but 85% of that is survivorship bias in
the universe; the market-neutral version is flat in every window, and what
remains does not clear costs.

Full account, including the controls that killed it and what the test cannot
reproduce (Kullamägi's intraday opening-range entry): **[`RESULTS.md`](RESULTS.md)**.

```
swing/panel.py   panel + features + the four setup masks + Bonde's Market Monitor breadth
swing/setups.py  numba trade simulator (ADR-capped stop, scale-out, MA trail) + portfolio
scripts/40_scout.py  raw edge, parameter sensitivity, survivorship controls
```

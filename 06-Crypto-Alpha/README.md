# Crypto On-Chain Alpha

Fourth study in this repo, and the first where a *fundamental* data source
rather than price history carried the result.

**Universe:** 18 crypto assets (BTC, ETH, SOL-family, and 15 majors/alts),
2010-07 → 2026-05, from the [`coinmetrics/data`](https://github.com/coinmetrics/data)
community tier — which supplies MVRV, exchange in/outflows, exchange supply,
active addresses, transaction counts and hash rate alongside price.

**Result:** on-chain valuation + exchange-flow signals deliver **Sharpe 0.80 on
an untouched 2024–2026 test window** — matching Bitcoin's risk-adjusted return
at a fifth of the volatility and a quarter of the drawdown, with positive skew.
Price-based trend, which dominated the training window, was rejected by
validation and failed out of sample.

See **[`RESULTS.md`](RESULTS.md)** for the full account, per-signal breakdown,
leverage ladder, and what is still missing (MSTR/BMNR/IBIT/ETHA and all
options data).

```
crypto/config.py   splits, costs, signal-lag policy
crypto/ingest.py   coinmetrics CSVs -> panel (handles the PriceUSD vs
                   ReferenceRate split; drops the revisable final row)
crypto/signals.py  price, on-chain and cross-sectional forecasts, all lagged
crypto/book.py     spot-crypto portfolio: vol targeting, buffer, costs
scripts/30_study.py  sweep -> validate -> one frozen test run
tests/             leakage and accounting tests
```

# Analytics-DB

Data engineering and quantitative research workspace.

| Directory | Contents |
|---|---|
| `01-Ingestion` | Databricks ingestion notebooks (HTTP, JDBC, API sources) |
| `02-Transform` | Medallion-architecture transformation notebooks |
| `03-Alpha-Research` | Daily-bar breakout-pullback study over 6,009 US stocks — research apparatus plus a documented negative result (see its README §7) |
| `04-Intraday-Alpha` | 10-minute intraday study on real intraday bars (19 symbols, 2018-2026) with observable fills, walk-forward ML sizing layer, and pre-declared selection rules |
| `05-Multi-Asset` | Trend + carry + skew across 157 futures instruments, 1970-2024. **The positive result**: out-of-sample Sharpe 0.72 as first frozen, 1.03 with structural fixes; blended with the index it beats QQQ by ~8 points a year at the same volatility |
| `06-Crypto-Alpha` | On-chain valuation and exchange-flow signals over 18 crypto assets from Coin Metrics. Test 2024-01 to 2026-05: Sharpe 0.80, max drawdown −11.7%, near-zero correlation to the futures book |
| `07-Swing-Momentum` | Qullamaggie and Stockbee swing setups, tested to a negative result and then re-tested properly (v2: pullback entry, relative strength, sector rotation, regime gate). Also `agent/` — a compliance agent that grades trades against his rules with deterministic guardrails plus an LLM judge |

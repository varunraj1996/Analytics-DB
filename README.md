# Analytics-DB — a medallion lakehouse on Azure Databricks

An end-to-end pricing-analytics lakehouse built on Azure Databricks with Unity Catalog
(`adb_rtp` catalog), following the bronze → silver → gold medallion pattern. Built as a
hands-on learning project.

## What it does

**Ingestion (`01-Ingestion/`) → bronze**
- Daily market-pricing CSVs pulled over HTTP, with a process-run log table that drives
  incremental loads (each run picks up the next unprocessed file date).
- A streaming source of daily-pricing data.
- Reference tables read from Azure SQL over JDBC.
- Geolocation and historical weather data from the Open-Meteo APIs, landed as JSON.

**Transformation (`02-Transform/`) → silver and gold**
- Typed, cleaned silver tables with incremental (change-data-capture style) loads.
- A gold star schema for reporting: date, state, market, product and variety dimensions
  plus a daily-pricing fact table.
- Slowly changing dimensions, both Type 1 (overwrite with `MERGE`) and Type 2 (history
  with start/end dates).
- Geocoding: exploding nested API arrays into a flat silver geolocation table joined to markets.

## Secrets

Credentials are never stored in the notebooks. The storage-account key and the JDBC
connection string are read at runtime from a Databricks secret scope:

```python
dbutils.secrets.get(scope="adb-scope", key="storage-account-key")
dbutils.secrets.get(scope="adb-scope", key="pricing-ref-jdbc-url")
```

## Stack

Azure Databricks · Unity Catalog · Delta tables · PySpark · Spark SQL · ADLS Gen2 · Azure SQL

-- Databricks notebook source
-- MAGIC %python
-- MAGIC daily_pricing_streaming_source_folder_path = "abfss://adbbronze@adbstorageu.dfs.core.windows.net/daily-pricing-streaming-source-data"

-- COMMAND ----------

USE CATALOG adb_rtp;


-- COMMAND ----------

-- MAGIC %python
-- MAGIC daily_pricing_source_stream_DF = spark.sql("""SELECT
-- MAGIC current_timestamp() as DATETIME_OF_PRICING,
-- MAGIC cast(ROW_ID as bigint) ,
-- MAGIC STATE_NAME,
-- MAGIC MARKET_NAME,
-- MAGIC PRODUCTGROUP_NAME,
-- MAGIC PRODUCT_NAME,
-- MAGIC VARIETY,
-- MAGIC ORIGIN,
-- MAGIC cast(ARRIVAL_IN_TONNES as decimal(18,2)) + extract(seconds from current_timestamp() )/100 as ARRIVAL_IN_TONNES, 
-- MAGIC cast(MINIMUM_PRICE as decimal(36,2)) + extract(seconds from current_timestamp() )/100 as MINIMUM_PRICE,
-- MAGIC cast(MAXIMUM_PRICE as decimal(36,2)) + extract(seconds from current_timestamp() )/100 as MAXIMUM_PRICE,
-- MAGIC cast(MODAL_PRICE as decimal(36,2))+ extract(seconds from current_timestamp() )/100 as MODAL_PRICE,
-- MAGIC current_timestamp() as source_stream_load_datetime
-- MAGIC FROM adb_rtp.bronze.daily_pricing
-- MAGIC WHERE DATE_OF_PRICING='01/01/2023'""")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC daily_pricing_source_stream_DF = spark.sql(""" select current_timestamp() as DATETIME_OF_PRICING,
-- MAGIC cast(ROW_ID as bigint) ,
-- MAGIC STATE_NAME,
-- MAGIC MARKET_NAME,
-- MAGIC PRODUCTGROUP_NAME,
-- MAGIC PRODUCT_NAME,
-- MAGIC VARIETY,
-- MAGIC ORIGIN,
-- MAGIC ARRIVAL_IN_TONNES, 
-- MAGIC MINIMUM_PRICE,
-- MAGIC MAXIMUM_PRICE,
-- MAGIC MODAL_PRICE,
-- MAGIC current_timestamp() as source_stream_load_datetime from adb_rtp.bronze.daily_pricing limit 10000""")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC (daily_pricing_source_stream_DF
-- MAGIC .write
-- MAGIC .mode('append')
-- MAGIC .json(daily_pricing_streaming_source_folder_path)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.ls(daily_pricing_streaming_source_folder_path)

-- COMMAND ----------



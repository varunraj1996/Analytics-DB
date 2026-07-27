"""Coin Metrics community CSVs -> one crypto panel with on-chain features.

Source: ``github.com/coinmetrics/data`` (`csv/<asset>.csv`), the community
tier of the same data institutions license.  Daily, through 2026-05, and it
carries far more than price:

* ``CapMVRVCur``      market cap / realised cap - the canonical crypto
                      valuation ratio (16 of our assets have it)
* ``FlowInExUSD`` /
  ``FlowOutExUSD``    exchange inflow/outflow - coins moving onto an exchange
                      are supply looking for a bid (BTC and ETH only)
* ``SplyExNtv``       supply held on exchanges
* ``AdrActCnt``       active addresses - network usage
* ``TxCnt``           transaction count
* ``HashRate``        miner commitment (proof-of-work assets)

Two gotchas the loader handles:

1. Newer chains (SOL, AVAX, ATOM, NEAR, FIL) publish ``ReferenceRate`` rather
   than ``PriceUSD``; both are USD spot, so the loader takes whichever exists.
2. Coin Metrics *revises* recent rows and the last row is often partial. The
   final day is dropped, and every feature is used with a one-day lag on the
   signal side, because a metric stamped for day T is not knowable until T
   closes and is republished.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config

PRICE_CANDIDATES = ("PriceUSD", "ReferenceRateUSD", "ReferenceRate")

FIELDS = {
    "mvrv": "CapMVRVCur",
    "flow_in": "FlowInExUSD",
    "flow_out": "FlowOutExUSD",
    "sply_ex": "SplyExNtv",
    "addr_act": "AdrActCnt",
    "tx_cnt": "TxCnt",
    "hashrate": "HashRate",
    "mktcap": "CapMrktCurUSD",
    "volume": "volume_reported_spot_usd_1d",
    "supply": "SplyCur",
}

# Coins we will not trade regardless of data presence: stable-ish, wrapped, or
# too thin to model a spread for.
EXCLUDE = {"usdt", "usdc", "dai", "wbtc", "steth"}


def _price_col(cols) -> str | None:
    for c in PRICE_CANDIDATES:
        if c in cols:
            return c
    return None


def build_panel(min_days: int = 500) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(config.CM_DIR, "*.csv")))
    frames = []
    report = []
    for f in files:
        asset = os.path.basename(f)[:-4].lower()
        if asset in EXCLUDE:
            continue
        head = pd.read_csv(f, nrows=1)
        pc = _price_col(head.columns)
        if pc is None:
            continue
        want = ["time", pc] + [v for v in FIELDS.values() if v in head.columns]
        d = pd.read_csv(f, usecols=want, parse_dates=["time"])
        d = d.rename(columns={pc: "price", "time": "date"})
        d = d.rename(columns={v: k for k, v in FIELDS.items() if v in d.columns})
        d = d.dropna(subset=["price"])
        d = d[d["price"] > 0]
        if len(d) < min_days:
            continue
        d = d.sort_values("date")
        d = d.iloc[:-1]                       # last row is partial / revisable
        d["asset"] = asset.upper()
        frames.append(d)
        report.append({"asset": asset.upper(), "days": len(d),
                       "first": d["date"].min().date(), "last": d["date"].max().date(),
                       **{k: round(d[k].notna().mean(), 2) for k in FIELDS if k in d}})

    panel = pd.concat(frames, ignore_index=True)
    for c in panel.columns:
        if c not in ("date", "asset"):
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel = panel.sort_values(["asset", "date"], ignore_index=True)

    rep = pd.DataFrame(report).sort_values("days", ascending=False)
    print(rep.to_string(index=False))
    print(f"\n[ingest] panel {len(panel):,} rows, {panel['asset'].nunique()} assets, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    panel.to_parquet(config.PANEL, index=False)
    rep.to_parquet(config.COVERAGE, index=False)
    return panel


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(config.PANEL)


def wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="asset", values=field).sort_index()


def us_session_proxy(price: pd.Series) -> pd.Series:
    """Spot ETF proxy (IBIT / ETHA): the underlying sampled only on US trading
    days, so weekend moves arrive as a Monday gap - the single most important
    behavioural difference between holding the coin and holding the ETF.

    This is a *proxy*, not the ETF: it ignores the sponsor fee (~0.25%/yr) and
    premium/discount to NAV. It is fit for testing whether a rule survives the
    weekend-gap structure, and unfit for quoting ETF tracking performance.
    """
    bdays = pd.bdate_range(price.index.min(), price.index.max())
    return price.reindex(bdays).ffill(limit=3).dropna()


if __name__ == "__main__":
    build_panel()

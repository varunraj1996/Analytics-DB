"""Two vendors -> one clean 10-minute regular-session panel.

Source A (``intraday_pk``): per-day parquet files of 5-minute bars, UTC
timestamps, prices as *strings*, real share volume.  18 leveraged ETFs,
2020-07 to 2026-02.

Source B (``intraday_sg``): single tab-separated files of 5-minute bars, UTC,
AAPL / NFLX / TSLA 2018-2023.  Volume is in vendor lots, not shares - so
nothing downstream may compare volume across vendors.  It is kept only for
within-symbol ratios.

Both are converted to America/New_York, clipped to the 09:30-16:00 regular
session, and resampled 5min -> 10min with the usual OHLCV aggregation.  Days
with fewer than 30 of the 39 possible 10-minute bars are dropped: a sparse day
means the feed was missing prints, and a breakout study on missing prints
invents gaps that never traded.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config

NY = "America/New_York"


def _sessionise(df: pd.DataFrame) -> pd.DataFrame:
    """UTC 5-min bars -> NY regular-session 10-min bars."""
    ts = df["timestamp"].dt.tz_convert(NY)
    df = df.assign(ts=ts).sort_values("ts")
    t = df["ts"].dt.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    if df.empty:
        return df

    o = (df.set_index("ts")
         .resample("10min", label="left", closed="left")
         .agg(open=("open", "first"), high=("high", "max"),
              low=("low", "min"), close=("close", "last"),
              volume=("volume", "sum"))
         .dropna(subset=["open", "close"]))
    o = o.reset_index()
    o["date"] = o["ts"].dt.date
    o["bar"] = ((o["ts"].dt.hour * 60 + o["ts"].dt.minute) - (9 * 60 + 30)) // 10
    return o


def _finish(sym: str, frames: list[pd.DataFrame], min_bars: int) -> pd.DataFrame | None:
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (df["high"] >= df["low"])]

    good_days = df.groupby("date")["bar"].size()
    keep = good_days[good_days >= min_bars].index
    df = df[df["date"].isin(keep)]
    if df.empty:
        return None
    df["symbol"] = sym
    return df[["symbol", "date", "bar", "ts", "open", "high", "low", "close", "volume"]]


def load_pk(min_bars: int = 30) -> pd.DataFrame:
    out = []
    for sym in sorted(os.listdir(config.PK_DIR)):
        files = sorted(glob.glob(os.path.join(config.PK_DIR, sym, "*", "*", "*.parquet")))
        frames = []
        for f in files:
            d = pd.read_parquet(f)
            if len(d):
                frames.append(_sessionise(d))
        r = _finish(sym, frames, min_bars)
        if r is not None:
            out.append(r)
            print(f"[ingest] {sym:6s} {r['date'].nunique():5d} days "
                  f"{r['date'].min()} -> {r['date'].max()}")
    return pd.concat(out, ignore_index=True)


def load_sg(min_bars: int = 30) -> pd.DataFrame:
    out = []
    for sym, path in (("AAPL", "stock/aapl/AAPLUSUSD_M5.csv"),
                      ("NFLX", "stock/nflx/NFLXUSUSD_M5.csv"),
                      ("TSLA", "stock/tsla/TSLAUSUSD_M5.csv")):
        f = os.path.join(config.SG_DIR, path)
        d = pd.read_csv(f, sep="\t")
        d.columns = [c.lower() for c in d.columns]
        d["timestamp"] = pd.to_datetime(d["time"], utc=True)
        r = _finish(sym, [_sessionise(d)], min_bars)
        if r is not None:
            out.append(r)
            print(f"[ingest] {sym:6s} {r['date'].nunique():5d} days "
                  f"{r['date'].min()} -> {r['date'].max()}")
    return pd.concat(out, ignore_index=True)


def build_panel() -> pd.DataFrame:
    panel = pd.concat([load_pk(), load_sg()], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date", "bar"], ignore_index=True)
    for c in ("open", "high", "low", "close"):
        panel[c] = panel[c].astype(np.float64)
    panel.to_parquet(config.PANEL_PARQUET, index=False)
    print(f"[ingest] panel: {len(panel):,} bars  {panel['symbol'].nunique()} symbols  "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    return panel


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(config.PANEL_PARQUET)


if __name__ == "__main__":
    build_panel()

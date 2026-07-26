"""Bronze -> silver: turn ~8.5k per-ticker vendor text files into one panel.

Source format (one file per symbol, e.g. ``data/Stocks/aapl.us.txt``)::

    Date,Open,High,Low,Close,Volume,OpenInt
    1984-09-07,0.42388,0.42902,0.41874,0.42388,23220030,0

Prices are already adjusted for splits and dividends by the vendor, so the
series are directly usable for return computation.  What they are *not* is
clean: the tail of the file list contains empty files, single-row files,
zero-volume stubs and the occasional impossible print.  Everything in here is
about turning that into a trustworthy panel.
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config

_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _read_one(path: str) -> pd.DataFrame | None:
    ticker = os.path.basename(path).split(".")[0].upper()
    try:
        df = pd.read_csv(path, usecols=_COLS)
    except Exception:
        return None
    if len(df) < 300:
        # Not enough history to ever form a 40-day base plus a 200-day trend
        # filter plus an out-of-sample window.
        return None

    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    px = df[["open", "high", "low", "close"]]
    ok = (
        (px > 0).all(axis=1)
        & (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9)
        & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)
        & df["volume"].ge(0)
        & np.isfinite(df["volume"])
    )
    df = df[ok]
    if len(df) < 300:
        return None

    df = df.sort_values("date").drop_duplicates("date", keep="last")

    # A residual split/bad-print guard: the vendor is usually right, but a
    # handful of series contain a single day that moves >90% and immediately
    # reverses.  Those single prints would dominate any breakout study.
    c = df["close"].to_numpy(dtype=np.float64)
    if len(c) > 2:
        r = np.zeros_like(c)
        r[1:] = c[1:] / c[:-1] - 1.0
        bad = np.zeros(len(c), dtype=bool)
        spike = (np.abs(r) > 0.9)
        rev = np.zeros(len(c), dtype=bool)
        rev[:-1] = spike[1:] & (np.sign(r[1:]) != np.sign(np.r_[r[2:], 0.0]))
        bad |= spike & np.r_[rev[1:], False]
        if bad.any():
            df = df[~bad]
            if len(df) < 300:
                return None

    df["ticker"] = ticker
    return df


def build_panel(raw_dir: str | None = None, out: str | None = None,
                start: str | None = None, workers: int = 4) -> pd.DataFrame:
    raw_dir = raw_dir or config.RAW_DIR
    out = out or config.PANEL_PARQUET
    start = start or config.SAMPLE_START

    paths = sorted(glob.glob(os.path.join(raw_dir, "Stocks", "*.txt")))
    print(f"[data] {len(paths)} candidate stock files")

    frames = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, df in enumerate(ex.map(_read_one, paths, chunksize=64)):
            if df is not None:
                frames.append(df)
            if (i + 1) % 1000 == 0:
                print(f"[data]   parsed {i + 1}/{len(paths)}  kept {len(frames)}")

    panel = pd.concat(frames, ignore_index=True)
    del frames

    # Keep a warm-up buffer before the research sample so that 200-day
    # indicators are fully formed on day one of the train split.
    warmup = pd.Timestamp(start) - pd.Timedelta(days=420)
    panel = panel[panel["date"] >= warmup]

    panel = panel.sort_values(["ticker", "date"], ignore_index=True)
    for c in ("open", "high", "low", "close"):
        panel[c] = panel[c].astype(np.float32)
    panel["volume"] = panel["volume"].astype(np.float64)

    panel.to_parquet(out, index=False)
    print(f"[data] panel: {len(panel):,} rows  {panel['ticker'].nunique():,} tickers "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    return panel


def load_benchmark(symbol: str = "spy") -> pd.DataFrame:
    """Daily bars for a benchmark ETF, used for the market-regime filter."""
    path = os.path.join(config.RAW_DIR, "ETFs", f"{symbol}.us.txt")
    df = pd.read_csv(path, usecols=_COLS)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ignore_index=True)


def load_panel(path: str | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or config.PANEL_PARQUET)


if __name__ == "__main__":
    build_panel()

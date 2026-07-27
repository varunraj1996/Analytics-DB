"""pysystemtrade CSVs -> one aligned daily panel + instrument metadata.

Outputs
-------
``futures_panel.parquet``
    long frame: instrument, date, price (back-adjusted, instrument currency),
    carry_ann (annualised carry return estimate, fraction/yr), fx (instrument
    currency -> USD rate).
``futures_meta.parquet``
    instrument, asset_class, point_size, currency, spread_points, years.

Carry
-----
``multiple_prices_csv`` carries three synchronous series: PRICE (the priced
contract), CARRY (a nearer or further contract), and their YYYYMM contract
labels.  Annualised carry per Carver:

    raw = (PRICE - CARRY) / |years between PRICE_CONTRACT and CARRY_CONTRACT|

with the sign arranged so positive = the position earns roll yield when long.
It is then divided by price to make it a return per year, and smoothed - daily
carry marks are noisy around rolls.

Costs
-----
``spreadcosts.csv`` gives half the bid-ask spread in price points.  A full
round trip therefore costs ``2 * spread_points * point_size`` currency per
contract, which downstream converts to a fraction of traded notional.
Instruments with no entry (or a zero entry) are DROPPED rather than defaulted:
an unknown cost on an illiquid contract is exactly how a paper edge survives.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config


def _read_price(path: str) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["DATETIME"])
    s = d.set_index("DATETIME")["price"].dropna()
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")]


def _contract_years(label: float) -> float:
    y, m = divmod(int(label), 100)
    return y + (m - 1) / 12.0


def _read_carry(path: str) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["DATETIME"])
    d = d.dropna(subset=["PRICE", "CARRY", "PRICE_CONTRACT", "CARRY_CONTRACT"])
    if d.empty:
        return pd.Series(dtype=float)
    gap_years = (d["CARRY_CONTRACT"].map(_contract_years)
                 - d["PRICE_CONTRACT"].map(_contract_years))
    ok = gap_years.abs() > 1e-9
    d, gap_years = d[ok], gap_years[ok]
    # If the carry contract is LATER than the priced one, a long position rolls
    # from PRICE toward CARRY: positive (PRICE - CARRY) means downward-sloping
    # curve = positive roll yield for longs, and vice versa when earlier.
    ann_points = (d["PRICE"] - d["CARRY"]) / gap_years
    ann_ret = ann_points / d["PRICE"].abs().clip(lower=1e-9)
    s = pd.Series(ann_ret.to_numpy(), index=pd.DatetimeIndex(d["DATETIME"]).normalize())
    s = s[~s.index.duplicated(keep="last")]
    # rolls produce one-day spikes; a 63d median is Carver's own smoothing scale
    return s.rolling(63, min_periods=10).median().clip(-3, 3)


def _read_fx(cur: str, calendar: pd.DatetimeIndex) -> pd.Series:
    if cur == "USD":
        return pd.Series(1.0, index=calendar)
    p = os.path.join(config.PST, "fx_prices_csv", f"{cur}USD.csv")
    if not os.path.exists(p):
        return pd.Series(np.nan, index=calendar)
    d = pd.read_csv(p, parse_dates=["DATETIME"])
    d.columns = [c.lower() for c in d.columns]
    s = d.set_index("datetime")["price"].dropna()
    s.index = s.index.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(calendar, method="ffill")


def build_futures_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = pd.read_csv(os.path.join(config.PST, "csvconfig", "instrumentconfig.csv"))
    cfg = cfg.set_index("Instrument")
    spread = (pd.read_csv(os.path.join(config.PST, "csvconfig", "spreadcosts.csv"))
              .set_index("Instrument")["SpreadCost"])

    files = sorted(glob.glob(os.path.join(config.PST, "adjusted_prices_csv", "*.csv")))
    rows_meta = []
    price_map: dict[str, pd.Series] = {}
    carry_map: dict[str, pd.Series] = {}

    for f in files:
        inst = os.path.basename(f)[:-4]
        if inst.endswith(config.DUPLICATE_SUFFIXES):
            continue
        if inst not in cfg.index:
            continue
        sp = float(spread.get(inst, np.nan))
        if not np.isfinite(sp) or sp <= 0:
            continue

        px = _read_price(f)
        if len(px) < config.MIN_YEARS * 252:
            continue
        years = (px.index[-1] - px.index[0]).days / 365.25
        if years < config.MIN_YEARS:
            continue

        mp = os.path.join(config.PST, "multiple_prices_csv", f"{inst}.csv")
        carry = _read_carry(mp) if os.path.exists(mp) else pd.Series(dtype=float)

        price_map[inst] = px
        carry_map[inst] = carry
        rows_meta.append({
            "instrument": inst,
            "asset_class": str(cfg.loc[inst, "AssetClass"]),
            "point_size": float(cfg.loc[inst, "Pointsize"]),
            "currency": str(cfg.loc[inst, "Currency"]),
            "spread_points": sp,
            "years": years,
            "last_date": px.index[-1],
        })
        if len(rows_meta) % 40 == 0:
            print(f"[ingest] {len(rows_meta)} instruments loaded...")

    meta = pd.DataFrame(rows_meta).set_index("instrument")
    print(f"[ingest] kept {len(meta)} instruments "
          f"(dropped: no-cost/short-history/duplicate variants)")
    print(meta.groupby("asset_class")["years"].agg(["size", "median"]).to_string())

    # unified business-day calendar across the union of instrument histories
    lo = min(s.index[0] for s in price_map.values())
    hi = max(s.index[-1] for s in price_map.values())
    calendar = pd.bdate_range(lo, hi)

    frames = []
    fx_cache: dict[str, pd.Series] = {}
    for inst, px in price_map.items():
        cur = meta.loc[inst, "currency"]
        if cur not in fx_cache:
            fx_cache[cur] = _read_fx(cur, calendar)
        p = px.reindex(calendar).ffill(limit=config.MAX_FFILL_DAYS)
        c = carry_map[inst].reindex(calendar).ffill(limit=63)
        live = p.notna()
        frames.append(pd.DataFrame({
            "instrument": inst,
            "date": calendar[live],
            "price": p[live].to_numpy(),
            "carry_ann": c[live].to_numpy(),
            "fx": fx_cache[cur].reindex(calendar)[live].to_numpy(),
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(config.FUT_PANEL, index=False)
    meta.reset_index().to_parquet(config.FUT_META, index=False)
    print(f"[ingest] futures panel {len(panel):,} rows, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    return panel, meta


# ---------------------------------------------------------------------------
def build_sg_panel() -> pd.DataFrame:
    """The independent cross-check universe (spot FX, metals, indices, crypto).

    Costs here are modelled downstream as bps of notional by asset class since
    the dump has no per-instrument spread data.
    """
    spec = []
    for f in sorted(glob.glob(os.path.join(config.SG, "*", "*", "*_D1.csv"))):
        parts = f.split(os.sep)
        klass, name = parts[-3], parts[-2]
        spec.append((f, klass, name))
    frames = []
    for f, klass, name in spec:
        d = pd.read_csv(f, sep=None, engine="python")   # dump mixes tab and comma
        d.columns = [str(c).lower() for c in d.columns]
        if "close" not in d.columns and len(d.columns) >= 5:
            # headerless variant: time, open, high, low, close[, volume]
            d = pd.read_csv(f, sep=None, engine="python", header=None)
            d = d.iloc[:, :6] if d.shape[1] >= 6 else d
            d.columns = ["time", "open", "high", "low", "close", "volume"][:d.shape[1]]
        tcol = "time" if "time" in d.columns else d.columns[0]
        d["date"] = pd.to_datetime(d[tcol]).dt.normalize()
        d = d.dropna(subset=["close"])
        d = d[d["close"] > 0]
        if len(d) < 800:
            continue
        frames.append(pd.DataFrame({
            "instrument": name.upper(),
            "asset_class": klass,
            "date": d["date"],
            "price": d["close"].astype(float),
        }))
    panel = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(["instrument", "date"], keep="last")
             .sort_values(["instrument", "date"], ignore_index=True))
    panel.to_parquet(config.SG_PANEL, index=False)
    print(f"[ingest] sg panel {len(panel):,} rows, "
          f"{panel['instrument'].nunique()} instruments")
    return panel


if __name__ == "__main__":
    build_futures_panel()
    build_sg_panel()

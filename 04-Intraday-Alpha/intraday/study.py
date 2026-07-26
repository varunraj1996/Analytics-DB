"""Trade generation and evaluation over the 10-minute panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, engine10
from .config import IntradayParams


class Book:
    """Flat arrays per symbol plus session-level levels, built once."""

    def __init__(self, panel: pd.DataFrame):
        self.symbols = sorted(panel["symbol"].unique())
        self.per_sym: dict[str, dict] = {}
        for sym in self.symbols:
            d = panel[panel["symbol"] == sym].sort_values(["date", "bar"])
            day_codes, uniq_days = pd.factorize(d["date"], sort=True)
            arr = {
                "open": d["open"].to_numpy(np.float64),
                "high": d["high"].to_numpy(np.float64),
                "low": d["low"].to_numpy(np.float64),
                "close": d["close"].to_numpy(np.float64),
                "volume": d["volume"].to_numpy(np.float64),
                "bar": d["bar"].to_numpy(np.int64),
                "day_id": day_codes.astype(np.int64),
                "days": pd.DatetimeIndex(uniq_days),
                "ts": d["ts"].to_numpy(),
            }
            arr["atr"] = engine10.wilder_atr_bars(
                arr["high"], arr["low"], arr["close"], None, 39)
            self.per_sym[sym] = arr

    # ------------------------------------------------------------------
    def _levels(self, a: dict, p: IntradayParams, side: int):
        """Per-bar breakout level, opening-range width, prior-day close."""
        n = len(a["open"])
        level = np.full(n, np.nan)
        or_rng = np.full(n, np.nan)
        pdc = np.full(n, np.nan)

        day = a["day_id"]
        bar = a["bar"]
        high, low, close = a["high"], a["low"], a["close"]

        # session boundaries
        starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
        ends = np.r_[starts[1:], n]

        prior_c = np.nan
        prior_h = np.nan
        prior_l = np.nan
        for s, e in zip(starts, ends):
            # opening range: first `or_bars` bars of THIS session
            k = min(p.or_bars, e - s)
            oh = high[s:s + k].max()
            ol = low[s:s + k].min()
            for i in range(s, e):
                if bar[i] >= p.or_bars:
                    if p.level_kind == "or":
                        level[i] = oh if side > 0 else ol
                    else:                      # prior day high/low
                        level[i] = prior_h if side > 0 else prior_l
                    or_rng[i] = oh - ol
                    pdc[i] = prior_c
            prior_c = close[e - 1]
            prior_h = high[s:e].max()
            prior_l = low[s:e].min()
        return level, or_rng, pdc

    # ------------------------------------------------------------------
    def trades(self, p: IntradayParams,
               date_lo: str | None = None, date_hi: str | None = None) -> pd.DataFrame:
        frames = []
        for sym in self.symbols:
            a = self.per_sym[sym]
            level, or_rng, pdc = self._levels(a, p, p.side)

            cap = int((a["day_id"].max() + 1)) + 8
            t_entry = np.empty(cap, np.int64)
            t_exit = np.empty(cap, np.int64)
            t_epx = np.empty(cap, np.float64)
            t_xpx = np.empty(cap, np.float64)
            t_reason = np.empty(cap, np.int8)
            t_level = np.empty(cap, np.float64)
            t_atr = np.empty(cap, np.float64)

            nt = engine10.run_symbol(
                a["open"], a["high"], a["low"], a["close"], a["bar"],
                a["day_id"], a["atr"], level, pdc,
                np.int64(p.side), float(p.confirm_atr), float(p.pullback_atr),
                np.int64(p.max_wait_bars), np.int64(p.or_bars),
                float(p.stop_atr), float(p.target_atr),
                np.int64(p.max_hold_bars), p.eod_flat,
                float(p.min_or_range_atr), or_rng,
                p.trend_filter == "above_pdc",
                t_entry, t_exit, t_epx, t_xpx, t_reason, t_level, t_atr)
            if nt == 0:
                continue

            f = pd.DataFrame({
                "symbol": sym,
                "entry_i": t_entry[:nt], "exit_i": t_exit[:nt],
                "entry_px": t_epx[:nt], "exit_px": t_xpx[:nt],
                "reason": t_reason[:nt], "level": t_level[:nt],
                "atr": t_atr[:nt],
            })
            f["date"] = a["days"][a["day_id"][f["entry_i"]]]
            f["entry_ts"] = a["ts"][f["entry_i"]]
            f["bars_held"] = f["exit_i"] - f["entry_i"] + 1
            f["side"] = p.side
            frames.append(f)

        if not frames:
            return pd.DataFrame()
        t = pd.concat(frames, ignore_index=True)
        t["stop_frac"] = p.stop_atr * t["atr"] / t["entry_px"]
        t["ret_gross"] = t["side"] * (t["exit_px"] / t["entry_px"] - 1.0)
        t["ret"] = t["ret_gross"] - 2.0 * config.DEFAULT_COSTS.per_side_bps / 1e4
        if date_lo:
            t = t[t["date"] >= pd.Timestamp(date_lo)]
        if date_hi:
            t = t[t["date"] <= pd.Timestamp(date_hi)]
        return t.reset_index(drop=True)


# --------------------------------------------------------------------------
def daily_returns(t: pd.DataFrame, spec: config.PortfolioSpec | None = None) -> pd.Series:
    """Equal-risk portfolio: each trade risks ``risk_per_trade`` of equity to
    its stop, capped by weight and concurrency, aggregated per calendar day.

    Intraday, flat overnight, so daily P&L is just the sum of that day's
    trade P&Ls at their allocated sizes.  Concurrency is approximated by
    capping the number of trades per day at ``max_positions`` (they largely
    overlap in time within a session) - a slight simplification that errs
    conservative because it drops trades rather than resizing them.
    """
    spec = spec or config.DEFAULT_PORTFOLIO
    if len(t) == 0:
        return pd.Series(dtype=float)
    t = t.sort_values(["date", "entry_ts"])
    out = {}
    for day, grp in t.groupby("date"):
        g = grp.head(spec.max_positions)
        # risk-based weight: risk_frac / (stop distance as fraction of price)
        w = np.minimum(spec.risk_per_trade
                       / np.maximum(g["stop_frac"].to_numpy(), 1e-9),
                       spec.max_weight)
        # total gross exposure capped at gross_target
        tot = w.sum()
        if tot > spec.gross_target:
            w *= spec.gross_target / tot
        out[day] = float((w * g["ret"].to_numpy()).sum())
    return pd.Series(out).sort_index()


def metrics(dr: pd.Series) -> dict:
    if len(dr) < 30:
        return {}
    idx = pd.date_range(dr.index.min(), dr.index.max(), freq="B")
    full = dr.reindex(idx).fillna(0.0)
    eq = (1.0 + full).cumprod()
    yrs = len(full) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = full.std(ddof=1) * np.sqrt(252)
    sharpe = full.mean() * 252 / vol if vol > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "sharpe": float(sharpe), "vol": float(vol),
            "max_dd": float(dd), "days": len(full),
            "hit_days": float((full > 0).mean()),
            "final": float(eq.iloc[-1])}


def trade_stats(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return {"n": 0}
    return {
        "n": len(t),
        "bps_net": float(t["ret"].mean() * 1e4),
        "bps_gross": float(t["ret_gross"].mean() * 1e4),
        "win": float((t["ret"] > 0).mean()),
        "med_bps": float(t["ret"].median() * 1e4),
        "avg_bars": float(t["bars_held"].mean()),
        "per_day": float(len(t) / max(t["date"].nunique(), 1)),
        "stop%": float((t["reason"] == 1).mean()),
        "tgt%": float((t["reason"] == 2).mean()),
        "eod%": float((t["reason"] == 4).mean()),
    }

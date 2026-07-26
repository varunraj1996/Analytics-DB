"""The breakout -> pullback rule, plus everything needed to score a run.

The rule, stated once in plain language:

    A name that has been building a base for `base_len` sessions closes above
    the top of that base on expanding volume, in an uptrend, in a market that
    is itself in an uptrend.  We do **not** chase the breakout.  Instead a
    limit order is rested `pullback_atr` ATRs below the breakout close and left
    working for the next `entry_window` sessions.  If the market comes back to
    us intraday we are long; if it never does, we skip the trade entirely.
    Risk is a fixed ATR distance, reward a fixed ATR distance, with a time stop.

The short side is the exact mirror (breakdown, rally into resistance).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import config, engine
from .config import StrategyParams, UniverseSpec

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
def market_regime(bench: pd.DataFrame, calendar: np.ndarray, ma: int) -> np.ndarray:
    """Boolean per calendar day: is the market above its own moving average?

    Uses only information available at the close of the day in question, and
    is forward-filled onto the panel calendar (the benchmark occasionally has
    a session the panel does not, and vice versa).
    """
    b = bench.copy()
    b["ma"] = b["close"].rolling(ma).mean()
    b["ok"] = (b["close"] > b["ma"]).astype(np.int8)
    b = b.set_index("date")["ok"].reindex(pd.DatetimeIndex(calendar), method="ffill")
    return b.fillna(0).to_numpy().astype(bool)


def setup_mask(P, F, p: StrategyParams, uni: UniverseSpec,
               regime_ok: np.ndarray | None,
               member: np.ndarray | None = None) -> np.ndarray:
    """Vectorised breakout (or breakdown) detection over the whole panel."""
    atr = F["atr14"] if p.atr_len == 14 else F["atr20"]
    hh = F[f"hh{p.base_len}"]
    ll = F[f"ll{p.base_len}"]
    c, h, l, v = P.close, P.high, P.low, P.volume

    with np.errstate(invalid="ignore"):
        if p.side > 0:
            m = c > hh + p.breakout_margin * atr
        else:
            m = c < ll - p.breakout_margin * atr

        m &= v > p.vol_mult * F["avgvol50"]

        if p.trend_ma:
            ma = F[f"ma{p.trend_ma}"]
            m &= (c > ma) if p.side > 0 else (c < ma)

        if p.min_base_tightness > 0:
            m &= (hh - ll) <= p.min_base_tightness * atr

        if p.min_close_loc > 0:
            rng = np.maximum(h - l, 1e-9)
            loc = (c - l) / rng if p.side > 0 else (h - c) / rng
            m &= loc >= p.min_close_loc

        if p.max_ext_atr < 99:
            ext = (c - F["ma50"]) / atr
            m &= (ext <= p.max_ext_atr) if p.side > 0 else (-ext <= p.max_ext_atr)

        # tradability
        m &= F["addv21"] >= uni.min_addv_usd
        m &= F["avgvol50"] >= uni.min_share_vol
        m &= c >= uni.min_price
        m &= c <= uni.max_price
        m &= F["age"] >= uni.min_history_days
        m &= np.isfinite(atr) & (atr > 0)
        m &= np.isfinite(hh) & np.isfinite(ll)

        if regime_ok is not None and p.regime_ma:
            m &= regime_ok[P.day]

        if uni.sp500_only:
            if member is None:
                raise ValueError("sp500_only requires the membership mask")
            m &= member

    return np.nan_to_num(m, nan=False).astype(np.bool_)


# ---------------------------------------------------------------------------
def generate_trades(P, F, p: StrategyParams, uni: UniverseSpec,
                    regime_ok: np.ndarray | None,
                    day_lo: int = 0, day_hi: int = 10 ** 9,
                    member: np.ndarray | None = None) -> pd.DataFrame:
    """Run the rule and return one row per trade, with features attached."""
    mask = setup_mask(P, F, p, uni, regime_ok, member)
    mask &= (P.day >= day_lo) & (P.day <= day_hi)
    if not mask.any():
        return pd.DataFrame()

    atr = F["atr14"] if p.atr_len == 14 else F["atr20"]
    max_paths = int(mask.sum()) * (p.max_hold + 2) + 16

    (sig, ent, ext, epx, xpx, stop, tgt, atr_v, reason, mae, mfe,
     poff, plen, prow) = engine.simulate_trades(
        mask, P.open, P.high, P.low, P.close, atr, P.starts, P.ends,
        np.int64(p.side), float(p.pullback_atr), np.int64(p.entry_window),
        float(p.stop_atr), float(p.target_atr), np.int64(p.max_hold),
        float(p.trail_atr), np.int64(max_paths))

    if len(sig) == 0:
        return pd.DataFrame()

    t = pd.DataFrame({
        "sym": P.sym_id[sig],
        "ticker": P.symbols[P.sym_id[sig]],
        "sig_row": sig,
        "entry_row": ent,
        "exit_row": ext,
        "sig_day": P.day[sig],
        "entry_day": P.day[ent],
        "exit_day": P.day[ext],
        "sig_date": P.dates[sig],
        "entry_date": P.dates[ent],
        "entry_px": epx,
        "exit_px": xpx,
        "stop_px": stop,
        "target_px": tgt,
        "atr": atr_v,
        "reason": reason,
        "mae_r": mae,
        "mfe_r": mfe,
        "path_off": poff,
        "path_len": plen,
        "side": p.side,
    })
    t["bars_held"] = t["exit_row"] - t["entry_row"] + 1
    t["wait"] = t["entry_row"] - t["sig_row"]
    t["addv"] = F["addv21"][sig]
    # gross return and return in units of initial risk (R)
    t["ret"] = p.side * (t["exit_px"] / t["entry_px"] - 1.0)
    risk = (t["entry_px"] - t["stop_px"]).abs()
    t["r_mult"] = p.side * (t["exit_px"] - t["entry_px"]) / risk.replace(0, np.nan)

    t.attrs["path_row"] = prow
    return t


def attach_features(t: pd.DataFrame, P, F) -> pd.DataFrame:
    """Snapshot of everything known at the close of the signal day."""
    i = t["sig_row"].to_numpy()
    atr = F["atr14"]
    c = P.close
    out = t.copy()
    out["f_atr_pct"] = atr[i] / c[i]
    out["f_vol20"] = F["vol20"][i]
    out["f_vol_ratio"] = F["vol20"][i] / np.maximum(F["vol60"][i], 1e-9)
    out["f_volsurge"] = P.volume[i] / np.maximum(F["avgvol50"][i], 1.0)
    out["f_mom21"] = F["mom21"][i]
    out["f_mom63"] = F["mom63"][i]
    out["f_mom126"] = F["mom126"][i]
    out["f_mom252"] = F["mom252"][i]
    out["f_ext50"] = (c[i] - F["ma50"][i]) / atr[i]
    out["f_ext200"] = (c[i] - F["ma200"][i]) / atr[i]
    out["f_from52w"] = c[i] / np.maximum(F["hh252"][i], 1e-9) - 1.0
    rng = np.maximum(P.high[i] - P.low[i], 1e-9)
    out["f_closeloc"] = (c[i] - P.low[i]) / rng
    out["f_daygain"] = F["ret1"][i]
    out["f_barrange"] = rng / atr[i]
    for n in (20, 40, 60):
        out[f"f_base{n}"] = (F[f"hh{n}"][i] - F[f"ll{n}"][i]) / atr[i]
    out["f_logaddv"] = np.log10(np.maximum(F["addv21"][i], 1.0))
    out["f_logpx"] = np.log10(np.maximum(c[i], 0.01))
    out["f_age"] = np.log10(np.maximum(F["age"][i], 1.0))
    out["f_wait"] = out["wait"]
    out["f_gap"] = P.open[out["entry_row"].to_numpy()] / c[i] - 1.0
    out["f_pull_depth"] = (c[i] - out["entry_px"]) / atr[i] * out["side"]
    return out


FEATURE_COLS = [
    "f_atr_pct", "f_vol20", "f_vol_ratio", "f_volsurge", "f_mom21", "f_mom63",
    "f_mom126", "f_mom252", "f_ext50", "f_ext200", "f_from52w", "f_closeloc",
    "f_daygain", "f_barrange", "f_base20", "f_base40", "f_base60",
    "f_logaddv", "f_logpx", "f_age", "f_wait", "f_gap", "f_pull_depth",
]


# ---------------------------------------------------------------------------
def portfolio_from_trades(t: pd.DataFrame, P, calendar_len: int,
                          score: np.ndarray | None = None,
                          spec=None, costs=None,
                          day_lo: int = 0, day_hi: int | None = None,
                          participation: float = 0.01,
                          borrow_bps: float = 2.0):
    spec = spec or config.DEFAULT_PORTFOLIO
    costs = costs or config.DEFAULT_COSTS
    if len(t) == 0:
        return None

    day_hi = calendar_len - 1 if day_hi is None else day_hi
    sel = (t["entry_day"] >= day_lo) & (t["exit_day"] <= day_hi)
    t = t[sel].reset_index(drop=True)
    if len(t) == 0:
        return None
    if score is not None:
        score = np.asarray(score)[sel.to_numpy()]
    else:
        score = np.zeros(len(t))

    prow = t.attrs.get("path_row")
    if prow is None:
        raise ValueError("trade frame is missing its path index")
    # re-pack the path arrays for the surviving subset
    offs = t["path_off"].to_numpy()
    lens = t["path_len"].to_numpy()
    new_off = np.zeros(len(t), np.int64)
    tot = int(lens.sum())
    pday = np.empty(tot, np.int32)
    pclose = np.empty(tot, np.float64)
    q = 0
    for k in range(len(t)):
        rows = prow[offs[k]:offs[k] + lens[k]]
        new_off[k] = q
        pday[q:q + lens[k]] = P.day[rows]
        pclose[q:q + lens[k]] = P.close[rows]
        q += lens[k]

    order = np.lexsort((-score, t["entry_day"].to_numpy())).astype(np.int64)

    eq, expo, nopen, pnl, taken, w = engine.run_portfolio(
        order,
        t["entry_day"].to_numpy().astype(np.int64),
        t["exit_day"].to_numpy().astype(np.int64),
        t["entry_px"].to_numpy(), t["exit_px"].to_numpy(),
        t["stop_px"].to_numpy(), t["side"].to_numpy().astype(np.float64),
        t["addv"].to_numpy(), new_off, lens.astype(np.int32), pday, pclose,
        calendar_len, float(spec.starting_equity), float(spec.risk_per_trade),
        int(spec.max_positions), float(spec.max_weight), int(spec.max_new_per_day),
        float(costs.base_bps), float(costs.impact_coef),
        float(costs.min_cents_per_share), float(participation), float(borrow_bps))

    return {"equity": eq, "exposure": expo, "n_open": nopen,
            "trade_pnl": pnl, "taken": taken.astype(bool), "weight": w,
            "trades": t}


# ---------------------------------------------------------------------------
def metrics(equity: np.ndarray, calendar: np.ndarray,
            day_lo: int, day_hi: int, rf: float = 0.0) -> dict:
    eq = equity[day_lo:day_hi + 1]
    eq = eq[np.isfinite(eq)]
    if len(eq) < 20 or eq[0] <= 0:
        return {}
    r = np.diff(eq) / eq[:-1]
    yrs = len(eq) / TRADING_DAYS
    total = eq[-1] / eq[0]
    cagr = total ** (1 / yrs) - 1 if total > 0 else -1.0
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - rf) / vol if vol > 0 else 0.0
    dn = r[r < 0]
    sortino = (r.mean() * TRADING_DAYS) / (dn.std(ddof=1) * np.sqrt(TRADING_DAYS)) \
        if len(dn) > 2 and dn.std(ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = dd.min()
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_dd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "total_return": total - 1.0, "years": yrs,
        "final_equity": eq[-1],
    }


def trade_stats(t: pd.DataFrame, taken: np.ndarray | None = None) -> dict:
    d = t if taken is None else t[taken]
    if len(d) == 0:
        return {}
    r = d["r_mult"].to_numpy()
    ret = d["ret"].to_numpy()
    win = r > 0
    return {
        "n_trades": len(d),
        "win_rate": win.mean(),
        "avg_r": np.nanmean(r),
        "median_r": np.nanmedian(r),
        "avg_ret": np.nanmean(ret),
        "payoff": (np.nanmean(r[win]) / abs(np.nanmean(r[~win]))) if (~win).any() and win.any() else np.nan,
        "avg_bars": d["bars_held"].mean(),
        "fill_wait": d["wait"].mean(),
        "expectancy_bps": np.nanmean(ret) * 1e4,
    }

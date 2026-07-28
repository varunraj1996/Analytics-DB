"""Signals: price-based, on-chain, and cross-sectional.

Everything returns a forecast matrix (dates x assets) in capped z-units, and
everything is **lagged by ``config.SIGNAL_LAG`` days at the end**, because a
Coin Metrics daily metric for day T is published after T closes and is
subsequently revised. Not lagging on-chain data is the single most common way
crypto factor studies leak.

Normalisation is expanding-window per asset (never full-sample), so a forecast
on day T only knows what was knowable on day T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CAP = config.FC_CAP


# ---------------------------------------------------------------------------
def _cap(x: pd.DataFrame) -> pd.DataFrame:
    return x.clip(-CAP, CAP)


def _expanding_z(raw: pd.DataFrame, min_periods: int = 180) -> pd.DataFrame:
    """z-score against the asset's own expanding history."""
    mu = raw.expanding(min_periods=min_periods).mean()
    sd = raw.expanding(min_periods=min_periods).std()
    return _cap((raw - mu) / sd.replace(0.0, np.nan))


def _expanding_scale(raw: pd.DataFrame, min_periods: int = 120) -> pd.DataFrame:
    """Divide by expanding mean-absolute value (keeps sign, sets scale)."""
    s = raw.abs().expanding(min_periods=min_periods).median()
    return _cap(raw / s.replace(0.0, np.nan))


def lag(f: pd.DataFrame) -> pd.DataFrame:
    return f.shift(config.SIGNAL_LAG)


# ---------------------------------------------------------------------------
# price-based
# ---------------------------------------------------------------------------
def vol(price: pd.DataFrame, span: int = 35) -> pd.DataFrame:
    r = price.pct_change()
    v = r.ewm(span=span, min_periods=20).std()
    return v.clip(lower=1e-4)


def ewmac(price: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    lp = np.log(price)
    raw = (lp.ewm(span=fast, min_periods=fast).mean()
           - lp.ewm(span=slow, min_periods=slow).mean()) / vol(price)
    return lag(_expanding_scale(raw))


def breakout(price: pd.DataFrame, n: int) -> pd.DataFrame:
    hi = price.rolling(n, min_periods=n // 2).max()
    lo = price.rolling(n, min_periods=n // 2).min()
    raw = (price - (hi + lo) / 2) / (hi - lo).replace(0.0, np.nan)
    return lag(_expanding_scale(raw.ewm(span=max(n // 4, 2), min_periods=2).mean()))


# ---------------------------------------------------------------------------
# on-chain
# ---------------------------------------------------------------------------
def mvrv_value(mvrv: pd.DataFrame) -> pd.DataFrame:
    """Market cap / realised cap. High = holders sitting on large unrealised
    gains = historically a poor forward return. The signal is the *negative*
    z-score: a value/mean-reversion factor, and the best-documented on-chain
    effect there is."""
    return lag(-_expanding_z(np.log(mvrv.where(mvrv > 0))))


def exchange_netflow(flow_in: pd.DataFrame, flow_out: pd.DataFrame,
                     mktcap: pd.DataFrame, span: int = 7) -> pd.DataFrame:
    """Net USD moving ONTO exchanges, scaled by market cap. Inflow is supply
    arriving at the venue where it can be sold, so the forecast is negative
    net inflow. BTC and ETH only."""
    net = (flow_in - flow_out) / mktcap.replace(0.0, np.nan)
    return lag(-_expanding_z(net.ewm(span=span, min_periods=5).mean()))


def exchange_supply_trend(sply_ex: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Falling exchange balances = coins moving to self-custody = supply
    leaving the market. Forecast is the negative of the change."""
    chg = np.log(sply_ex.where(sply_ex > 0)).diff(n)
    return lag(-_expanding_scale(chg))


def address_momentum(addr: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Growth in active addresses - network adoption as a momentum proxy."""
    g = np.log(addr.where(addr > 0)).diff(n)
    return lag(_expanding_scale(g))


def nvt(mktcap: pd.DataFrame, tx: pd.DataFrame, span: int = 28) -> pd.DataFrame:
    """Network value to transactions - crypto's price/earnings. High NVT =
    expensive relative to chain usage, so the forecast is negative."""
    ratio = mktcap / tx.replace(0.0, np.nan)
    sm = ratio.ewm(span=span, min_periods=14).mean()
    return lag(-_expanding_z(np.log(sm.where(sm > 0))))


def hash_ribbon(hashrate: pd.DataFrame, fast: int = 30, slow: int = 60) -> pd.DataFrame:
    """Miner capitulation/recovery: the 30d hash MA crossing back above the
    60d has historically marked local bottoms in proof-of-work assets."""
    h = np.log(hashrate.where(hashrate > 0))
    raw = h.ewm(span=fast, min_periods=fast).mean() - h.ewm(span=slow, min_periods=slow).mean()
    return lag(_expanding_scale(raw))


# ---------------------------------------------------------------------------
# cross-sectional (rank within each day, demeaned -> dollar-neutral tilt)
# ---------------------------------------------------------------------------
def cross_sectional(raw: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Convert any signal into a within-day cross-sectional score in [-CAP,CAP].

    Ranks across assets on each date, centred so the average tilt is zero: the
    resulting book is long the top names and short the bottom, which strips out
    the market beta that dominates every crypto return series.
    """
    ok = raw.notna().sum(axis=1) >= min_names
    r = raw.rank(axis=1, pct=True)
    # rank-pct over k names averages to 0.5 + 1/(2k), not 0.5, so subtracting a
    # literal 0.5 leaves a small long bias that grows as names drop out.
    # Demean against the row's own mean instead.
    centred = r.sub(r.mean(axis=1), axis=0)
    span = centred.abs().max(axis=1).replace(0.0, np.nan)
    z = centred.div(span, axis=0) * CAP
    return z.where(ok, np.nan)


def xs_momentum(price: pd.DataFrame, n: int = 60, skip: int = 1) -> pd.DataFrame:
    ret = np.log(price).diff(n).shift(skip)
    return lag(cross_sectional(ret))


def xs_reversal(price: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    ret = np.log(price).diff(n)
    return lag(cross_sectional(-ret))


def xs_from(f: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionalise an already-lagged time-series forecast."""
    return cross_sectional(f)


# ---------------------------------------------------------------------------
def combine(parts: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    num = den = None
    for k, w in weights.items():
        if w == 0 or k not in parts:
            continue
        f = parts[k]
        m = f.notna().astype(float) * w
        v = f.fillna(0.0) * w
        num = v if num is None else num + v
        den = m if den is None else den + m
    if num is None:
        raise ValueError("no active signals")
    return _cap(num / den.replace(0.0, np.nan))


# ---------------------------------------------------------------------------
# volume — present for every asset in the dump, and unused until the data audit
# ---------------------------------------------------------------------------
def volume_expansion(volume: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Today's volume against its own recent average. Participation showing up
    is what separates a real move from drift."""
    lv = np.log(volume.where(volume > 0))
    return lag(_expanding_scale(lv - lv.rolling(n, min_periods=int(n * 0.7)).mean()))


def turnover(volume: pd.DataFrame, mktcap: pd.DataFrame) -> pd.DataFrame:
    """Volume as a share of market cap: attention per unit of size, which is
    comparable across assets in a way that raw volume is not."""
    t = (volume / mktcap.replace(0.0, np.nan))
    return lag(_expanding_z(np.log(t.where(t > 0))))


def volume_momentum(price: pd.DataFrame, volume: pd.DataFrame,
                    n: int = 60) -> pd.DataFrame:
    """Momentum confirmed by participation — a move on expanding volume scores
    higher than the same move on fading volume."""
    lv = np.log(volume.where(volume > 0))
    return lag(_expanding_scale(price.pct_change(n)
                                * (lv - lv.rolling(n, min_periods=n // 2).mean())))


def coverage_gate(fc: pd.DataFrame, parts: dict, names, min_signals: int):
    """Blank the forecast where too few of its inputs actually exist.

    The data audit found this matters more than any signal choice. Exchange
    flow is published for BTC and ETH only, and eleven of twenty-one assets
    carry fewer than two of the six on-chain signals, so a naive blend ranks
    an asset scored on one sparse input against one scored on six. This makes
    the cross-section comparable by refusing to rank what it cannot measure.
    """
    navail = sum(parts[s].reindex_like(fc).notna().astype(int)
                 for s in names if s in parts)
    return fc.where(navail >= min_signals)


# ---------------------------------------------------------------------------
# candidates from the field audit — families the original six never touched
# ---------------------------------------------------------------------------
def dilution_carry(iss_ntv: pd.DataFrame, supply: pd.DataFrame,
                   n: int = 30) -> pd.DataFrame:
    """Negative annualised issuance rate: low-emission chains earn a premium.

    The closest thing to a carry factor available on-chain — holders of a
    chain printing 8% a year are diluted by 8% a year, and that is a cost
    the price has to overcome. Slow-moving by construction.
    """
    rate = 365.0 * iss_ntv.rolling(n, min_periods=int(n * 0.7)).mean() / supply
    return lag(-_expanding_z(np.log(rate.where(rate > 0))))


def fee_share_security_budget(fee_ntv: pd.DataFrame, iss_ntv: pd.DataFrame,
                              n: int = 30) -> pd.DataFrame:
    """Fees as a share of miner revenue: blockspace demand without price.

    Both terms are in native units, so the ratio never touches market cap or
    price — which is what makes it orthogonal to every valuation signal in
    the book. A rising share means users are bidding for blockspace rather
    than the chain paying for its own security by printing.
    """
    share = fee_ntv / (fee_ntv + iss_ntv).replace(0.0, np.nan)
    return lag(_expanding_z(np.log(share.where(share > 0)
                                   .ewm(span=n, min_periods=n // 2).mean())))


def freefloat_trend(mktcap_est: pd.DataFrame, mktcap: pd.DataFrame,
                    n: int = 30) -> pd.DataFrame:
    """Negative drift in free float vs full market cap: unlock overhang.

    A rising ratio means locked or vesting supply is reaching the market —
    persistent sell pressure that is invisible to price-based signals and to
    valuation ratios alike.
    """
    ff = (mktcap_est / mktcap.replace(0.0, np.nan))
    lf = np.log(ff.where(ff > 0))
    return lag(-_expanding_scale(lf - lf.shift(n)))


def transfers_per_tx(tfr_cnt: pd.DataFrame, tx_cnt: pd.DataFrame,
                     n: int = 30) -> pd.DataFrame:
    """Batching intensity: transfers packed per transaction.

    Exchanges and custodians batch withdrawals; retail does not. Rising
    batching is a proxy for institutional and exchange flow dominating the
    chain's activity mix.
    """
    b = (tfr_cnt / tx_cnt.replace(0.0, np.nan))
    lb = np.log(b.where(b > 0))
    return lag(_expanding_scale(lb - lb.shift(n)))


def realized_cap_inflow(mktcap: pd.DataFrame, mvrv: pd.DataFrame,
                        n: int = 30) -> pd.DataFrame:
    """Growth in realised cap — net capital entering at a fresh cost basis.

    Realised cap is not published directly but is exactly reconstructable,
    since Coin Metrics defines MVRV as market cap over realised cap. Unlike
    the MVRV *level* (a slow valuation tilt) its growth rate is fast.
    """
    rc = mktcap / mvrv.replace(0.0, np.nan)
    lr = np.log(rc.where(rc > 0))
    return lag(_expanding_scale(lr - lr.shift(n)))


def activation_rate(addr_act: pd.DataFrame, addr_bal: pd.DataFrame,
                    n: int = 30) -> pd.DataFrame:
    """Share of the holder base transacting — the only dormancy proxy here.

    Coin-days-destroyed and SOPR are absent from this tier, so the fraction
    of funded addresses that move on a given day is the nearest available
    read on whether long-term holders are waking up.
    """
    a = (addr_act / addr_bal.replace(0.0, np.nan))
    return lag(_expanding_z(np.log(a.where(a > 0)
                                   .ewm(span=n, min_periods=n // 2).mean())))

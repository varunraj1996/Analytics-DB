"""Fetch the data this environment cannot reach. Run this on your own machine.

The data audit (RESULTS.md Addendum 4) found the binding constraint is not
the model, it is the universe: Coin Metrics' free tier publishes no price
series for SOL, AVAX, SUI, APT and every other asset that led the 2023-25
cycle, so the book could only choose among pre-2018 chains that as a group
went nowhere. Those hosts are blocked by this sandbox's egress policy, so
the fetch has to happen somewhere else.

What matters: the alt sleeve does **not** need the exotic on-chain fields.
Eleven of twenty-one assets already carried fewer than two on-chain signals,
and the lift from 0.75 to 1.11 Sharpe came from volume, turnover and
volume-confirmed momentum — all of which need only price, market cap and
volume. CoinGecko supplies exactly those three, free, no key, full history,
for every asset. So this script fills the real gap.

    pip install requests
    python3 31_fetch_local.py --out ./cm_extra
    #   ... copy ./cm_extra/*.csv into the scratchpad cm/ directory,
    #   then: python3 -m crypto.ingest

Output matches the Coin Metrics CSV schema the ingest already reads, so new
assets slot into the panel with no code change:

    time, PriceUSD, CapMrktCurUSD, volume_reported_spot_usd_1d

Equities (IBIT, ETHA, MSTR, BMNR) go through --equities, which needs
yfinance and writes an OHLCV csv per ticker.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request

CG = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

# CoinGecko ids for the assets the audit found missing. Everything the
# current panel already covers is deliberately omitted.
COINS = {
    "sol": "solana", "avax": "avalanche-2", "near": "near",
    "apt": "aptos", "sui": "sui", "arb": "arbitrum", "op": "optimism",
    "ton": "the-open-network", "shib": "shiba-inu", "pepe": "pepe",
    "hbar": "hedera-hashgraph", "vet": "vechain", "ftm": "fantom",
    "inj": "injective-protocol", "sei": "sei-network",
    "tia": "celestia", "grt": "the-graph", "rndr": "render-token",
    "atom": "cosmos", "fil": "filecoin", "jup": "jupiter-exchange-solana",
    "wif": "dogwifcoin", "bonk": "bonk",
}

EQUITIES = ["IBIT", "ETHA", "MSTR", "BMNR", "COIN", "GBTC", "BITO"]


def fetch_json(url: str, tries: int = 4):
    """GET with backoff. CoinGecko's free tier rate-limits aggressively."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as exc:                      # noqa: BLE001
            if i == tries - 1:
                print(f"    failed: {exc}")
                return None
            time.sleep(2 ** (i + 1))
    return None


def fetch_coin(sym: str, coin_id: str, out_dir: str) -> bool:
    """One asset -> one Coin-Metrics-shaped CSV of price, mktcap, volume."""
    url = f"{CG.format(id=coin_id)}?vs_currency=usd&days=max&interval=daily"
    d = fetch_json(url)
    if not d or "prices" not in d:
        return False
    # each series is [[epoch_ms, value], ...] on the same daily grid
    by_day: dict[str, dict] = {}
    for key, col in (("prices", "PriceUSD"),
                     ("market_caps", "CapMrktCurUSD"),
                     ("total_volumes", "volume_reported_spot_usd_1d")):
        for ms, v in d.get(key, []):
            day = time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))
            by_day.setdefault(day, {})[col] = v

    rows = sorted(by_day.items())
    if len(rows) < 400:
        print(f"    only {len(rows)} days, skipping")
        return False
    path = os.path.join(out_dir, f"{sym}.csv")
    cols = ["time", "PriceUSD", "CapMrktCurUSD", "volume_reported_spot_usd_1d"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for day, vals in rows:
            w.writerow({"time": day, **{c: vals.get(c, "") for c in cols[1:]}})
    print(f"    {len(rows)} days -> {path}")
    return True


def fetch_equities(tickers, out_dir: str) -> None:
    try:
        import yfinance as yf
    except ImportError:
        print("equities need yfinance:  pip install yfinance")
        return
    for t in tickers:
        try:
            df = yf.download(t, period="max", interval="1d",
                             auto_adjust=False, progress=False)
            if df is None or df.empty:
                print(f"  {t}: no data")
                continue
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            path = os.path.join(out_dir, f"{t}.csv")
            df.to_csv(path, index=False)
            print(f"  {t}: {len(df)} days {df['Date'].min().date()} -> "
                  f"{df['Date'].max().date()} -> {path}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  {t}: failed {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./cm_extra")
    ap.add_argument("--sleep", type=float, default=8.0,
                    help="seconds between calls; CoinGecko free tier is strict")
    ap.add_argument("--equities", action="store_true",
                    help="also fetch IBIT/ETHA/MSTR/BMNR via yfinance")
    ap.add_argument("--only", nargs="*", help="subset of symbols")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    want = {k: v for k, v in COINS.items()
            if not args.only or k in set(args.only)}
    print(f"fetching {len(want)} crypto assets into {args.out}")
    ok = 0
    for sym, cid in want.items():
        print(f"  {sym.upper()} ({cid})")
        ok += bool(fetch_coin(sym, cid, args.out))
        time.sleep(args.sleep)
    print(f"\n{ok}/{len(want)} crypto assets written")

    if args.equities:
        print(f"\nfetching {len(EQUITIES)} equities")
        fetch_equities(EQUITIES, args.out)

    print("\nnext: copy the crypto csvs into the scratchpad cm/ directory and run")
    print("      python3 -m crypto.ingest")


if __name__ == "__main__":
    main()

"""Point-in-time S&P 500 membership.

Source: ``fja05680/sp500`` - a dated list of index constituents starting
1996-01-02, including tickers that were later delisted or acquired.  Because
the list is dated, membership can be evaluated as of each bar without any
knowledge of what the index looks like today, which is what makes it usable as
a survivorship-bias-free "known to be a real, large, liquid company" flag.

This is the control universe.  The wide universe is defined by liquidity
screens alone; if an edge only exists in the wide universe and vanishes here,
that edge is probably an artefact of back-adjusted micro-cap prices.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config

MEMBERS_CSV = "S&P 500 Historical Components & Changes (Updated).csv"


def load_membership(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(config.SP500_DIR, MEMBERS_CSV)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ignore_index=True)


def membership_mask(P, path: str | None = None) -> np.ndarray:
    """Boolean per panel row: was this symbol in the S&P 500 on that date?

    The source file only records dates on which the constituent list changed,
    so membership is held constant between change dates.
    """
    df = load_membership(path)
    sym_index = {s: i for i, s in enumerate(P.symbols)}
    n_sym = len(P.symbols)

    change_days = np.searchsorted(P.calendar, df["date"].to_numpy(), "left")
    # membership state per (change event, symbol)
    events: list[tuple[int, np.ndarray]] = []
    for day, tickers in zip(change_days, df["tickers"]):
        if day >= len(P.calendar):
            continue
        flags = np.zeros(n_sym, dtype=bool)
        for tk in str(tickers).split(","):
            # the source uses '.' for share classes (BRK.B); the vendor files
            # use a dash, and a handful of names simply are not in the panel.
            j = sym_index.get(tk) or sym_index.get(tk.replace(".", "-"))
            if j is not None:
                flags[j] = True
        events.append((int(day), flags))

    if not events:
        return np.zeros(P.n, dtype=bool)

    ev_days = np.array([e[0] for e in events])
    ev_flags = np.stack([e[1] for e in events])

    # for every panel row, the most recent change event at or before its date
    idx = np.searchsorted(ev_days, P.day, "right") - 1
    valid = idx >= 0
    out = np.zeros(P.n, dtype=bool)
    out[valid] = ev_flags[idx[valid], P.sym_id[valid]]
    return out


def coverage_report(P, mask: np.ndarray) -> pd.DataFrame:
    """How many index members we actually have prices for, by year."""
    yrs = pd.DatetimeIndex(P.dates).year
    df = pd.DataFrame({"year": yrs, "member": mask, "sym": P.sym_id})
    g = df[df["member"]].groupby("year")["sym"].nunique()
    return g.rename("members_with_prices").to_frame()

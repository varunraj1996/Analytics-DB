"""Render the equity curve and split boundaries to a PNG."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/user/Analytics-DB/03-Alpha-Research")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha import config, data, workspace

ws = workspace.load()
res = json.load(open(os.path.join(config.RESULTS_DIR, "test_results.json")))
curves = {w: np.load(os.path.join(config.RESULTS_DIR, f"equity_{w}.npy"))
          for w in ("train", "valid", "test")
          if os.path.exists(os.path.join(config.RESULTS_DIR, f"equity_{w}.npy"))}

fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=False,
                         gridspec_kw={"height_ratios": [2.2, 1]})
cal = pd.DatetimeIndex(ws.calendar)
colors = {"train": "#6b7280", "valid": "#2563eb", "test": "#dc2626"}

ax = axes[0]
for w, (lo, hi) in (("train", ws.train), ("valid", ws.valid), ("test", ws.test)):
    if w not in curves:
        continue
    eq = curves[w][lo:hi + 1]
    ax.semilogy(cal[lo:hi + 1], eq / eq[0], color=colors[w], lw=1.4,
                label=f"{w}  CAGR {res[w]['cagr']:.1%}  Sharpe {res[w]['sharpe']:.2f}"
                      f"  DD {res[w]['max_dd']:.1%}")
spy = data.load_benchmark("spy").set_index("date")["close"].reindex(cal, method="ffill")
for w, (lo, hi) in (("train", ws.train), ("valid", ws.valid), ("test", ws.test)):
    b = spy.to_numpy()[lo:hi + 1]
    ax.semilogy(cal[lo:hi + 1], b / b[0], color="#9ca3af", lw=0.9, ls="--",
                label="SPY" if w == "train" else None)
ax.set_ylabel("growth of $1 (log)")
ax.set_title("Breakout-pullback intraday strategy - each window rebased to 1.0")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25, which="both")

ax = axes[1]
for w, (lo, hi) in (("train", ws.train), ("valid", ws.valid), ("test", ws.test)):
    if w not in curves:
        continue
    eq = curves[w][lo:hi + 1]
    dd = eq / np.maximum.accumulate(eq) - 1
    ax.fill_between(cal[lo:hi + 1], dd * 100, 0, color=colors[w], alpha=0.55, lw=0)
ax.set_ylabel("drawdown %")
ax.grid(alpha=0.25)

plt.tight_layout()
out = os.path.join(config.RESULTS_DIR, "equity.png")
plt.savefig(out, dpi=130)
print(f"chart -> {out}")

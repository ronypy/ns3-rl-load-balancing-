"""Generate README figures from results/raw.  Usage: python -m analysis.plots"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW, FIG = ROOT / "results/raw", ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)
ORDER = ["fixed", "random", "rule", "dqn", "noisy_dueling_dqn"]


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def fig_problem(ev):
    g = ev[ev.policy == "fixed"]
    cols = [f"load_c{i}" for i in range(7)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(7), g[cols].mean(), yerr=[ci(g[c]) for c in cols], capsize=3)
    ax.axhline(1 / 7, ls="--", c="gray", label="even load")
    ax.set(xlabel="cell (0 = centre)", ylabel="share of UEs", title="Fixed power: hotspot cell is overloaded")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig1_problem.png", dpi=150)


def fig_learning():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in ["dqn", "noisy_dueling_dqn"]:
        fs = sorted(RAW.glob(f"train_{name}_seed*.csv"))
        if not fs:
            continue
        R = np.stack([pd.read_csv(f).sort_values("episode").ret.rolling(10, min_periods=1).mean().values for f in fs])
        m, h = R.mean(0), np.array([ci(R[:, i]) for i in range(R.shape[1])])
        ax.plot(m, label=name); ax.fill_between(range(len(m)), m - h, m + h, alpha=0.2)
    ax.set(xlabel="episode", ylabel="return (10-ep moving avg)", title="Learning curves (mean, 95% CI over seeds)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig2_learning.png", dpi=150)


def fig_compare(ev):
    metrics = [("edge_tput", "cell-edge tput (Mbps)"), ("jain", "load fairness (Jain)"), ("pingpong_rate", "ping-pong rate")]
    pols = [p for p in ORDER if p in set(ev.policy)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (m, lab) in zip(axs, metrics):
        mean = [ev[ev.policy == p][m].mean() for p in pols]; err = [ci(ev[ev.policy == p][m]) for p in pols]
        ax.bar(range(len(pols)), mean, yerr=err, capsize=3)
        ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols, rotation=30, ha="right"); ax.set_title(lab)
    fig.tight_layout(); fig.savefig(FIG / "fig3_comparison.png", dpi=150)


if __name__ == "__main__":
    ev = pd.read_csv(RAW / "eval_episodes.csv")
    fig_problem(ev); fig_learning(); fig_compare(ev)
    print("figures written to", FIG)

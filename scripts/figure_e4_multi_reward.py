"""Figure 5 — E4 multi-reward variance isolation (D038 + 5/5 gap-fill).

Compares two flavours of "P × 3" on E4:

(A) Single-seed E4 from `eureka_results.csv`: 1 LLM call × 3 PPO seeds.
    Captures PPO seed variance for a fixed reward.
(B) Multi-reward gap-fill from `eureka_multi_reward.csv`: 3 LLM calls
    × 1 PPO seed each. Captures LLM-call variance for a fixed PPO seed.

The headline: PPO-seed variance is small (settle CV ≈ 2%, energy
CV ≈ 8%). LLM-call variance is meaningfully larger on settle (CV
≈ 10%) but comparable on energy. The reward function space sampled
by the LLM is narrow on energy but stretches the time-vs-energy
trade-off — consistent with D029 axis (P creative-but-robust).

Output: results/figure_e4_multi_reward.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_CSV = REPO_ROOT / "results" / "eureka_results.csv"
MULTI_CSV = REPO_ROOT / "results" / "eureka_multi_reward.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_e4_multi_reward.png"


def _by_cond(path: Path) -> dict:
    out: dict[str, list[dict]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out.setdefault(row["condition"], []).append(row)
    return out


def _cv_pct(arr: np.ndarray) -> float:
    if len(arr) < 2 or arr.mean() == 0:
        return 0.0
    return float(arr.std(ddof=1) / abs(arr.mean()) * 100)


def main() -> None:
    single = _by_cond(SINGLE_CSV)
    multi = _by_cond(MULTI_CSV)

    b_settle = np.array(
        [float(r["mean_settling_time_s@0.3"]) for r in single["B"]]
    )
    b_energy = np.array(
        [float(r["mean_control_energy@0.3"]) for r in single["B"]]
    )
    p1_settle = np.array(
        [float(r["mean_settling_time_s@0.3"]) for r in single["P"]]
    )
    p1_energy = np.array(
        [float(r["mean_control_energy@0.3"]) for r in single["P"]]
    )
    p2_settle = np.array(
        [float(r["mean_settling_time_s@0.3"]) for r in multi["P"]]
    )
    p2_energy = np.array(
        [float(r["mean_control_energy@0.3"]) for r in multi["P"]]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    rng = np.random.default_rng(0)

    for ax, b_arr, p1_arr, p2_arr, ylabel, title in (
        (axes[0], b_settle, p1_settle, p2_settle,
         "Mean settling time (s) @ eval radius 0.30",
         "Settling time"),
        (axes[1], b_energy, p1_energy, p2_energy,
         "Mean control energy @ eval radius 0.30",
         "Control energy"),
    ):
        positions = [0, 1, 2]
        groups = [b_arr, p1_arr, p2_arr]
        labels = [
            "B (3 PPO seeds,\nbaseline reward)",
            "P (1 LLM call ×\n3 PPO seeds)",
            "P (3 LLM calls ×\n1 PPO seed)",
        ]
        colors = ["#7f7f7f", "#1f77b4", "#3a86ff"]
        for pos, arr, color in zip(positions, groups, colors):
            ax.scatter(
                np.full(len(arr), pos) + rng.uniform(-0.05, 0.05, len(arr)),
                arr,
                s=85, c=color, edgecolor="black", linewidth=0.6, zorder=3,
            )
            ax.errorbar(
                pos, arr.mean(),
                yerr=arr.std(ddof=1) if len(arr) > 1 else 0.0,
                fmt="_", color=color, capsize=12, elinewidth=2,
                markeredgewidth=2, markersize=22, zorder=2,
            )

        # Expand y-limits FIRST so the CV annotations land in the safe
        # headroom above the data points.
        data_max = max(arr.max() for arr in groups)
        data_min = min(arr.min() for arr in groups)
        span = data_max - data_min
        ax.set_ylim(data_min - 0.12 * span, data_max + 0.32 * span)

        # CV labels placed just above each group's max value, inside the
        # chart, well clear of the panel title.
        for pos, arr in zip(positions, groups):
            cv = _cv_pct(arr)
            ax.text(
                pos, data_max + 0.18 * span, f"CV = {cv:.1f}%",
                ha="center", va="center", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Experiment 4 — variance isolation: "
        "PPO-seed variance vs LLM-call variance under a fixed PPO seed",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

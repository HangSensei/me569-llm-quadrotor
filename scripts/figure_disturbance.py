"""Figure — Disturbance robustness sweep for B/P/Q.

Reads `results/disturbance_results.csv` (B/P/Q × sigma ∈ {0, 0.05, 0.10,
0.20}) and produces a 2-panel figure: (left) hover success rate vs
sigma, (right) mean settling time vs sigma. Each condition is one line.

Output: results/figure_disturbance.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "disturbance_results.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_disturbance.png"

COLORS = {"B": "#7f7f7f", "P": "#1f77b4", "Q": "#d62728"}
MARKERS = {"B": "o", "P": "s", "Q": "^"}
LABELS = {
    "B": "B — textbook Bryson cost",
    "P": "P — Qwen 3.6-Plus cost",
    "Q": "Q — Qwen3.5-4B cost",
}


def main() -> None:
    by_cond: dict[str, list[dict]] = {}
    with RESULTS_CSV.open() as f:
        for row in csv.DictReader(f):
            by_cond.setdefault(row["condition"], []).append(row)

    conditions = [c for c in ["B", "P", "Q"] if c in by_cond]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    for cond in conditions:
        rows = sorted(by_cond[cond], key=lambda r: float(r["sigma"]))
        sigmas = np.array([float(r["sigma"]) for r in rows])
        success = np.array(
            [float(r["hover_success_rate"]) for r in rows]
        )
        settle_raw = [
            float(r["mean_settling_time_s"]) for r in rows
        ]
        settle = np.array(settle_raw)

        axes[0].plot(
            sigmas, success,
            color=COLORS[cond], marker=MARKERS[cond], markersize=8,
            linewidth=2, label=LABELS[cond],
        )
        # mask NaN for settle
        valid = ~np.isnan(settle)
        axes[1].plot(
            sigmas[valid], settle[valid],
            color=COLORS[cond], marker=MARKERS[cond], markersize=8,
            linewidth=2, label=LABELS[cond],
        )

    axes[0].set_xlabel("Process noise σ")
    axes[0].set_ylabel("Hover success rate")
    axes[0].set_title("Success rate vs process noise", fontsize=11)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower left", fontsize=9)

    axes[1].set_xlabel("Process noise σ")
    axes[1].set_ylabel("Mean settling time (s)")
    axes[1].set_title("Settling time vs process noise (only successful runs)",
                      fontsize=11)
    axes[1].grid(linestyle=":", alpha=0.5)

    fig.suptitle(
        "Disturbance robustness — closed-loop hover under additive process noise",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

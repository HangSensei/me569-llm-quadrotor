"""Figure 1 — E1 (SINDy basis) headline: P/Q reach the noise floor; B cannot.

Two-panel log-scale bar chart of one-step MSE and 50-step rollout MSE
for B/P/Q from `results/e1_results.csv`. Output:

    results/figure_e1_noise_floor.png

The headline narrative: P and Q drive prediction MSE down to the
finite-difference noise floor of the dataset (≈ 0.5 for one-step,
≈ 1e-2 for 50-step rollout). The polynomial baseline cannot represent
the sin/cos thrust-projection structure at any degree, so it sits
~30× above for one-step and ~1000× above for 50-step rollout.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "e1_results.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_e1_noise_floor.png"

COLORS = {"B": "#7f7f7f", "P": "#1f77b4", "Q": "#d62728"}
LABELS = {
    "B": "B — polynomial baseline\n(degree 3, 165 features)",
    "P": "P — Qwen 3.6-Plus\nfrontier cloud LLM",
    "Q": "Q — Qwen3.5-4B\nedge LLM (local MLX)",
}


def main() -> None:
    rows = {}
    with RESULTS_CSV.open() as f:
        for row in csv.DictReader(f):
            rows[row["condition"]] = row

    conditions = ["B", "P", "Q"]
    one_step = [float(rows[c]["one_step_mse"]) for c in conditions]
    rollout50 = [float(rows[c]["rollout_mse_50"]) for c in conditions]
    active = [int(rows[c]["active_terms"]) for c in conditions]
    n_basis = [int(rows[c]["n_basis"]) for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, vals, ylabel, title in (
        (axes[0], one_step, "one-step MSE", "One-step prediction MSE"),
        (axes[1], rollout50, "50-step rollout MSE", "50-step rollout MSE"),
    ):
        x = np.arange(len(conditions))
        bars = ax.bar(
            x, vals,
            color=[COLORS[c] for c in conditions],
            edgecolor="black", linewidth=0.7,
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        # Two-line tick labels: condition letter + "active/total" annotation,
        # so the active-term count never collides with the value-on-top label.
        tick_labels = [
            f"{c}\n({ac}/{nb} active)"
            for c, ac, nb in zip(conditions, active, n_basis)
        ]
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")
        # Value-on-top labels with generous breathing room above the bars.
        for xi, v in zip(x, vals):
            ax.text(
                xi, v * 1.8, f"{v:.3g}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        # Expand y-limits so the bold value labels never collide with the
        # panel title even on the tallest bar.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 5)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[c], ec="black", lw=0.7)
        for c in conditions
    ]
    fig.legend(
        legend_handles, [LABELS[c] for c in conditions],
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.03), fontsize=9,
    )
    fig.suptitle(
        "Experiment 1 — SINDy basis selection: LLM bases reach the noise floor",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

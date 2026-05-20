"""Figure 2 — E2 multi-seed CV: P shows variance, Q is deterministic (D033).

Reads `results/e3_multi_seed_clean.csv` (3 P + 3 Q runs under the
sanitized prompt). Plots control energy as scatter dots per condition,
with B as a single reference line.

The headline: P (Qwen 3.6-Plus, temperature 0.3) samples a meaningfully
different cost shape on each call (energy CV ≈ 65%). Q (Qwen3.5-4B,
greedy decoding) returns the IDENTICAL stage cost on every call
(energy CV = 0%). This is the D029 creative-stochastic vs
memorization-deterministic axis.

Output: results/figure_e2_multi_seed_cv.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "e3_multi_seed_clean.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_e2_multi_seed_cv.png"


def _cv_pct(arr: np.ndarray) -> float:
    if len(arr) < 2 or arr.mean() == 0:
        return 0.0
    return float(arr.std(ddof=1) / abs(arr.mean()) * 100)


def main() -> None:
    rows = {"B": [], "P": [], "Q": []}
    with RESULTS_CSV.open() as f:
        for row in csv.DictReader(f):
            rows[row["condition"]].append(row)

    p_settle = np.array([float(r["mean_settling_time_s"]) for r in rows["P"]])
    q_settle = np.array([float(r["mean_settling_time_s"]) for r in rows["Q"]])
    b_settle = float(rows["B"][0]["mean_settling_time_s"])
    p_energy = np.array([float(r["mean_control_energy"]) for r in rows["P"]])
    q_energy = np.array([float(r["mean_control_energy"]) for r in rows["Q"]])
    b_energy = float(rows["B"][0]["mean_control_energy"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    for ax, b_val, p_arr, q_arr, ylabel, title in (
        (axes[0], b_settle, p_settle, q_settle,
         "Mean settling time (s)", "Settling time over 3 LLM calls"),
        (axes[1], b_energy, p_energy, q_energy,
         "Mean control energy", "Control energy over 3 LLM calls"),
    ):
        rng = np.random.default_rng(0)
        # Slight x-jitter so identical Q dots are visible
        ax.scatter(
            np.full(3, 1) + rng.uniform(-0.05, 0.05, 3), p_arr,
            s=80, c="#1f77b4", edgecolor="black", linewidth=0.6,
            zorder=3, label="P (Qwen 3.6-Plus)",
        )
        ax.scatter(
            np.full(3, 2) + rng.uniform(-0.05, 0.05, 3), q_arr,
            s=80, c="#d62728", edgecolor="black", linewidth=0.6,
            zorder=3, label="Q (Qwen3.5-4B)",
        )
        ax.errorbar(
            1, p_arr.mean(), yerr=p_arr.std(ddof=1),
            fmt="_", color="#1f77b4", capsize=12, elinewidth=2,
            markeredgewidth=2, markersize=22, zorder=2,
        )
        ax.errorbar(
            2, q_arr.mean(), yerr=q_arr.std(ddof=1),
            fmt="_", color="#d62728", capsize=12, elinewidth=2,
            markeredgewidth=2, markersize=22, zorder=2,
        )
        ax.axhline(b_val, color="#7f7f7f", linestyle="--", linewidth=1.2,
                   alpha=0.8, zorder=1, label=f"B (Bryson) = {b_val:.3g}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["P", "Q"], fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlim(0.4, 2.6)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Expand y-limits so CV labels never crash into the panel title.
        data_max = max(p_arr.max(), q_arr.max(), b_val)
        data_min = min(p_arr.min(), q_arr.min(), b_val)
        span = data_max - data_min
        ax.set_ylim(data_min - 0.15 * span, data_max + 0.35 * span)

        # Annotate CV inside the chart area, anchored just under the panel
        # title with extra headroom from the data.
        cv_p = _cv_pct(p_arr)
        cv_q = _cv_pct(q_arr)
        cv_y = data_max + 0.22 * span
        ax.text(
            1, cv_y, f"CV = {cv_p:.1f}%",
            ha="center", va="center", fontsize=10, color="#1f77b4",
            fontweight="bold",
        )
        ax.text(
            2, cv_y, f"CV = {cv_q:.1f}%",
            ha="center", va="center", fontsize=10, color="#d62728",
            fontweight="bold",
        )

    # Figure-level legend at the bottom — keeps the chart area clean and
    # cannot collide with either CV labels or data points.
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, leg_labels,
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )

    fig.suptitle(
        "Experiment 2 — D033: Q is prompt-deterministic (CV = 0%); "
        "P samples a meaningful distribution",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

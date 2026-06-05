"""Figure 4 — E3 (Koopman) multi-seed P + Q failure annotation.

Reads:
  - `results/koopman_results_clean.csv` (single-seed B, single-seed P clean)
  - `results/koopman_multi_seed.csv` (P × 3 multi-call gap-fill from 5/5)

Plots one-step MSE and 50-step rollout MSE for B (single bar) and
P (3 dots + mean ± std), with Q annotated as failed (3 attempts under
the clean prompt produced no fenced code;).

Headline narrative: the LLM-generated observable consistently beats the
polynomial baseline by ~30× on one-step prediction and ~20× on 50-step
rollout. The variance across 3 P calls is small relative to the gap to B.
Q is fragile under precise clean prompts (4 independent failures across
E3 + E4 — +).

Output: results/figure_e3_multi_seed.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_CSV = REPO_ROOT / "results" / "koopman_results_clean.csv"
MULTI_CSV = REPO_ROOT / "results" / "koopman_multi_seed.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_e3_multi_seed.png"


def main() -> None:
    # Single-seed B (and single-seed P, just for reference)
    single = {}
    with SINGLE_CSV.open() as f:
        for row in csv.DictReader(f):
            single[row["condition"]] = row

    # Multi-seed P (3 rows, fresh from 5/5 gap-fill)
    p_runs = []
    with MULTI_CSV.open() as f:
        for row in csv.DictReader(f):
            p_runs.append(row)

    b_one = float(single["B"]["one_step_mse"])
    b_roll = float(single["B"]["rollout_mse_50"])
    p_one = np.array([float(r["one_step_mse"]) for r in p_runs])
    p_roll = np.array([float(r["rollout_mse_50"]) for r in p_runs])
    p_n_obs = [int(r["n_obs"]) for r in p_runs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, b_val, p_arr, ylabel, title in (
        (axes[0], b_one, p_one,
         "Validation one-step MSE", "One-step prediction MSE"),
        (axes[1], b_roll, p_roll,
         "Validation 50-step rollout MSE", "50-step rollout MSE"),
    ):
        # B as single bar at x=0
        ax.bar(
            [0], [b_val], width=0.55,
            color="#7f7f7f", edgecolor="black", linewidth=0.7,
            label="B — polynomial baseline (deg 2, 27 dims)",
        )

        # P as 3 dots + mean ± std at x=1
        rng = np.random.default_rng(0)
        ax.scatter(
            np.full(3, 1) + rng.uniform(-0.06, 0.06, 3), p_arr,
            s=85, c="#1f77b4", edgecolor="black", linewidth=0.6, zorder=3,
            label="P — Qwen 3.6-Plus (3 independent LLM calls)",
        )
        ax.errorbar(
            1, p_arr.mean(), yerr=p_arr.std(ddof=1),
            fmt="_", color="#1f77b4", capsize=12, elinewidth=2,
            markeredgewidth=2, markersize=22, zorder=2,
        )

        ax.set_yscale("log")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["B", "P", "Q"], fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlim(-0.6, 2.6)
        ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")

        # Expand the y-axis BEFORE placing annotations so nothing collides
        # with the panel title.
        all_vals = np.concatenate([[b_val], p_arr])
        ax.set_ylim(all_vals.min() / 30, b_val * 4)

        # Numeric value labels — placed just above each datum/bar in the
        # log-space coordinate, well clear of the panel title.
        ax.text(
            0, b_val * 1.3, f"{b_val:.3g}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        ax.text(
            1, p_arr.max() * 1.3,
            f"mean {p_arr.mean():.3g}\nstd {p_arr.std(ddof=1):.3g}",
            ha="center", va="bottom", fontsize=8, color="#1f77b4",
        )

        # Improvement annotation (P-vs-B factor) — between B and P in the
        # lower-middle region of the chart.
        factor = b_val / p_arr.mean()
        ax.text(
            0.5, p_arr.mean() * 1.5, f"$\\approx {factor:.0f}\\times$ lower",
            ha="center", va="bottom", fontsize=9,
            color="#1f77b4", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#1f77b4", lw=0.8),
        )

        # Q annotation at x=2 — placed mid-axis on log scale.
        ax.text(
            2, np.sqrt(p_arr.mean() * b_val),
            "Q\n3 attempts\nno code fence",
            ha="center", va="center", fontsize=9, color="#d62728",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fde0e0",
                      ec="#d62728", lw=1.2),
        )

    # Single legend on the left panel, placed below the chart in the
    # margin where it can't overlap data or titles.
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    # Note on n_obs values placed inside the right panel, low and right.
    axes[1].text(
        0.98, 0.02,
        f"P n$_{{obs}}$ $\\in$ {{{', '.join(str(n) for n in p_n_obs)}}}",
        transform=axes[1].transAxes,
        ha="right", va="bottom", fontsize=8, color="#444",
    )

    fig.suptitle(
        "Experiment 3 — Koopman/EDMDc: P consistently outperforms B; "
        "Q is fragile",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

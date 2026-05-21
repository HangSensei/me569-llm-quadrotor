"""Figure 3 — E2 prompt-leak: OLD vs CLEAN prompt comparison (D033 evidence).

Reads `results/e3_results.csv` (OLD prompt with the Bryson-shape worked
example) and `results/e3_results_clean.csv` (CLEAN prompt with the
example removed). For each condition (B/P/Q), shows control energy
under OLD vs CLEAN prompts.

The headline:
- B: identical (textbook baseline doesn't see the prompt) → control.
- P: gets BETTER with the CLEAN prompt (1.108 → 0.581 energy). The
  Bryson-shape example was an anti-helpful anchor for the frontier model.
- Q: gets DRAMATICALLY WORSE (0.475 → 2.594 energy). Under OLD,
  Q reproduced Bryson char-for-char (memorization). Under CLEAN, Q locked
  onto a different deterministic answer at 5.46× the Bryson energy.

This is direct evidence for the D029 creative-stochastic vs
memorization-deterministic axis.

Output: results/figure_e2_prompt_leak.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_CSV = REPO_ROOT / "results" / "e3_results.csv"
CLEAN_CSV = REPO_ROOT / "results" / "e3_results_clean.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_e2_prompt_leak.png"


def _by_cond(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["condition"]] = row
    return out


def main() -> None:
    old = _by_cond(OLD_CSV)
    clean = _by_cond(CLEAN_CSV)

    conditions = ["B", "P", "Q"]
    energy_old = [float(old[c]["mean_control_energy"]) for c in conditions]
    energy_clean = [float(clean[c]["mean_control_energy"]) for c in conditions]
    settle_old = [float(old[c]["mean_settling_time_s"]) for c in conditions]
    settle_clean = [float(clean[c]["mean_settling_time_s"]) for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    width = 0.35
    x = np.arange(len(conditions))

    for ax, old_vals, clean_vals, ylabel, title in (
        (axes[0], energy_old, energy_clean,
         "Mean control energy", "Control energy"),
        (axes[1], settle_old, settle_clean,
         "Mean settling time (s)", "Settling time"),
    ):
        b_old = ax.bar(
            x - width/2, old_vals, width,
            color="#ffb000", edgecolor="black", linewidth=0.6,
            label="OLD prompt (Bryson example leaked)",
        )
        b_clean = ax.bar(
            x + width/2, clean_vals, width,
            color="#3a86ff", edgecolor="black", linewidth=0.6,
            label="CLEAN prompt (example removed)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for xi, v in zip(x - width/2, old_vals):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        for xi, v in zip(x + width/2, clean_vals):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        # Pad y-limits so the bar-value labels never touch the panel title.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.18)

    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Pull the two Q annotations into the chart body, above the centre, where
    # there is empty space between B and the tall Q-CLEAN bar — no overlap
    # with either the panel title or the bar value labels.
    e_old_max = max(energy_old)
    e_clean_max = max(energy_clean)
    e_ymax = max(e_old_max, e_clean_max)
    axes[0].annotate(
        "Q under OLD prompt\nreproduces Bryson\nchar-for-char",
        xy=(2 - width/2, energy_old[2]),
        xytext=(0.55, e_ymax * 0.78),
        fontsize=8.5, color="#a14a00", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#a14a00", lw=0.8,
                        connectionstyle="arc3,rad=-0.2"),
    )
    axes[0].annotate(
        "Q under CLEAN\nlocks onto a different\nfixed cost (5.46$\\times$B)",
        xy=(2 + width/2, energy_clean[2]),
        xytext=(0.95, e_ymax * 0.45),
        fontsize=8.5, color="#1d3a8a", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#1d3a8a", lw=0.8,
                        connectionstyle="arc3,rad=0.2"),
    )

    fig.suptitle(
        "Experiment 2 — prompt-leak audit: removing the Bryson-shape "
        "worked example flips Q's behaviour",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

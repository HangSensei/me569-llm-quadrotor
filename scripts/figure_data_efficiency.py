"""Figure — data-efficiency: validation one-step MSE vs number of samples.

Two log-log panels:
  Left  (E1 SINDy basis): B (poly deg 3), P (Qwen 3.6-Plus), Q (Qwen3.5-4B), R (tuned RBF).
  Right (E3 Koopman/EDMDc): B (poly deg 2), P (Qwen 3.6-Plus), R (tuned RBF).
Mean +/- 1 std band over the 3 subsample seeds.

Output: results/figure_data_efficiency.png
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "data_efficiency.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_data_efficiency.png"

COLORS = {"B": "#7f7f7f", "P": "#1f77b4", "Q": "#d62728", "R": "#2ca02c"}
LABELS = {
    "B": "B (polynomial)",
    "P": "P (Qwen 3.6-Plus)",
    "Q": "Q (Qwen3.5-4B)",
    "R": "R (RBF, tuned)",
}


def _load():
    # data[experiment][condition][n_samples] = [mse, mse, ...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with RESULTS_CSV.open() as f:
        for row in csv.DictReader(f):
            data[row["experiment"]][row["condition"]][int(row["n_samples"])].append(
                float(row["val_one_step_mse"])
            )
    return data


def _plot_panel(ax, exp_data, conditions, title):
    for cond in conditions:
        if cond not in exp_data:
            continue
        ns = sorted(exp_data[cond].keys())
        means = np.array([np.mean(exp_data[cond][n]) for n in ns])
        stds = np.array([np.std(exp_data[cond][n], ddof=1) if len(exp_data[cond][n]) > 1
                         else 0.0 for n in ns])
        ax.plot(ns, means, "-o", color=COLORS[cond], label=LABELS[cond],
                markersize=5, linewidth=1.6, zorder=3)
        ax.fill_between(ns, np.clip(means - stds, 1e-300, None), means + stds,
                        color=COLORS[cond], alpha=0.18, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training transitions $N$")
    ax.set_ylabel("Validation one-step MSE")
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    # The curves are flat plateaus at well-separated y-levels (B high, P/Q low),
    # leaving the vertical middle empty — anchor the legend there so it never
    # sits on a data line.
    ax.legend(fontsize=9, framealpha=0.95, loc="center right")


def main() -> None:
    data = _load()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _plot_panel(axes[0], data["E1"], ["B", "P", "Q", "R"], "E1 — SINDy basis selection")
    _plot_panel(axes[1], data["E3"], ["B", "P", "R"], "E3 — Koopman / EDMDc observable")
    fig.suptitle(
        "Data efficiency — LLM-chosen bases vs polynomial (B) and tuned RBF (R); "
        "the gap is error-floor height, not sample count",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

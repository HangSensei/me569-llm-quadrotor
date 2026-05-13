"""Generate the headline figures for E4 (Eureka-style RL reward design).

Reads ``results/eureka_results.csv`` (per-seed rows for B/P/Q × 3 seeds)
and ``results/eureka_training_curves.npz`` (learning curves) and produces:

- ``results/eureka_comparison.png``: 3-panel bar chart with mean ± std
  error bars for hover_success_rate, mean_settling_time, and
  mean_control_energy at the E2-comparable eval radius (0.30).
- ``results/eureka_training_curves.png``: rolling-mean episode reward
  vs PPO timesteps, mean ± std band over the 3 seeds, per condition.

Both figures are caption-ready for Final Report §4.5.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "eureka_results.csv"
CURVES_NPZ = REPO_ROOT / "results" / "eureka_training_curves.npz"
COMPARISON_PNG = REPO_ROOT / "results" / "eureka_comparison.png"
CURVES_PNG = REPO_ROOT / "results" / "eureka_training_curves.png"


COLOR_BY_CONDITION = {"B": "#7f7f7f", "P": "#1f77b4", "Q": "#d62728"}
LABEL_BY_CONDITION = {
    "B": "B — baseline (Bryson)",
    "P": "P — Qwen 3.6-Plus",
    "Q": "Q — Qwen3.5-4B",
}


def _load_rows() -> dict[str, list[dict]]:
    by_cond: dict[str, list[dict]] = {}
    with RESULTS_CSV.open() as f:
        for row in csv.DictReader(f):
            by_cond.setdefault(row["condition"], []).append(row)
    return by_cond


def _to_float(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if not np.isnan(x):
            out.append(x)
    return np.asarray(out)


def make_comparison_figure() -> None:
    by_cond = _load_rows()
    conditions = [c for c in ["B", "P", "Q"] if c in by_cond]
    if not conditions:
        raise SystemExit(f"No usable rows in {RESULTS_CSV}")

    metrics = [
        ("hover_success_rate@0.3", "Hover success rate", "(higher = better)"),
        ("mean_settling_time_s@0.3", "Mean settling time (s)", "(lower = better)"),
        ("mean_control_energy@0.3", "Mean control energy", "(lower = better)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    for ax, (key, ylabel, sub) in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        colors = []
        for cond in conditions:
            vals = _to_float(by_cond[cond], key)
            if len(vals) == 0:
                means.append(float("nan"))
                stds.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            labels.append(LABEL_BY_CONDITION[cond])
            colors.append(COLOR_BY_CONDITION[cond])

        x = np.arange(len(conditions))
        bars = ax.bar(
            x, means, yerr=stds, color=colors,
            capsize=8, edgecolor="black", linewidth=0.6,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([c for c in conditions], fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n{sub}", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for xi, mi, si in zip(x, means, stds):
            label_y = (mi + si) if (mi == mi and si > 0) else mi
            ax.text(
                xi, label_y, f"{mi:.3f}\n±{si:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle(
        "Experiment 4 — PPO with B / P / Q rewards "
        "(eval @ E2-matched init_radius = 0.30, 3 PPO seeds, "
        "300k timesteps each)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(COMPARISON_PNG, dpi=150)
    print(f"Wrote {COMPARISON_PNG}")


def make_training_curves_figure() -> None:
    if not CURVES_NPZ.exists():
        print(f"Skip: {CURVES_NPZ} does not exist")
        return
    data = np.load(CURVES_NPZ)
    keys = list(data.keys())

    by_cond: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for k in keys:
        if k.endswith("_steps"):
            cond = k.split("_seed")[0]
            seed_str = k.split("_seed")[1].split("_steps")[0]
            steps = data[k]
            rewards = data[f"{cond}_seed{seed_str}_rewards"]
            by_cond.setdefault(cond, []).append((steps, rewards))

    conditions = [c for c in ["B", "P", "Q"] if c in by_cond]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.0))

    for cond in conditions:
        seeds_data = by_cond[cond]
        # Find common step grid (assume same across seeds; if not, take min length)
        min_len = min(len(s) for s, _ in seeds_data)
        steps = seeds_data[0][0][:min_len]
        rewards_stack = np.array([r[:min_len] for _, r in seeds_data])
        # Mask NaN
        mean = np.nanmean(rewards_stack, axis=0)
        std = np.nanstd(rewards_stack, axis=0)
        color = COLOR_BY_CONDITION[cond]
        ax.plot(steps, mean, color=color, linewidth=1.5,
                label=LABEL_BY_CONDITION[cond])
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("PPO training timesteps")
    ax.set_ylabel("Rolling-mean episode reward")
    ax.set_title(
        "Experiment 4 — PPO training curves "
        "(mean ± std over 3 seeds per condition)",
        fontsize=11,
    )
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CURVES_PNG, dpi=150)
    print(f"Wrote {CURVES_PNG}")


def main() -> None:
    make_comparison_figure()
    make_training_curves_figure()


if __name__ == "__main__":
    main()

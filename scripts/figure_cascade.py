"""Figure — Cascade: does E1's structural win propagate to closed-loop control?

Reads `results/cascade_results.csv` (B/P/Q closed-loop metrics with the
E1-fitted SINDy plant model in MPC + Bryson stage cost). Plots:

- Hover success rate (B/P/Q bars, 0 to 1)
- Mean settling time (B/P/Q, with NaN-friendly handling)
- Mean control energy

Annotates B failures (build crash or MPC infeasibility) with text.
A reference horizontal line shows the ground-truth-MPC + Bryson cost
result from `results/e3_results.csv` for the B condition (the 5/2
single-seed E2 number, 0.475 energy / 0.82 settle / 100% success).

Output: results/figure_cascade.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
CASCADE_CSV = REPO_ROOT / "results" / "cascade_results.csv"
REF_CSV = REPO_ROOT / "results" / "e3_results.csv"  # ground-truth-MPC reference
OUTPUT_PNG = REPO_ROOT / "results" / "figure_cascade.png"

COLORS = {"B": "#7f7f7f", "P": "#1f77b4", "Q": "#d62728"}


def _read_rows(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["condition"]] = row
    return out


def _f(s: str) -> float:
    if s in ("", None):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def main() -> None:
    rows = _read_rows(CASCADE_CSV)
    ref = _read_rows(REF_CSV)
    ref_b = ref.get("B")  # ground-truth MPC + Bryson reference

    conditions = [c for c in ["B", "P", "Q"] if c in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

    metric_specs = [
        ("hover_success_rate", "Hover success rate", (0, 1.15),
         "(higher = better)"),
        ("mean_settling_time_s", "Mean settling time (s)", None,
         "(lower = better)"),
        ("mean_control_energy", "Mean control energy", None,
         "(lower = better)"),
    ]
    ref_keys = {
        "hover_success_rate": "hover_success_rate",
        "mean_settling_time_s": "mean_settling_time_s",
        "mean_control_energy": "mean_control_energy",
    }

    for ax, (key, ylabel, ylim, sub) in zip(axes, metric_specs):
        vals = []
        labels_x = []
        colors = []
        build_failed_idx = []
        for i, cond in enumerate(conditions):
            r = rows[cond]
            failed = int(_f(r.get("build_failed", "0")) or 0) == 1
            v = _f(r.get(key, "")) if not failed else float("nan")
            vals.append(v)
            labels_x.append(cond)
            colors.append(COLORS[cond])
            if failed:
                build_failed_idx.append(i)
        x = np.arange(len(conditions))
        bars = ax.bar(
            x, vals, color=colors, edgecolor="black", linewidth=0.7,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Reference dashed line for ground-truth MPC + Bryson
        if ref_b is not None:
            ref_v = _f(ref_b.get(ref_keys[key], ""))
            if not np.isnan(ref_v):
                ax.axhline(
                    ref_v, color="black", linestyle="--", linewidth=1.0,
                    alpha=0.7,
                )
                # Blend axes-x (fraction) with data-y so the label always
                # sits at the right edge of the axes interior without
                # getting clipped by bbox_inches="tight".
                trans = mtransforms.blended_transform_factory(
                    ax.transAxes, ax.transData
                )
                ax.text(
                    0.88, ref_v,
                    f"GT-MPC ref = {ref_v:.3g}",
                    ha="right", va="bottom", fontsize=8, color="black",
                    alpha=0.85, transform=trans,
                )

        # Annotate values + build-failed markers
        for xi, v, cond in zip(x, vals, conditions):
            if np.isnan(v):
                ax.text(
                    xi, ax.get_ylim()[1] * 0.5, "build\nfailed",
                    ha="center", va="center", fontsize=10, color="#d62728",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fde0e0",
                              ec="#d62728", lw=1.0),
                )
            else:
                ax.text(
                    xi, v, f"{v:.3g}",
                    ha="center", va="bottom", fontsize=10,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n{sub}", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Cascade — closed-loop hover with E1-fitted SINDy plant model in MPC "
        "(Bryson stage cost held constant)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

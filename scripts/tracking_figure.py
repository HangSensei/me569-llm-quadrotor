"""Generate the trajectory-tracking figures for the Final Report.

Reads ``results/tracking_step.csv`` and ``results/tracking_figure8.csv``
(produced by ``scripts/run_tracking.py``) and writes:

- ``results/tracking_trajectories.png``: top-down (p_x, p_z) view of the
  3 figure-8 rollouts overlaid on the reference path. Each rollout
  starts from a different disturbed initial state, all converge onto
  the reference within roughly half a period.
- ``results/tracking_errors.png``: tracking-error magnitude vs time
  for both the step reference (1 rollout) and the figure-8 reference
  (3 rollouts), as a 2-panel figure.

These are caption-ready and feed Final Report §4 (the
trajectory-tracking demonstration subsection).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
STEP_CSV = REPO_ROOT / "results" / "tracking_step.csv"
FIG8_CSV = REPO_ROOT / "results" / "tracking_figure8.csv"
TRAJ_PNG = REPO_ROOT / "results" / "tracking_trajectories.png"
ERR_PNG = REPO_ROOT / "results" / "tracking_errors.png"


def _load_rollouts(path: Path) -> dict[int, dict]:
    """Read a tracking CSV into {rollout_idx: {col: np.array}}."""
    by_rollout: dict[int, list[dict]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            idx = int(row["rollout"])
            by_rollout.setdefault(idx, []).append(row)
    out: dict[int, dict] = {}
    for idx, rows in by_rollout.items():
        cols = rows[0].keys()
        arrays = {c: np.array([float(r[c]) for r in rows]) for c in cols}
        out[idx] = arrays
    return out


def make_trajectory_figure(fig8: dict[int, dict]) -> None:
    """Top-down view of figure-8 rollouts vs reference path."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Reference path: take the union of (ref_p_x, ref_p_z) across all rollouts
    # and de-duplicate by sorting on time. Easier: just use rollout 0's
    # reference, which spans one full period.
    ref0 = fig8[0]
    ax.plot(
        ref0["ref_p_x"], ref0["ref_p_z"],
        color="black", linestyle="--", linewidth=1.5, label="Reference (figure-8)",
    )

    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    for idx in sorted(fig8.keys()):
        roll = fig8[idx]
        x0 = (roll["p_x"][0], roll["p_z"][0])
        ax.plot(
            roll["p_x"], roll["p_z"],
            color=palette[idx % len(palette)], linewidth=1.4,
            label=f"Rollout {idx} (start = ({x0[0]:.1f}, {x0[1]:.1f}))",
        )
        ax.scatter(
            [x0[0]], [x0[1]],
            color=palette[idx % len(palette)], marker="o", s=40, zorder=5,
            edgecolor="black", linewidth=0.8,
        )

    ax.set_xlabel("p_x  (m)")
    ax.set_ylabel("p_z  (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Trajectory tracking: 3 disturbed-start rollouts converge onto a "
        "figure-8 reference\n(baseline tracking MPC, 8 s simulated)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(TRAJ_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {TRAJ_PNG}")


def make_error_figure(step: dict[int, dict], fig8: dict[int, dict]) -> None:
    """Two-panel: step error and figure-8 errors vs time."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=False)

    # Left: step reference, single rollout.
    step0 = step[0]
    axes[0].plot(step0["t"], step0["pos_error"], color="#1f77b4", linewidth=1.5,
                 label="position error")
    axes[0].plot(step0["t"], step0["att_error"], color="#d62728", linewidth=1.0,
                 linestyle="--", label="attitude error |theta|")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Error magnitude")
    axes[0].set_title("(a) Step reference: 0 → (0.5, 0.3)")
    axes[0].grid(linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].axvline(0.5, color="gray", linewidth=0.8, alpha=0.6)
    axes[0].text(0.55, axes[0].get_ylim()[1] * 0.92,
                 "step at t=0.5s", color="gray", fontsize=8, va="top")

    # Right: figure-8 reference, 3 rollouts (position error only for clarity).
    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    for idx in sorted(fig8.keys()):
        roll = fig8[idx]
        axes[1].plot(
            roll["t"], roll["pos_error"],
            color=palette[idx % len(palette)], linewidth=1.4,
            label=f"Rollout {idx}",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position error (m)")
    axes[1].set_title("(b) Figure-8 reference, 3 disturbed-start rollouts")
    axes[1].grid(linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Tracking error vs time (baseline MPC, no LLM)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ERR_PNG, dpi=150)
    print(f"Wrote {ERR_PNG}")


def main() -> None:
    step = _load_rollouts(STEP_CSV)
    fig8 = _load_rollouts(FIG8_CSV)
    make_trajectory_figure(fig8)
    make_error_figure(step, fig8)


if __name__ == "__main__":
    main()

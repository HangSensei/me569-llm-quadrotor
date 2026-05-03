"""Generate the headline figure for E3 (Koopman / EDMDc).

Reads BOTH ``results/koopman_results.csv`` (the OLD prompt — SINDy-framed
system description, kept for D036 audit comparison) and
``results/koopman_results_clean.csv`` (the audited clean prompt with the
dedicated ``KOOPMAN_SYSTEM_DESCRIPTION``). Produces a 2-panel
horizontal-bar comparison on (a) val one-step MSE and (b) 50-step
free-run rollout MSE, log scale, with both prompt variants side-by-side
per condition and a clear annotation that Q under the clean prompt
failed to produce a usable code fence after three attempts (D036).

Writes ``results/koopman_comparison.png``. Caption-ready for Final Report.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_CSV = REPO_ROOT / "results" / "koopman_results.csv"
CLEAN_CSV = REPO_ROOT / "results" / "koopman_results_clean.csv"
FIG_PATH = REPO_ROOT / "results" / "koopman_comparison.png"


COLOR_B = "#7f7f7f"   # gray
COLOR_P = "#1f77b4"   # blue
COLOR_Q = "#d62728"   # red
COLOR_BY_CONDITION = {"B": COLOR_B, "P": COLOR_P, "Q": COLOR_Q}


def _load(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def _val(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    old = _load(OLD_CSV)
    clean = _load(CLEAN_CSV)

    # Row order top-to-bottom: B-old, B-clean, P-old, P-clean, Q-old, Q-clean.
    # B-old and B-clean are arithmetically identical (polynomial baseline,
    # not affected by the prompt audit) but we plot both for visual symmetry.
    spec = [
        ("B (orig prompt)", "B", old),
        ("B (clean prompt)", "B", clean),
        ("P (orig prompt)", "P", old),
        ("P (clean prompt)", "P", clean),
        ("Q (orig prompt)", "Q", old),
        ("Q (clean prompt)", "Q", clean),
    ]
    labels: list[str] = []
    one_step: list[float] = []
    rollout: list[float] = []
    colors: list[str] = []
    annotations: list[str] = []
    failed: list[bool] = []

    for label, cond, table in spec:
        row = table.get(cond)
        n_obs_str = row.get("n_obs", "?") if row else "?"
        os_val = _val(row, "one_step_mse")
        rl_val = _val(row, "rollout_mse_50")
        if row is None:
            full_label = f"{label} — failed"
            os_val = float("nan")
            rl_val = float("nan")
            ann = "no code emitted"
            failed.append(True)
        else:
            full_label = f"{label} (N={n_obs_str})"
            ann = ""
            failed.append(False)
        labels.append(full_label)
        one_step.append(os_val if os_val is not None else float("nan"))
        rollout.append(rl_val if rl_val is not None else float("nan"))
        colors.append(COLOR_BY_CONDITION[cond])
        annotations.append(ann)

    n_rows = len(labels)
    y = np.arange(n_rows)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)

    # Plot helper that handles NaN (failed rows): draw a red hatched marker
    # on the left edge of the axis so the failure is visible without
    # contaminating the log-scale extent.
    def plot_panel(ax, vals, xlabel, title):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", which="both", linestyle=":", alpha=0.5)

        finite = [v for v in vals if v == v and v > 0]
        if finite:
            xmin = min(finite) / 4
            xmax = max(finite) * 4
            ax.set_xlim(xmin, xmax)

        for yi, vi, ann, fail, col in zip(y, vals, annotations, failed, colors):
            if not fail and vi == vi and vi > 0:
                ax.barh(yi, vi, color=col)
                ax.text(vi * 1.05, yi, f"{vi:.2e}",
                        va="center", ha="left", fontsize=9)
            else:
                # Draw a hatched 'X' marker spanning the axis to make the
                # failure visually obvious.
                xmin, xmax = ax.get_xlim()
                ax.barh(
                    yi, xmax - xmin, left=xmin,
                    color="white", edgecolor=col, hatch="xxx",
                    alpha=0.35,
                )
                ax.text(
                    (xmin * xmax) ** 0.5, yi, ann or "failed",
                    va="center", ha="center", fontsize=9, color=col,
                    fontweight="bold",
                )

    plot_panel(
        axes[0], one_step,
        "Validation one-step MSE (log scale)",
        "(a) One-step prediction MSE",
    )
    plot_panel(
        axes[1], rollout,
        "50-step free-run rollout MSE (log scale)",
        "(b) 50-step rollout MSE",
    )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)

    fig.suptitle(
        "Experiment 3 — Koopman/EDMDc observable comparison "
        "(orig vs clean prompt, single-seed)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()

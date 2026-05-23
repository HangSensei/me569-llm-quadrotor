"""Figure — prompt-robustness profile (P vs Q across E1-E4).

Per task, grouped emission-rate bars (0-1) for P and Q, annotated with the
quality CV among successful paraphrases (or "n/a"). Q's zero-emission bars
on the structural tasks (E3/E4) are the visual punchline.

Output: results/figure_prompt_robustness.png
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from me569_project.llm.robustness_metrics import code_emission_rate, quality_cv

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "prompt_robustness.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figure_prompt_robustness.png"

TASKS = ["E1", "E2", "E3", "E4"]
TASK_LABELS = {"E1": "E1\nSINDy", "E2": "E2\nMPC cost",
               "E3": "E3\nKoopman", "E4": "E4\nReward"}
COND_COLOR = {"P": "#1f77b4", "Q": "#d62728"}


def _load():
    # rows[(task, cond)] = list of (emitted_bool, quality_or_None)
    rows = defaultdict(list)
    with RESULTS_CSV.open() as f:
        for r in csv.DictReader(f):
            emitted = r["emitted_code"] in ("1", "True", "true")
            q = r["primary_metric_value"]
            qv = float(q) if (q not in ("", "nan", "None")) else None
            rows[(r["task"], r["condition"])].append((emitted, qv))
    return rows


def main() -> None:
    rows = _load()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS))
    width = 0.38

    for cond in ["P", "Q"]:
        emit_rates, cvs = [], []
        for task in TASKS:
            entries = rows.get((task, cond), [])
            emit_rates.append(code_emission_rate([e for e, _ in entries]))
            good = [q for e, q in entries if e and q is not None]
            cvs.append(quality_cv(good))
        offset = (-width / 2) if cond == "P" else (width / 2)
        ax.bar(x + offset, emit_rates, width, color=COND_COLOR[cond],
               edgecolor="black", linewidth=0.6, label=f"{cond}")
        for xi, rate, cv in zip(x + offset, emit_rates, cvs):
            cv_txt = "CV n/a" if cv is None else f"CV {cv * 100:.0f}%"
            ax.text(xi, rate + 0.02, f"{rate:.1f}\n{cv_txt}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("Code-emission rate (of K=5 paraphrases)")
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=0.8)
    ax.set_title("Prompt robustness — emission rate (bars) + quality CV (labels), "
                 "P vs Q across tasks", fontsize=11, pad=10)
    ax.legend(title="condition", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

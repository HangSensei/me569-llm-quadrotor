"""Figure — Cost / latency trade-off between cloud P and edge Q.

Aggregates LLM-call timing across all four experiments. Cloud P (Qwen
3.6-Plus via Alibaba DashScope) charges by token; edge Q (Qwen3.5-4B
local on Apple Silicon via mlx-vlm) is free at the API level but pays a
one-time 4.8 GB model load and ~95 s/call generation cost.

The figure is the central client-style answer to "which LLM should I use
for control-pipeline code generation?": per-call latency tells the
engineer how slow iteration will be; total cost tells the budget owner
how much one experiment costs.

Output: ``results/figure_cost_latency.png``
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PNG = REPO_ROOT / "results" / "figure_cost_latency.png"


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _pull(rows: list[dict], cond: str, key: str) -> list[float]:
    """Pull a numeric column for one condition, handling empty cells.

    Falls back to ``fit_or_call_time_s`` when ``llm_call_time_s`` is the
    requested key but the row uses the older E2 schema where the LLM
    call time is stored under that legacy column. For B rows in the
    legacy schema, ``fit_or_call_time_s`` is the closed-loop eval
    time, which we exclude.
    """
    out: list[float] = []
    fallback = "fit_or_call_time_s" if key == "llm_call_time_s" else None
    for r in rows:
        if r.get("condition") != cond:
            continue
        v = r.get(key, "")
        if (v == "" or v is None) and fallback is not None:
            # Only take the fallback for LLM conditions (P/Q): for B in the
            # legacy E2 CSV, fit_or_call_time_s is the eval time, not a call.
            if cond in ("P", "Q"):
                v = r.get(fallback, "")
        if v == "" or v is None:
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if np.isnan(x):
            continue
        out.append(x)
    return out


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main() -> None:
    # Per-experiment LLM-call timing (seconds). Pull from headline CSVs +
    # multi-seed gap-fill CSVs to get robust means.
    src = {
        "E1\nSINDy\nbasis": [
            REPO_ROOT / "results" / "e1_results.csv",
        ],
        "E2\nMPC\ncost": [
            REPO_ROOT / "results" / "e3_results_clean.csv",
        ],
        "E3\nKoopman": [
            REPO_ROOT / "results" / "koopman_results_clean.csv",
            REPO_ROOT / "results" / "koopman_multi_seed.csv",
            # Include OLD-prompt result so Q's E3 attempt-time is visible.
            # Even though the OLD-prompt Q output was eliminated under audit
            # (D036), the LLM-call wall-time itself is a fair latency
            # data point for the cost/latency comparison.
            REPO_ROOT / "results" / "koopman_results.csv",
        ],
        "E4\nEureka RL": [
            REPO_ROOT / "results" / "eureka_results.csv",
            REPO_ROOT / "results" / "eureka_multi_reward.csv",
        ],
    }

    p_means, p_stds, q_means, q_stds = [], [], [], []
    p_tokens_total, q_total = 0.0, 0.0
    n_p_calls = 0
    n_q_calls = 0
    for label, paths in src.items():
        all_rows: list[dict] = []
        for p in paths:
            all_rows.extend(_read_rows(p))
        p_times = _pull(all_rows, "P", "llm_call_time_s")
        q_times = _pull(all_rows, "Q", "llm_call_time_s")
        p_mean, p_std = _mean_std(p_times)
        q_mean, q_std = _mean_std(q_times)
        p_means.append(p_mean)
        p_stds.append(p_std)
        q_means.append(q_mean)
        q_stds.append(q_std)
        # Token / call counts
        p_tokens = sum(
            float(r.get("llm_total_tokens", 0) or 0) for r in all_rows
            if r.get("condition") == "P"
        )
        p_tokens_total += p_tokens
        n_p_calls += len(p_times)
        n_q_calls += len(q_times)
        q_total += sum(q_times)

    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.5, 2.5, 0.2, 1.4, 1.4])
    ax_p = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_tab = fig.add_subplot(gs[0, 3:])
    ax_tab.axis("off")

    # Shorten the x-axis labels: split the experiment ID and its descriptive
    # tag onto two visually-distinct lines that take less horizontal space.
    labels = list(src.keys())
    x = np.arange(len(labels))

    # Panel 1 — P (cloud)
    bars_p = ax_p.bar(
        x, p_means, yerr=p_stds, color="#1f77b4",
        edgecolor="black", linewidth=0.7, capsize=5,
        error_kw={"linewidth": 1.0, "capthick": 1.0},
    )
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(labels, fontsize=8.5)
    ax_p.set_ylabel("Mean wall time per LLM call (s)")
    ax_p.set_title("P — Qwen 3.6-Plus (cloud, DashScope API)", fontsize=10,
                   pad=10)
    ax_p.grid(axis="y", linestyle=":", alpha=0.5)
    # Headroom so the value-on-top labels don't crash into the title.
    ymin_p, ymax_p = ax_p.get_ylim()
    ax_p.set_ylim(ymin_p, ymax_p * 1.20)
    for xi, m, s in zip(x, p_means, p_stds):
        if not np.isnan(m):
            ax_p.text(xi, m + s + 0.3, f"{m:.1f}",
                      ha="center", va="bottom", fontsize=9)

    # Panel 2 — Q (edge)
    bars_q = ax_q.bar(
        x, q_means, yerr=q_stds, color="#d62728",
        edgecolor="black", linewidth=0.7, capsize=5,
        error_kw={"linewidth": 1.0, "capthick": 1.0},
    )
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(labels, fontsize=8.5)
    ax_q.set_ylabel("Mean wall time per LLM call (s)")
    ax_q.set_title("Q — Qwen3.5-4B (edge, MLX local on M4)", fontsize=10,
                   pad=10)
    ax_q.grid(axis="y", linestyle=":", alpha=0.5)
    ymin_q, ymax_q = ax_q.get_ylim()
    ax_q.set_ylim(ymin_q, ymax_q * 1.18)
    for xi, m, s in zip(x, q_means, q_stds):
        if not np.isnan(m):
            ax_q.text(xi, m + s + 5, f"{m:.0f}",
                      ha="center", va="bottom", fontsize=9)
    # Annotate the empty E4 bar so it isn't read as missing data.
    ax_q.text(
        3, ymax_q * 0.5, "E4 attempt\nfailed parsing",
        ha="center", va="center", fontsize=8, color="#d62728",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fde0e0",
                  ec="#d62728", lw=1.0),
    )

    # Panel 3 — Summary table
    rows = [
        ["Latency / call", "~17 s", "~95–300 s"],
        ["Tokens (avg)", "~1.5 k", "n/a (raw)"],
        ["Cost / call", "~$0.001", "$0 (offline)"],
        ["Total calls", f"{n_p_calls}", f"{n_q_calls}"],
        ["Total LLM-time", f"~{n_p_calls * 17 / 60:.1f} min",
         f"~{q_total / 60:.1f} min"],
        ["Setup", "API key only", "+4.8 GB local model"],
    ]
    table = ax_tab.table(
        cellText=rows,
        colLabels=["Metric", "P (cloud)", "Q (edge)"],
        loc="center",
        cellLoc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    ax_tab.set_title("Cost / latency summary", fontsize=10)

    fig.suptitle(
        "Cloud-vs-edge trade-off across the four LLM-augmented experiments",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

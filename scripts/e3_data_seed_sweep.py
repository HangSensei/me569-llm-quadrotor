"""Experiment 3 multi-data-seed sweep.

This script fixes the saved B / P / Q stage costs and varies only the
initial-state seed used by the closed-loop evaluation harness.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from me569_project.mpc.baseline_cost import make_baseline_stage_cost
from me569_project.mpc.evaluation import evaluate_controller
from me569_project.mpc.mpc_math import load_stage_cost_callable
from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
from me569_project.quadrotor_dynamics import QuadrotorParams


REPO_ROOT = Path(__file__).resolve().parent.parent
COSTS_DIR = REPO_ROOT / "llm_artifacts" / "e3_costs"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INIT_SEEDS = [0, 7, 13, 42, 99]
NUM_INITIAL_STATES = 20
MAX_STEPS = 500


@dataclass(frozen=True)
class Row:
    condition: str
    init_seed: int
    success_rate: float
    mean_settling_time_s: float
    mean_control_energy: float


def _load_stage_cost(path: Path) -> Callable:
    return load_stage_cost_callable(path.read_text())


def _evaluate(
    condition: str,
    stage_cost: Callable,
    params: QuadrotorParams,
    init_seed: int,
) -> Row:
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=stage_cost,
        config=MPCConfig(),
    )
    result = evaluate_controller(
        mpc=mpc,
        params=params,
        num_initial_states=NUM_INITIAL_STATES,
        max_steps=MAX_STEPS,
        seed=init_seed,
    )

    settling = result.mean_settling_time_s
    if not np.isfinite(settling):
        settling = float("nan")

    return Row(
        condition=condition,
        init_seed=init_seed,
        success_rate=result.hover_success_rate,
        mean_settling_time_s=settling,
        mean_control_energy=result.mean_control_energy,
    )


def _summary(rows: list[Row]) -> None:
    print()
    print("=" * 92)
    print("E3 multi-data-seed summary")
    print("=" * 92)
    for condition in ("B", "P", "Q"):
        cond_rows = [row for row in rows if row.condition == condition]
        if not cond_rows:
            print(f"{condition}: no rows")
            continue

        success = np.array([r.success_rate for r in cond_rows], dtype=float)
        settle = np.array([r.mean_settling_time_s for r in cond_rows], dtype=float)
        energy = np.array([r.mean_control_energy for r in cond_rows], dtype=float)

        print(f"{condition}  (N={len(cond_rows)} seeds)")
        print(f"  success_rate      : {success.mean():.3f} ± {success.std(ddof=0):.3f}")
        print(f"  mean_settling_time: {settle.mean():.3f} ± {settle.std(ddof=0):.3f}")
        print(f"  mean_control_energy: {energy.mean():.4f} ± {energy.std(ddof=0):.4f}")
        print()
    print("=" * 92)


def _write_csv(rows: list[Row], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> int:
    params = QuadrotorParams()
    print(f"Loading stage costs from {COSTS_DIR} ...")
    b_cost = make_baseline_stage_cost(params)
    p_cost = _load_stage_cost(COSTS_DIR / "p_qwen_plus_run_01.py")
    q_cost = _load_stage_cost(COSTS_DIR / "q_qwen_local_run_01.py")

    cost_by_condition = {
        "B": b_cost,
        "P": p_cost,
        "Q": q_cost,
    }

    rows: list[Row] = []
    for init_seed in INIT_SEEDS:
        print()
        print(f"Evaluating init_seed={init_seed} ...")
        for condition in ("B", "P", "Q"):
            print(f"  [{condition}] building + evaluating ...")
            row = _evaluate(
                condition=condition,
                stage_cost=cost_by_condition[condition],
                params=params,
                init_seed=init_seed,
            )
            rows.append(row)
            print(
                f"    success={row.success_rate * 100:.1f}% "
                f"settle={row.mean_settling_time_s:.3f}s "
                f"energy={row.mean_control_energy:.4f}"
            )

    if len(rows) != len(INIT_SEEDS) * 3:
        raise RuntimeError(
            f"Expected {len(INIT_SEEDS) * 3} rows, collected {len(rows)}"
        )

    out_path = RESULTS_DIR / "e3_data_seed_sweep.csv"
    _write_csv(rows, out_path)
    _summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

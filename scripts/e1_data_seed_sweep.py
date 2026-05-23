"""Experiment 1 multi-data-seed sweep.

This script re-fits the saved E1 basis functions on multiple random
train/validation splits to check whether the B / P / Q ordering is
stable across data seeds.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn
from me569_project.sysid.polynomial_basis import make_polynomial_basis_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
BASES_DIR = REPO_ROOT / "llm_artifacts" / "e1_bases"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SEEDS = [0, 7, 13, 42, 99]
NUM_TRAIN = 200
NUM_VAL = 50
STEPS_PER_TRAJ = 400
THRESHOLD = 0.1


@dataclass(frozen=True)
class Row:
    condition: str
    data_seed: int
    n_basis: int
    active_terms: int
    one_step_mse: float
    rollout_mse_10: float
    rollout_mse_50: float
    fit_time_s: float


def _load_basis(path: Path) -> Callable:
    return load_callable(path.read_text(), function_name="basis")


def _fit_condition(
    condition: str,
    basis_fn: Callable,
    train_data,
    val_data,
    data_seed: int,
) -> Row:
    t0 = time.time()
    model = fit_with_basis_fn(
        train_data,
        basis_fn=basis_fn,
        threshold=THRESHOLD,
    )
    fit_time_s = time.time() - t0

    return Row(
        condition=condition,
        data_seed=data_seed,
        n_basis=model.n_basis,
        active_terms=model.active_terms,
        one_step_mse=model.one_step_mse(val_data),
        rollout_mse_10=model.rollout_mse(val_data, horizon=10),
        rollout_mse_50=model.rollout_mse(val_data, horizon=50),
        fit_time_s=fit_time_s,
    )


def _summary(rows: list[Row]) -> None:
    print()
    print("=" * 92)
    print("E1 multi-data-seed summary")
    print("=" * 92)
    for condition in ("B", "P", "Q"):
        cond_rows = [row for row in rows if row.condition == condition]
        if not cond_rows:
            print(f"{condition}: no rows")
            continue

        one_step = np.array([r.one_step_mse for r in cond_rows], dtype=float)
        roll10 = np.array([r.rollout_mse_10 for r in cond_rows], dtype=float)
        roll50 = np.array([r.rollout_mse_50 for r in cond_rows], dtype=float)
        fit_time = np.array([r.fit_time_s for r in cond_rows], dtype=float)
        active = np.array([r.active_terms for r in cond_rows], dtype=float)

        print(f"{condition}  (N={len(cond_rows)} seeds)")
        print(
            f"  one_step_mse   : {one_step.mean():.4e} ± {one_step.std(ddof=0):.2e}"
        )
        print(
            f"  rollout_mse_10 : {roll10.mean():.4e} ± {roll10.std(ddof=0):.2e}"
        )
        print(
            f"  rollout_mse_50 : {roll50.mean():.4e} ± {roll50.std(ddof=0):.2e}"
        )
        print(
            f"  active_terms   : {active.mean():.1f} ± {active.std(ddof=0):.1f}"
        )
        print(
            f"  fit_time_s     : {fit_time.mean():.1f} ± {fit_time.std(ddof=0):.1f}"
        )
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
    print(f"Loading bases from {BASES_DIR} ...")
    b_basis, _ = make_polynomial_basis_fn(degree=3, n_input=8)
    p_basis = _load_basis(BASES_DIR / "p_qwen_plus_run_01.py")
    q_basis = _load_basis(BASES_DIR / "q_qwen_local_run_01.py")

    basis_by_condition = {
        "B": b_basis,
        "P": p_basis,
        "Q": q_basis,
    }

    rows: list[Row] = []
    for data_seed in DATA_SEEDS:
        train_seed = data_seed
        val_seed = data_seed + 1000
        print()
        print(
            f"Generating data_seed={data_seed} "
            f"(train_seed={train_seed}, val_seed={val_seed}) ..."
        )
        train_data = generate_trajectories(
            num_trajectories=NUM_TRAIN,
            steps_per_trajectory=STEPS_PER_TRAJ,
            seed=train_seed,
        )
        val_data = generate_trajectories(
            num_trajectories=NUM_VAL,
            steps_per_trajectory=STEPS_PER_TRAJ,
            seed=val_seed,
        )

        for condition in ("B", "P", "Q"):
            print(f"  Fitting {condition} on seed {data_seed} ...")
            row = _fit_condition(
                condition=condition,
                basis_fn=basis_by_condition[condition],
                train_data=train_data,
                val_data=val_data,
                data_seed=data_seed,
            )
            rows.append(row)
            print(
                f"    n_basis={row.n_basis} active={row.active_terms} "
                f"one_step={row.one_step_mse:.4e} "
                f"roll10={row.rollout_mse_10:.4e} "
                f"roll50={row.rollout_mse_50:.4e} "
                f"fit={row.fit_time_s:.1f}s"
            )

    if len(rows) != len(DATA_SEEDS) * 3:
        raise RuntimeError(
            f"Expected {len(DATA_SEEDS) * 3} rows, collected {len(rows)}"
        )

    out_path = RESULTS_DIR / "e1_data_seed_sweep.csv"
    _write_csv(rows, out_path)
    _summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

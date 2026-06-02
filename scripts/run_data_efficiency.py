"""Data-efficiency sweep (grader 1b): validation MSE vs number of samples.

Reuses the saved E1 basis functions (B/P/Q) and E3 (Koopman/EDMDc)
observable functions (B/P) -- no new LLM calls. Subsamples the existing
200k-transition training pool to a log ladder, refits, and records
validation one-step MSE on a fixed held-out set.

Run end-to-end:
    uv run python scripts/run_data_efficiency.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from me569_project.data.trajectory_generator import (
    TrajectoryData,
    generate_trajectories,
)
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.edmdc import edmdc_one_step_mse, fit_edmdc
from me569_project.sysid.koopman_baselines import polynomial_observable
from me569_project.sysid.llm_sysid_model import (
    build_design_matrix,
    fit_from_design_matrix,
)
from me569_project.sysid.polynomial_basis import make_polynomial_basis_fn
from me569_project.sysid.rbf_basis import build_rbf_e1, build_rbf_e3

REPO_ROOT = Path(__file__).resolve().parent.parent
E1_BASES = REPO_ROOT / "llm_artifacts" / "e1_bases"
KOOPMAN_OBS = REPO_ROOT / "llm_artifacts" / "koopman_observables"
RESULTS_CSV = REPO_ROOT / "results" / "data_efficiency.csv"

LADDER = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
SUBSAMPLE_SEEDS = [0, 1, 2]
NUM_TRAIN_TRAJ = 500   # 500 * 400 = 200_000 transitions (matches published runs)
NUM_VAL_TRAJ = 100     # matches run_e1_full / run_koopman validation set
STEPS = 400
THRESHOLD = 0.1


def subsample_indices(n_total: int, n_sub: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n_sub`` distinct row indices in [0, n_total) without replacement."""
    if n_sub > n_total:
        raise ValueError(f"n_sub ({n_sub}) exceeds n_total ({n_total})")
    return rng.choice(n_total, size=n_sub, replace=False)


def _load_basis(path: Path) -> Callable:
    return load_callable(path.read_text(), function_name="basis")


def _load_observable(path: Path) -> Callable:
    return load_callable(path.read_text(), function_name="observables")


def sweep_condition_e1(
    condition: str,
    basis_fn: Callable,
    feature_names: list[str] | None,
    train_pool: TrajectoryData,
    val_data: TrajectoryData,
    ladder: list[int],
    seeds: list[int],
    threshold: float = THRESHOLD,
) -> list[dict]:
    """Build the full E1 design matrix once, then refit on row subsamples.

    The STLSQ threshold used by ``fit_from_design_matrix`` is configurable via
    ``threshold`` (default: ``THRESHOLD``).
    """
    Phi, Y, n_basis = build_design_matrix(train_pool, basis_fn)
    n_total = Phi.shape[0]
    rows: list[dict] = []
    for n in ladder:
        if n > n_total:
            continue
        for seed in seeds:
            idx = subsample_indices(n_total, n, np.random.default_rng(seed))
            model = fit_from_design_matrix(
                Phi[idx], Y[idx], basis_fn, n_basis,
                feature_names=feature_names, threshold=threshold,
            )
            mse = model.one_step_mse(val_data)
            rows.append({
                "experiment": "E1",
                "condition": condition,
                "n_samples": n,
                "seed": seed,
                "val_one_step_mse": mse,
            })
            print(f"  [E1/{condition}] N={n:>7d} seed={seed} mse={mse:.4e}")
    return rows


def sweep_condition_e3(
    condition: str,
    observable_fn: Callable,
    train_pool: TrajectoryData,
    val_data: TrajectoryData,
    ladder: list[int],
    seeds: list[int],
) -> list[dict]:
    """Refit EDMDc on row subsamples of the flattened transition pool."""
    X, U, X_next = train_pool.flatten()
    Xv, Uv, Xv_next = val_data.flatten()
    n_total = X.shape[0]
    rows: list[dict] = []
    for n in ladder:
        if n > n_total:
            continue
        for seed in seeds:
            idx = subsample_indices(n_total, n, np.random.default_rng(seed))
            res = fit_edmdc(
                X[idx], U[idx], X_next[idx], observable_fn,
                observable_name=f"{condition}-deff",  # label only; result is discarded after one_step_mse
            )
            mse = edmdc_one_step_mse(res, Xv, Uv, Xv_next)
            rows.append({
                "experiment": "E3",
                "condition": condition,
                "n_samples": n,
                "seed": seed,
                "val_one_step_mse": mse,
            })
            print(f"  [E3/{condition}] N={n:>7d} seed={seed} mse={mse:.4e}")
    return rows


def main() -> int:
    print(f"Generating train pool ({NUM_TRAIN_TRAJ} traj) + val ({NUM_VAL_TRAJ} traj) ...")
    train = generate_trajectories(NUM_TRAIN_TRAJ, steps_per_trajectory=STEPS, seed=0)
    val = generate_trajectories(NUM_VAL_TRAJ, steps_per_trajectory=STEPS, seed=42)
    print(f"  {train.num_transitions} train transitions, {val.num_transitions} val")

    rows: list[dict] = []

    # E1: B (poly deg 3), P, Q
    b_basis, b_names = make_polynomial_basis_fn(degree=3, n_input=8)
    p_basis = _load_basis(E1_BASES / "p_qwen_plus_run_01.py")
    q_basis = _load_basis(E1_BASES / "q_qwen_local_run_01.py")
    print("E1 sweep ...")
    rows += sweep_condition_e1("B", b_basis, b_names, train, val, LADDER, SUBSAMPLE_SEEDS)
    rows += sweep_condition_e1("P", p_basis, None, train, val, LADDER, SUBSAMPLE_SEEDS)
    rows += sweep_condition_e1("Q", q_basis, None, train, val, LADDER, SUBSAMPLE_SEEDS)
    r_basis = build_rbf_e1(REPO_ROOT, for_cascade=False)
    rows += sweep_condition_e1("R", r_basis, None, train, val, LADDER, SUBSAMPLE_SEEDS)

    # E3: B (poly deg 2), P  (Q failed E3 categorically -> excluded)
    p_obs = _load_observable(KOOPMAN_OBS / "p_qwen_plus_run_01.py")
    print("E3 sweep ...")
    rows += sweep_condition_e3("B", polynomial_observable, train, val, LADDER, SUBSAMPLE_SEEDS)
    rows += sweep_condition_e3("P", p_obs, train, val, LADDER, SUBSAMPLE_SEEDS)
    r_obs = build_rbf_e3(REPO_ROOT)
    rows += sweep_condition_e3("R", r_obs, train, val, LADDER, SUBSAMPLE_SEEDS)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "condition", "n_samples", "seed", "val_one_step_mse"]
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

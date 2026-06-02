# scripts/run_rbf_baseline.py
"""Open-loop MSE comparison table on the test split (seed 42) for grader (1a).

Recomputes ALL conditions on the same test set so every column is
apples-to-apples with R:
  E1 (SINDy):   B (poly deg3), P (Qwen-Plus), Q (Qwen3.5-4B), R (tuned RBF)
                -> one_step + rollout_mse_10 + rollout_mse_50
  E3 (Koopman): B (poly deg2), P (Qwen-Plus), R (tuned RBF)
                -> one_step + rollout_mse_50

Run:  uv run python scripts/run_rbf_baseline.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.edmdc import (
    edmdc_one_step_mse,
    edmdc_rollout_mse,
    fit_edmdc,
)
from me569_project.sysid.koopman_baselines import polynomial_observable
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn
from me569_project.sysid.polynomial_basis import make_polynomial_basis_fn
from me569_project.sysid.rbf_basis import build_rbf_e1, build_rbf_e3

REPO_ROOT = Path(__file__).resolve().parent.parent
E1_BASES = REPO_ROOT / "llm_artifacts" / "e1_bases"
KOOPMAN_OBS = REPO_ROOT / "llm_artifacts" / "koopman_observables"
RESULTS_CSV = REPO_ROOT / "results" / "rbf_baseline.csv"


def _flatten(data) -> tuple:
    X = data.states[:, :-1, :].reshape(-1, 6)
    U = data.actions.reshape(-1, 2)
    X_next = data.next_states.reshape(-1, 6)  # == states[:, 1:, :]; matches run_koopman
    return X, U, X_next


def _load(path, function_name: str):
    """Read + sandbox-load a saved artifact, with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    return load_callable(path.read_text(), function_name=function_name)


def _e1_row(condition, basis, names, train, test) -> dict:
    model = fit_with_basis_fn(train, basis, names, threshold=0.1)
    row = {
        "experiment": "E1", "condition": condition,
        "one_step_mse": model.one_step_mse(test),
        "rollout_mse_10": model.rollout_mse(test, horizon=10),
        "rollout_mse_50": model.rollout_mse(test, horizon=50),
    }
    print(f"  [E1/{condition}] one={row['one_step_mse']:.4e} "
          f"r10={row['rollout_mse_10']:.4e} r50={row['rollout_mse_50']:.4e}")
    return row


def _e3_row(condition, obs, train, test) -> dict:
    res = fit_edmdc(*_flatten(train), obs, observable_name=f"{condition}")
    Xv, Uv, Xv_next = _flatten(test)
    row = {
        "experiment": "E3", "condition": condition,
        "one_step_mse": edmdc_one_step_mse(res, Xv, Uv, Xv_next),
        "rollout_mse_10": "",
        "rollout_mse_50": edmdc_rollout_mse(
            res, test.states[:, 0, :], test.actions, test.states[:, 1:, :],
            horizon=50),
    }
    print(f"  [E3/{condition}] one={row['one_step_mse']:.4e} "
          f"r50={row['rollout_mse_50']:.4e}")
    return row


def main() -> int:
    print("Generating train(seed0,500) + test(seed42,100) ...")
    train = generate_trajectories(500, steps_per_trajectory=400, seed=0)
    test = generate_trajectories(100, steps_per_trajectory=400, seed=42)

    rows = []

    # E1
    b_basis, b_names = make_polynomial_basis_fn(degree=3, n_input=8)
    rows.append(_e1_row("B", b_basis, b_names, train, test))
    p_basis = _load(E1_BASES / "p_qwen_plus_run_01.py", "basis")
    rows.append(_e1_row("P", p_basis, None, train, test))
    q_basis = _load(E1_BASES / "q_qwen_local_run_01.py", "basis")
    rows.append(_e1_row("Q", q_basis, None, train, test))
    # NOTE: open-loop R uses the M=400 artifact (e1_rbf.npz). The cascade
    # (run_cascade.py) uses the separate cascade-capped M<=100 artifact
    # (e1_rbf_cascade.npz) for Ipopt tractability — so the R model here and
    # the R model in cascade_results.csv are NOT the same size. Keep this
    # distinction explicit in any writeup.
    rows.append(_e1_row("R", build_rbf_e1(REPO_ROOT), None, train, test))

    # E3
    rows.append(_e3_row("B", polynomial_observable, train, test))
    p_obs = _load(KOOPMAN_OBS / "p_qwen_plus_run_01.py", "observables")
    rows.append(_e3_row("P", p_obs, train, test))
    rows.append(_e3_row("R", build_rbf_e3(REPO_ROOT), train, test))

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "condition", "one_step_mse",
              "rollout_mse_10", "rollout_mse_50"]
    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

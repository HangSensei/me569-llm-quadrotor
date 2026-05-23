"""E1 diagnostic grid: verify the B-vs-P-vs-Q gap is real, not an artifact.

Runs a (basis × threshold × data-seed) grid of sparse regression fits
and prints a summary table. Key hypotheses tested:

    H1  STLSQ threshold=0.1 is too aggressive for the 165-feature
        polynomial baseline.
        → Scan threshold ∈ {0.001, 0.01, 0.1}.
    H2  Polynomial degree 3 is unfairly over-parameterized.
        → Compare degrees 1, 2, 3.
    H3  The LLM's physically-motivated basis is the thing that matters.
        → Include a hand-written 7-feature ``perfect`` basis; if it
          matches or beats the LLM, the LLM is at the noise floor.
    H4  The original single-seed result is just variance noise.
        → Rerun every condition with 3 independent data seeds.
    H5  Finite-difference x_dot error creates an irreducible MSE floor.
        → The perfect basis number quantifies this floor.

The script does NOT call any LLM. Both P and Q bases are loaded from
``llm_artifacts/e1_bases/`` which holds the raw code from the first
committed E1 run. This keeps the diagnostic cheap and deterministic.

Usage::

    python scripts/e1_diagnostics.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn
from me569_project.sysid.polynomial_basis import make_polynomial_basis_fn
from me569_project.sysid.reference_basis import (
    expected_perfect_basis_7_Xi,
    perfect_quadrotor_basis_10,
    perfect_quadrotor_basis_7,
)

# Saved LLM bases live at the repo root under llm_artifacts/e1_bases/.
# We load them through the same sandbox path that a live LLM response
# would flow through, so the diagnostic exercises the real production
# parser as a side effect.
REPO_ROOT = Path(__file__).resolve().parent.parent
BASES_DIR = REPO_ROOT / "llm_artifacts" / "e1_bases"


def _load_saved_basis(stem: str) -> Callable:
    path = BASES_DIR / f"{stem}.py"
    code = path.read_text()
    return load_callable(code, function_name="basis")


p_basis_run_01 = _load_saved_basis("p_qwen_plus_run_01")
q_basis_run_01 = _load_saved_basis("q_qwen_local_run_01")


@dataclass
class DiagnosticRow:
    basis_name: str
    threshold: float
    data_seed: int
    n_basis: int
    active_terms: int
    one_step_mse: float
    rollout_mse_10: float
    rollout_mse_50: float
    fit_time_s: float


# ---------------------------------------------------------------
# Basis factory: returns (name, callable) for each condition to test
# ---------------------------------------------------------------


def build_basis_catalog() -> list[tuple[str, Callable]]:
    poly_d1, _ = make_polynomial_basis_fn(degree=1, n_input=8)
    poly_d2, _ = make_polynomial_basis_fn(degree=2, n_input=8)
    poly_d3, _ = make_polynomial_basis_fn(degree=3, n_input=8)
    return [
        ("perfect_7", perfect_quadrotor_basis_7),
        ("perfect_10", perfect_quadrotor_basis_10),
        ("poly_d1", poly_d1),
        ("poly_d2", poly_d2),
        ("poly_d3", poly_d3),
        ("p_qwen_plus_01", p_basis_run_01),
        ("q_qwen_local_01", q_basis_run_01),
    ]


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------


def run_one(
    basis_name: str,
    basis_fn: Callable,
    threshold: float,
    data_seed: int,
    train_data,
    val_data,
) -> DiagnosticRow:
    t0 = time.time()
    model = fit_with_basis_fn(
        train_data, basis_fn=basis_fn, threshold=threshold
    )
    fit_time = time.time() - t0

    m1 = model.one_step_mse(val_data)
    m10 = model.rollout_mse(val_data, horizon=10)
    m50 = model.rollout_mse(val_data, horizon=50)

    return DiagnosticRow(
        basis_name=basis_name,
        threshold=threshold,
        data_seed=data_seed,
        n_basis=model.n_basis,
        active_terms=model.active_terms,
        one_step_mse=m1,
        rollout_mse_10=m10,
        rollout_mse_50=m50,
        fit_time_s=fit_time,
    )


def run_grid(
    data_seeds: list[int],
    thresholds: list[float],
    num_train: int = 200,
    num_val: int = 50,
    steps_per_traj: int = 400,
) -> list[DiagnosticRow]:
    """Run the full diagnostic grid and return all result rows."""
    rows: list[DiagnosticRow] = []
    catalog = build_basis_catalog()

    for data_seed in data_seeds:
        print(f"\n=== Data seed {data_seed} ===")
        print(
            f"Generating {num_train} train + {num_val} val trajectories "
            f"(seed train={data_seed}, seed val={data_seed + 1000})..."
        )
        train = generate_trajectories(
            num_trajectories=num_train,
            steps_per_trajectory=steps_per_traj,
            seed=data_seed,
        )
        val = generate_trajectories(
            num_trajectories=num_val,
            steps_per_trajectory=steps_per_traj,
            seed=data_seed + 1000,
        )

        for basis_name, basis_fn in catalog:
            for threshold in thresholds:
                try:
                    row = run_one(
                        basis_name=basis_name,
                        basis_fn=basis_fn,
                        threshold=threshold,
                        data_seed=data_seed,
                        train_data=train,
                        val_data=val,
                    )
                    rows.append(row)
                    print(
                        f"  [{basis_name:<15s}] thr={threshold:<6.3f} "
                        f"n_b={row.n_basis:>4d} act={row.active_terms:>4d} "
                        f"1s={row.one_step_mse:>10.3e} "
                        f"r10={row.rollout_mse_10:>10.3e} "
                        f"r50={row.rollout_mse_50:>10.3e} "
                        f"({row.fit_time_s:.1f}s)"
                    )
                except Exception as e:
                    print(
                        f"  [{basis_name:<15s}] thr={threshold:<6.3f} "
                        f"FAILED: {type(e).__name__}: {e}"
                    )
    return rows


# ---------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------


def summarize_by_basis_threshold(rows: list[DiagnosticRow]) -> None:
    """Print a mean / std table grouped by (basis_name, threshold)."""
    print("\n" + "=" * 96)
    print("Summary (mean ± std across data seeds)")
    print("=" * 96)
    print(
        f"{'basis':<18s} {'thr':>7s} {'n_b':>6s} {'act':>6s} "
        f"{'one_step':>20s} {'roll_50':>20s}"
    )
    print("-" * 96)

    # Group
    groups: dict[tuple[str, float], list[DiagnosticRow]] = {}
    for row in rows:
        key = (row.basis_name, row.threshold)
        groups.setdefault(key, []).append(row)

    # Print in a stable order: catalog order, then threshold ascending
    catalog_order = [name for name, _ in build_basis_catalog()]
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (catalog_order.index(k[0]), k[1]),
    )

    for key in sorted_keys:
        name, thr = key
        grp = groups[key]
        one_step_vals = np.array([r.one_step_mse for r in grp])
        roll50_vals = np.array([r.rollout_mse_50 for r in grp])
        n_b = grp[0].n_basis  # constant across seeds
        act_mean = np.mean([r.active_terms for r in grp])

        one_step_str = f"{one_step_vals.mean():.3e}±{one_step_vals.std():.2e}"
        roll50_str = f"{roll50_vals.mean():.3e}±{roll50_vals.std():.2e}"

        print(
            f"{name:<18s} {thr:>7.3f} {n_b:>6d} {act_mean:>6.1f} "
            f"{one_step_str:>20s} {roll50_str:>20s}"
        )
    print("=" * 96)


def print_verdicts(rows: list[DiagnosticRow]) -> None:
    """Walk through the hypotheses H1–H5 and print a verdict for each."""
    # Helper: mean metric for a (basis, threshold) group
    def group_mean(name: str, thr: float, metric: str) -> float:
        vals = [
            getattr(r, metric)
            for r in rows
            if r.basis_name == name and abs(r.threshold - thr) < 1e-9
        ]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nHYPOTHESIS VERDICTS")
    print("-" * 96)

    # H1: STLSQ threshold=0.1 too aggressive for poly_d3
    poly_d3_thr001 = group_mean("poly_d3", 0.001, "one_step_mse")
    poly_d3_thr01 = group_mean("poly_d3", 0.01, "one_step_mse")
    poly_d3_thr1 = group_mean("poly_d3", 0.1, "one_step_mse")
    print(
        f"H1 (threshold hurts poly_d3): "
        f"one_step_mse @ thr=0.001 = {poly_d3_thr001:.3e}, "
        f"@0.01 = {poly_d3_thr01:.3e}, @0.1 = {poly_d3_thr1:.3e}"
    )

    # H2: polynomial degree does / does not matter
    p_d1 = group_mean("poly_d1", 0.1, "one_step_mse")
    p_d2 = group_mean("poly_d2", 0.1, "one_step_mse")
    p_d3 = group_mean("poly_d3", 0.1, "one_step_mse")
    print(
        f"H2 (poly degree scan @ thr=0.1): d1={p_d1:.3e} d2={p_d2:.3e} d3={p_d3:.3e}"
    )

    # H3: perfect basis vs LLM
    perfect7 = group_mean("perfect_7", 0.1, "one_step_mse")
    p_llm = group_mean("p_qwen_plus_01", 0.1, "one_step_mse")
    q_llm = group_mean("q_qwen_local_01", 0.1, "one_step_mse")
    print(
        f"H3 (perfect vs LLM @ thr=0.1): "
        f"perfect_7={perfect7:.3e}  P={p_llm:.3e}  Q={q_llm:.3e}"
    )

    # H5: FD floor
    print(
        f"H5 (FD / perfect floor) one_step_mse = {perfect7:.3e} "
        f"— this is the lower bound the LLM could reach with a correct basis."
    )
    print("-" * 96)


def main() -> int:
    # Smaller grid than the production E1 run so the diagnostic stays snappy:
    # 200 train + 50 val trajectories per seed, 7 basis × 3 thresholds × 3 seeds
    # = 63 fits total.
    data_seeds = [0, 7, 13]
    thresholds = [0.001, 0.01, 0.1]

    rows = run_grid(data_seeds=data_seeds, thresholds=thresholds)
    summarize_by_basis_threshold(rows)
    print_verdicts(rows)

    # Save raw rows to CSV for later analysis
    out_path = Path("results/e1_diagnostics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "basis_name",
                "threshold",
                "data_seed",
                "n_basis",
                "active_terms",
                "one_step_mse",
                "rollout_mse_10",
                "rollout_mse_50",
                "fit_time_s",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)
    print(f"\nWrote {len(rows)} diagnostic rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

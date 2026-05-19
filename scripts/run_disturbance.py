"""Disturbance robustness sweep — B/P/Q × {sigma=0, 0.05, 0.10, 0.20}.

Replays the E2 hover-task evaluation (ground-truth dynamics + each
condition's stage cost) but injects additive Gaussian process noise into
the state update at every RK4 step:

    x_{k+1} = rk4_step(x_k, u_k, dt) + sigma * sqrt(dt) * randn(6)

The noise scaling by ``sqrt(dt)`` makes the disturbance approximately
discretization-invariant (Euler-Maruyama). At sigma=0 the result should
match the E2 single-seed numbers exactly; at sigma=0.20 we expect crashes.

Conditions:
- B: textbook Bryson stage cost from `me569_project.mpc.baseline_cost`.
- P: Qwen 3.6-Plus stage cost from `llm_artifacts/e3_costs_clean/p_qwen_plus_run_01.py`.
- Q: Qwen3.5-4B stage cost from `llm_artifacts/e3_costs_clean/q_qwen_local_run_01.py`.

Output:
- ``results/disturbance_results.csv`` — one row per (condition, sigma).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from me569_project.mpc.baseline_cost import make_baseline_stage_cost
from me569_project.mpc.evaluation import _sample_initial_states
from me569_project.mpc.mpc_math import load_stage_cost_callable
from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "disturbance_results.csv"
E2_ARTIFACTS = REPO_ROOT / "llm_artifacts" / "e3_costs_clean"

SIGMAS = [0.0, 0.05, 0.10, 0.20]
NUM_INIT_STATES = 20
INIT_RADIUS = 0.30
MAX_STEPS = 500
DT = 0.02
POS_TOL = 0.1
ATT_TOL = 0.1
SETTLE_WINDOW = 50
WORKSPACE_BOUND = 5.0


def disturbed_rollout(
    initial_state: np.ndarray,
    mpc: PlanarQuadrotorMPC,
    params: QuadrotorParams,
    sigma: float,
    rng: np.random.Generator,
) -> dict:
    """One closed-loop rollout with additive process noise."""
    state = np.asarray(initial_state, dtype=np.float64).copy()
    mpc.reset()

    settled_step = -1
    in_box_streak = 0
    crashed = False
    control_energy = 0.0
    u_hover_vec = np.array([params.u_hover, params.u_hover])
    sqrt_dt = float(np.sqrt(DT))

    for step in range(MAX_STEPS):
        u = mpc.make_step(state)
        control_energy += float(np.sum((u - u_hover_vec) ** 2)) * DT
        state = rk4_step(state, u, DT, params)
        if sigma > 0.0:
            state = state + sigma * sqrt_dt * rng.standard_normal(6)

        if (
            abs(state[0]) > WORKSPACE_BOUND
            or abs(state[1]) > WORKSPACE_BOUND
            or not np.all(np.isfinite(state))
        ):
            crashed = True
            break

        in_box = (
            np.linalg.norm(state[:2]) < POS_TOL and abs(state[2]) < ATT_TOL
        )
        in_box_streak = in_box_streak + 1 if in_box else 0
        if settled_step < 0 and in_box_streak >= SETTLE_WINDOW:
            settled_step = step - SETTLE_WINDOW + 1

    settled = settled_step >= 0 and not crashed
    return {
        "settled": settled,
        "settling_time_s": settled_step * DT if settled else float("nan"),
        "control_energy": control_energy,
        "final_position_error": float(np.linalg.norm(state[:2])),
        "final_attitude_error": float(abs(state[2])),
        "crashed": crashed,
        "timeout": (not crashed and not settled),
    }


def evaluate_with_noise(
    mpc: PlanarQuadrotorMPC,
    params: QuadrotorParams,
    sigma: float,
    seed: int = 0,
) -> dict:
    """Aggregate metrics over NUM_INIT_STATES rollouts with shared noise seed."""
    initial_states = _sample_initial_states(
        NUM_INIT_STATES, INIT_RADIUS, seed=seed
    )
    rng = np.random.default_rng(seed * 1000 + int(sigma * 1000))

    per_rollout = [
        disturbed_rollout(x0, mpc, params, sigma, rng)
        for x0 in initial_states
    ]

    n_settled = sum(1 for r in per_rollout if r["settled"])
    n_crashed = sum(1 for r in per_rollout if r["crashed"])
    n_timeout = sum(1 for r in per_rollout if r["timeout"])
    settle_times = [
        r["settling_time_s"] for r in per_rollout if r["settled"]
    ]
    return {
        "num_rollouts": NUM_INIT_STATES,
        "hover_success_rate": n_settled / NUM_INIT_STATES,
        "num_settled": n_settled,
        "mean_settling_time_s": (
            float(np.mean(settle_times)) if settle_times else float("nan")
        ),
        "mean_control_energy": float(
            np.mean([r["control_energy"] for r in per_rollout])
        ),
        "mean_final_position_error": float(
            np.mean([r["final_position_error"] for r in per_rollout])
        ),
        "mean_final_attitude_error": float(
            np.mean([r["final_attitude_error"] for r in per_rollout])
        ),
        "failed_due_to_crash": n_crashed,
        "failed_due_to_timeout": n_timeout,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"\nWrote {len(rows)} row(s) to {path}")


def load_p_cost():
    p_path = E2_ARTIFACTS / "p_qwen_plus_run_01.py"
    if not p_path.exists():
        return None
    return load_stage_cost_callable(p_path.read_text())


def load_q_cost():
    q_path = E2_ARTIFACTS / "q_qwen_local_run_01.py"
    if not q_path.exists():
        return None
    return load_stage_cost_callable(q_path.read_text())


def main() -> int:
    params = QuadrotorParams()
    config = MPCConfig()

    conditions: dict[str, callable] = {}
    conditions["B"] = make_baseline_stage_cost(params)
    p_cost = load_p_cost()
    if p_cost is not None:
        conditions["P"] = p_cost
    else:
        print("[P] cost file missing; skipping.")
    q_cost = load_q_cost()
    if q_cost is not None:
        conditions["Q"] = q_cost
    else:
        print("[Q] cost file missing; skipping.")

    print("=" * 76)
    print(
        f"Disturbance sweep: {len(conditions)} conditions × "
        f"{len(SIGMAS)} noise levels × {NUM_INIT_STATES} init states"
    )
    print("=" * 76)

    rows: list[dict] = []
    for cond, cost_fn in conditions.items():
        print(f"\n[{cond}] Building MPC ...")
        t0 = time.time()
        mpc = PlanarQuadrotorMPC(params=params, stage_cost=cost_fn, config=config)
        print(f"[{cond}]   built in {time.time() - t0:.2f}s")

        for sigma in SIGMAS:
            print(f"[{cond}] sigma={sigma:.2f} ...")
            t1 = time.time()
            metrics = evaluate_with_noise(mpc, params, sigma=sigma)
            elapsed = time.time() - t1
            print(
                f"[{cond}|sigma={sigma:.2f}]   eval in {elapsed:.1f}s: "
                f"success={metrics['hover_success_rate']:.2f}, "
                f"crashes={metrics['failed_due_to_crash']}, "
                f"settle_mean={metrics['mean_settling_time_s']:.2f}"
            )
            rows.append({
                "condition": cond,
                "sigma": sigma,
                "eval_time_s": elapsed,
                **metrics,
            })

    print()
    print("=" * 76)
    print("Disturbance Robustness Summary (success rate)")
    print("=" * 76)
    print(f"{'cond':4s}", end="")
    for sigma in SIGMAS:
        print(f"  σ={sigma:.2f}", end="")
    print()
    by_cond: dict[str, dict] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], {})[r["sigma"]] = r["hover_success_rate"]
    for cond in conditions:
        print(f"{cond:4s}", end="")
        for sigma in SIGMAS:
            v = by_cond.get(cond, {}).get(sigma, float("nan"))
            print(f"  {v:6.2f}", end="")
        print()
    print("=" * 76)

    write_csv(rows, RESULTS_CSV)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

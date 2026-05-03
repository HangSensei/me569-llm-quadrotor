"""Run a closed-loop trajectory-tracking demo with the baseline MPC controller.

The Project Proposal §5C committed to "2-3 trajectory- and convergence-comparison
figures" for the Final Report. The hover-task convergence figure is already
covered by E1 (sysid noise floor) and E3 (Koopman observable comparison).
This script produces the remaining trajectory-tracking figure: the planar
quadrotor under MPC tracking a non-trivial time-varying reference (figure-8
in the position plane) starting from disturbed initial conditions.

The controller is the existing baseline tracking MPC (Bryson's-rule weights
on the error state, hover-relative thrust penalty). No LLM is involved in
this experiment — the goal is to demonstrate that the closed-loop framework
extends from the hover task (E2) to genuine reference tracking, and to
produce caption-ready figures for the Final Report.

Outputs:
- ``results/tracking_step.csv``       : 1 rollout, step reference, per-step state + action.
- ``results/tracking_figure8.csv``    : 3 rollouts (3 disturbed initial conditions), figure-8 reference.
- ``results/tracking_trajectories.png``: top-down (p_x, p_z) view of the figure-8 rollouts overlaid on the reference.
- ``results/tracking_errors.png``     : tracking-error magnitude vs time, both references.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from me569_project.mpc.reference_trajectories import (
    figure_eight_reference,
    step_reference,
)
from me569_project.mpc.tracking_cost import make_tracking_stage_cost
from me569_project.mpc.tracking_mpc import (
    PlanarQuadrotorTrackingMPC,
    TrackingMPCConfig,
)
from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def simulate_tracking(
    reference_fn,
    initial_state: np.ndarray,
    horizon: int = 20,
    dt: float = 0.02,
    duration_s: float = 8.0,
    workspace_bound: float = 5.0,
) -> dict:
    """Run one closed-loop tracking rollout and return per-step time series.

    Returns a dict with keys ``time``, ``state``, ``action``, ``reference``,
    each a numpy array. ``state`` has shape (T+1, 6); ``action`` and
    ``reference`` have shape (T, 6) and (T+1, 6) respectively. Crashes are
    not raised — the rollout just terminates early and returns whatever
    was collected.
    """
    params = QuadrotorParams()
    stage_cost = make_tracking_stage_cost(reference_fn=reference_fn, params=params)
    mpc = PlanarQuadrotorTrackingMPC(
        params=params,
        reference_fn=reference_fn,
        stage_cost=stage_cost,
        config=TrackingMPCConfig(horizon=horizon, dt=dt),
    )

    n_steps = int(round(duration_s / dt))
    state = np.asarray(initial_state, dtype=np.float64).copy()

    times = np.zeros(n_steps + 1, dtype=np.float64)
    states = np.zeros((n_steps + 1, 6), dtype=np.float64)
    actions = np.zeros((n_steps, 2), dtype=np.float64)
    refs = np.zeros((n_steps + 1, 6), dtype=np.float64)

    states[0] = state
    refs[0] = reference_fn(0.0)
    crashed = False

    for k in range(n_steps):
        u = mpc.make_step(state)
        actions[k] = u
        state = rk4_step(state, u, dt, params)
        states[k + 1] = state
        times[k + 1] = (k + 1) * dt
        refs[k + 1] = reference_fn(times[k + 1])
        if abs(state[0]) > workspace_bound or abs(state[1]) > workspace_bound:
            crashed = True
            states = states[: k + 2]
            actions = actions[: k + 1]
            refs = refs[: k + 2]
            times = times[: k + 2]
            break

    return {
        "time": times,
        "state": states,
        "action": actions,
        "reference": refs,
        "crashed": crashed,
        "n_steps": len(times) - 1,
    }


def write_rollout_csv(path: Path, rollouts: list[tuple[int, dict]]) -> None:
    """Write per-step rows from a list of (rollout_index, rollout_dict).

    Columns: rollout, t, p_x, p_z, theta, v_x, v_z, omega, u_1, u_2,
    ref_p_x, ref_p_z, pos_error, att_error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rollout", "t",
        "p_x", "p_z", "theta", "v_x", "v_z", "omega",
        "u_1", "u_2",
        "ref_p_x", "ref_p_z",
        "pos_error", "att_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, roll in rollouts:
            n = roll["n_steps"]
            for k in range(n):
                row = {
                    "rollout": idx,
                    "t": float(roll["time"][k]),
                    "p_x": float(roll["state"][k, 0]),
                    "p_z": float(roll["state"][k, 1]),
                    "theta": float(roll["state"][k, 2]),
                    "v_x": float(roll["state"][k, 3]),
                    "v_z": float(roll["state"][k, 4]),
                    "omega": float(roll["state"][k, 5]),
                    "u_1": float(roll["action"][k, 0]),
                    "u_2": float(roll["action"][k, 1]),
                    "ref_p_x": float(roll["reference"][k, 0]),
                    "ref_p_z": float(roll["reference"][k, 1]),
                    "pos_error": float(
                        np.linalg.norm(roll["state"][k, :2] - roll["reference"][k, :2])
                    ),
                    "att_error": float(
                        abs(roll["state"][k, 2] - roll["reference"][k, 2])
                    ),
                }
                writer.writerow(row)
    print(f"  wrote {path} ({sum(r['n_steps'] for _, r in rollouts)} rows)")


def main() -> None:
    print("=" * 76)
    print("Trajectory tracking demo (baseline MPC, no LLM)")
    print("=" * 76)

    # 1) Step reference: simple proof-of-life rollout.
    print("\n[step] Tracking a step from origin to (0.5, 0.3):")
    step_fn = lambda t: step_reference(t, step_time=0.5, target_x=0.5, target_z=0.3)
    t0 = time.time()
    step_roll = simulate_tracking(
        reference_fn=step_fn,
        initial_state=np.zeros(6),
        duration_s=4.0,
    )
    print(f"[step]   done in {time.time() - t0:.1f}s "
          f"(steps={step_roll['n_steps']}, crashed={step_roll['crashed']})")
    write_rollout_csv(RESULTS_DIR / "tracking_step.csv", [(0, step_roll)])

    # 2) Figure-8 reference: 3 rollouts from disturbed initial conditions.
    print("\n[figure8] Tracking a figure-8 from 3 disturbed initial conditions:")
    fig8_fn = lambda t: figure_eight_reference(t, period=8.0, amplitude=1.0)

    init_conditions = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),    # nominal start
        np.array([-0.5, 0.3, 0.1, 0.0, 0.0, 0.0]),   # offset, slight tilt
        np.array([0.3, -0.4, -0.1, 0.0, 0.0, 0.0]),  # opposite offset, tilt
    ]
    fig8_rollouts = []
    for idx, x0 in enumerate(init_conditions):
        t0 = time.time()
        roll = simulate_tracking(
            reference_fn=fig8_fn,
            initial_state=x0,
            duration_s=8.0,    # one full figure-8 period
        )
        elapsed = time.time() - t0
        mean_err = float(np.mean(np.linalg.norm(
            roll["state"][:, :2] - roll["reference"][:, :2], axis=1
        )))
        print(f"[figure8] rollout {idx} from {x0[:3].tolist()} done in "
              f"{elapsed:.1f}s "
              f"(steps={roll['n_steps']}, crashed={roll['crashed']}, "
              f"mean pos error={mean_err:.4f})")
        fig8_rollouts.append((idx, roll))
    write_rollout_csv(
        RESULTS_DIR / "tracking_figure8.csv", fig8_rollouts
    )

    print("\nDone. CSV outputs in results/. Run scripts/tracking_figure.py "
          "to generate the figures.")


if __name__ == "__main__":
    sys.exit(main())

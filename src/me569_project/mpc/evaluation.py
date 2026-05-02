"""Closed-loop evaluation harness for Experiment 3 MPC controllers.

Given a built ``PlanarQuadrotorMPC`` controller, run it on a fixed set
of randomized initial states and compute three aggregate metrics:

- **hover_success_rate** (0.0 – 1.0): fraction of rollouts that
  reach the convergence criterion (``‖position‖ < pos_tol`` and
  ``|theta| < att_tol``) AND stay there for ``settle_window`` steps
  before the episode times out.
- **mean_settling_time_s**: mean wall-clock-equivalent time (in
  seconds) for the *converged* rollouts to first enter and stay in
  the convergence region. NaN if no rollout converged.
- **mean_control_energy**: mean ``∫‖u - u_hover‖² dt`` over the
  rollouts (averaged across all rollouts, including failed ones).

Initial state distribution matches ``PlanarQuadrotorEnv.reset``
defaults so the same seed produces the same start states across
B / P / Q conditions for a fair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from me569_project.mpc.mpc_wrapper import PlanarQuadrotorMPC
from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics from a closed-loop evaluation run."""

    num_rollouts: int
    hover_success_rate: float       # in [0, 1]
    num_successful: int
    mean_settling_time_s: float     # NaN if no rollouts converged
    median_settling_time_s: float   # NaN if no rollouts converged
    mean_control_energy: float
    mean_final_position_error: float
    mean_final_attitude_error: float
    failed_due_to_crash: int        # rollouts that left the workspace
    failed_due_to_timeout: int      # rollouts that ran out of steps


@dataclass(frozen=True)
class TrackingResult:
    """Aggregate metrics from a closed-loop tracking evaluation run."""

    num_rollouts: int
    tracking_success_rate: float
    num_successful: int
    mean_settling_time_s: float
    mean_tracking_error: float
    max_tracking_error: float
    mean_control_energy: float
    mean_final_position_error: float
    mean_final_attitude_error: float
    failed_due_to_crash: int
    failed_due_to_timeout: int


def _sample_initial_states(
    num_states: int,
    init_radius: float,
    seed: int,
) -> np.ndarray:
    """Sample ``num_states`` initial states from the env's default box."""
    rng = np.random.default_rng(seed)
    r = init_radius
    low = np.array([-r, -r, -0.2, -0.5, -0.5, -0.5])
    high = np.array([r, r, 0.2, 0.5, 0.5, 0.5])
    return rng.uniform(low=low, high=high, size=(num_states, 6))


def _evaluate_single_rollout(
    initial_state: np.ndarray,
    mpc: PlanarQuadrotorMPC,
    params: QuadrotorParams,
    max_steps: int,
    dt: float,
    pos_tol: float,
    att_tol: float,
    settle_window: int,
    workspace_bound: float,
) -> dict:
    """Run one closed-loop rollout and return per-rollout metrics."""
    state = np.asarray(initial_state, dtype=np.float64).copy()
    mpc.reset()

    settled_step = -1     # first step where state has been "in box" for window
    in_box_streak = 0
    crashed = False
    control_energy = 0.0
    u_hover_vec = np.array([params.u_hover, params.u_hover])

    for step in range(max_steps):
        u = mpc.make_step(state)
        control_energy += float(np.sum((u - u_hover_vec) ** 2)) * dt
        state = rk4_step(state, u, dt, params)

        if (
            abs(state[0]) > workspace_bound
            or abs(state[1]) > workspace_bound
        ):
            crashed = True
            break

        in_box = (
            np.linalg.norm(state[:2]) < pos_tol
            and abs(state[2]) < att_tol
        )
        if in_box:
            in_box_streak += 1
            if settled_step < 0 and in_box_streak >= settle_window:
                settled_step = step - settle_window + 1
        else:
            in_box_streak = 0

    final_pos_error = float(np.linalg.norm(state[:2]))
    final_att_error = float(abs(state[2]))

    return {
        "settled_step": settled_step,
        "in_box_streak_at_end": in_box_streak,
        "crashed": crashed,
        "final_pos_error": final_pos_error,
        "final_att_error": final_att_error,
        "control_energy": control_energy,
    }


def evaluate_controller(
    mpc: PlanarQuadrotorMPC,
    params: QuadrotorParams | None = None,
    num_initial_states: int = 20,
    max_steps: int = 500,
    dt: float = 0.02,
    init_radius: float = 0.3,
    pos_tol: float = 0.1,
    att_tol: float = 0.1,
    settle_window: int = 50,
    workspace_bound: float = 5.0,
    seed: int = 0,
) -> EvaluationResult:
    """Run a controller on ``num_initial_states`` random rollouts and
    return the aggregate ``EvaluationResult``.

    Default settings:
    - ``num_initial_states = 20`` randomized starts
    - ``max_steps = 500`` (10 s at dt=0.02)
    - ``pos_tol = 0.1`` m, ``att_tol = 0.1`` rad
    - ``settle_window = 50`` steps (~1 s of staying in box for "settled")
    - ``workspace_bound = 5`` m (rollout fails if |p_x| or |p_z| > 5)
    """
    if params is None:
        params = QuadrotorParams()

    initial_states = _sample_initial_states(
        num_initial_states, init_radius, seed
    )

    per_rollout: list[dict] = []
    for x0 in initial_states:
        per_rollout.append(
            _evaluate_single_rollout(
                initial_state=x0,
                mpc=mpc,
                params=params,
                max_steps=max_steps,
                dt=dt,
                pos_tol=pos_tol,
                att_tol=att_tol,
                settle_window=settle_window,
                workspace_bound=workspace_bound,
            )
        )

    successful = [r for r in per_rollout if r["settled_step"] >= 0 and not r["crashed"]]
    num_successful = len(successful)
    success_rate = num_successful / num_initial_states

    if successful:
        settling_steps = np.array([r["settled_step"] for r in successful])
        mean_settle_s = float(np.mean(settling_steps) * dt)
        median_settle_s = float(np.median(settling_steps) * dt)
    else:
        mean_settle_s = float("nan")
        median_settle_s = float("nan")

    mean_energy = float(np.mean([r["control_energy"] for r in per_rollout]))
    mean_final_pos = float(np.mean([r["final_pos_error"] for r in per_rollout]))
    mean_final_att = float(np.mean([r["final_att_error"] for r in per_rollout]))

    crashed = sum(1 for r in per_rollout if r["crashed"])
    timeout = sum(
        1 for r in per_rollout if not r["crashed"] and r["settled_step"] < 0
    )

    return EvaluationResult(
        num_rollouts=num_initial_states,
        hover_success_rate=success_rate,
        num_successful=num_successful,
        mean_settling_time_s=mean_settle_s,
        median_settling_time_s=median_settle_s,
        mean_control_energy=mean_energy,
        mean_final_position_error=mean_final_pos,
        mean_final_attitude_error=mean_final_att,
        failed_due_to_crash=crashed,
        failed_due_to_timeout=timeout,
    )


def _evaluate_single_tracking_rollout(
    initial_state: np.ndarray,
    mpc,
    reference_fn,
    params: QuadrotorParams,
    max_steps: int,
    dt: float,
    pos_tol: float,
    att_tol: float,
    settle_window: int,
    workspace_bound: float,
) -> dict:
    """Run one closed-loop rollout against a time-varying reference."""
    state = np.asarray(initial_state, dtype=np.float64).copy()
    mpc.reset()

    settled_step = -1
    in_box_streak = 0
    crashed = False
    control_energy = 0.0
    u_hover_vec = np.array([params.u_hover, params.u_hover])
    tracking_errors: list[float] = []

    for step in range(max_steps):
        u = mpc.make_step(state)
        control_energy += float(np.sum((u - u_hover_vec) ** 2)) * dt
        state = rk4_step(state, u, dt, params)

        if abs(state[0]) > workspace_bound or abs(state[1]) > workspace_bound:
            crashed = True
            break

        ref = np.asarray(reference_fn((step + 1) * dt), dtype=np.float64).reshape(6)
        pos_error = float(np.linalg.norm(state[:2] - ref[:2]))
        att_error = float(abs(state[2] - ref[2]))
        tracking_errors.append(float(np.linalg.norm(state - ref)))

        in_box = pos_error < pos_tol and att_error < att_tol
        if in_box:
            in_box_streak += 1
            if settled_step < 0 and in_box_streak >= settle_window:
                settled_step = step - settle_window + 1
        else:
            in_box_streak = 0

    final_ref = np.asarray(reference_fn(max_steps * dt), dtype=np.float64).reshape(6)
    final_pos_error = float(np.linalg.norm(state[:2] - final_ref[:2]))
    final_att_error = float(abs(state[2] - final_ref[2]))

    if tracking_errors:
        mean_tracking_error = float(np.mean(tracking_errors))
        max_tracking_error = float(np.max(tracking_errors))
    else:
        mean_tracking_error = float(np.linalg.norm(state - final_ref))
        max_tracking_error = mean_tracking_error

    return {
        "settled_step": settled_step,
        "crashed": crashed,
        "mean_tracking_error": mean_tracking_error,
        "max_tracking_error": max_tracking_error,
        "final_pos_error": final_pos_error,
        "final_att_error": final_att_error,
        "control_energy": control_energy,
    }


def evaluate_tracking(
    mpc,
    reference_fn,
    params: QuadrotorParams | None = None,
    num_initial_states: int = 5,
    max_steps: int = 500,
    dt: float = 0.02,
    init_radius: float = 0.3,
    pos_tol: float = 0.15,
    att_tol: float = 0.15,
    settle_window: int = 25,
    workspace_bound: float = 5.0,
    seed: int = 0,
) -> TrackingResult:
    """Evaluate a controller on a tracking task with time-varying references."""
    if params is None:
        params = QuadrotorParams()

    initial_states = _sample_initial_states(num_initial_states, init_radius, seed)
    per_rollout: list[dict] = []
    for x0 in initial_states:
        per_rollout.append(
            _evaluate_single_tracking_rollout(
                initial_state=x0,
                mpc=mpc,
                reference_fn=reference_fn,
                params=params,
                max_steps=max_steps,
                dt=dt,
                pos_tol=pos_tol,
                att_tol=att_tol,
                settle_window=settle_window,
                workspace_bound=workspace_bound,
            )
        )

    successful = [r for r in per_rollout if r["settled_step"] >= 0 and not r["crashed"]]
    num_successful = len(successful)
    success_rate = num_successful / num_initial_states
    if successful:
        settling_steps = np.array([r["settled_step"] for r in successful])
        mean_settle_s = float(np.mean(settling_steps) * dt)
    else:
        mean_settle_s = float("nan")

    crashed = sum(1 for r in per_rollout if r["crashed"])
    timeout = sum(
        1 for r in per_rollout if not r["crashed"] and r["settled_step"] < 0
    )

    return TrackingResult(
        num_rollouts=num_initial_states,
        tracking_success_rate=success_rate,
        num_successful=num_successful,
        mean_settling_time_s=mean_settle_s,
        mean_tracking_error=float(np.mean([r["mean_tracking_error"] for r in per_rollout])),
        max_tracking_error=float(np.max([r["max_tracking_error"] for r in per_rollout])),
        mean_control_energy=float(np.mean([r["control_energy"] for r in per_rollout])),
        mean_final_position_error=float(np.mean([r["final_pos_error"] for r in per_rollout])),
        mean_final_attitude_error=float(np.mean([r["final_att_error"] for r in per_rollout])),
        failed_due_to_crash=crashed,
        failed_due_to_timeout=timeout,
    )

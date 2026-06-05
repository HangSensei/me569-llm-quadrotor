"""SINDy wrapper and metrics for Experiment 1 (SysID).

This module hides the PySINDy 2.x API surface behind a compact function
``fit_sindy`` and a ``SINDyFitResult`` dataclass holding the fitted
model plus the metadata needed downstream for the B/P/Q comparison.

The baseline (B) configuration uses ``PolynomialLibrary(degree=3)`` per the project spec §3.1. Experiments P (Qwen-Plus) and Q (Qwen3.5-4B)
will call ``fit_sindy`` with a custom ``feature_library`` built from the
LLM-generated basis function list.

Metrics
-------
``one_step_mse(result, val_data)``
    Mean squared error of the fitted model's one-step state-derivative
    prediction against the finite-difference derivative computed from the
    validation trajectories.

``rollout_mse(result, val_data, horizon)``
    Mean squared error of a closed-loop rollout: starting from each
    validation trajectory's initial state, integrate the fitted SINDy
    model forward for ``horizon`` steps using the recorded control
    sequence, and compare against the ground-truth state at step
    ``horizon``.

Both metrics are aggregate scalars (averaged over all state components
and all trajectories) so they can be logged in a single results CSV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pysindy as ps

from me569_project.data.trajectory_generator import TrajectoryData


FEATURE_NAMES = ["p_x", "p_z", "theta", "v_x", "v_z", "omega"]
CONTROL_NAMES = ["u_1", "u_2"]


@dataclass
class SINDyFitResult:
    """Container for a fitted SINDy model and its metadata."""

    model: ps.SINDy
    feature_library: Any
    active_terms: int
    equations: list[str] = field(default_factory=list)

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return the predicted state derivative x_dot at the given (state, action).

        Shapes: state (6,), action (2,), return (6,).
        """
        state_2d = np.asarray(state, dtype=np.float64).reshape(1, 6)
        action_2d = np.asarray(action, dtype=np.float64).reshape(1, 2)
        x_dot = self.model.predict(state_2d, u=action_2d)
        return np.asarray(x_dot, dtype=np.float64).reshape(6)


def fit_sindy(
    data: TrajectoryData,
    feature_library: Any | None = None,
    threshold: float = 0.1,
) -> SINDyFitResult:
    """Fit a SINDy model to trajectory data.

    Parameters
    ----------
    data : TrajectoryData
        Output of ``generate_trajectories``.
    feature_library : PySINDy feature library, optional
        Basis function library. Defaults to ``PolynomialLibrary(degree=3)``
        which is the baseline.
    threshold : float
        Sparsity threshold for the STLSQ optimizer. Higher = sparser model.

    Returns
    -------
    SINDyFitResult
    """
    if feature_library is None:
        feature_library = ps.PolynomialLibrary(degree=3)

    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=ps.STLSQ(threshold=threshold),
        differentiation_method=ps.FiniteDifference(),
    )

    N = data.num_trajectories
    x_list = [data.states[n, :-1, :] for n in range(N)]  # (T, 6)
    u_list = [data.actions[n] for n in range(N)]         # (T, 2)

    # PySINDy 2.x: passing a list of arrays automatically treats them as
    # multiple trajectories. Both x and u must be lists of the same length.
    # `feature_names` must have length n_states + n_controls so that
    # equations() can render both state-variable and control-variable terms.
    model.fit(
        x_list,
        u=u_list,
        t=data.dt,
        feature_names=FEATURE_NAMES + CONTROL_NAMES,
    )

    coefs = model.coefficients()
    active_terms = int(np.count_nonzero(coefs))

    try:
        equations = list(model.equations())
    except Exception:
        equations = []

    return SINDyFitResult(
        model=model,
        feature_library=feature_library,
        active_terms=active_terms,
        equations=equations,
    )


def one_step_mse(result: SINDyFitResult, val_data: TrajectoryData) -> float:
    """Mean squared error of predicted x_dot against finite-difference ground truth.

    Aggregated across all validation trajectories, all time steps, and all
    state components. Returns a single nonnegative float.
    """
    total_err_sq = 0.0
    total_count = 0

    dt = val_data.dt
    N = val_data.num_trajectories

    for n in range(N):
        states = val_data.states[n]        # (T+1, 6)
        actions = val_data.actions[n]      # (T, 2)

        # Ground-truth derivative via finite differences on the trajectory
        x_dot_true = (states[1:] - states[:-1]) / dt  # (T, 6)

        # Predicted derivative from SINDy model
        x_dot_pred = result.model.predict(states[:-1], u=actions)
        x_dot_pred = np.asarray(x_dot_pred, dtype=np.float64)

        diff_sq = (x_dot_true - x_dot_pred) ** 2
        total_err_sq += float(diff_sq.sum())
        total_count += diff_sq.size

    return total_err_sq / total_count if total_count > 0 else 0.0


def rollout_mse(
    result: SINDyFitResult,
    val_data: TrajectoryData,
    horizon: int,
) -> float:
    """Mean squared error of an H-step rollout using the fitted SINDy model.

    Starting from each validation trajectory's initial state, integrate the
    SINDy-learned dynamics forward for ``horizon`` simulation steps using
    the recorded control sequence. Compare against the ground-truth state
    at step ``horizon``. Errors are averaged across trajectories and state
    components.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    dt = val_data.dt
    N = val_data.num_trajectories
    T = val_data.steps_per_trajectory

    if horizon > T:
        raise ValueError(
            f"horizon ({horizon}) exceeds trajectory length ({T})"
        )

    total_err_sq = 0.0
    total_count = 0

    for n in range(N):
        state = val_data.states[n, 0].astype(np.float64)
        for t in range(horizon):
            action = val_data.actions[n, t]
            x_dot = result.predict(state, action)
            state = state + dt * x_dot  # forward Euler; dt matches SINDy fit dt

        true_state = val_data.states[n, horizon]
        diff_sq = (state - true_state) ** 2
        total_err_sq += float(diff_sq.sum())
        total_count += diff_sq.size

    return total_err_sq / total_count if total_count > 0 else 0.0

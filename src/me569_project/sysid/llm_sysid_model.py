"""Basis-function-driven SysID model for Experiment 1 LLM conditions.

Why a second model class?
-------------------------
The baseline (B) uses pysindy.SINDy with PolynomialLibrary. In principle
we could plug a custom feature library into pysindy for the LLM
conditions (P and Q) too, but the pysindy ``BaseFeatureLibrary`` API is
heavy (AxesArray wrapping, n-dimensional sample handling, x_sequence
decorators) and brittle for a one-off LLM-generated callable.

Instead, we do the sparse regression directly:

1. Build the feature matrix Phi row-by-row by calling the LLM-generated
   ``basis_fn`` on each ``[state, control]`` transition.
2. Build the target matrix Y as finite-difference state derivatives.
3. Run ``pysindy.STLSQ`` on ``(Phi, Y)`` to get the coefficient matrix Xi.

This keeps the sparse-regression machinery identical to the baseline
(STLSQ with the same threshold), so the B / P / Q comparison is fair in
the only axis that matters: which basis library does better, not which
optimizer does better.

API surface
-----------
- ``fit_with_basis_fn(data, basis_fn, ...)``: fit and return a
  ``LLMBasisSysIDModel`` ready to evaluate.
- ``LLMBasisSysIDModel``: lightweight container with ``predict``,
  ``one_step_mse``, ``rollout_mse``, ``active_terms``, and the fitted
  coefficient matrix ``Xi``.

The model's ``predict(state, action) -> x_dot`` mirrors the signature of
``SINDyFitResult.predict`` in ``sindy_wrapper.py``, so both baseline and
LLM conditions can feed into the same E1 runner without special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from pysindy.optimizers import STLSQ

from me569_project.data.trajectory_generator import TrajectoryData


@dataclass
class LLMBasisSysIDModel:
    """Sparse linear-in-basis-features model: x_dot ~= Phi(x, u) @ Xi.

    Attributes
    ----------
    basis_fn : Callable[[np.ndarray], list | np.ndarray]
        Callable that takes a concatenated [state, control] vector of
        length ``n_input_features`` and returns a list or array of length
        ``n_basis``.
    Xi : np.ndarray, shape (n_basis, n_state)
        Fitted coefficient matrix. ``x_dot_hat = Phi @ Xi``.
    n_basis : int
        Number of output features from the basis function.
    n_state : int
        State dimension (6 for the Planar Quadrotor).
    n_control : int
        Control dimension (2 for the Planar Quadrotor).
    threshold : float
        STLSQ sparsity threshold used during fitting.
    feature_names : list[str]
        Human-readable names for the basis columns (optional, used only
        for reporting).
    """

    basis_fn: Callable
    Xi: np.ndarray
    n_basis: int
    n_state: int = 6
    n_control: int = 2
    threshold: float = 0.1
    feature_names: list[str] = field(default_factory=list)

    @property
    def active_terms(self) -> int:
        return int(np.count_nonzero(self.Xi))

    def _phi(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        xu = np.concatenate([state, action]).astype(np.float64)
        phi = np.asarray(self.basis_fn(xu), dtype=np.float64).reshape(-1)
        if phi.shape[0] != self.n_basis:
            raise ValueError(
                f"basis_fn returned {phi.shape[0]} features, "
                f"expected {self.n_basis}"
            )
        return phi

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return predicted state derivative x_dot of shape (6,)."""
        phi = self._phi(state, action)
        return phi @ self.Xi  # (n_basis,) @ (n_basis, n_state) -> (n_state,)

    def one_step_mse(self, val_data: TrajectoryData) -> float:
        """MSE between finite-difference x_dot and predicted x_dot over
        all validation transitions.
        """
        dt = val_data.dt
        N = val_data.num_trajectories
        T = val_data.steps_per_trajectory

        total_err_sq = 0.0
        total_count = 0

        for n in range(N):
            states = val_data.states[n]
            actions = val_data.actions[n]
            x_dot_true = (states[1:] - states[:-1]) / dt  # (T, 6)

            for t in range(T):
                x_dot_pred = self.predict(states[t], actions[t])
                diff_sq = (x_dot_true[t] - x_dot_pred) ** 2
                total_err_sq += float(diff_sq.sum())
                total_count += diff_sq.size

        return total_err_sq / total_count if total_count > 0 else 0.0

    def rollout_mse(self, val_data: TrajectoryData, horizon: int) -> float:
        """MSE between ground-truth state at step ``horizon`` and simulated
        state using the learned model with recorded control sequence.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if horizon > val_data.steps_per_trajectory:
            raise ValueError(
                f"horizon ({horizon}) exceeds trajectory length "
                f"({val_data.steps_per_trajectory})"
            )

        dt = val_data.dt
        N = val_data.num_trajectories

        total_err_sq = 0.0
        total_count = 0

        for n in range(N):
            state = val_data.states[n, 0].astype(np.float64)
            for t in range(horizon):
                action = val_data.actions[n, t]
                x_dot = self.predict(state, action)
                state = state + dt * x_dot  # forward Euler matching fit dt

            true_state = val_data.states[n, horizon]
            diff_sq = (state - true_state) ** 2
            total_err_sq += float(diff_sq.sum())
            total_count += diff_sq.size

        return total_err_sq / total_count if total_count > 0 else 0.0


def _build_phi_and_y(
    data: TrajectoryData,
    basis_fn: Callable,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build feature matrix Phi and finite-difference target Y from data."""
    N = data.num_trajectories
    T = data.steps_per_trajectory
    dt = data.dt

    # Probe basis size with a dummy call
    probe_xu = np.zeros(6 + 2, dtype=np.float64)
    probe_out = np.asarray(basis_fn(probe_xu), dtype=np.float64).reshape(-1)
    n_basis = probe_out.shape[0]

    Phi = np.zeros((N * T, n_basis), dtype=np.float64)
    Y = np.zeros((N * T, 6), dtype=np.float64)

    row = 0
    for n in range(N):
        states = data.states[n]
        actions = data.actions[n]
        x_dot_true = (states[1:] - states[:-1]) / dt  # (T, 6)

        for t in range(T):
            xu = np.concatenate([states[t], actions[t]])
            phi = np.asarray(basis_fn(xu), dtype=np.float64).reshape(-1)
            if phi.shape[0] != n_basis:
                raise ValueError(
                    f"basis_fn returned inconsistent length: expected "
                    f"{n_basis}, got {phi.shape[0]} at trajectory {n}, step {t}"
                )
            Phi[row] = phi
            Y[row] = x_dot_true[t]
            row += 1

    return Phi, Y, n_basis


def fit_with_basis_fn(
    data: TrajectoryData,
    basis_fn: Callable,
    feature_names: list[str] | None = None,
    threshold: float = 0.1,
) -> LLMBasisSysIDModel:
    """Fit x_dot ~= Phi(x, u) @ Xi using STLSQ sparse regression.

    Parameters
    ----------
    data : TrajectoryData
        Training trajectories from ``generate_trajectories``.
    basis_fn : Callable
        Function ``basis_fn(xu) -> list | np.ndarray`` taking a
        concatenated ``[state, control]`` vector of length 8 and
        returning a list of ``n_basis`` scalar features.
    feature_names : list[str], optional
        Human-readable names for the basis columns, used only for
        reporting. Auto-generated as ``["basis_0", ...]`` if omitted.
    threshold : float
        STLSQ sparsity threshold. Same optimizer and threshold as the
        baseline fit for a fair B/P/Q comparison.

    Returns
    -------
    LLMBasisSysIDModel
        Fitted model ready for prediction and metric computation.
    """
    Phi, Y, n_basis = _build_phi_and_y(data, basis_fn)

    optimizer = STLSQ(threshold=threshold)
    optimizer.fit(Phi, Y)
    # sklearn convention: coef_ shape is (n_targets, n_features) = (n_state, n_basis)
    Xi = np.asarray(optimizer.coef_, dtype=np.float64).T  # transpose to (n_basis, n_state)

    if feature_names is None:
        feature_names = [f"basis_{i}" for i in range(n_basis)]
    elif len(feature_names) != n_basis:
        raise ValueError(
            f"feature_names has length {len(feature_names)}, "
            f"basis_fn produced {n_basis} features"
        )

    return LLMBasisSysIDModel(
        basis_fn=basis_fn,
        Xi=Xi,
        n_basis=n_basis,
        threshold=threshold,
        feature_names=feature_names,
    )

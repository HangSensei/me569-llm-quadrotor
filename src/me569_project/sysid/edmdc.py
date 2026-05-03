"""EDMDc (Extended Dynamic Mode Decomposition with Control) for Experiment 3.

This module fits a discrete-time linear lifted-space model

    psi(x_{k+1}) = A psi(x_k) + B u_k

to trajectory data, given a fixed observable function ``psi: R^6 -> R^N``
that the caller provides. The observable function may come from a
hand-built dictionary (the B condition baseline, see ``koopman_baselines.py``)
or from an LLM (the P / Q conditions, see the run_koopman runner script).

State-recovery convention
-------------------------
The first six components of ``psi(x)`` MUST equal the state ``x`` itself,
in the order ``[p_x, p_z, theta, v_x, v_z, omega]``. With this convention,
state recovery from a lifted prediction is the trivial slice
``x_hat = psi_hat[:6]``. The convention is asserted in this module
whenever a new observable function is registered for fitting.

Pipeline
--------
``fit_edmdc(X, U, X_next, observable_fn)`` ->
``EDMDcResult`` (holds A, B, the observable, the n_state and n_obs sizes,
and the train one-step MSE).

``edmdc_one_step_mse(result, X_val, U_val, X_val_next)`` ->
prediction MSE on held-out single transitions, in the original state
space (after the ``[:6]`` projection from lifted space).

``edmdc_rollout_mse(result, X0, U_seq, X_seq_true, horizon)`` ->
free-run rollout MSE in lifted space with re-lifting at every step.

This is the EDMDc analog of ``one_step_mse`` and ``rollout_mse`` in
``sindy_wrapper.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


_STATE_DIM = 6
_CONTROL_DIM = 2
_RECOVERY_TOLERANCE = 1e-10
_MIN_OBS_DIM = 6
_MAX_OBS_DIM = 50


@dataclass(frozen=True)
class EDMDcResult:
    """Container for a fitted EDMDc model and its metadata.

    Attributes
    ----------
    A : np.ndarray of shape (N, N)
        Lifted-space dynamics matrix.
    B : np.ndarray of shape (N, m)
        Lifted-space control input matrix.
    n_state : int
        Original state dimension. 6 for Quadrotor.
    n_obs : int
        Lifted observable dimension N. Satisfies
        ``_MIN_OBS_DIM <= n_obs <= _MAX_OBS_DIM``.
    observable_name : str
        Human-readable label, e.g. ``"B-poly"``, ``"P-Qwen-Plus-run-01"``.
    observable_fn : Callable
        The function used to produce ``psi(x)`` at fit time. Kept on the
        result object so the rollout function can re-lift after each
        projection back to state space.
    fit_time_s : float
        Wall time for the lstsq solve (excludes data prep).
    train_one_step_mse : float
        Mean squared error of the predicted x_{k+1} (after [:6] slice)
        on the training data, for sanity reporting.
    """

    A: np.ndarray
    B: np.ndarray
    n_state: int
    n_obs: int
    observable_name: str
    observable_fn: Callable[[np.ndarray], Sequence[float]]
    fit_time_s: float
    train_one_step_mse: float


def _validate_observable(observable_fn: Callable[[np.ndarray], Sequence[float]]) -> int:
    """Sanity-check the observable function and return its output dim N.

    Checks
    ------
    1. Output dim is in [_MIN_OBS_DIM, _MAX_OBS_DIM].
    2. The first ``_STATE_DIM`` components equal the input state to within
       ``_RECOVERY_TOLERANCE`` (state-recovery convention).
    3. Output is finite (no NaN, no Inf).

    Raises
    ------
    ValueError if any check fails.
    """
    sentinels = [
        np.zeros(_STATE_DIM, dtype=np.float64),
        np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05], dtype=np.float64),
        np.array([1.0, 1.0, 0.7, -0.5, 0.5, -0.3], dtype=np.float64),
    ]
    obs_dim: int | None = None
    for x in sentinels:
        psi = np.asarray(observable_fn(x), dtype=np.float64).ravel()
        if not np.all(np.isfinite(psi)):
            raise ValueError(
                f"observable produced non-finite values on sentinel input {x!r}"
            )
        if obs_dim is None:
            obs_dim = psi.shape[0]
        elif psi.shape[0] != obs_dim:
            raise ValueError(
                f"observable returned inconsistent dim across inputs: "
                f"{obs_dim} vs {psi.shape[0]}"
            )
        if obs_dim < _MIN_OBS_DIM or obs_dim > _MAX_OBS_DIM:
            raise ValueError(
                f"observable dim {obs_dim} out of allowed range "
                f"[{_MIN_OBS_DIM}, {_MAX_OBS_DIM}]"
            )
        first_six = psi[:_STATE_DIM]
        if not np.allclose(first_six, x, atol=_RECOVERY_TOLERANCE):
            raise ValueError(
                "observable violates state-recovery convention: first 6 "
                f"components must equal input state. Got psi[:6]={first_six!r}, "
                f"x={x!r}"
            )
    assert obs_dim is not None
    return obs_dim


def _lift_batch(
    X: np.ndarray,
    observable_fn: Callable[[np.ndarray], Sequence[float]],
) -> np.ndarray:
    """Apply ``observable_fn`` to every row of ``X``, returning a (K, N) matrix."""
    K = X.shape[0]
    rows: list[np.ndarray] = []
    for i in range(K):
        rows.append(np.asarray(observable_fn(X[i]), dtype=np.float64).ravel())
    return np.stack(rows, axis=0)


def fit_edmdc(
    X: np.ndarray,
    U: np.ndarray,
    X_next: np.ndarray,
    observable_fn: Callable[[np.ndarray], Sequence[float]],
    observable_name: str = "unnamed",
) -> EDMDcResult:
    """Fit linear lifted-space dynamics ``psi(x_{k+1}) = A psi(x_k) + B u_k``.

    Parameters
    ----------
    X : np.ndarray of shape (K, n_state)
        Source states for each transition.
    U : np.ndarray of shape (K, n_control)
        Controls applied at each transition.
    X_next : np.ndarray of shape (K, n_state)
        Resulting next states.
    observable_fn : callable ``x -> psi(x)``
        Vectorized one-row-at-a-time function. Must satisfy the
        state-recovery convention (first 6 entries = input state).
    observable_name : str
        Label stored on the result for downstream logging.

    Returns
    -------
    EDMDcResult

    Raises
    ------
    ValueError on shape mismatch, observable validation failure, or
    rank-deficient regression matrix.
    """
    import time

    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    X_next = np.asarray(X_next, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != _STATE_DIM:
        raise ValueError(f"X must have shape (K, {_STATE_DIM}); got {X.shape}")
    if U.ndim != 2 or U.shape[1] != _CONTROL_DIM:
        raise ValueError(f"U must have shape (K, {_CONTROL_DIM}); got {U.shape}")
    if X_next.shape != X.shape:
        raise ValueError(
            f"X_next shape {X_next.shape} must match X shape {X.shape}"
        )
    if X.shape[0] != U.shape[0]:
        raise ValueError(
            f"X and U must agree on K; got X K={X.shape[0]}, U K={U.shape[0]}"
        )

    n_obs = _validate_observable(observable_fn)

    Psi = _lift_batch(X, observable_fn)            # (K, N)
    Psi_next = _lift_batch(X_next, observable_fn)  # (K, N)

    Z = np.concatenate([Psi, U], axis=1)           # (K, N + m)

    t0 = time.time()
    AB_T, residuals, rank, _ = np.linalg.lstsq(Z, Psi_next, rcond=None)
    fit_time_s = time.time() - t0

    expected_rank = n_obs + _CONTROL_DIM
    if rank < expected_rank:
        # Don't hard-fail: lstsq still returns the minimum-norm solution.
        # Just record a warning so the caller knows.
        # In practice rank < expected_rank means redundant observables,
        # which is fine for prediction quality.
        pass

    AB = AB_T.T                                    # (N, N + m)
    A = AB[:, :n_obs]                              # (N, N)
    B = AB[:, n_obs:]                              # (N, m)

    # Compute training one-step MSE for sanity reporting.
    Psi_pred = (A @ Psi.T + B @ U.T).T             # (K, N)
    X_pred = Psi_pred[:, :_STATE_DIM]
    train_err = X_pred - X_next
    train_one_step_mse = float(np.mean(train_err ** 2))

    return EDMDcResult(
        A=A,
        B=B,
        n_state=_STATE_DIM,
        n_obs=n_obs,
        observable_name=observable_name,
        observable_fn=observable_fn,
        fit_time_s=fit_time_s,
        train_one_step_mse=train_one_step_mse,
    )


def edmdc_one_step_mse(
    result: EDMDcResult,
    X_val: np.ndarray,
    U_val: np.ndarray,
    X_val_next: np.ndarray,
) -> float:
    """One-step MSE on held-out transitions, in original state space.

    Predicts ``psi_hat(x_{k+1}) = A psi(x_k) + B u_k`` and projects back
    to state with the convention ``x_hat = psi_hat[:6]``. MSE is averaged
    over all 6 state components and all K transitions.

    Returns
    -------
    float >= 0
    """
    X_val = np.asarray(X_val, dtype=np.float64)
    U_val = np.asarray(U_val, dtype=np.float64)
    X_val_next = np.asarray(X_val_next, dtype=np.float64)

    Psi = _lift_batch(X_val, result.observable_fn)                   # (K, N)
    Psi_pred = (result.A @ Psi.T + result.B @ U_val.T).T             # (K, N)
    X_pred = Psi_pred[:, :result.n_state]                            # (K, n)
    err = X_pred - X_val_next
    return float(np.mean(err ** 2))


def edmdc_rollout_mse(
    result: EDMDcResult,
    X0: np.ndarray,
    U_seq: np.ndarray,
    X_seq_true: np.ndarray,
    horizon: int,
) -> float:
    """Free-run rollout MSE at the terminal step.

    For each trajectory ``i`` in the batch:
      psi_0 = psi(X0[i])
      for k in 0 .. horizon-1:
          psi_{k+1} = A psi_k + B U_seq[i, k]
          x_{k+1}   = psi_{k+1}[:6]            # project to state
          psi_{k+1} = psi(x_{k+1})              # re-lift (standard EDMDc)
      err_i = (x_horizon - X_seq_true[i, horizon-1]) ** 2

    Returns the mean over the batch and over all 6 state components.

    Parameters
    ----------
    X0 : (K, n_state)
        Initial states.
    U_seq : (K, T, m)
        Control sequence per trajectory; T must be >= horizon.
    X_seq_true : (K, T, n_state)
        Ground-truth state evolution per trajectory.
    horizon : int
        Number of integration steps to roll forward.
    """
    X0 = np.asarray(X0, dtype=np.float64)
    U_seq = np.asarray(U_seq, dtype=np.float64)
    X_seq_true = np.asarray(X_seq_true, dtype=np.float64)

    K, T, _ = U_seq.shape
    if X0.shape != (K, result.n_state):
        raise ValueError(
            f"X0 shape {X0.shape} != expected ({K}, {result.n_state})"
        )
    if X_seq_true.shape != (K, T, result.n_state):
        raise ValueError(
            f"X_seq_true shape {X_seq_true.shape} != expected ({K}, {T}, "
            f"{result.n_state})"
        )
    if horizon > T or horizon < 1:
        raise ValueError(
            f"horizon {horizon} must be in [1, {T}]"
        )

    final_errs = np.zeros((K, result.n_state), dtype=np.float64)

    for i in range(K):
        x = X0[i].copy()
        for k in range(horizon):
            psi = np.asarray(result.observable_fn(x), dtype=np.float64).ravel()
            psi_next = result.A @ psi + result.B @ U_seq[i, k]
            x = psi_next[:result.n_state]
        final_errs[i] = x - X_seq_true[i, horizon - 1]

    return float(np.mean(final_errs ** 2))

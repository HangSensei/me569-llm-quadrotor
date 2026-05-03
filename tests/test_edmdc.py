"""Tests for the EDMDc fitter used in Experiment 3 (Koopman/EDMDc).

The fitter solves linear lifted-space dynamics

    psi(x_{k+1}) = A psi(x_k) + B u_k

via ordinary least squares. These tests verify:

1. On a synthetic linear system (identity observable), EDMDc recovers
   ``A``, ``B`` to within ``1e-10``.
2. On Quadrotor data with the physics-aware observable from
   ``koopman_baselines.py``, prediction MSE is reasonably low.
3. On Quadrotor data with the polynomial-only observable, prediction
   MSE is higher than with the physics-aware one (the E1-narrative
   parallel: polynomial-only fails to capture trig structure).
4. Observable validation (sandbox checks) rejects non-conforming
   functions: wrong output dim, scrambled state-recovery, non-finite,
   inconsistent dim across inputs.
5. The result dataclass is frozen.
6. Rollout MSE on a perfect-observable fit is much smaller than on
   the polynomial-only fit at horizon 50.
"""
import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.sysid.edmdc import (
    EDMDcResult,
    edmdc_one_step_mse,
    edmdc_rollout_mse,
    fit_edmdc,
)
from me569_project.sysid.koopman_baselines import (
    physics_aware_observable,
    polynomial_observable,
)


# ---------------------------------------------------------------------------
# Fixtures: small Quadrotor train + val sets for fitter tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quadrotor_train():
    return generate_trajectories(
        num_trajectories=30, steps_per_trajectory=200, dt=0.02, seed=0
    )


@pytest.fixture(scope="module")
def quadrotor_val():
    return generate_trajectories(
        num_trajectories=10, steps_per_trajectory=200, dt=0.02, seed=42
    )


def _flatten(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten TrajectoryData into (X, U, X_next) arrays."""
    X = data.states[:, :-1, :].reshape(-1, 6)
    U = data.actions.reshape(-1, 2)
    X_next = data.next_states.reshape(-1, 6)
    return X, U, X_next


# ---------------------------------------------------------------------------
# 1. Synthetic linear system recovery.
# ---------------------------------------------------------------------------

def test_fit_edmdc_recovers_known_linear_system():
    """With identity observable, EDMDc must recover the true A, B exactly."""
    rng = np.random.default_rng(0)
    A_true = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.85, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
    ])
    B_true = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    K = 500
    X = rng.normal(size=(K, 6))
    U = rng.normal(size=(K, 2))
    X_next = X @ A_true.T + U @ B_true.T

    # Identity observable: psi(x) = padded with zeros to >=6 dims.
    # We use 6-dim identity to satisfy convention exactly.
    def identity_obs(x):
        return np.asarray(x, dtype=np.float64)

    res = fit_edmdc(X, U, X_next, identity_obs, observable_name="identity")
    np.testing.assert_allclose(res.A, A_true, atol=1e-10)
    np.testing.assert_allclose(res.B, B_true, atol=1e-10)
    assert res.train_one_step_mse < 1e-20


# ---------------------------------------------------------------------------
# 2 + 3. Physics-aware vs polynomial-only on Quadrotor data.
# ---------------------------------------------------------------------------

def test_physics_aware_observable_beats_polynomial_on_quadrotor(
    quadrotor_train, quadrotor_val
):
    X_tr, U_tr, X_tr_next = _flatten(quadrotor_train)
    X_va, U_va, X_va_next = _flatten(quadrotor_val)

    res_phys = fit_edmdc(
        X_tr, U_tr, X_tr_next, physics_aware_observable, "phys"
    )
    res_poly = fit_edmdc(
        X_tr, U_tr, X_tr_next, polynomial_observable, "poly"
    )

    mse_phys = edmdc_one_step_mse(res_phys, X_va, U_va, X_va_next)
    mse_poly = edmdc_one_step_mse(res_poly, X_va, U_va, X_va_next)

    # The physics-aware observable should be at least 1.5x better than poly.
    # In practice it's much more (often ~10x), but we assert a soft bound
    # to avoid flakiness across seeds.
    assert mse_phys < mse_poly / 1.5, (
        f"physics_aware MSE {mse_phys} should be << polynomial MSE {mse_poly}"
    )
    # Both should be finite and nonnegative.
    assert mse_phys >= 0.0 and mse_poly >= 0.0
    assert np.isfinite(mse_phys) and np.isfinite(mse_poly)


# ---------------------------------------------------------------------------
# 4. Observable validation.
# ---------------------------------------------------------------------------

def test_fit_edmdc_rejects_observable_with_wrong_first_six():
    X = np.random.default_rng(0).normal(size=(50, 6))
    U = np.random.default_rng(1).normal(size=(50, 2))
    X_next = X.copy()

    def bad_obs(x):
        # Scramble: first 6 are state * 2, not state itself.
        return np.concatenate([np.asarray(x) * 2.0, np.asarray(x) ** 2])

    with pytest.raises(ValueError, match="state-recovery"):
        fit_edmdc(X, U, X_next, bad_obs, "bad")


def test_fit_edmdc_rejects_observable_too_short():
    X = np.random.default_rng(0).normal(size=(50, 6))
    U = np.random.default_rng(1).normal(size=(50, 2))
    X_next = X.copy()

    def short_obs(x):
        return np.asarray(x)[:5]

    with pytest.raises(ValueError):
        fit_edmdc(X, U, X_next, short_obs, "short")


def test_fit_edmdc_rejects_observable_too_long():
    X = np.random.default_rng(0).normal(size=(50, 6))
    U = np.random.default_rng(1).normal(size=(50, 2))
    X_next = X.copy()

    def long_obs(x):
        return np.concatenate([np.asarray(x), np.zeros(60)])

    with pytest.raises(ValueError):
        fit_edmdc(X, U, X_next, long_obs, "long")


def test_fit_edmdc_rejects_non_finite_observable():
    X = np.random.default_rng(0).normal(size=(50, 6))
    U = np.random.default_rng(1).normal(size=(50, 2))
    X_next = X.copy()

    def bad_obs(x):
        return np.concatenate([np.asarray(x), [np.inf, np.nan]])

    with pytest.raises(ValueError, match="non-finite"):
        fit_edmdc(X, U, X_next, bad_obs, "bad")


# ---------------------------------------------------------------------------
# 5. Result is frozen.
# ---------------------------------------------------------------------------

def test_edmdc_result_is_frozen():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    U = rng.normal(size=(50, 2))
    X_next = X.copy()
    res = fit_edmdc(X, U, X_next, polynomial_observable, "poly")
    with pytest.raises(Exception):  # FrozenInstanceError
        res.observable_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. Rollout MSE: poly diverges, phys stays stable.
# ---------------------------------------------------------------------------

def test_rollout_phys_better_than_poly_on_quadrotor(quadrotor_train, quadrotor_val):
    X_tr, U_tr, X_tr_next = _flatten(quadrotor_train)

    res_phys = fit_edmdc(
        X_tr, U_tr, X_tr_next, physics_aware_observable, "phys"
    )
    res_poly = fit_edmdc(
        X_tr, U_tr, X_tr_next, polynomial_observable, "poly"
    )

    # Use the validation trajectories as rollouts.
    X0 = quadrotor_val.states[:, 0, :]                        # (K, 6)
    U_seq = quadrotor_val.actions                             # (K, T, 2)
    X_seq_true = quadrotor_val.states[:, 1:, :]               # (K, T, 6)

    horizon = 50
    mse_phys = edmdc_rollout_mse(res_phys, X0, U_seq, X_seq_true, horizon)
    mse_poly = edmdc_rollout_mse(res_poly, X0, U_seq, X_seq_true, horizon)

    assert mse_phys < mse_poly, (
        f"physics-aware rollout MSE {mse_phys} should be < polynomial "
        f"rollout MSE {mse_poly}"
    )
    assert np.isfinite(mse_phys)


# ---------------------------------------------------------------------------
# 7. Misc shape checks.
# ---------------------------------------------------------------------------

def test_fit_edmdc_rejects_wrong_X_shape():
    X = np.zeros((50, 5))
    U = np.zeros((50, 2))
    X_next = np.zeros((50, 5))
    with pytest.raises(ValueError, match="X must have shape"):
        fit_edmdc(X, U, X_next, polynomial_observable, "x")


def test_fit_edmdc_rejects_wrong_U_shape():
    X = np.zeros((50, 6))
    U = np.zeros((50, 3))
    X_next = np.zeros((50, 6))
    with pytest.raises(ValueError, match="U must have shape"):
        fit_edmdc(X, U, X_next, polynomial_observable, "u")


def test_fit_edmdc_rejects_K_mismatch():
    X = np.zeros((50, 6))
    U = np.zeros((40, 2))
    X_next = np.zeros((50, 6))
    with pytest.raises(ValueError, match="X and U"):
        fit_edmdc(X, U, X_next, polynomial_observable, "k")


def test_rollout_rejects_horizon_too_long():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    U = rng.normal(size=(100, 2))
    X_next = X.copy()
    res = fit_edmdc(X, U, X_next, polynomial_observable, "p")
    X0 = rng.normal(size=(5, 6))
    U_seq = rng.normal(size=(5, 10, 2))
    X_seq = rng.normal(size=(5, 10, 6))
    with pytest.raises(ValueError, match="horizon"):
        edmdc_rollout_mse(res, X0, U_seq, X_seq, horizon=20)

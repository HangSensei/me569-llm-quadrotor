"""Tests for the hand-written reference basis used in E1 diagnostics."""
import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.quadrotor_dynamics import QuadrotorParams
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn
from me569_project.sysid.reference_basis import (
    PERFECT_BASIS_7_FEATURE_NAMES,
    PERFECT_BASIS_10_FEATURE_NAMES,
    expected_perfect_basis_7_Xi,
    perfect_quadrotor_basis_10,
    perfect_quadrotor_basis_7,
)


def test_perfect_basis_7_returns_7_values():
    xu = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.5, 4.9, 5.0])
    out = perfect_quadrotor_basis_7(xu)
    assert len(out) == 7


def test_perfect_basis_7_feature_values():
    """Verify each feature is computed exactly as documented."""
    xu = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.5, 4.9, 5.0])
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    expected = [
        1.0,
        v_x,
        v_z,
        omega,
        (u_1 + u_2) * np.sin(theta),
        (u_1 + u_2) * np.cos(theta),
        u_2 - u_1,
    ]
    assert perfect_quadrotor_basis_7(xu) == pytest.approx(expected)


def test_perfect_basis_10_returns_10_values():
    xu = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.5, 4.9, 5.0])
    out = perfect_quadrotor_basis_10(xu)
    assert len(out) == 10


def test_perfect_basis_10_includes_all_7_features_plus_three_linear():
    xu = np.array([0.1, 0.2, 0.05, 0.3, 0.4, 0.5, 4.9, 5.0])
    out7 = perfect_quadrotor_basis_7(xu)
    out10 = perfect_quadrotor_basis_10(xu)
    assert out10[:7] == out7
    assert out10[7] == pytest.approx(xu[0])   # p_x
    assert out10[8] == pytest.approx(xu[1])   # p_z
    assert out10[9] == pytest.approx(xu[2])   # theta


def test_feature_name_lists_match_basis_output_length():
    assert len(PERFECT_BASIS_7_FEATURE_NAMES) == 7
    assert len(PERFECT_BASIS_10_FEATURE_NAMES) == 10


def test_expected_Xi_has_correct_shape():
    params = QuadrotorParams()
    Xi = expected_perfect_basis_7_Xi(params)
    assert Xi.shape == (7, 6)


def test_expected_Xi_known_entries_for_default_params():
    """Check the non-zero entries against the hand-derived values."""
    params = QuadrotorParams()
    Xi = expected_perfect_basis_7_Xi(params)

    # Non-zero entries per the module docstring
    assert Xi[0, 4] == pytest.approx(-9.81)
    assert Xi[1, 0] == pytest.approx(1.0)
    assert Xi[2, 1] == pytest.approx(1.0)
    assert Xi[3, 2] == pytest.approx(1.0)
    assert Xi[4, 3] == pytest.approx(-1.0)  # -1/m, m=1
    assert Xi[5, 4] == pytest.approx(1.0)   # 1/m
    assert Xi[6, 5] == pytest.approx(25.0)  # L/I_yy = 0.25/0.01

    # Every other entry must be zero
    mask = np.ones_like(Xi, dtype=bool)
    mask[0, 4] = False
    mask[1, 0] = False
    mask[2, 1] = False
    mask[3, 2] = False
    mask[4, 3] = False
    mask[5, 4] = False
    mask[6, 5] = False
    assert np.all(Xi[mask] == 0.0)


def test_fitting_perfect_basis_recovers_expected_coefficients():
    """End-to-end: fit the 7-feature perfect basis to generated data and
    verify the recovered Xi matches the expected coefficients.

    Tolerance: atol=0.2 on the dominant entries accounts for the
    finite-difference truncation error (target_dt ~ 0.02 means the FD
    target x_dot is off by O(dt * x_ddot), which for gravity-scale
    accelerations introduces up to ~0.1 per entry).
    """
    params = QuadrotorParams()
    train = generate_trajectories(
        num_trajectories=50,
        steps_per_trajectory=400,
        dt=0.02,
        seed=0,
    )
    result = fit_with_basis_fn(
        train, basis_fn=perfect_quadrotor_basis_7, threshold=0.1
    )
    Xi_hat = result.Xi
    Xi_true = expected_perfect_basis_7_Xi(params)

    assert Xi_hat.shape == Xi_true.shape

    # Key structural checks: the dominant entries should be recovered
    assert abs(Xi_hat[0, 4] - (-9.81)) < 0.5     # gravity
    assert abs(Xi_hat[1, 0] - 1.0) < 0.1          # p_x_dot ~ v_x
    assert abs(Xi_hat[2, 1] - 1.0) < 0.1          # p_z_dot ~ v_z
    assert abs(Xi_hat[3, 2] - 1.0) < 0.1          # theta_dot ~ omega
    assert abs(Xi_hat[4, 3] - (-1.0)) < 0.2       # v_x_dot coefficient
    assert abs(Xi_hat[5, 4] - 1.0) < 0.2          # v_z_dot coefficient
    assert abs(Xi_hat[6, 5] - 25.0) < 2.0         # omega_dot coefficient


def test_fitting_perfect_basis_is_near_zero_mse():
    """The perfect basis should give one-step MSE dominated by the
    finite-difference truncation floor — orders of magnitude below the
    polynomial baseline's ~16.
    """
    train = generate_trajectories(
        num_trajectories=50,
        steps_per_trajectory=400,
        dt=0.02,
        seed=0,
    )
    val = generate_trajectories(
        num_trajectories=20,
        steps_per_trajectory=400,
        dt=0.02,
        seed=42,
    )
    result = fit_with_basis_fn(
        train, basis_fn=perfect_quadrotor_basis_7, threshold=0.1
    )
    mse = result.one_step_mse(val)
    assert mse < 1.0, f"Perfect basis MSE should be small, got {mse:.3f}"


def test_fitting_perfect_basis_10_zeros_out_redundant_features():
    """When the library contains ``p_x``, ``p_z``, ``theta`` on top of the
    minimal 7, STLSQ with threshold 0.1 should drive those three columns
    to zero (or very small) for all 6 state-derivative rows.
    """
    train = generate_trajectories(
        num_trajectories=50,
        steps_per_trajectory=400,
        dt=0.02,
        seed=0,
    )
    result = fit_with_basis_fn(
        train, basis_fn=perfect_quadrotor_basis_10, threshold=0.1
    )
    Xi = result.Xi
    assert Xi.shape == (10, 6)
    # Rows 7, 8, 9 correspond to p_x, p_z, theta which should not contribute
    redundant_block = Xi[7:10, :]
    assert np.max(np.abs(redundant_block)) < 0.5, (
        f"Expected redundant features to have near-zero coefficients, "
        f"got max |coef| = {np.max(np.abs(redundant_block)):.3f}"
    )

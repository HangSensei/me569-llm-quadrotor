"""Tests for the planar quadrotor continuous-time dynamics and RK4 integrator.

State convention (per spec §2.1):
    x = [p_x, p_z, theta, v_x, v_z, omega]

Control convention:
    u = [u_1, u_2]   # left and right rotor thrusts, Newtons, in [0, u_max]

Dynamics:
    p_x_dot = v_x
    p_z_dot = v_z
    theta_dot = omega
    v_x_dot = -(u_1 + u_2) sin(theta) / m
    v_z_dot = (u_1 + u_2) cos(theta) / m - g
    omega_dot = (u_2 - u_1) L / I_yy
"""
import numpy as np
import pytest

from me569_project.quadrotor_dynamics import (
    QuadrotorParams,
    quadrotor_dynamics,
    rk4_step,
)


def test_free_fall_with_zero_thrust():
    """Zero thrust, level pose, zero velocity. Only gravity acts.

    All derivatives should be zero except v_z_dot, which should equal -g.
    """
    params = QuadrotorParams()
    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.0])

    dx = quadrotor_dynamics(state, u, params)

    assert dx.shape == (6,)
    assert dx.dtype == np.float64
    assert dx[0] == 0.0  # p_x_dot = v_x = 0
    assert dx[1] == 0.0  # p_z_dot = v_z = 0
    assert dx[2] == 0.0  # theta_dot = omega = 0
    assert dx[3] == 0.0  # v_x_dot = 0 (no thrust, level pose)
    assert dx[4] == pytest.approx(-params.g)  # v_z_dot = -g
    assert dx[5] == 0.0  # omega_dot = 0 (symmetric zero thrust)


def test_hover_thrust_yields_zero_acceleration():
    """At u_1 = u_2 = m*g/2 from rest, all derivatives must vanish.

    This is the defining equilibrium: both rotors supply half of the weight,
    the quadrotor is level, and nothing moves.
    """
    params = QuadrotorParams()
    state = np.zeros(6)
    u = np.array([params.u_hover, params.u_hover])

    dx = quadrotor_dynamics(state, u, params)

    np.testing.assert_allclose(dx, np.zeros(6), atol=1e-12)


def test_asymmetric_thrust_creates_positive_torque():
    """u_2 > u_1 should yield a positive angular acceleration (nose pitches up).

    Sign convention: right rotor (u_2) stronger -> positive omega_dot.
    """
    params = QuadrotorParams()
    state = np.zeros(6)
    u = np.array([params.u_hover, params.u_hover + 0.1])  # right rotor slightly stronger

    dx = quadrotor_dynamics(state, u, params)

    assert dx[5] > 0, "Expected positive omega_dot when u_2 > u_1"


def test_pitched_forward_produces_forward_acceleration():
    """With the quadrotor pitched nose-down (theta < 0) at hover thrust,
    the horizontal projection of thrust should push it forward (+x direction).

    Why: thrust vector rotates with theta. At level (theta=0), thrust is
    purely vertical. At theta=-0.2 rad (nose down), thrust has a positive
    x component.
    """
    params = QuadrotorParams()
    state = np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0])  # pitched nose down 0.2 rad
    u = np.array([params.u_hover, params.u_hover])

    dx = quadrotor_dynamics(state, u, params)

    assert dx[3] > 0, "Expected positive v_x_dot when pitched nose-down at hover thrust"


def test_state_vector_unpacking_matches_convention():
    """Sanity check on state vector indexing: position [0:3], velocity [3:6]."""
    params = QuadrotorParams()
    state = np.array([1.0, 2.0, 0.3, 0.5, -0.5, 0.1])
    u = np.array([params.u_hover, params.u_hover])

    dx = quadrotor_dynamics(state, u, params)

    # p_x_dot = v_x, p_z_dot = v_z, theta_dot = omega
    assert dx[0] == pytest.approx(state[3])  # v_x
    assert dx[1] == pytest.approx(state[4])  # v_z
    assert dx[2] == pytest.approx(state[5])  # omega


def test_params_defaults():
    """Default QuadrotorParams values match the spec §2.1 table."""
    params = QuadrotorParams()
    assert params.m == 1.0
    assert params.I_yy == 0.01
    assert params.L == 0.25
    assert params.g == 9.81
    assert params.u_max == pytest.approx(2.0 * 1.0 * 9.81)
    assert params.u_hover == pytest.approx(1.0 * 9.81 / 2.0)


# ============================================================================
# RK4 integrator tests
# ============================================================================


def test_rk4_free_fall_matches_analytical_solution():
    """Free fall from rest integrated for 1 s should match p_z(t) and v_z(t).

    Analytical: p_z(t) = p_z0 - 0.5*g*t^2,  v_z(t) = -g*t.

    RK4 with dt=0.01 should match to better than 1e-4 relative error
    because the exact solution is a quadratic and RK4 is 4th-order.
    """
    params = QuadrotorParams()
    state = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.0])
    dt = 0.01
    t_total = 1.0
    n_steps = int(round(t_total / dt))

    for _ in range(n_steps):
        state = rk4_step(state, u, dt, params)

    expected_p_z = 10.0 - 0.5 * params.g * t_total * t_total
    expected_v_z = -params.g * t_total
    assert state[1] == pytest.approx(expected_p_z, rel=1e-4)
    assert state[4] == pytest.approx(expected_v_z, rel=1e-4)
    # Horizontal, pitch, and angular motion should remain exactly zero
    assert state[0] == 0.0
    assert state[2] == 0.0
    assert state[3] == 0.0
    assert state[5] == 0.0


def test_rk4_hover_stays_at_origin():
    """At hover thrust starting from the origin, the state should stay
    numerically at zero for 10 seconds (500 steps at dt=0.02).

    This is a numerical stability check: any accumulated drift here would
    indicate either a sign bug in the dynamics or a step-integration bug.
    """
    params = QuadrotorParams()
    state = np.zeros(6)
    u = np.array([params.u_hover, params.u_hover])
    dt = 0.02
    n_steps = 500

    for _ in range(n_steps):
        state = rk4_step(state, u, dt, params)

    np.testing.assert_allclose(state, np.zeros(6), atol=1e-10)


def test_rk4_zero_dt_returns_state_unchanged():
    """With dt=0 the next state should equal the current state exactly."""
    params = QuadrotorParams()
    state = np.array([1.0, 2.0, 0.3, 0.5, -0.5, 0.1])
    u = np.array([params.u_hover, params.u_hover])

    next_state = rk4_step(state, u, 0.0, params)

    np.testing.assert_array_equal(next_state, state)


def test_rk4_returns_correct_shape_and_dtype():
    """One RK4 step should return a (6,) float64 array regardless of input."""
    params = QuadrotorParams()
    state = np.array([0.1, 0.2, 0.05, 0.0, 0.0, 0.0])
    u = np.array([params.u_hover, params.u_hover])

    next_state = rk4_step(state, u, 0.01, params)

    assert next_state.shape == (6,)
    assert next_state.dtype == np.float64


def test_rk4_consistency_with_pure_euler_at_small_dt():
    """At extremely small dt, RK4 and forward Euler should agree to high precision.

    This is a sanity check: at dt=1e-6, truncation error of both methods
    is negligible, so they must produce essentially identical next states.
    """
    params = QuadrotorParams()
    state = np.array([0.5, 1.0, 0.1, 0.2, -0.1, 0.05])
    u = np.array([params.u_hover + 0.2, params.u_hover - 0.1])
    dt = 1e-6

    rk4_next = rk4_step(state, u, dt, params)
    euler_next = state + dt * quadrotor_dynamics(state, u, params)

    np.testing.assert_allclose(rk4_next, euler_next, atol=1e-14)

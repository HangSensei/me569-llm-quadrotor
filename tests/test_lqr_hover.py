"""Tests for the LQR hover controller.

The controller linearizes the Planar Quadrotor around the hover equilibrium
(x* = 0, u* = [m*g/2, m*g/2]) and solves the continuous-time algebraic
Riccati equation to compute the optimal feedback gain K. It then applies
u(x) = u_hover - K x, clipped into [0, u_max].

These tests verify:
- At the equilibrium state, the controller outputs exactly hover thrust.
- Small perturbations are driven to zero within a few seconds.
- The gain matrix has the correct shape and is finite.
- Output is always in the action bounds.
"""
import numpy as np
import pytest

from me569_project.controllers.lqr_hover import LQRHoverController
from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step


def test_lqr_returns_hover_thrust_at_origin():
    """At the equilibrium state, control should be exactly [u_hover, u_hover]."""
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)

    u = ctrl.act(np.zeros(6))

    np.testing.assert_allclose(u, [params.u_hover, params.u_hover], atol=1e-12)


def test_lqr_gain_matrix_shape_and_finiteness():
    """K must be (2, 6) and entirely finite."""
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)

    assert ctrl.K.shape == (2, 6)
    assert np.all(np.isfinite(ctrl.K))


def test_lqr_stabilizes_small_position_perturbation():
    """A 10 cm horizontal + 10 cm vertical perturbation should decay to near zero in 10 s."""
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)

    state = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    dt = 0.02

    for _ in range(500):  # 10 seconds
        u = ctrl.act(state)
        state = rk4_step(state, u, dt, params)

    # Position and velocity should both be well under the initial perturbation
    assert np.linalg.norm(state[:2]) < 0.02  # < 2 cm residual
    assert np.linalg.norm(state[3:5]) < 0.05  # small residual velocity


def test_lqr_stabilizes_small_attitude_perturbation():
    """A small pitch perturbation should decay to near zero with LQR."""
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)

    state = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])  # 0.05 rad ~ 2.9 deg
    dt = 0.02

    for _ in range(500):  # 10 seconds
        u = ctrl.act(state)
        state = rk4_step(state, u, dt, params)

    assert abs(state[2]) < 0.005  # attitude < 0.3 deg
    assert abs(state[5]) < 0.05   # angular velocity small


def test_lqr_output_is_always_in_action_bounds():
    """Even for a large perturbation the clipped output must stay in [0, u_max]."""
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)

    extreme_state = np.array([2.0, 2.0, 0.5, 1.0, 1.0, 1.0])
    u = ctrl.act(extreme_state)

    assert u.shape == (2,)
    assert np.all(u >= 0.0)
    assert np.all(u <= params.u_max)


def test_lqr_accepts_custom_Q_R_weights():
    """Passing explicit Q and R should change the gain matrix."""
    params = QuadrotorParams()
    default = LQRHoverController(params)
    aggressive = LQRHoverController(
        params,
        Q=np.diag([100.0, 100.0, 50.0, 10.0, 10.0, 5.0]),
        R=np.diag([0.01, 0.01]),
    )

    # Different weights should yield different gains
    assert not np.allclose(default.K, aggressive.K)


def test_lqr_acts_on_flat_1d_state():
    """act() must accept a plain numpy 1d array (not a 2d column vector)."""
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)

    state_1d = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    u = ctrl.act(state_1d)

    assert u.shape == (2,)
    assert np.all(np.isfinite(u))

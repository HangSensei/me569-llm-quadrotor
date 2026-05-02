"""Tests for the hand-written E3 B-condition baseline stage cost."""
import casadi
import numpy as np
import pytest

from me569_project.mpc.baseline_cost import (
    DEFAULT_Q_DIAG,
    DEFAULT_R_DIAG,
    make_baseline_stage_cost,
)
from me569_project.quadrotor_dynamics import QuadrotorParams


def test_default_Q_R_match_spec_D011():
    assert DEFAULT_Q_DIAG == (10.0, 10.0, 5.0, 1.0, 1.0, 0.5)
    assert DEFAULT_R_DIAG == (0.1, 0.1)


def test_baseline_cost_zero_at_hover_origin():
    params = QuadrotorParams()
    fn = make_baseline_stage_cost(params=params)

    x_origin = np.zeros(6)
    u_hover = np.array([params.u_hover, params.u_hover])

    assert fn(x_origin, u_hover) == pytest.approx(0.0, abs=1e-12)


def test_baseline_cost_positive_off_origin():
    params = QuadrotorParams()
    fn = make_baseline_stage_cost(params=params)

    x_off = np.array([0.5, -0.3, 0.1, 0.0, 0.0, 0.0])
    u_hover = np.array([params.u_hover, params.u_hover])
    cost = fn(x_off, u_hover)
    assert cost > 0.0
    # Position dominates: 10 * 0.25 + 10 * 0.09 + 5 * 0.01 = 3.45
    assert cost == pytest.approx(3.45, rel=1e-9)


def test_baseline_cost_penalizes_thrust_deviation():
    params = QuadrotorParams()
    fn = make_baseline_stage_cost(params=params)

    x_origin = np.zeros(6)
    u_max_thrust = np.array([params.u_max, params.u_max])

    cost = fn(x_origin, u_max_thrust)
    # Each rotor deviates by (u_max - u_hover) = (2mg - mg/2) = 1.5 mg = 14.715
    # Cost: 0.1 * 14.715^2 * 2 ~= 43.3
    expected = 2 * 0.1 * (params.u_max - params.u_hover) ** 2
    assert cost == pytest.approx(expected, rel=1e-9)


def test_baseline_cost_works_on_casadi_symbol():
    """The baseline stage cost must be CasADi-compatible so it can be
    plugged into PlanarQuadrotorMPC.
    """
    params = QuadrotorParams()
    fn = make_baseline_stage_cost(params=params)

    x_sym = casadi.SX.sym("x", 6)
    u_sym = casadi.SX.sym("u", 2)
    expr = fn(x_sym, u_sym)
    f = casadi.Function("f", [x_sym, u_sym], [expr])

    val = float(f(np.zeros(6), [params.u_hover, params.u_hover]))
    assert val == pytest.approx(0.0, abs=1e-12)


def test_baseline_cost_drives_mpc_to_origin():
    """An MPC built around the baseline cost should drive a small
    perturbation back toward the origin within ~1 second.
    """
    from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
    from me569_project.quadrotor_dynamics import rk4_step

    params = QuadrotorParams()
    cost_fn = make_baseline_stage_cost(params=params)
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=cost_fn,
        config=MPCConfig(horizon=20),
    )

    state = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    dt = 0.02

    for _ in range(50):  # 1 second
        u = mpc.make_step(state)
        state = rk4_step(state, u, dt, params)

    # Position error should be smaller than the initial 0.14 norm
    pos_norm = float(np.linalg.norm(state[:2]))
    assert pos_norm < 0.14, f"position not converging, residual {pos_norm:.3f}"


def test_custom_Q_R_change_cost():
    params = QuadrotorParams()
    fn_default = make_baseline_stage_cost(params=params)
    fn_aggressive = make_baseline_stage_cost(
        params=params,
        Q_diag=(100.0, 100.0, 50.0, 10.0, 10.0, 5.0),
    )

    x = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([params.u_hover, params.u_hover])

    assert fn_aggressive(x, u) > fn_default(x, u)

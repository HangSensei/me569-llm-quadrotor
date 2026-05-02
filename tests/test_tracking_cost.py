"""Tests for the quadratic tracking cost helper."""
import casadi
import numpy as np
import pytest

from me569_project.mpc.tracking_cost import make_tracking_stage_cost
from me569_project.quadrotor_dynamics import QuadrotorParams


def test_tracking_cost_is_zero_at_reference():
    params = QuadrotorParams()
    fn = make_tracking_stage_cost(reference_fn=lambda _t: np.zeros(6), params=params)

    error = np.zeros(6)
    u_hover = np.array([params.u_hover, params.u_hover])

    assert fn(error, u_hover) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_positive_off_reference():
    params = QuadrotorParams()
    fn = make_tracking_stage_cost(reference_fn=lambda _t: np.zeros(6), params=params)

    error = np.array([0.5, -0.3, 0.1, 0.0, 0.0, 0.0])
    u_hover = np.array([params.u_hover, params.u_hover])

    assert fn(error, u_hover) > 0.0


def test_tracking_cost_works_with_casadi():
    params = QuadrotorParams()
    fn = make_tracking_stage_cost(reference_fn=lambda _t: np.zeros(6), params=params)

    e_sym = casadi.SX.sym("e", 6)
    u_sym = casadi.SX.sym("u", 2)
    expr = fn(e_sym, u_sym)
    f = casadi.Function("f", [e_sym, u_sym], [expr])

    val = float(f(np.zeros(6), [params.u_hover, params.u_hover]))
    assert val == pytest.approx(0.0, abs=1e-12)

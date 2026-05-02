"""Tests for the E3 stage cost math shim and sandbox loader.

Verifies that:
1. The CasADi math functions work on both numpy and CasADi inputs.
2. ``load_stage_cost_callable`` loads functions in a sandbox where
   numpy is intentionally absent and CasADi math is pre-injected.
3. The same loaded function evaluates correctly on numpy arrays
   (for unit testing) and on CasADi SX symbols (for MPC compilation).
4. LLM code that tries to use ``numpy.sin`` fails cleanly with the
   expected error type, instead of silently producing a function that
   breaks in symbolic mode.
"""
import casadi
import numpy as np
import pytest

from me569_project.llm.sandbox import SandboxError
from me569_project.mpc.mpc_math import (
    MPC_MATH_NAMESPACE,
    cos,
    exp,
    fabs,
    load_stage_cost_callable,
    log,
    sin,
    sqrt,
)


# ----------------------------------------------------------
# Tests on the math shims directly
# ----------------------------------------------------------


def test_namespace_dict_contains_six_functions():
    expected = {"sin", "cos", "exp", "sqrt", "log", "fabs"}
    assert set(MPC_MATH_NAMESPACE.keys()) == expected


def test_sin_works_on_python_scalar():
    assert sin(0.0) == pytest.approx(0.0)
    assert sin(np.pi / 2) == pytest.approx(1.0)


def test_sin_works_on_numpy_array():
    x = np.array([0.0, np.pi / 2, np.pi])
    out = sin(x)
    # CasADi returns a DM-like object on numpy input; convert for compare.
    out_arr = np.array(out).flatten()
    np.testing.assert_allclose(out_arr, [0.0, 1.0, 0.0], atol=1e-12)


def test_sin_works_on_casadi_symbol():
    x = casadi.SX.sym("x")
    expr = sin(x)
    # The result should be a CasADi expression we can evaluate.
    f = casadi.Function("f", [x], [expr])
    assert float(f(0.0)) == pytest.approx(0.0)
    assert float(f(np.pi / 2)) == pytest.approx(1.0)


def test_cos_squared_plus_sin_squared_is_one_in_casadi():
    """A real symbolic computation that the LLM might generate."""
    theta = casadi.SX.sym("theta")
    expr = sin(theta) ** 2 + cos(theta) ** 2
    f = casadi.Function("f", [theta], [expr])
    for val in [0.0, 0.3, -0.7, np.pi]:
        assert float(f(val)) == pytest.approx(1.0)


def test_fabs_works_on_negative_input():
    assert float(fabs(-3.7)) == pytest.approx(3.7)
    assert float(fabs(2.5)) == pytest.approx(2.5)


def test_sqrt_log_exp_round_trip():
    x = casadi.SX.sym("x")
    expr = log(exp(sqrt(x) ** 2))
    f = casadi.Function("f", [x], [expr])
    assert float(f(7.5)) == pytest.approx(7.5)


# ----------------------------------------------------------
# Tests on load_stage_cost_callable
# ----------------------------------------------------------


def test_load_stage_cost_returns_callable():
    code = """
def stage_cost(x, u):
    return x[0] ** 2 + u[0] ** 2
"""
    fn = load_stage_cost_callable(code)
    assert callable(fn)


def test_load_stage_cost_runs_on_numpy_arrays():
    code = """
def stage_cost(x, u):
    return x[0] ** 2 + 0.5 * x[1] ** 2 + 0.1 * u[0] ** 2
"""
    fn = load_stage_cost_callable(code)
    x = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([3.0, 0.0])
    result = fn(x, u)
    # 1*1 + 0.5*4 + 0.1*9 = 1 + 2 + 0.9 = 3.9
    assert float(result) == pytest.approx(3.9)


def test_load_stage_cost_uses_casadi_sin_natively():
    """The injected ``sin`` should be CasADi's, so a stage cost using
    ``sin(theta)`` works on a CasADi symbolic state.
    """
    code = """
def stage_cost(x, u):
    # Penalize attitude with a smooth (1 - cos(theta)) term
    return x[0] ** 2 + (1 - cos(x[2])) + 0.1 * u[0] ** 2
"""
    fn = load_stage_cost_callable(code)

    # Build a CasADi SX vector for state and control and evaluate.
    x_sym = casadi.SX.sym("x", 6)
    u_sym = casadi.SX.sym("u", 2)
    expr = fn(x_sym, u_sym)
    f = casadi.Function("f", [x_sym, u_sym], [expr])

    # At theta=0 the (1 - cos) term is 0; at theta=pi it is 2.
    val_origin = float(f([0.0] * 6, [0.0, 0.0]))
    val_inverted = float(f([0.0, 0.0, np.pi, 0.0, 0.0, 0.0], [0.0, 0.0]))
    assert val_origin == pytest.approx(0.0, abs=1e-12)
    assert val_inverted == pytest.approx(2.0)


def test_load_stage_cost_blocks_numpy_import():
    code = """
import numpy as np

def stage_cost(x, u):
    return np.sin(x[2])
"""
    with pytest.raises(SandboxError, match="disallowed"):
        load_stage_cost_callable(code)


def test_load_stage_cost_blocks_np_name_when_no_import():
    """Even without an import statement, ``np`` should not be available
    in the namespace, so referencing it raises NameError -> SandboxError.
    """
    code = """
def stage_cost(x, u):
    return np.sin(x[2]) + u[0] ** 2
"""
    fn = load_stage_cost_callable(code)
    # The function loads fine (no syntax error in defining it), but
    # *calling* it should raise because ``np`` isn't bound.
    with pytest.raises(NameError):
        fn(np.zeros(6), np.zeros(2))


def test_load_stage_cost_uses_custom_function_name():
    code = """
def my_cost(state, control):
    return state[0] ** 2
"""
    fn = load_stage_cost_callable(code, function_name="my_cost")
    assert float(fn(np.array([2.0, 0, 0, 0, 0, 0]), np.zeros(2))) == pytest.approx(4.0)


def test_load_stage_cost_realistic_quadrotor_hover_cost():
    """A realistic LLM-style hover cost that mixes positions, attitude,
    velocities, and a control regularizer.
    """
    code = """
def stage_cost(x, u):
    # x = [p_x, p_z, theta, v_x, v_z, omega]
    # Position penalty
    pos = x[0] ** 2 + x[1] ** 2
    # Smooth attitude penalty (zero at theta=0, monotonic for small theta)
    att = (1 - cos(x[2]))
    # Velocity penalty
    vel = x[3] ** 2 + x[4] ** 2 + 0.1 * x[5] ** 2
    # Control deviation from hover thrust (4.905 = m*g/2)
    ctrl = (u[0] - 4.905) ** 2 + (u[1] - 4.905) ** 2
    return pos + 0.5 * att + 0.1 * vel + 0.01 * ctrl
"""
    fn = load_stage_cost_callable(code)

    # Numerical test: at exact hover state and hover thrust, cost ~ 0
    x_hover = np.zeros(6)
    u_hover = np.array([4.905, 4.905])
    cost_at_hover = float(fn(x_hover, u_hover))
    assert cost_at_hover == pytest.approx(0.0, abs=1e-12)

    # Off-hover cost should be positive
    x_off = np.array([0.5, 0.5, 0.2, 0.1, -0.1, 0.0])
    cost_off = float(fn(x_off, u_hover))
    assert cost_off > 0.0

    # Symbolic test: build the same expression with CasADi and verify
    x_sym = casadi.SX.sym("x", 6)
    u_sym = casadi.SX.sym("u", 2)
    expr = fn(x_sym, u_sym)
    f_sym = casadi.Function("f", [x_sym, u_sym], [expr])
    cost_sym_at_hover = float(f_sym([0.0] * 6, [4.905, 4.905]))
    assert cost_sym_at_hover == pytest.approx(0.0, abs=1e-12)

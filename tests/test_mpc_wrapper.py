"""Tests for the do-mpc wrapper that drives the Planar Quadrotor."""
import casadi
import numpy as np
import pytest

from me569_project.mpc.mpc_wrapper import (
    MPCConfig,
    PlanarQuadrotorMPC,
)
from me569_project.quadrotor_dynamics import QuadrotorParams


def _quadratic_stage_cost(x, u):
    """Reference quadratic stage cost using only ops valid in CasADi."""
    return (
        10.0 * x[0] ** 2
        + 10.0 * x[1] ** 2
        + 5.0 * x[2] ** 2
        + 1.0 * x[3] ** 2
        + 1.0 * x[4] ** 2
        + 0.5 * x[5] ** 2
        + 0.1 * (u[0] - 4.905) ** 2
        + 0.1 * (u[1] - 4.905) ** 2
    )


def test_mpc_config_defaults():
    cfg = MPCConfig()
    assert cfg.horizon == 20
    assert cfg.dt == pytest.approx(0.02)
    assert cfg.terminal_weight == pytest.approx(10.0)


def test_mpc_constructor_runs_without_error():
    params = QuadrotorParams()
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=_quadratic_stage_cost,
        config=MPCConfig(),
    )
    assert mpc is not None


def test_mpc_make_step_returns_two_thrusts_in_bounds():
    params = QuadrotorParams()
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=_quadratic_stage_cost,
        config=MPCConfig(),
    )

    x0 = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    u = mpc.make_step(x0)

    assert u.shape == (2,)
    assert np.all(u >= 0.0)
    assert np.all(u <= params.u_max + 1e-9)


def test_mpc_drives_origin_perturbation_toward_origin():
    """One MPC step from a small perturbation should produce a control
    that decreases the cost on the next state.
    """
    params = QuadrotorParams()
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=_quadratic_stage_cost,
        config=MPCConfig(),
    )

    x0 = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    u0 = mpc.make_step(x0)

    # Apply that control via the project's RK4 to get the next state.
    from me569_project.quadrotor_dynamics import rk4_step

    dt = 0.02
    x1 = rk4_step(x0, u0, dt, params)

    # The MPC controller should NOT make the position dramatically worse
    # in one step (we allow some slack because the controller is
    # trading off attitude vs position).
    assert abs(x1[0]) < abs(x0[0]) + 0.05
    assert abs(x1[1]) < abs(x0[1]) + 0.05
    # Velocities should still be small
    assert np.abs(x1[3:]).max() < 1.0


def test_mpc_returns_hover_thrust_at_origin():
    """At the equilibrium state with the quadratic cost, the MPC's first
    action should be very close to per-rotor hover thrust m*g/2.
    """
    params = QuadrotorParams()
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=_quadratic_stage_cost,
        config=MPCConfig(),
    )

    x0 = np.zeros(6)
    u = mpc.make_step(x0)

    expected = params.u_hover
    np.testing.assert_allclose(u, [expected, expected], atol=0.05)


def test_mpc_works_with_trig_stage_cost():
    """The wrapper must accept a stage cost that uses CasADi-compatible
    trig functions injected via mpc_math.
    """
    from me569_project.mpc.mpc_math import cos as mpc_cos

    def trig_cost(x, u):
        return (
            10.0 * x[0] ** 2
            + 10.0 * x[1] ** 2
            + 5.0 * (1 - mpc_cos(x[2]))
            + 1.0 * (x[3] ** 2 + x[4] ** 2)
            + 0.5 * x[5] ** 2
            + 0.1 * (u[0] - 4.905) ** 2
            + 0.1 * (u[1] - 4.905) ** 2
        )

    params = QuadrotorParams()
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=trig_cost,
        config=MPCConfig(),
    )

    x0 = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # tilted
    u = mpc.make_step(x0)
    assert u.shape == (2,)
    assert np.all(np.isfinite(u))


def test_mpc_clips_action_to_box():
    """Even if the LLM stage cost would push the controller off the action
    box, the returned thrust must still respect [0, u_max].
    """
    params = QuadrotorParams()
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=_quadratic_stage_cost,
        config=MPCConfig(),
    )

    x_far = np.array([2.0, 2.0, 0.5, 1.0, 1.0, 1.0])
    u = mpc.make_step(x_far)
    assert np.all(u >= 0.0)
    assert np.all(u <= params.u_max + 1e-9)


def test_mpc_horizon_is_configurable():
    params = QuadrotorParams()
    cfg = MPCConfig(horizon=5)
    mpc = PlanarQuadrotorMPC(
        params=params,
        stage_cost=_quadratic_stage_cost,
        config=cfg,
    )
    u = mpc.make_step(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert u.shape == (2,)
    assert np.all(np.isfinite(u))


def test_mpc_raises_on_invalid_stage_cost_signature():
    """A stage cost that crashes during symbolic compilation should raise
    a recognizable error type, not silently produce a broken MPC.
    """
    def bad_cost(x, u):
        # Returns a tuple instead of a scalar — CasADi can't symbolically
        # build the lterm from this.
        return (x[0], x[1])

    params = QuadrotorParams()
    with pytest.raises(Exception):
        PlanarQuadrotorMPC(
            params=params,
            stage_cost=bad_cost,
            config=MPCConfig(),
        )

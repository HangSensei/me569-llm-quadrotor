"""Planar quadrotor continuous-time dynamics.

This module defines the physical parameters and the nonlinear continuous-time
ODE for a 2D quadrotor (the "Planar Quadrotor"). It is the foundation for all
downstream work in this project: data generation for SINDy (Experiment 1), the
ground-truth model inside MPC (Experiment 3), and the Gymnasium environment
used for RL stretch goals.

State convention:
    x = [p_x, p_z, theta, v_x, v_z, omega] in R^6
where
    p_x, p_z : horizontal and vertical position (m), p_z positive = up
    theta    : pitch angle (rad), 0 = level, positive = nose up on the right
    v_x, v_z : linear velocities (m/s)
    omega    : angular velocity (rad/s)

Control convention:
    u = [u_1, u_2] in R^2, u_i in [0, u_max]
where u_1, u_2 are the left and right rotor thrusts in Newtons.

Continuous-time dynamics:
    p_x_dot   = v_x
    p_z_dot   = v_z
    theta_dot = omega
    v_x_dot   = -(u_1 + u_2) sin(theta) / m
    v_z_dot   =  (u_1 + u_2) cos(theta) / m - g
    omega_dot = (u_2 - u_1) L / I_yy

This is the standard formulation used in Brunton & Kutz (2019) chapter 5.6 and
in e.g. Tedrake's "Underactuated Robotics" quadrotor chapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadrotorParams:
    """Physical parameters of the Planar Quadrotor.

    Defaults match the spec §2.1 table and are chosen to give a clean hover
    thrust of m*g/2 ~= 4.905 N per rotor with an upper bound u_max = 2*m*g.
    """

    m: float = 1.0        # mass (kg)
    I_yy: float = 0.01    # moment of inertia about the pitch axis (kg * m^2)
    L: float = 0.25       # rotor arm length (m)
    g: float = 9.81       # gravity (m/s^2)

    @property
    def u_max(self) -> float:
        """Maximum thrust per rotor (Newtons)."""
        return 2.0 * self.m * self.g

    @property
    def u_hover(self) -> float:
        """Per-rotor thrust required to hover level at zero velocity."""
        return self.m * self.g / 2.0


def quadrotor_dynamics(
    state: np.ndarray,
    u: np.ndarray,
    params: QuadrotorParams,
) -> np.ndarray:
    """Compute dx/dt for a given state and control.

    Parameters
    ----------
    state : np.ndarray of shape (6,)
        Current state [p_x, p_z, theta, v_x, v_z, omega].
    u : np.ndarray of shape (2,)
        Current control [u_1, u_2] (left and right rotor thrusts in Newtons).
        No clipping is applied here; the env layer is responsible for clipping
        into [0, u_max].
    params : QuadrotorParams
        Physical parameters.

    Returns
    -------
    np.ndarray of shape (6,), dtype float64
        Time derivative of the state, same ordering as the input state.
    """
    p_x, p_z, theta, v_x, v_z, omega = state
    u_1, u_2 = u

    total_thrust = u_1 + u_2
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array(
        [
            v_x,
            v_z,
            omega,
            -total_thrust * sin_theta / params.m,
            total_thrust * cos_theta / params.m - params.g,
            (u_2 - u_1) * params.L / params.I_yy,
        ],
        dtype=np.float64,
    )


def rk4_step(
    state: np.ndarray,
    u: np.ndarray,
    dt: float,
    params: QuadrotorParams,
) -> np.ndarray:
    """Advance the state by one step using the classical 4th-order Runge-Kutta scheme.

    The control u is held constant across the step (zero-order hold), which
    is the standard assumption for discrete-time control of a continuous plant.

    Parameters
    ----------
    state : np.ndarray of shape (6,)
        Current state [p_x, p_z, theta, v_x, v_z, omega].
    u : np.ndarray of shape (2,)
        Control input, held constant over the interval [t, t+dt].
    dt : float
        Integration step size in seconds. Typical values 0.01–0.02 for
        this system; larger values degrade accuracy quickly because the
        dynamics become stiff when the quadrotor is near saturation.
    params : QuadrotorParams
        Physical parameters, passed through to ``quadrotor_dynamics``.

    Returns
    -------
    np.ndarray of shape (6,), dtype float64
        State at time t + dt.

    Notes
    -----
    RK4 error scales as O(dt^5) per step, O(dt^4) over a fixed horizon.
    For dt = 0.01 and a 1 s horizon, relative error on the free-fall
    solution is around 1e-6, well within the tolerance used by the test
    suite (1e-4).
    """
    k1 = quadrotor_dynamics(state, u, params)
    k2 = quadrotor_dynamics(state + 0.5 * dt * k1, u, params)
    k3 = quadrotor_dynamics(state + 0.5 * dt * k2, u, params)
    k4 = quadrotor_dynamics(state + dt * k3, u, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

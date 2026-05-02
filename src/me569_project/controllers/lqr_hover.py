"""LQR controller linearized around the Planar Quadrotor hover equilibrium.

Linearization
-------------
At the hover equilibrium x* = 0, u* = [m*g/2, m*g/2], the Jacobians of the
continuous dynamics f(x, u) are:

    A = df/dx |_{x*, u*} =
        [[0, 0,  0, 1, 0, 0],   # p_x_dot = v_x
         [0, 0,  0, 0, 1, 0],   # p_z_dot = v_z
         [0, 0,  0, 0, 0, 1],   # theta_dot = omega
         [0, 0, -g, 0, 0, 0],   # v_x_dot ~ -g * theta (small-angle)
         [0, 0,  0, 0, 0, 0],   # v_z_dot ~ 0 to first order (at theta=0)
         [0, 0,  0, 0, 0, 0]]   # omega_dot doesn't depend on state linearly

    B = df/du |_{x*, u*} =
        [[0,      0],
         [0,      0],
         [0,      0],
         [0,      0],
         [1/m,    1/m],          # v_z_dot = (u1+u2)/m - g
         [-L/I,   L/I]]           # omega_dot = (u2-u1)*L/I

The horizontal control chain is u -> omega -> theta -> v_x -> p_x (a chain
of integrators driven by differential thrust). Vertical control is a pure
2-integrator chain driven by symmetric thrust. The pair (A, B) is
controllable, so the continuous algebraic Riccati equation has a unique
positive-definite solution P, and the optimal feedback gain is

    K = R^-1 B^T P

The control law is

    u(x) = u_hover - K x,   clipped into [0, u_max]^2.

Default Q and R
---------------
The default weights follow the baseline values from spec §3.1 / D011:

    Q = diag(10, 10, 5, 1, 1, 0.5)
    R = diag(0.1, 0.1)

This puts the most cost on positional error, moderate cost on attitude, and
light regularization on velocity and control effort. Bryson's rule is the
rationale (weights ~ 1 / max_acceptable_value^2), with the default values
tuned so that the closed-loop system has a ~2 s settling time from a
small perturbation without saturating the thrust envelope.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from me569_project.quadrotor_dynamics import QuadrotorParams


class LQRHoverController:
    """Continuous-time LQR around the hover equilibrium, applied via
    zero-order hold.

    Parameters
    ----------
    params : QuadrotorParams
        Physical parameters used to build A, B.
    Q : np.ndarray of shape (6, 6), optional
        State cost matrix. Default follows spec §3.1 / D011.
    R : np.ndarray of shape (2, 2), optional
        Control cost matrix. Default follows spec §3.1 / D011.
    """

    def __init__(
        self,
        params: QuadrotorParams,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ) -> None:
        self.params = params

        # Linearization around hover
        g = params.g
        m = params.m
        I_yy = params.I_yy
        L = params.L

        A = np.zeros((6, 6), dtype=np.float64)
        A[0, 3] = 1.0   # p_x_dot = v_x
        A[1, 4] = 1.0   # p_z_dot = v_z
        A[2, 5] = 1.0   # theta_dot = omega
        A[3, 2] = -g    # v_x_dot ~= -g * theta (small-angle)

        B = np.zeros((6, 2), dtype=np.float64)
        B[4, 0] = 1.0 / m
        B[4, 1] = 1.0 / m
        B[5, 0] = -L / I_yy
        B[5, 1] = L / I_yy

        self.A = A
        self.B = B

        self.Q = (
            Q
            if Q is not None
            else np.diag([10.0, 10.0, 5.0, 1.0, 1.0, 0.5])
        )
        self.R = R if R is not None else np.diag([0.1, 0.1])

        # Solve continuous-time algebraic Riccati equation: A'P + PA - PBR^-1 B'P + Q = 0
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.P = P
        self.K = np.linalg.solve(self.R, B.T @ P)

        self.u_hover_vec = np.array(
            [params.u_hover, params.u_hover], dtype=np.float64
        )

    def act(self, state: np.ndarray) -> np.ndarray:
        """Return the LQR control action for the current state.

        u(x) = u_hover - K x, clipped into [0, u_max]^2.

        Parameters
        ----------
        state : np.ndarray of shape (6,)
            Current quadrotor state.

        Returns
        -------
        np.ndarray of shape (2,), dtype float64
            Rotor thrusts [u_1, u_2], each in [0, u_max].
        """
        state = np.asarray(state, dtype=np.float64).reshape(6)
        delta_u = -self.K @ state
        u = self.u_hover_vec + delta_u
        return np.clip(u, 0.0, self.params.u_max)

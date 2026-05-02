"""Hand-written quadratic stage cost — the B condition for Experiment 3.

This is the textbook MPC stage cost for hover stabilization on the
Planar Quadrotor:

    ℓ(x, u) = xᵀ Q x + (u - u_hover)ᵀ R (u - u_hover)

with Q and R chosen via Bryson's rule from the spec D011 / Section 3.1
defaults. The Q matrix penalizes deviations from the origin in
position, attitude, and velocity; the R matrix penalizes deviations
from per-rotor hover thrust ``u_hover = m * g / 2``.

This file is intentionally minimal. The B condition is the *baseline*
the LLM-generated stage costs (P / Q conditions) get compared against,
and the comparison is only fair if the baseline is a textbook standard
without any post-hoc tuning. The Q and R values are exactly the ones
in the spec, with no adjustment.
"""
from __future__ import annotations

from typing import Callable

from me569_project.quadrotor_dynamics import QuadrotorParams


# Default Q and R weights from spec D011 / Section 3.1.
# Q rows correspond to [p_x, p_z, theta, v_x, v_z, omega].
DEFAULT_Q_DIAG: tuple[float, ...] = (10.0, 10.0, 5.0, 1.0, 1.0, 0.5)
DEFAULT_R_DIAG: tuple[float, ...] = (0.1, 0.1)


def make_baseline_stage_cost(
    params: QuadrotorParams | None = None,
    Q_diag: tuple[float, ...] = DEFAULT_Q_DIAG,
    R_diag: tuple[float, ...] = DEFAULT_R_DIAG,
) -> Callable:
    """Return the closure ``stage_cost(x, u) -> scalar`` for the B condition.

    The closure is suitable for both:
    - Numerical evaluation on numpy arrays (for unit tests / debug).
    - Symbolic propagation by CasADi inside the do-mpc wrapper (only
      ``+``, ``-``, ``*``, ``**``, indexing — all CasADi-compatible).

    Parameters
    ----------
    params : QuadrotorParams, optional
        Used only to compute ``u_hover = m * g / 2``. Defaults to the
        standard ``QuadrotorParams()``.
    Q_diag : tuple of 6 floats, optional
        Diagonal of the state-cost matrix. Default from D011.
    R_diag : tuple of 2 floats, optional
        Diagonal of the control-deviation cost matrix. Default from D011.
    """
    if params is None:
        params = QuadrotorParams()
    if len(Q_diag) != 6:
        raise ValueError(f"Q_diag must have length 6, got {len(Q_diag)}")
    if len(R_diag) != 2:
        raise ValueError(f"R_diag must have length 2, got {len(R_diag)}")

    q0, q1, q2, q3, q4, q5 = Q_diag
    r0, r1 = R_diag
    u_hover = params.u_hover

    def stage_cost(x, u):
        # State penalty
        state_term = (
            q0 * x[0] ** 2
            + q1 * x[1] ** 2
            + q2 * x[2] ** 2
            + q3 * x[3] ** 2
            + q4 * x[4] ** 2
            + q5 * x[5] ** 2
        )
        # Control deviation from hover thrust
        ctrl_term = (
            r0 * (u[0] - u_hover) ** 2
            + r1 * (u[1] - u_hover) ** 2
        )
        return state_term + ctrl_term

    return stage_cost

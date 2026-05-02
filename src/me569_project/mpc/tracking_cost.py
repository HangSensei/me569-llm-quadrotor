"""Tracking-cost helpers for time-varying reference MPC."""
from __future__ import annotations

from typing import Callable

from me569_project.mpc.baseline_cost import DEFAULT_Q_DIAG, DEFAULT_R_DIAG
from me569_project.quadrotor_dynamics import QuadrotorParams


def make_tracking_stage_cost(
    reference_fn,
    params: QuadrotorParams | None = None,
    Q_diag: tuple[float, ...] | None = None,
    R_diag: tuple[float, ...] | None = None,
) -> Callable:
    """Return a quadratic tracking cost on error state and hover-relative thrust.

    The returned callable has signature ``stage_cost(e, u)`` where ``e`` is the
    tracking error ``x - x_ref``. The reference trajectory itself is injected
    symbolically inside ``PlanarQuadrotorTrackingMPC`` through do-mpc TVPs.
    ``reference_fn`` is accepted here so the factory mirrors the tracking API,
    even though the symbolic controller, not this closure, owns the reference.
    """
    del reference_fn

    if params is None:
        params = QuadrotorParams()
    if Q_diag is None:
        Q_diag = DEFAULT_Q_DIAG
    if R_diag is None:
        R_diag = DEFAULT_R_DIAG
    if len(Q_diag) != 6:
        raise ValueError(f"Q_diag must have length 6, got {len(Q_diag)}")
    if len(R_diag) != 2:
        raise ValueError(f"R_diag must have length 2, got {len(R_diag)}")

    q0, q1, q2, q3, q4, q5 = Q_diag
    r0, r1 = R_diag
    u_hover = params.u_hover

    def stage_cost(e, u):
        state_term = (
            q0 * e[0] ** 2
            + q1 * e[1] ** 2
            + q2 * e[2] ** 2
            + q3 * e[3] ** 2
            + q4 * e[4] ** 2
            + q5 * e[5] ** 2
        )
        ctrl_term = (
            r0 * (u[0] - u_hover) ** 2
            + r1 * (u[1] - u_hover) ** 2
        )
        return state_term + ctrl_term

    return stage_cost

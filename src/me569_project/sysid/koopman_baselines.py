"""Hand-built baseline observable functions for E3 (Koopman / EDMDc).

The B condition of E3 uses ``polynomial_observable`` — a degree-2
polynomial dictionary on the state, which has the same family of
limitations on Quadrotor as the polynomial baseline did for E1: it
cannot represent the bilinear ``(u_1 + u_2) sin(theta)`` /
``(u_1 + u_2) cos(theta)`` thrust-projection structure no matter how
many monomials we add.

A second, smarter baseline ``physics_aware_observable`` is provided
for testing purposes: it includes the ``sin(theta)`` and ``cos(theta)``
terms a controls engineer would write down by inspection. This is NOT
the B condition; it's used in unit tests to confirm that EDMDc can
reach a much lower MSE when given the right observables, validating
the fitter independently of the LLM-generated observables.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def polynomial_observable(x: np.ndarray) -> np.ndarray:
    """Degree-2 polynomial dictionary on state. 27 dims total.

    Layout: [x (6), x_i^2 for i in 0..5 (6), x_i * x_j for i<j (15)].
    First six elements equal the state itself, satisfying the
    state-recovery convention.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape[0] != 6:
        raise ValueError(f"polynomial_observable expects 6-dim state; got {x.shape}")

    px, pz, th, vx, vz, om = x

    sq = [px * px, pz * pz, th * th, vx * vx, vz * vz, om * om]                        # 6
    cross = [
        px * pz, px * th, px * vx, px * vz, px * om,                                    # 5
        pz * th, pz * vx, pz * vz, pz * om,                                             # 4
        th * vx, th * vz, th * om,                                                      # 3
        vx * vz, vx * om,                                                               # 2
        vz * om,                                                                        # 1
    ]                                                                                   # 15 cross
    out = [px, pz, th, vx, vz, om, *sq, *cross]                                         # 6 + 6 + 15 = 27
    return np.array(out, dtype=np.float64)


def physics_aware_observable(x: np.ndarray) -> np.ndarray:
    """Test-only observable that includes sin(theta), cos(theta), and bilinear-ready cross-products.

    NOT used as the B condition of E3 — it's an oracle for the EDMDc
    unit tests. We want to demonstrate that the *fitter* is correct on
    a Quadrotor by giving it observables that span the bilinear
    coupling, and then check the resulting MSE is much lower than the
    polynomial baseline's MSE.

    Layout: [state (6), sin th, cos th (2), v_x sin th, v_x cos th,
    v_z sin th, v_z cos th, omega sin th, omega cos th (6),
    sin(2 theta), cos(2 theta) (2)] = 16 dims.

    Returning an EDMDc observable like this is more generous than the
    LLM is allowed: the prompt will not contain any trig hints.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape[0] != 6:
        raise ValueError(
            f"physics_aware_observable expects 6-dim state; got {x.shape}"
        )

    px, pz, th, vx, vz, om = x
    s, c = np.sin(th), np.cos(th)
    s2, c2 = np.sin(2 * th), np.cos(2 * th)

    out = [
        px, pz, th, vx, vz, om,
        s, c,
        vx * s, vx * c,
        vz * s, vz * c,
        om * s, om * c,
        s2, c2,
    ]
    return np.array(out, dtype=np.float64)

"""Hand-written reference basis functions for Experiment 1 sanity checks.

The LLM-generated bases (P = Qwen-Plus, Q = Qwen3.5-4B) produce metrics
one to three orders of magnitude better than the polynomial baseline.
Before trusting that gap as a real scientific finding we need a
ground-truth reference point: what MSE does a **physically perfect**
basis achieve against the same data? If the perfect basis hits some
floor well below the LLM numbers, the LLM still has room to improve.
If the perfect basis gives the same numbers as the LLM, the LLM is
already at the noise floor.

Two reference bases are provided:

- ``perfect_quadrotor_basis_7(xu)`` — the minimal 7-feature library
  that exactly spans the continuous dynamics. STLSQ fit against
  finite-difference state derivatives should recover a near-diagonal
  coefficient matrix with known entries.

- ``perfect_quadrotor_basis_10(xu)`` — the same 7 features plus linear
  state terms (p_x, p_z, theta) that the true dynamics do NOT depend
  on. STLSQ should discard those extra features as redundant.

Dynamics (from ``quadrotor_dynamics.py``):

    p_x_dot   = v_x
    p_z_dot   = v_z
    theta_dot = omega
    v_x_dot   = -(u_1 + u_2) sin(theta) / m
    v_z_dot   =  (u_1 + u_2) cos(theta) / m - g
    omega_dot = (u_2 - u_1) L / I_yy

The minimal library that spans the right-hand sides is therefore:

    phi_0 = 1
    phi_1 = v_x
    phi_2 = v_z
    phi_3 = omega
    phi_4 = (u_1 + u_2) * sin(theta)
    phi_5 = (u_1 + u_2) * cos(theta)
    phi_6 = (u_2 - u_1)

With default physical parameters (m = 1.0, I_yy = 0.01, L = 0.25,
g = 9.81), the **expected STLSQ coefficients** are:

    p_x_dot   = 0   * phi_0 + 1    * phi_1 + 0    * phi_2 + 0    * phi_3 + 0    * phi_4 + 0    * phi_5 + 0    * phi_6
    p_z_dot   = 0   * phi_0 + 0    * phi_1 + 1    * phi_2 + 0    * phi_3 + 0    * phi_4 + 0    * phi_5 + 0    * phi_6
    theta_dot = 0   * phi_0 + 0    * phi_1 + 0    * phi_2 + 1    * phi_3 + 0    * phi_4 + 0    * phi_5 + 0    * phi_6
    v_x_dot   = 0   * phi_0 + 0    * phi_1 + 0    * phi_2 + 0    * phi_3 + -1.0 * phi_4 + 0    * phi_5 + 0    * phi_6
    v_z_dot   = -9.81 * phi_0 + 0  * phi_1 + 0    * phi_2 + 0    * phi_3 + 0    * phi_4 + 1.0  * phi_5 + 0    * phi_6
    omega_dot = 0   * phi_0 + 0    * phi_1 + 0    * phi_2 + 0    * phi_3 + 0    * phi_4 + 0    * phi_5 + 25.0 * phi_6

Or as a ``Xi`` matrix of shape ``(n_basis, n_state) = (7, 6)``:

    Xi = [
        [ 0.0,  0.0,  0.0,  0.0,  -9.81, 0.0 ],   # constant
        [ 1.0,  0.0,  0.0,  0.0,   0.0, 0.0 ],   # v_x
        [ 0.0,  1.0,  0.0,  0.0,   0.0, 0.0 ],   # v_z
        [ 0.0,  0.0,  1.0,  0.0,   0.0, 0.0 ],   # omega
        [ 0.0,  0.0,  0.0, -1.0,   0.0, 0.0 ],   # (u1+u2) sin(theta)
        [ 0.0,  0.0,  0.0,  0.0,   1.0, 0.0 ],   # (u1+u2) cos(theta)
        [ 0.0,  0.0,  0.0,  0.0,   0.0, 25.0],   # (u2-u1)
    ]

The tests in ``tests/test_reference_basis.py`` verify that a fit on
real generated data recovers this matrix to within the finite-difference
truncation error.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from me569_project.quadrotor_dynamics import QuadrotorParams


def perfect_quadrotor_basis_7(xu: np.ndarray) -> list[float]:
    """The 7-feature minimal basis that exactly spans the quadrotor dynamics.

    Ordering is fixed so the expected coefficient matrix (documented in
    the module docstring and checked in the tests) is reproducible:

        [1, v_x, v_z, omega, (u_1+u_2)*sin(theta), (u_1+u_2)*cos(theta), (u_2-u_1)]
    """
    xu_arr = np.asarray(xu, dtype=np.float64).reshape(-1)
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu_arr
    u_sum = u_1 + u_2
    u_diff = u_2 - u_1
    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))
    return [
        1.0,
        float(v_x),
        float(v_z),
        float(omega),
        float(u_sum) * sin_theta,
        float(u_sum) * cos_theta,
        float(u_diff),
    ]


PERFECT_BASIS_7_FEATURE_NAMES: list[str] = [
    "1",
    "v_x",
    "v_z",
    "omega",
    "(u_1+u_2) sin(theta)",
    "(u_1+u_2) cos(theta)",
    "(u_2-u_1)",
]


def perfect_quadrotor_basis_10(xu: np.ndarray) -> list[float]:
    """``perfect_quadrotor_basis_7`` plus three redundant linear state terms.

    The added features (``p_x``, ``p_z``, ``theta``) do NOT appear in the
    ground-truth dynamics. A correct STLSQ fit should drive their
    coefficients to exactly zero, demonstrating that the sparse
    regression can distinguish relevant from irrelevant features when
    the library contains both.
    """
    xu_arr = np.asarray(xu, dtype=np.float64).reshape(-1)
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu_arr
    u_sum = u_1 + u_2
    u_diff = u_2 - u_1
    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))
    return [
        1.0,
        float(v_x),
        float(v_z),
        float(omega),
        float(u_sum) * sin_theta,
        float(u_sum) * cos_theta,
        float(u_diff),
        float(p_x),
        float(p_z),
        float(theta),
    ]


PERFECT_BASIS_10_FEATURE_NAMES: list[str] = (
    PERFECT_BASIS_7_FEATURE_NAMES + ["p_x", "p_z", "theta"]
)


def expected_perfect_basis_7_Xi(params: QuadrotorParams) -> np.ndarray:
    """Return the known-correct Xi matrix that the perfect 7-feature basis
    should produce when fit against the ground-truth dynamics.

    Shape: (7, 6). Rows correspond to basis features, columns to state
    derivative components, matching the ``LLMBasisSysIDModel.Xi`` layout.
    """
    m = params.m
    I_yy = params.I_yy
    L = params.L
    g = params.g

    Xi = np.zeros((7, 6), dtype=np.float64)
    # constant phi_0 -> only v_z_dot (gravity)
    Xi[0, 4] = -g
    # v_x phi_1 -> p_x_dot
    Xi[1, 0] = 1.0
    # v_z phi_2 -> p_z_dot
    Xi[2, 1] = 1.0
    # omega phi_3 -> theta_dot
    Xi[3, 2] = 1.0
    # (u_1+u_2) sin(theta) phi_4 -> v_x_dot with coefficient -1/m
    Xi[4, 3] = -1.0 / m
    # (u_1+u_2) cos(theta) phi_5 -> v_z_dot with coefficient 1/m
    Xi[5, 4] = 1.0 / m
    # (u_2-u_1) phi_6 -> omega_dot with coefficient L/I_yy
    Xi[6, 5] = L / I_yy

    return Xi

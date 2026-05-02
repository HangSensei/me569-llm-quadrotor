"""P-condition basis function from the first real Qwen-Plus call.

Captured from the committed run in commit 711a681. Prompt: honest
SINDY_BASIS prompt without equation leak. Qwen-Plus backend: qwen-plus
alias (Qwen 3.6 per D005). Threshold used for the committed fit: 0.1.

Committed E1 metrics for this basis:
    n_basis     = 29    (one basis below; the extracted code had two
                         def blocks and Python's exec picked the last
                         one, which is reproduced verbatim here)
    active_terms= 12
    one_step_mse= 5.558e-01
    roll_mse_10 = 6.513e-04
    roll_mse_50 = 9.256e-03
"""
import numpy as np


def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    # Physical basis features for planar quadrotor dynamics:
    # - Constant term for bias/offset terms (e.g., gravity)
    # - Linear state terms for damping, inertia, etc.
    # - Trigonometric terms for thrust projection: cos(theta), sin(theta)
    # - Control terms: individual thrusts and differential thrust (u_1 - u_2) for torque
    # - Quadratic velocity terms for aerodynamic drag (common in robotics models)
    # - Coupling terms: thrust * sin/cos for acceleration components
    # - Angular acceleration depends on torque = L*(u_2 - u_1) and inertia
    # - No explicit position dependence in dynamics (no spatially varying forces)
    # - Avoid redundant high-order polynomials; prioritize physically motivated terms

    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    u_sum = u_1 + u_2
    u_diff = u_2 - u_1

    features = [
        1.0,                           # constant (for gravity, bias)
        ctheta,                        # cos(theta): appears in v_x_dot (thrust projection)
        stheta,                        # sin(theta): appears in v_z_dot (thrust projection)
        u_1,                           # left thrust
        u_2,                           # right thrust
        u_sum,                         # total thrust (affects vertical acceleration)
        u_diff,                        # differential thrust (torque input)
        v_x,                           # horizontal velocity (damping)
        v_z,                           # vertical velocity (damping)
        omega,                         # angular velocity (rotational damping)
        u_1 * ctheta,                  # left thrust projected horizontally
        u_2 * ctheta,                  # right thrust projected horizontally
        u_1 * stheta,                  # left thrust projected vertically
        u_2 * stheta,                  # right thrust projected vertically
        v_x * v_x,                     # quadratic drag in x
        v_z * v_z,                     # quadratic drag in z
        omega * omega,                 # quadratic rotational drag
        v_x * omega,                   # cross-coupling (Coriolis-like)
        v_z * omega,                   # cross-coupling (Coriolis-like)
        u_1 * v_x,                     # thrust-velocity coupling (drag modulation)
        u_2 * v_x,                     # thrust-velocity coupling
        u_1 * v_z,                     # thrust-velocity coupling
        u_2 * v_z,                     # thrust-velocity coupling
        u_diff * ctheta,               # differential thrust with cosine (torque-related coupling)
        u_diff * stheta,               # differential thrust with sine (torque-related coupling)
        theta,                         # small-angle approximation terms (linear in theta)
        v_x * theta,                   # velocity-angle coupling
        v_z * theta,                   # velocity-angle coupling
        omega * theta,                 # angular velocity-angle coupling
    ]

    return features

"""Q-condition basis function from the first real Qwen3.5-4B mlx-vlm call.

Captured from the committed run in commit ff0bb3b. Prompt: honest
SINDY_BASIS prompt without equation leak. Model: mlx-community/Qwen3.5-4B-MLX-8bit.
Threshold used for the committed fit: 0.1.

Committed E1 metrics for this basis:
    n_basis     = 29
    active_terms= 11
    one_step_mse= 7.868e-01
    roll_mse_10 = 5.998e-03
    roll_mse_50 = 4.543e-02

Note: the LLM output contained two ``def basis`` blocks (the first
partial, the second complete) which the extractor concatenated. Python
exec takes the last definition, which is the complete 29-feature
version reproduced here.
"""
import numpy as np


def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu

    # Physical constants
    g = 9.81
    m = 1.0
    I_yy = 0.01
    L = 0.25

    # Basic state and control features
    features = [
        1.0,  # constant term
        p_x,  # horizontal position
        p_z,  # vertical position
        v_x,  # horizontal velocity
        v_z,  # vertical velocity
        omega,  # angular velocity
        u_1,  # left thrust
        u_2,  # right thrust
    ]

    # Trigonometric projections for thrust coupling (critical for v_x_dot, v_z_dot)
    features.extend([
        np.sin(theta),
        np.cos(theta),
        np.sin(theta) ** 2,
        np.cos(theta) ** 2,
    ])

    # Control combinations (total thrust and differential thrust)
    features.extend([
        u_1 + u_2,  # total thrust (vertical component)
        u_1 - u_2,  # differential thrust (horizontal component and torque)
    ])

    # State-control interactions for potential nonlinearities
    features.extend([
        v_x * u_1,
        v_x * u_2,
        v_z * u_1,
        v_z * u_2,
        omega * u_1,
        omega * u_2,
    ])

    # Higher-order control terms (for saturation and nonlinear effects)
    features.extend([
        u_1 ** 2,
        u_2 ** 2,
        u_1 ** 3,
        u_2 ** 3,
    ])

    # Cross terms between control and state
    features.extend([
        p_x * u_1,
        p_x * u_2,
        p_z * u_1,
        p_z * u_2,
    ])

    # Gravity term (constant, but included for completeness)
    features.append(g)

    return features

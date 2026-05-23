"""P-condition basis from prompt variant 'minimal'. Saved by scripts/e1_prompt_ablation.py.
"""
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    return [
        1.0,
        p_x, p_z, theta, v_x, v_z, omega,
        u_1, u_2,
        p_x**2, p_z**2, theta**2, v_x**2, v_z**2, omega**2,
        u_1**2, u_2**2,
        p_x * p_z, p_x * theta, p_x * v_x, p_x * v_z, p_x * omega,
        p_z * theta, p_z * v_x, p_z * v_z, p_z * omega,
        theta * v_x, theta * v_z, theta * omega,
        v_x * v_z, v_x * omega,
        v_z * omega,
        p_x * u_1, p_x * u_2,
        p_z * u_1, p_z * u_2,
        theta * u_1, theta * u_2,
        v_x * u_1, v_x * u_2,
        v_z * u_1, v_z * u_2,
        omega * u_1, omega * u_2,
        np.sin(theta), np.cos(theta),
        u_1 * u_2
    ]

"""P-condition basis from prompt variant 'current'. Saved by scripts/e1_prompt_ablation.py.
"""
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    # Physical basis features for planar quadrotor dynamics:
    # - Constant term for bias/offset terms (e.g., gravity in v_z_dot)
    # - Linear state terms (e.g., drag, damping: v_x, v_z, omega)
    # - Trigonometric terms for thrust projection: cos(theta), sin(theta)
    # - Control terms: u_1, u_2 (thrust magnitudes)
    # - Coupling terms: u_1*cos(theta), u_2*cos(theta) → horizontal acceleration
    #                   u_1*sin(theta), u_2*sin(theta) → vertical acceleration
    # - Differential thrust for torque: (u_2 - u_1) → angular acceleration
    # - Quadratic velocity terms for aerodynamic drag (common modeling choice)
    # - Cross-coupling: v_x*omega, v_z*omega (Coriolis-like in rotating frame)
    # - Theta-related nonlinearities: theta itself (for small-angle approx), theta^2
    # - Products of controls and states: u_1*v_x, u_2*v_x, etc. (less likely but possible)
    # - We avoid high-order polynomials without physical motivation; prioritize physics-based terms.
    
    # Precompute common nonlinear functions
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    u_sum = u_1 + u_2
    u_diff = u_2 - u_1
    
    return [
        1.0,                           # constant (for gravity offset in v_z_dot, etc.)
        v_x,                           # linear drag in x
        v_z,                           # linear drag in z
        omega,                         # angular damping
        cos_th,                        # thrust projection to horizontal
        sin_th,                        # thrust projection to vertical
        u_1,                           # left thrust
        u_2,                           # right thrust
        u_sum * cos_th,                # total horizontal thrust component
        u_sum * sin_th,                # total vertical thrust component
        u_diff,                        # differential thrust → torque
        v_x**2,                        # quadratic drag in x
        v_z**2,                        # quadratic drag in z
        omega**2,                      # quadratic angular damping / centripetal
        v_x * omega,                   # Coriolis coupling (x-velocity × rotation)
        v_z * omega,                   # Coriolis coupling (z-velocity × rotation)
        theta,                         # small-angle approximation term (linear in orientation)
        u_1 * cos_th,                  # left rotor horizontal component
        u_2 * cos_th,                  # right rotor horizontal component
        u_1 * sin_th,                  # left rotor vertical component
        u_2 * sin_th,                  # right rotor vertical component
        v_x * v_z,                     # cross drag term (possible interaction)
        u_1 * v_x,                     # control-state coupling (e.g., induced velocity effects)
        u_2 * v_x,
        u_1 * v_z,
        u_2 * v_z,
        theta * v_x,                   # orientation-velocity coupling
        theta * v_z,
        theta * omega,
        u_diff * theta,                # torque modulation by attitude
    ]

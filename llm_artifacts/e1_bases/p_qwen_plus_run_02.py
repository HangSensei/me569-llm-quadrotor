"""P-condition basis from Qwen-Plus run 02. Temperature 0.3, max_tokens 1500. Saved by scripts/e1_multi_seed.py.
"""
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    # Physical basis features for planar quadrotor dynamics:
    # - Constant term for bias/offset terms (e.g., gravity)
    # - Linear state terms: velocities (damping), angular velocity (rotational damping)
    # - Trigonometric terms: sin(theta), cos(theta) for thrust projection coupling
    # - Control terms: individual thrusts and differential thrust (u_1 - u_2) for torque
    # - Thrust-weighted trig terms: u_1*cos(theta), u_1*sin(theta), u_2*cos(theta), u_2*sin(theta)
    # - Quadratic velocity terms: v_x^2, v_z^2, v_x*v_z, omega^2 (aerodynamic drag approximations)
    # - Cross terms: v_x*omega, v_z*omega (Coriolis-like effects in rotating frame)
    # - Differential thrust squared: (u_1 - u_2)^2 (torque-squared for nonlinear actuator effects)
    # - Product of trig and differential thrust: (u_1 - u_2)*sin(theta), (u_1 - u_2)*cos(theta)
    # - State-control interactions: theta*u_1, theta*u_2 (small-angle approximations extended)
    # - Gravity-related constant: g = 9.81 appears as constant offset in v_z_dot
    # We avoid high-order polynomials without physical motivation; prioritize physics-based terms.
    
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    u_diff = u_1 - u_2
    u_sum = u_1 + u_2
    
    return [
        1.0,                           # constant (for gravity, bias)
        v_x,                           # linear drag / inertia coupling
        v_z,                           # linear drag / inertia coupling
        omega,                         # angular velocity (rotational dynamics)
        sin_th,                        # pitch-dependent vertical thrust component
        cos_th,                        # pitch-dependent horizontal thrust component
        u_1,                           # left rotor thrust
        u_2,                           # right rotor thrust
        u_diff,                        # differential thrust → torque
        u_sum,                         # total thrust → vertical force
        u_1 * cos_th,                  # horizontal component of left thrust
        u_1 * sin_th,                  # vertical component of left thrust
        u_2 * cos_th,                  # horizontal component of right thrust
        u_2 * sin_th,                  # vertical component of right thrust
        u_diff * sin_th,               # coupling: torque * sin(theta) for lateral force
        u_diff * cos_th,               # coupling: torque * cos(theta) for longitudinal force
        v_x * v_x,                     # quadratic drag in x
        v_z * v_z,                     # quadratic drag in z
        omega * omega,                 # rotational drag / centripetal
        v_x * omega,                   # Coriolis-like coupling in body frame
        v_z * omega,                   # Coriolis-like coupling in body frame
        u_diff * u_diff,               # torque squared (nonlinear actuator/torque saturation effect)
        theta,                         # small-angle approximation term (linearized coupling)
        v_x * sin_th,                  # velocity-thrust direction coupling
        v_z * cos_th,                  # velocity-thrust direction coupling
        u_1 * theta,                   # linearized thrust-pitch interaction
        u_2 * theta,                   # linearized thrust-pitch interaction
        u_sum * sin_th,                # total thrust vertical component (redundant but helpful for SINDy)
        u_sum * cos_th,                # total thrust horizontal component (redundant but helpful for SINDy)
        v_x * v_z                      # cross-velocity drag term
    ]

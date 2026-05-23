"""P-condition basis from Qwen-Plus run 03. Temperature 0.3, max_tokens 1500. Saved by scripts/e1_multi_seed.py.
"""
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    # Physical parameters
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # Essential trigonometric terms for thrust projection
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Thrust sum and difference (physically meaningful combinations)
    u_sum = u_1 + u_2
    u_diff = u_1 - u_2
    
    # Linear velocity terms (damping is often linear in practice)
    # Angular velocity terms
    
    # Basis features selected based on physical principles:
    # - Constant term for bias/offset
    # - State variables for linear dynamics
    # - Trig functions for rotational coupling in translational dynamics
    # - Control inputs and their combinations (sum for lift, diff for torque)
    # - Quadratic velocity terms for aerodynamic drag (common approximation)
    # - Products of trig and control for projected thrust components
    # - Angular acceleration terms from torque
    # - Gravity term (constant)
    
    return [
        1.0,                           # bias term
        v_x,                           # linear drag-like term in x-dot
        v_z,                           # linear drag-like term in z-dot
        omega,                         # angular rate coupling
        sin_theta,                     # pitch-dependent horizontal force component
        cos_theta,                     # pitch-dependent vertical force component
        u_sum,                         # total thrust (affects vertical acceleration)
        u_diff,                        # differential thrust (affects angular acceleration)
        u_sum * sin_theta,             # horizontal component of total thrust
        u_sum * cos_theta,             # vertical component of total thrust
        u_diff * theta,                # torque proportional to pitch angle (small-angle approx)
        v_x * v_x,                     # quadratic drag in x
        v_z * v_z,                     # quadratic drag in z
        omega * omega,                 # rotational drag / centripetal effects
        v_x * omega,                   # cross-coupling between translation and rotation
        v_z * omega,                   # cross-coupling between translation and rotation
        theta,                         # small-angle restoring or coupling
        theta * theta,                 # nonlinear angular coupling
        u_1,                           # individual rotor thrust (may appear in some terms)
        u_2,                           # individual rotor thrust
        v_x * sin_theta,               # velocity-modulated thrust projection
        v_z * cos_theta,               # velocity-modulated thrust projection
        omega * sin_theta,             # angular rate modulated by orientation
        omega * cos_theta,             # angular rate modulated by orientation
        u_1 * sin_theta,               # left rotor horizontal component
        u_2 * sin_theta,               # right rotor horizontal component
        u_1 * cos_theta,               # left rotor vertical component
        u_2 * cos_theta,               # right rotor vertical component
        g,                             # gravitational acceleration constant
        u_sum * theta,                 # linearized thrust coupling with angle
    ]

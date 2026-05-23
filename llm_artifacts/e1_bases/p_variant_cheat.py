"""P-condition basis from prompt variant 'cheat'. Saved by scripts/e1_prompt_ablation.py.
"""
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    # Physical basis features for planar quadrotor dynamics
    # Based on known physics: gravity, thrust projection (sin/cos), differential thrust torque
    # Also include relevant linear terms and constants for completeness
    
    # Constant term for bias/offset terms
    const = 1.0
    
    # State variables (linear terms)
    px, pz, th, vx, vz, om = p_x, p_z, theta, v_x, v_z, omega
    
    # Control inputs
    u1, u2 = u_1, u_2
    
    # Trigonometric terms for thrust projection
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    
    # Sum and difference of thrusts (physically meaningful combinations)
    u_sum = u1 + u2
    u_diff = u2 - u1
    
    # Products of thrusts with trig functions (appear in v_x_dot and v_z_dot)
    u_sum_sin = u_sum * sin_th
    u_sum_cos = u_sum * cos_th
    
    # Linear velocity terms (for possible drag/damping, though not in ground truth,
    # included for robustness as SINDy may need to identify zero coefficients)
    vx_squared = vx * vx
    vz_squared = vz * vz
    om_squared = om * om
    
    # Cross terms between velocities and angles (possible higher-order effects)
    vx_sin = vx * sin_th
    vz_cos = vz * cos_th
    om_th = om * theta
    
    # Control squared terms (for saturation effects, though not in ground truth)
    u1_sq = u1 * u1
    u2_sq = u2 * u2
    
    # Combined terms that capture physical couplings
    u1_sin = u1 * sin_th
    u2_sin = u2 * sin_th
    u1_cos = u1 * cos_th
    u2_cos = u2 * cos_th
    
    # Return basis features that span the true dynamics:
    # p_x_dot = v_x                    -> needs vx
    # p_z_dot = v_z                    -> needs vz
    # theta_dot = omega                -> needs om
    # v_x_dot = -(u1+u2)*sin(theta)/m  -> needs u_sum_sin
    # v_z_dot = (u1+u2)*cos(theta)/m - g -> needs u_sum_cos, const
    # omega_dot = (u2-u1)*L/I_yy       -> needs u_diff
    
    # We include all physically motivated terms, plus some common extensions
    return [
        const,           # constant term (for gravity offset in v_z_dot)
        vx,              # p_x_dot
        vz,              # p_z_dot  
        om,              # theta_dot
        u_sum_sin,       # v_x_dot coupling
        u_sum_cos,       # v_z_dot coupling
        u_diff,          # omega_dot coupling
        sin_th,          # pure angle dependence
        cos_th,          # pure angle dependence
        u1,              # individual thrust (may help with identification)
        u2,              # individual thrust
        u1_sin,          # u1*sin(theta)
        u2_sin,          # u2*sin(theta)
        u1_cos,          # u1*cos(theta)
        u2_cos,          # u2*cos(theta)
        vx_squared,      # possible quadratic damping
        vz_squared,      # possible quadratic damping
        om_squared,      # possible quadratic damping
        theta,           # linear angle term
        vx * om,         # cross term
        vz * om,         # cross term
        u_sum,           # total thrust (baseline)
        u_diff * sin_th, # differential thrust with angle
        u_diff * cos_th, # differential thrust with angle
        u1 * u2,         # thrust interaction
        th * u_sum,      # angle-thrust coupling
        vx * sin_th,     # velocity-angle coupling
        vz * cos_th,     # velocity-angle coupling
    ]

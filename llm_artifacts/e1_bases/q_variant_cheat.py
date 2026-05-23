"""Q-condition basis from prompt variant 'cheat'. Saved by scripts/e1_prompt_ablation.py.
"""
import numpy as np

def basis(xu):
    """
    Generate basis features for SINDy identification of planar quadrotor dynamics.
    
    The basis spans the right-hand sides of the continuous-time dynamics:
    - p_x_dot = v_x
    - p_z_dot = v_z
    - theta_dot = omega
    - v_x_dot = -(u_1 + u_2) * sin(theta) / m
    - v_z_dot = (u_1 + u_2) * cos(theta) / m - g
    - omega_dot = (u_2 - u_1) * L / I_yy
    
    Parameters:
    xu: numpy array of length 8 [p_x, p_z, theta, v_x, v_z, omega, u_1, u_2]
    
    Returns:
    list of basis feature values
    """
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    
    # Physical constants (will be learned as coefficients by SINDy)
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # State variables (for position and velocity derivatives)
    features = [
        1.0,  # constant term for bias
        p_x,  # position x
        p_z,  # position z
        theta,  # pitch angle
        v_x,  # horizontal velocity
        v_z,  # vertical velocity
        omega,  # angular velocity
    ]
    
    # Thrust combinations
    features.extend([
        u_1,  # left rotor thrust
        u_2,  # right rotor thrust
        u_1 + u_2,  # total thrust
        u_2 - u_1,  # differential thrust (torque)
    ])
    
    # Trigonometric functions of pitch angle
    features.extend([
        np.sin(theta),  # for v_x dynamics
        np.cos(theta),  # for v_z dynamics
    ])
    
    # Product terms (thrust * trig) - critical for v_x_dot and v_z_dot
    features.extend([
        (u_1 + u_2) * np.sin(theta),  # v_x dynamics numerator
        (u_1 + u_2) * np.cos(theta),  # v_z dynamics numerator
    ])
    
    # Additional polynomial terms for potential nonlinearities
    features.extend([
        v_x * v_z,  # cross-coupling
        v_x * omega,  # gyroscopic coupling
        v_z * omega,  # gyroscopic coupling
        theta * omega,  # pitch-velocity coupling
        u_1 * u_2,  # thrust interaction
    ])
    
    # Higher-order terms for robustness
    features.extend([
        np.sin(theta) ** 2,  # squared sine
        np.cos(theta) ** 2,  # squared cosine
        np.sin(theta) * np.cos(theta),  # sin*cos coupling
    ])
    
    return features

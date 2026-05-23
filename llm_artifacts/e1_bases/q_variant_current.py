"""Q-condition basis from prompt variant 'current'. Saved by scripts/e1_prompt_ablation.py.
"""
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    
    # Physical constants
    g = 9.81
    m = 1.0
    I_yy = 0.01
    L = 0.25
    
    # Basic features
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
    
    # Trigonometric projections for thrust coupling
    features.extend([
        np.sin(theta),
        np.cos(theta),
        np.sin(theta)**2,
        np.cos(theta)**2,
    ])
    
    # Control combinations
    features.extend([
        u_1 + u_2,  # total thrust
        u_1 - u_2,  # differential thrust
        u_1 * u_2,  # interaction term
    ])
    
    # State-control interactions
    features.extend([
        v_x * u_1,
        v_x * u_2,
        v_z * u_1,
        v_z * u_2,
        omega * u_1,
        omega * u_2,
    ])
    
    # Higher-order terms
    features.extend([
        u_1**2,
        u_2**2,
        u_1**3,
        u_2**3,
    ])
    
    return features

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
        np.sin(theta)**2,
        np.cos(theta)**2,
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
        u_1**2,
        u_2**2,
        u_1**3,
        u_2**3,
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

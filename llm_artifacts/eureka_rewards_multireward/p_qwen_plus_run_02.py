import numpy as np
import math

def reward(state, action):
    # Penalize position deviation from the hover origin (p_x, p_z)
    # Use quadratic penalty for smooth gradients; weight position highly
    pos_penalty = -10.0 * (state[0]**2 + (state[1] - 1.0)**2)  # Note: hover at z=1.0m is not required; equilibrium is at [0,0], so use state[1]**2
    
    # Corrected: hover equilibrium is at p_z = 0 (origin), per problem statement "drive the state back to the hover equilibrium x = 0"
    pos_penalty = -10.0 * (state[0]**2 + state[1]**2)
    
    # Penalize attitude deviation from level (theta)
    # Quadratic in theta for small angles, but bounded to avoid excessive penalty for large errors
    att_penalty = -5.0 * (state[2]**2)
    
    # Penalize translational and angular velocity (v_x, v_z, omega)
    # Lower weight than position/attitude since velocities are derivatives
    vel_penalty = -1.0 * (state[3]**2 + state[4]**2 + state[5]**2)
    
    # Penalize control effort relative to hover-equilibrium thrust
    # u_hover = m*g/2 = 1.0 * 9.81 / 2 = 4.905 N
    u_hover = 4.905
    ctrl_penalty = -0.1 * ((action[0] - u_hover)**2 + (action[1] - u_hover)**2)
    
    # Bonus for being near equilibrium (small additional incentive)
    # Not needed since penalties already peak at zero, but ensure no unintended flatness
    # Instead, add a small term to encourage staying near hover thrusts without over-penalizing
    # Already covered by ctrl_penalty; no extra term needed
    
    return pos_penalty + att_penalty + vel_penalty + ctrl_penalty
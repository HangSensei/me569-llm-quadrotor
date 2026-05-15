import numpy as np
import math

def reward(state, action):
    # Penalize position deviation from the hover origin (quadratic penalty)
    pos_x = state[0]
    pos_z = state[1]
    pos_penalty = -1.0 * (pos_x**2 + (pos_z - 0.0)**2)  # target z=0 for hover
    
    # Penalize attitude deviation from level (quadratic penalty on theta)
    theta = state[2]
    att_penalty = -2.0 * theta**2
    
    # Penalize translational and angular velocity (quadratic penalties)
    v_x = state[3]
    v_z = state[4]
    omega = state[5]
    vel_penalty = -0.5 * (v_x**2 + v_z**2 + 0.1 * omega**2)
    
    # Penalize control effort relative to hover-equilibrium thrust (u_hover = 4.905 N)
    u_hover = 4.905
    u1 = action[0]
    u2 = action[1]
    ctrl_penalty = -0.02 * ((u1 - u_hover)**2 + (u2 - u_hover)**2)
    
    # Bonus for being near equilibrium (small positive term to encourage convergence)
    # Use smooth Gaussian-like bonus centered at origin (but avoid large magnitude)
    equilibrium_bonus = 0.1 * np.exp(-0.5 * (pos_x**2 + pos_z**2 + theta**2 + 
                                             0.1*v_x**2 + 0.1*v_z**2 + 0.01*omega**2))
    
    return pos_penalty + att_penalty + vel_penalty + ctrl_penalty + equilibrium_bonus
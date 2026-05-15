import numpy as np
import math

def reward(state, action):
    # Penalize position deviation from the hover origin
    pos = -10.0 * (state[0]**2 + (state[1] - 0.0)**2)
    # Penalize attitude deviation from level
    att = -20.0 * (state[2]**2)
    # Penalize translational and angular velocity
    vel = -5.0 * (state[3]**2 + state[4]**2 + 0.1 * state[5]**2)
    # Penalize control effort relative to the hover-equilibrium thrust
    u_hover = 4.905
    ctrl = -0.1 * ((action[0] - u_hover)**2 + (action[1] - u_hover)**2)
    return pos + att + vel + ctrl
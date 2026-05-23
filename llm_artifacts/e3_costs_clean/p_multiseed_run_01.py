"""P-condition stage cost from multi-seed run 01. Qwen-Plus call at temperature 0.3. Saved by e3_multi_seed.py.
"""
def stage_cost(x, u):
    # Penalize position deviation from the origin (p_x, p_z)
    # Use quadratic penalty: stronger weight on vertical position since gravity dominates
    pos = 10.0 * x[0]**2 + 20.0 * x[1]**2
    
    # Penalize pitch deviation from level (theta)
    # Use quadratic penalty but moderate weight to allow necessary tilting for horizontal motion
    att = 5.0 * x[2]**2
    
    # Penalize translational and angular velocity (v_x, v_z, omega)
    # Weaker than position penalties, but v_z slightly weighted more due to gravity coupling
    vel = 2.0 * x[3]**2 + 4.0 * x[4]**2 + 1.0 * x[5]**2
    
    # Penalize control effort relative to hover-equilibrium thrust (u_hover = m*g/2 = 4.905)
    u_hover = 4.905
    ctrl = 0.1 * ((u[0] - u_hover)**2 + (u[1] - u_hover)**2)
    
    return pos + att + vel + ctrl

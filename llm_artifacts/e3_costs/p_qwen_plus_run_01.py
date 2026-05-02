"""P-condition stage cost from Qwen-Plus run 01 (Qwen 3.6). Saved by scripts/run_e3_full.py.
"""
def stage_cost(x, u):
    # Position errors: penalize horizontal and vertical deviation from hover (origin)
    # p_z error is critical for altitude hold; p_x less critical but still important
    pos_cost = 15.0 * x[0]**2 + 20.0 * x[1]**2
    
    # Attitude error: penalize pitch angle, but allow moderate tilting for horizontal motion
    # Use theta^2 (not sin/cos) for smoothness and convexity near equilibrium
    att_cost = 8.0 * x[2]**2
    
    # Velocity errors: penalize velocities less than positions, but angular velocity slightly more
    # since uncontrolled rotation can destabilize quickly
    vel_cost = 2.0 * x[3]**2 + 2.0 * x[4]**2 + 3.0 * x[5]**2
    
    # Control effort: penalize deviation from hover thrust (u_hover = m*g/2 = 4.905 N)
    # Light penalty to allow necessary differential thrust for control
    u_hover = 4.905
    ctrl_cost = 0.05 * (u[0] - u_hover)**2 + 0.05 * (u[1] - u_hover)**2
    
    return pos_cost + att_cost + vel_cost + ctrl_cost

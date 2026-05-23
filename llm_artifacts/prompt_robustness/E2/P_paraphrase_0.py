"""P E2 paraphrase 0. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def stage_cost(x, u):
#     # Position penalties (p_x, p_z) - penalize more heavily
#     pos_penalty = 10.0 * (x[0]**2 + x[1]**2)
#     
#     # Attitude penalty (theta) - moderate penalty to allow tilting
#     att_penalty = 2.0 * x[2]**2
#     
#     # Velocity penalties (v_x, v_z, omega) - penalize less than position
#     vel_penalty = 1.0 * (x[3]**2 + x[4]**2) + 0.5 * x[5]**2
#     
#     # Control effort penalty - light penalty
#     # Reference hover thrusts: u1_ref = u2_ref = m*g/2 = 4.905
#     u1_ref = 4.905
#     u2_ref = 4.905
#     ctrl_penalty = 0.01 * ((u[0] - u1_ref)**2 + (u[1] - u2_ref)**2)
#     
#     return pos_penalty + att_penalty + vel_penalty + ctrl_penalty
# ```
# --- extracted code ---
def stage_cost(x, u):
    # Position penalties (p_x, p_z) - penalize more heavily
    pos_penalty = 10.0 * (x[0]**2 + x[1]**2)
    
    # Attitude penalty (theta) - moderate penalty to allow tilting
    att_penalty = 2.0 * x[2]**2
    
    # Velocity penalties (v_x, v_z, omega) - penalize less than position
    vel_penalty = 1.0 * (x[3]**2 + x[4]**2) + 0.5 * x[5]**2
    
    # Control effort penalty - light penalty
    # Reference hover thrusts: u1_ref = u2_ref = m*g/2 = 4.905
    u1_ref = 4.905
    u2_ref = 4.905
    ctrl_penalty = 0.01 * ((u[0] - u1_ref)**2 + (u[1] - u2_ref)**2)
    
    return pos_penalty + att_penalty + vel_penalty + ctrl_penalty

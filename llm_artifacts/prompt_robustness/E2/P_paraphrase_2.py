"""P E2 paraphrase 2. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def stage_cost(x, u):
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     eq_u = m * g / 2.0
#     
#     # Position and attitude deviations (higher weight)
#     pos_cost = 10.0 * (x[0] * x[0] + x[1] * x[1]) + 5.0 * x[2] * x[2]
#     
#     # Velocity deviations (lower weight)
#     vel_cost = 1.0 * (x[3] * x[3] + x[4] * x[4]) + 0.5 * x[5] * x[5]
#     
#     # Control effort (gentle penalty)
#     ctrl_cost = 0.01 * ((u[0] - eq_u) * (u[0] - eq_u) + (u[1] - eq_u) * (u[1] - eq_u))
#     
#     return pos_cost + vel_cost + ctrl_cost
# ```
# --- extracted code ---
def stage_cost(x, u):
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    eq_u = m * g / 2.0
    
    # Position and attitude deviations (higher weight)
    pos_cost = 10.0 * (x[0] * x[0] + x[1] * x[1]) + 5.0 * x[2] * x[2]
    
    # Velocity deviations (lower weight)
    vel_cost = 1.0 * (x[3] * x[3] + x[4] * x[4]) + 0.5 * x[5] * x[5]
    
    # Control effort (gentle penalty)
    ctrl_cost = 0.01 * ((u[0] - eq_u) * (u[0] - eq_u) + (u[1] - eq_u) * (u[1] - eq_u))
    
    return pos_cost + vel_cost + ctrl_cost

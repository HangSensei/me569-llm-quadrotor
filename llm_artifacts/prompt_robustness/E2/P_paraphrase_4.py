"""P E2 paraphrase 4. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def stage_cost(x, u):
#     # Hover equilibrium: x = [0, 0, 0, 0, 0, 0], u = [4.905, 4.905]
#     # Position errors (dominant)
#     pos_err = x[0]*x[0] + x[1]*x[1]
#     # Pitch error (moderate, not over-penalized)
#     pitch_err = x[2]*x[2]
#     # Velocity errors (subordinate to position/pitch)
#     vel_err = x[3]*x[3] + x[4]*x[4] + x[5]*x[5]
#     # Control effort (small penalty)
#     u1_eq = 4.905
#     u2_eq = 4.905
#     ctrl_err = (u[0] - u1_eq)*(u[0] - u1_eq) + (u[1] - u2_eq)*(u[1] - u2_eq)
#     # Weighted sum: position dominates, then pitch, then velocities, then control
#     return 10.0*pos_err + 2.0*pitch_err + 1.0*vel_err + 0.01*ctrl_err
# ```
# --- extracted code ---
def stage_cost(x, u):
    # Hover equilibrium: x = [0, 0, 0, 0, 0, 0], u = [4.905, 4.905]
    # Position errors (dominant)
    pos_err = x[0]*x[0] + x[1]*x[1]
    # Pitch error (moderate, not over-penalized)
    pitch_err = x[2]*x[2]
    # Velocity errors (subordinate to position/pitch)
    vel_err = x[3]*x[3] + x[4]*x[4] + x[5]*x[5]
    # Control effort (small penalty)
    u1_eq = 4.905
    u2_eq = 4.905
    ctrl_err = (u[0] - u1_eq)*(u[0] - u1_eq) + (u[1] - u2_eq)*(u[1] - u2_eq)
    # Weighted sum: position dominates, then pitch, then velocities, then control
    return 10.0*pos_err + 2.0*pitch_err + 1.0*vel_err + 0.01*ctrl_err

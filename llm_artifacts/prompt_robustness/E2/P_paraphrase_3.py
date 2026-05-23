"""P E2 paraphrase 3. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def stage_cost(x, u):
#     px = x[0]
#     pz = x[1]
#     theta = x[2]
#     vx = x[3]
#     vz = x[4]
#     omega = x[5]
#     u1 = u[0]
#     u2 = u[1]
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     hover_thrust = m * g / 2.0
#     # Position and attitude errors (heavily penalized)
#     pos_att_err = px**2 + pz**2 + theta**2
#     # Velocity errors (moderately penalized)
#     vel_err = vx**2 + vz**2 + omega**2
#     # Control effort (lightly penalized, but avoid excessive thrust)
#     ctrl_err = (u1 - hover_thrust)**2 + (u2 - hover_thrust)**2
#     # Penalize extreme tilt to keep feasible (soft constraint via cos(theta) penalty)
#     # When theta is large, cos(theta) drops, so 1 - cos(theta) grows
#     tilt_penalty = 1.0 - cos(theta)
#     # Combine with appropriate weights
#     # Position/attitude: weight 10.0
#     # Velocity: weight 1.0
#     # Control: weight 0.01
#     # Tilt penalty: weight 5.0
#     return 10.0 * pos_att_err + 1.0 * vel_err + 0.01 * ctrl_err + 5.0 * tilt_penalty
# ```
# --- extracted code ---
def stage_cost(x, u):
    px = x[0]
    pz = x[1]
    theta = x[2]
    vx = x[3]
    vz = x[4]
    omega = x[5]
    u1 = u[0]
    u2 = u[1]
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    hover_thrust = m * g / 2.0
    # Position and attitude errors (heavily penalized)
    pos_att_err = px**2 + pz**2 + theta**2
    # Velocity errors (moderately penalized)
    vel_err = vx**2 + vz**2 + omega**2
    # Control effort (lightly penalized, but avoid excessive thrust)
    ctrl_err = (u1 - hover_thrust)**2 + (u2 - hover_thrust)**2
    # Penalize extreme tilt to keep feasible (soft constraint via cos(theta) penalty)
    # When theta is large, cos(theta) drops, so 1 - cos(theta) grows
    tilt_penalty = 1.0 - cos(theta)
    # Combine with appropriate weights
    # Position/attitude: weight 10.0
    # Velocity: weight 1.0
    # Control: weight 0.01
    # Tilt penalty: weight 5.0
    return 10.0 * pos_att_err + 1.0 * vel_err + 0.01 * ctrl_err + 5.0 * tilt_penalty

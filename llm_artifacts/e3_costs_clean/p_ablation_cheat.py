"""P-condition prompt ablation variant 'cheat'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# ```python
# def stage_cost(x, u):
#     # Penalize position deviation from origin (p_x, p_z) with stronger weight on vertical position
#     pos = 10.0 * x[0]**2 + 20.0 * x[1]**2
#     # Penalize pitch angle deviation (theta) — use sin^2 or theta^2; theta^2 is smooth and appropriate for small angles
#     att = 5.0 * x[2]**2
#     # Penalize velocities: horizontal/vertical velocity and angular velocity
#     vel = 2.0 * x[3]**2 + 2.0 * x[4]**2 + 1.0 * x[5]**2
#     # Penalize control effort relative to hover equilibrium thrust (u_hover = m*g/2 = 4.905)
#     u_hover = 4.905
#     ctrl = 0.1 * ((u[0] - u_hover)**2 + (u[1] - u_hover)**2)
#     return pos + att + vel + ctrl
# ```
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
    # Penalize position deviation from origin (p_x, p_z) with stronger weight on vertical position
    pos = 10.0 * x[0]**2 + 20.0 * x[1]**2
    # Penalize pitch angle deviation (theta) — use sin^2 or theta^2; theta^2 is smooth and appropriate for small angles
    att = 5.0 * x[2]**2
    # Penalize velocities: horizontal/vertical velocity and angular velocity
    vel = 2.0 * x[3]**2 + 2.0 * x[4]**2 + 1.0 * x[5]**2
    # Penalize control effort relative to hover equilibrium thrust (u_hover = m*g/2 = 4.905)
    u_hover = 4.905
    ctrl = 0.1 * ((u[0] - u_hover)**2 + (u[1] - u_hover)**2)
    return pos + att + vel + ctrl

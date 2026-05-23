"""P-condition prompt ablation variant 'cheat'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# ```python
# def stage_cost(x, u):
#     # Position errors: penalize horizontal and vertical deviation from origin
#     pos_cost = 15.0 * x[0]**2 + 15.0 * x[1]**2
#     
#     # Attitude error: penalize pitch deviation, but less than position to allow tilting for translation
#     att_cost = 3.0 * x[2]**2
#     
#     # Velocity errors: penalize velocities moderately, with angular velocity slightly weighted
#     vel_cost = 2.0 * x[3]**2 + 2.0 * x[4]**2 + 1.0 * x[5]**2
#     
#     # Control effort: penalize deviation from hover thrust (u_hover = m*g/2 = 4.905 N)
#     # Use small weights to allow sufficient control authority while preferring efficient solutions
#     u_hover = 4.905
#     ctrl_cost = 0.05 * (u[0] - u_hover)**2 + 0.05 * (u[1] - u_hover)**2
#     
#     # Additional penalty on total thrust deviation to encourage balanced operation
#     # and avoid excessive total thrust that could cause instability
#     total_thrust = u[0] + u[1]
#     hover_total = 2.0 * u_hover  # = m*g = 9.81 N
#     thrust_balance_cost = 0.01 * (total_thrust - hover_total)**2
#     
#     return pos_cost + att_cost + vel_cost + ctrl_cost + thrust_balance_cost
# ```
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
    # Position errors: penalize horizontal and vertical deviation from origin
    pos_cost = 15.0 * x[0]**2 + 15.0 * x[1]**2
    
    # Attitude error: penalize pitch deviation, but less than position to allow tilting for translation
    att_cost = 3.0 * x[2]**2
    
    # Velocity errors: penalize velocities moderately, with angular velocity slightly weighted
    vel_cost = 2.0 * x[3]**2 + 2.0 * x[4]**2 + 1.0 * x[5]**2
    
    # Control effort: penalize deviation from hover thrust (u_hover = m*g/2 = 4.905 N)
    # Use small weights to allow sufficient control authority while preferring efficient solutions
    u_hover = 4.905
    ctrl_cost = 0.05 * (u[0] - u_hover)**2 + 0.05 * (u[1] - u_hover)**2
    
    # Additional penalty on total thrust deviation to encourage balanced operation
    # and avoid excessive total thrust that could cause instability
    total_thrust = u[0] + u[1]
    hover_total = 2.0 * u_hover  # = m*g = 9.81 N
    thrust_balance_cost = 0.01 * (total_thrust - hover_total)**2
    
    return pos_cost + att_cost + vel_cost + ctrl_cost + thrust_balance_cost

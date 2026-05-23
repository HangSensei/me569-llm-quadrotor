"""P E4 paraphrase 2. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def reward(state, action):
#     import numpy as np
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     
#     p_x, p_z, theta, v_x, v_z, omega = state
#     u_1, u_2 = action
#     
#     # Target hover thrusts (each rotor supports half weight plus margin for control)
#     u_hover = m * g / 2.0  # 4.905 N
#     
#     # Position penalty: quadratic penalty on horizontal and vertical deviation
#     # Emphasize z-position (altitude) slightly more than x-position
#     pos_penalty = -0.5 * (p_x**2 + 1.5 * p_z**2)
#     
#     # Attitude penalty: quadratic penalty on angle and angular velocity
#     # Prioritize small angles (theta near 0) and low angular rates
#     attitude_penalty = -0.3 * (theta**2 + 0.5 * omega**2)
#     
#     # Velocity penalty: lighter penalty than position/attitude, but still present
#     vel_penalty = -0.1 * (v_x**2 + v_z**2)
#     
#     # Thrust penalty: smooth quadratic penalty around hover thrust
#     # Encourage both rotors to be near u_hover, with moderate penalty scaling
#     thrust_penalty = -0.05 * ((u_1 - u_hover)**2 + (u_2 - u_hover)**2)
#     
#     # Small bonus for being near zero angular velocity and small angle (smooth incentive)
#     # This is already covered in attitude_penalty, so no additional bonus needed
#     
#     return pos_penalty + attitude_penalty + vel_penalty + thrust_penalty
# ```
# --- extracted code ---
def reward(state, action):
    import numpy as np
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    p_x, p_z, theta, v_x, v_z, omega = state
    u_1, u_2 = action
    
    # Target hover thrusts (each rotor supports half weight plus margin for control)
    u_hover = m * g / 2.0  # 4.905 N
    
    # Position penalty: quadratic penalty on horizontal and vertical deviation
    # Emphasize z-position (altitude) slightly more than x-position
    pos_penalty = -0.5 * (p_x**2 + 1.5 * p_z**2)
    
    # Attitude penalty: quadratic penalty on angle and angular velocity
    # Prioritize small angles (theta near 0) and low angular rates
    attitude_penalty = -0.3 * (theta**2 + 0.5 * omega**2)
    
    # Velocity penalty: lighter penalty than position/attitude, but still present
    vel_penalty = -0.1 * (v_x**2 + v_z**2)
    
    # Thrust penalty: smooth quadratic penalty around hover thrust
    # Encourage both rotors to be near u_hover, with moderate penalty scaling
    thrust_penalty = -0.05 * ((u_1 - u_hover)**2 + (u_2 - u_hover)**2)
    
    # Small bonus for being near zero angular velocity and small angle (smooth incentive)
    # This is already covered in attitude_penalty, so no additional bonus needed
    
    return pos_penalty + attitude_penalty + vel_penalty + thrust_penalty

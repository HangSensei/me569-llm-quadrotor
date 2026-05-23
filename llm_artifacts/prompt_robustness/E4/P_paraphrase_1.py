"""P E4 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# import math
# 
# def reward(state, action):
#     p_x, p_z, theta, v_x, v_z, omega = state
#     u_1, u_2 = action
#     
#     # Constants
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     hover_thrust = m * g / 2.0  # 4.905 N each
#     
#     # Position and attitude errors (heavily weighted)
#     pos_error_sq = p_x**2 + (p_z - 0.0)**2  # z-reference is 0 for hover
#     angle_error_sq = theta**2
#     
#     # Velocity errors (lighter weight)
#     vel_error_sq = v_x**2 + v_z**2 + omega**2
#     
#     # Thrust deviation penalty (quadratic, centered at hover_thrust)
#     thrust_error_sq = (u_1 - hover_thrust)**2 + (u_2 - hover_thrust)**2
#     
#     # Smooth penalty for extreme thrusts (prevent numerical instability and unrealistic control)
#     # Soft saturation: penalize |u_i| > 15 N quadratically beyond threshold
#     thrust_sat_penalty = 0.0
#     for u in [u_1, u_2]:
#         if abs(u) > 15.0:
#             thrust_sat_penalty += (abs(u) - 15.0)**2
#     
#     # Composite reward: maximize negative penalties (so larger = better)
#     # Use strong weighting on position/attitude, moderate on velocity, lighter on thrust error
#     reward_val = (
#         -10.0 * pos_error_sq
#         -5.0 * angle_error_sq
#         -0.5 * vel_error_sq
#         -0.1 * thrust_error_sq
#         -0.05 * thrust_sat_penalty
#     )
#     
#     return float(reward_val)
# ```
# --- extracted code ---
import numpy as np
import math

def reward(state, action):
    p_x, p_z, theta, v_x, v_z, omega = state
    u_1, u_2 = action
    
    # Constants
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    hover_thrust = m * g / 2.0  # 4.905 N each
    
    # Position and attitude errors (heavily weighted)
    pos_error_sq = p_x**2 + (p_z - 0.0)**2  # z-reference is 0 for hover
    angle_error_sq = theta**2
    
    # Velocity errors (lighter weight)
    vel_error_sq = v_x**2 + v_z**2 + omega**2
    
    # Thrust deviation penalty (quadratic, centered at hover_thrust)
    thrust_error_sq = (u_1 - hover_thrust)**2 + (u_2 - hover_thrust)**2
    
    # Smooth penalty for extreme thrusts (prevent numerical instability and unrealistic control)
    # Soft saturation: penalize |u_i| > 15 N quadratically beyond threshold
    thrust_sat_penalty = 0.0
    for u in [u_1, u_2]:
        if abs(u) > 15.0:
            thrust_sat_penalty += (abs(u) - 15.0)**2
    
    # Composite reward: maximize negative penalties (so larger = better)
    # Use strong weighting on position/attitude, moderate on velocity, lighter on thrust error
    reward_val = (
        -10.0 * pos_error_sq
        -5.0 * angle_error_sq
        -0.5 * vel_error_sq
        -0.1 * thrust_error_sq
        -0.05 * thrust_sat_penalty
    )
    
    return float(reward_val)

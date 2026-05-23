"""P E4 paraphrase 4. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# import math
# 
# def reward(state, action):
#     p_x, p_z, theta, v_x, v_z, omega = state
#     u_1, u_2 = action
#     
#     # Physical constants
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     hover_thrust_per_rotor = 4.905  # m*g/2 = 1.0*9.81/2
#     
#     # Position and attitude errors (penalize deviation from origin and level orientation)
#     pos_error = p_x**2 + p_z**2
#     theta_error = theta**2  # small-angle approximation valid for penalization
#     
#     # Velocity errors (lower priority than position/attitude)
#     vel_error = v_x**2 + v_z**2 + omega**2
#     
#     # Thrust deviation penalty (smooth penalty for deviating from hover thrust)
#     thrust_deviation = (u_1 - hover_thrust_per_rotor)**2 + (u_2 - hover_thrust_per_rotor)**2
#     
#     # Combine with appropriate weights to prioritize position/attitude > velocity > thrust
#     # Use negative signs since we want to maximize reward (i.e., minimize penalties)
#     reward_val = (
#         -1.0 * pos_error          # Strong penalty for position error
#         -0.5 * theta_error       # Moderate penalty for attitude error
#         -0.1 * vel_error         # Light penalty for velocity error
#         -0.05 * thrust_deviation # Gentle penalty for thrust deviation
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
    
    # Physical constants
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    hover_thrust_per_rotor = 4.905  # m*g/2 = 1.0*9.81/2
    
    # Position and attitude errors (penalize deviation from origin and level orientation)
    pos_error = p_x**2 + p_z**2
    theta_error = theta**2  # small-angle approximation valid for penalization
    
    # Velocity errors (lower priority than position/attitude)
    vel_error = v_x**2 + v_z**2 + omega**2
    
    # Thrust deviation penalty (smooth penalty for deviating from hover thrust)
    thrust_deviation = (u_1 - hover_thrust_per_rotor)**2 + (u_2 - hover_thrust_per_rotor)**2
    
    # Combine with appropriate weights to prioritize position/attitude > velocity > thrust
    # Use negative signs since we want to maximize reward (i.e., minimize penalties)
    reward_val = (
        -1.0 * pos_error          # Strong penalty for position error
        -0.5 * theta_error       # Moderate penalty for attitude error
        -0.1 * vel_error         # Light penalty for velocity error
        -0.05 * thrust_deviation # Gentle penalty for thrust deviation
    )
    
    return float(reward_val)

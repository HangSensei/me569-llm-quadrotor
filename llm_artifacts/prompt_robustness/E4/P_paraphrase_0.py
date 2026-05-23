"""P E4 paraphrase 0. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# 
# def reward(state, action):
#     p_x, p_z, theta, v_x, v_z, omega = state
#     u_1, u_2 = action
#     
#     # Hover reference: position (0,0), attitude 0, velocities 0, thrusts ~4.905 N each
#     hover_thrust = 4.905
#     
#     # Position and attitude errors (weighted higher)
#     pos_error_sq = p_x**2 + p_z**2
#     att_error_sq = theta**2
#     
#     # Velocity errors (lower weight)
#     vel_error_sq = v_x**2 + v_z**2 + omega**2
#     
#     # Control deviation from hover thrust (penalize both magnitude and imbalance)
#     thrust_deviation_sq = (u_1 - hover_thrust)**2 + (u_2 - hover_thrust)**2
#     thrust_imbalance_sq = (u_1 - u_2)**2
#     
#     # Combine with prioritized weights: position/attitude > velocity > control
#     # Use negative terms for errors (since we maximize reward)
#     reward = (
#         -0.5 * pos_error_sq          # dominant penalty for position
#         -0.3 * att_error_sq          # strong penalty for attitude
#         -0.1 * vel_error_sq          # moderate penalty for velocity
#         -0.05 * thrust_deviation_sq  # mild penalty for thrust deviation
#         -0.02 * thrust_imbalance_sq  # very mild penalty for imbalance
#     )
#     
#     return float(reward)
# ```
# --- extracted code ---
import numpy as np

def reward(state, action):
    p_x, p_z, theta, v_x, v_z, omega = state
    u_1, u_2 = action
    
    # Hover reference: position (0,0), attitude 0, velocities 0, thrusts ~4.905 N each
    hover_thrust = 4.905
    
    # Position and attitude errors (weighted higher)
    pos_error_sq = p_x**2 + p_z**2
    att_error_sq = theta**2
    
    # Velocity errors (lower weight)
    vel_error_sq = v_x**2 + v_z**2 + omega**2
    
    # Control deviation from hover thrust (penalize both magnitude and imbalance)
    thrust_deviation_sq = (u_1 - hover_thrust)**2 + (u_2 - hover_thrust)**2
    thrust_imbalance_sq = (u_1 - u_2)**2
    
    # Combine with prioritized weights: position/attitude > velocity > control
    # Use negative terms for errors (since we maximize reward)
    reward = (
        -0.5 * pos_error_sq          # dominant penalty for position
        -0.3 * att_error_sq          # strong penalty for attitude
        -0.1 * vel_error_sq          # moderate penalty for velocity
        -0.05 * thrust_deviation_sq  # mild penalty for thrust deviation
        -0.02 * thrust_imbalance_sq  # very mild penalty for imbalance
    )
    
    return float(reward)

"""Q E4 paraphrase 3. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# The user wants me to design a dense reward function for a planar quadrotor hover task using PPO. Let me break down the requirements:
# 
# 1. State: [p_x, p_z, theta, v_x, v_z, omega] in R^6
# 2. Action: [u_1, u_2] in R^2 (Newtons)
# 3. Physical constants: m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2
# 4. Equilibrium: state = 0 with u_1 = u_2 ~= 4.905 N
# 5. Episodes begin from a disturbance
# 6. Need to design continuous shaping reward (not crash penalty, survival bonus, or termination)
# 
# Key requirements:
# - Maximize by PPO
# - Higher = better
# - Encode penalties as negative terms
# - Deterministic and pure
# - May import only numpy and math
# - Must stay finite on any reasonable input
# - Smooth and differentiable
# 
# Reward components I should include:
# 1. Position error (p_x, p_z) - should be penalized (negative)
# 2. Tilt error (theta) - should be penalized (negative)
# 3. Velocity error (v_x, v_z) - should be penalized (negative)
# 4. Control deviation from hover - should be penalized (negative)
# 5. Smoothness/differentiability - use squared terms
# 
# Let me think about the weights:
# - Position errors should be weighted heavily (want to stay near origin)
# - Tilt errors should be penalized (want to stay level)
# - Velocity errors should be penalized (want to hover)
# - Control deviation should be penalized (want to use minimal control effort)
# 
# For the hover equilibrium:
# - u_1 = u_2 = 4.905 N (to counteract gravity)
# 
# 
# - theta = 0 (level)
# - p_x = 0, p_z = 0 (at origin)
# - v_x = 0, v_z = 0 (no velocity)
# - omega = 0 (no angular velocity)
# 
# I'll design the reward function with appropriate weights for each component. The reward should encourage staying near the origin while maintaining stable hover conditions. I'll use squared error terms for smoothness and differentiability.
# 
# The reward will combine position, tilt, velocity, and control effort penalties with carefully tuned weights to guide the quadrotor toward stable hovering.
# </think>
# 
# ```python
# import numpy as np
# import math
# 
# def reward(state, action):
#     """
#     Dense reward function for planar quadrotor hover task.
#     state = [p_x, p_z, theta, v_x, v_z, omega] in R^6
#     action = [u_1, u_2] in R^2 (Newtons)
#     
#     Returns a single finite scalar (higher = better).
#     """
#     p_x, p_z, theta, v_x, v_z, omega = state
#     u_1, u_2 = action
#     
#     # Physical constants
#     m = 1.0
#     g = 9.81
#     L = 0.25
#     I_yy = 0.01
#     
#     # Hover equilibrium thrust
#     hover_thrust = g * m / 2.0  # ~4.905 N
#     
#     # Position error penalty (weighted heavily)
#     pos_error = p_x**2 + p_z**2
#     pos_weight = 10.0
#     pos_penalty = -pos_weight * pos_error
#     
#     # Tilt error penalty (theta should be near 0)
#     tilt_error = theta**2
#     tilt_weight = 5.0
#     tilt_penalty = -tilt_weight * tilt_error
#     
#     # Velocity error penalty (should hover, v_x = v_z = 0)
#     vel_error = v_x**2 + v_z**2
#     vel_weight = 1.0
#     vel_penalty = -vel_weight * vel_error
#     
#     # Angular velocity penalty (should be stable, omega = 0)
#     ang_vel_error = omega**2
#     ang_vel_weight = 0.5
#     ang_vel_penalty = -ang_vel_weight * ang_vel_error
#     
#     # Control deviation penalty (deviate from hover thrust)
#     hover_thrust_target = hover_thrust
#     control_error = (u_1 - hover_thrust_target)**2 + (u_2 - hover_thrust_target)**2
#     control_weight = 0.1
#     control_penalty = -control_weight * control_error
#     
#     # Combined reward (sum of all penalties, higher = better)
#     reward_value = pos_penalty + tilt_penalty + vel_penalty + ang_vel_penalty + control_penalty
#     
#     # Ensure finite output (clip extreme values)
#     reward_value = np.clip(reward_value, -1000.0, 1000.0)
#     
#     return float(reward_value)
# ```
# --- extracted code ---
import numpy as np
import math

def reward(state, action):
    """
    Dense reward function for planar quadrotor hover task.
    state = [p_x, p_z, theta, v_x, v_z, omega] in R^6
    action = [u_1, u_2] in R^2 (Newtons)
    
    Returns a single finite scalar (higher = better).
    """
    p_x, p_z, theta, v_x, v_z, omega = state
    u_1, u_2 = action
    
    # Physical constants
    m = 1.0
    g = 9.81
    L = 0.25
    I_yy = 0.01
    
    # Hover equilibrium thrust
    hover_thrust = g * m / 2.0  # ~4.905 N
    
    # Position error penalty (weighted heavily)
    pos_error = p_x**2 + p_z**2
    pos_weight = 10.0
    pos_penalty = -pos_weight * pos_error
    
    # Tilt error penalty (theta should be near 0)
    tilt_error = theta**2
    tilt_weight = 5.0
    tilt_penalty = -tilt_weight * tilt_error
    
    # Velocity error penalty (should hover, v_x = v_z = 0)
    vel_error = v_x**2 + v_z**2
    vel_weight = 1.0
    vel_penalty = -vel_weight * vel_error
    
    # Angular velocity penalty (should be stable, omega = 0)
    ang_vel_error = omega**2
    ang_vel_weight = 0.5
    ang_vel_penalty = -ang_vel_weight * ang_vel_error
    
    # Control deviation penalty (deviate from hover thrust)
    hover_thrust_target = hover_thrust
    control_error = (u_1 - hover_thrust_target)**2 + (u_2 - hover_thrust_target)**2
    control_weight = 0.1
    control_penalty = -control_weight * control_error
    
    # Combined reward (sum of all penalties, higher = better)
    reward_value = pos_penalty + tilt_penalty + vel_penalty + ang_vel_penalty + control_penalty
    
    # Ensure finite output (clip extreme values)
    reward_value = np.clip(reward_value, -1000.0, 1000.0)
    
    return float(reward_value)

"""P E4 paraphrase 3. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# 
# def reward(state, action):
#     p_x, p_z, theta, v_x, v_z, omega = state
#     u_1, u_2 = action
#     
#     # Hover equilibrium thrust: each motor supports half weight => u_eq = m*g/2 = 4.905 N
#     u_eq = 4.905
#     
#     # Position errors (penalize deviation from origin; prioritize z over x, and both over tilt)
#     pos_penalty = 10.0 * (p_x**2 + 4.0 * p_z**2)  # stronger penalty on vertical position error
#     
#     # Orientation error (theta in radians; penalize tilt quadratically)
#     theta_penalty = 20.0 * theta**2
#     
#     # Velocity errors (lighter penalty than position/tilt, but still meaningful)
#     vel_penalty = 2.0 * (v_x**2 + v_z**2 + 0.1 * omega**2)
#     
#     # Control effort penalty: deviation from hover thrusts (smooth, quadratic)
#     ctrl_penalty = 0.5 * ((u_1 - u_eq)**2 + (u_2 - u_eq)**2)
#     
#     # Bonus for being near equilibrium thrusts (implicit in ctrl_penalty, but we want net reward > 0 near optimum)
#     # So we offset with a constant baseline that makes reward ~0 at perfect hover, and positive near it
#     # At perfect hover: state=0, action=[u_eq, u_eq] => all penalties zero => reward = 0
#     # We add a small smooth bonus for being close to hover to encourage convergence, but keep it bounded
#     # Instead, structure as negative penalties only — PPO benefits from dense negative shaping; baseline handled by algorithm
#     
#     # Ensure smoothness and finiteness: all terms are quadratic → infinitely differentiable, finite everywhere
#     total_penalty = pos_penalty + theta_penalty + vel_penalty + ctrl_penalty
#     
#     # Return negative penalty (since higher reward is better)
#     return -total_penalty
# ```
# --- extracted code ---
import numpy as np

def reward(state, action):
    p_x, p_z, theta, v_x, v_z, omega = state
    u_1, u_2 = action
    
    # Hover equilibrium thrust: each motor supports half weight => u_eq = m*g/2 = 4.905 N
    u_eq = 4.905
    
    # Position errors (penalize deviation from origin; prioritize z over x, and both over tilt)
    pos_penalty = 10.0 * (p_x**2 + 4.0 * p_z**2)  # stronger penalty on vertical position error
    
    # Orientation error (theta in radians; penalize tilt quadratically)
    theta_penalty = 20.0 * theta**2
    
    # Velocity errors (lighter penalty than position/tilt, but still meaningful)
    vel_penalty = 2.0 * (v_x**2 + v_z**2 + 0.1 * omega**2)
    
    # Control effort penalty: deviation from hover thrusts (smooth, quadratic)
    ctrl_penalty = 0.5 * ((u_1 - u_eq)**2 + (u_2 - u_eq)**2)
    
    # Bonus for being near equilibrium thrusts (implicit in ctrl_penalty, but we want net reward > 0 near optimum)
    # So we offset with a constant baseline that makes reward ~0 at perfect hover, and positive near it
    # At perfect hover: state=0, action=[u_eq, u_eq] => all penalties zero => reward = 0
    # We add a small smooth bonus for being close to hover to encourage convergence, but keep it bounded
    # Instead, structure as negative penalties only — PPO benefits from dense negative shaping; baseline handled by algorithm
    
    # Ensure smoothness and finiteness: all terms are quadratic → infinitely differentiable, finite everywhere
    total_penalty = pos_penalty + theta_penalty + vel_penalty + ctrl_penalty
    
    # Return negative penalty (since higher reward is better)
    return -total_penalty

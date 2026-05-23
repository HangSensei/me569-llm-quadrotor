"""P-condition prompt ablation variant 'minimal'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# ```python
# def stage_cost(x, u):
#     # Hover state: [0, 0, 0, 0, 0, 0]
#     # Hover control: [4.905, 4.905]
#     px, pz, theta, vx, vz, omega = x
#     u1, u2 = u
#     
#     # Position error squared (with scaling)
#     pos_cost = px*px + pz*pz
#     
#     # Angle error squared (with scaling)
#     angle_cost = theta*theta
#     
#     # Velocity error squared
#     vel_cost = vx*vx + vz*vz + omega*omega
#     
#     # Control effort (deviation from hover thrust)
#     ctrl_cost = (u1 - 4.905)*(u1 - 4.905) + (u2 - 4.905)*(u2 - 4.905)
#     
#     # Small penalty for large angles to encourage small angles (using sin^2 for smoothness)
#     # sin(theta)^2 is approximately theta^2 near 0, but grows bounded for larger angles
#     angle_nonlinear_cost = sin(theta)*sin(theta)
#     
#     return pos_cost + angle_cost + vel_cost + ctrl_cost + angle_nonlinear_cost
# ```
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
    # Hover state: [0, 0, 0, 0, 0, 0]
    # Hover control: [4.905, 4.905]
    px, pz, theta, vx, vz, omega = x
    u1, u2 = u
    
    # Position error squared (with scaling)
    pos_cost = px*px + pz*pz
    
    # Angle error squared (with scaling)
    angle_cost = theta*theta
    
    # Velocity error squared
    vel_cost = vx*vx + vz*vz + omega*omega
    
    # Control effort (deviation from hover thrust)
    ctrl_cost = (u1 - 4.905)*(u1 - 4.905) + (u2 - 4.905)*(u2 - 4.905)
    
    # Small penalty for large angles to encourage small angles (using sin^2 for smoothness)
    # sin(theta)^2 is approximately theta^2 near 0, but grows bounded for larger angles
    angle_nonlinear_cost = sin(theta)*sin(theta)
    
    return pos_cost + angle_cost + vel_cost + ctrl_cost + angle_nonlinear_cost

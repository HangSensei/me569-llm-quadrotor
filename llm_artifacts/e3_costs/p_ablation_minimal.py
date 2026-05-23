"""P-condition prompt ablation variant 'minimal'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# ```python
# def stage_cost(x, u):
#     # Reference (hover) state and control
#     x_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     u_ref = [4.905, 4.905]
#     
#     # State deviation squared (with reasonable weights)
#     # Position: p_x, p_z (indices 0,1) - penalize more heavily
#     # Angle: theta (index 2) - penalize moderately
#     # Velocities: v_x, v_z, omega (indices 3,4,5) - penalize moderately
#     state_cost = (
#         10.0 * (x[0] - x_ref[0])**2 +      # p_x
#         10.0 * (x[1] - x_ref[1])**2 +      # p_z
#         5.0  * (x[2] - x_ref[2])**2 +      # theta
#         1.0  * (x[3] - x_ref[3])**2 +      # v_x
#         1.0  * (x[4] - x_ref[4])**2 +      # v_z
#         2.0  * (x[5] - x_ref[5])**2        # omega
#     )
#     
#     # Control deviation squared (with reasonable weights)
#     # Both thrusts are equally important
#     control_cost = (
#         0.1 * (u[0] - u_ref[0])**2 +
#         0.1 * (u[1] - u_ref[1])**2
#     )
#     
#     return state_cost + control_cost
# ```
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
    # Reference (hover) state and control
    x_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u_ref = [4.905, 4.905]
    
    # State deviation squared (with reasonable weights)
    # Position: p_x, p_z (indices 0,1) - penalize more heavily
    # Angle: theta (index 2) - penalize moderately
    # Velocities: v_x, v_z, omega (indices 3,4,5) - penalize moderately
    state_cost = (
        10.0 * (x[0] - x_ref[0])**2 +      # p_x
        10.0 * (x[1] - x_ref[1])**2 +      # p_z
        5.0  * (x[2] - x_ref[2])**2 +      # theta
        1.0  * (x[3] - x_ref[3])**2 +      # v_x
        1.0  * (x[4] - x_ref[4])**2 +      # v_z
        2.0  * (x[5] - x_ref[5])**2        # omega
    )
    
    # Control deviation squared (with reasonable weights)
    # Both thrusts are equally important
    control_cost = (
        0.1 * (u[0] - u_ref[0])**2 +
        0.1 * (u[1] - u_ref[1])**2
    )
    
    return state_cost + control_cost

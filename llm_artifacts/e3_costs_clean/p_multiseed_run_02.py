"""P-condition stage cost from multi-seed run 02. Qwen-Plus call at temperature 0.3. Saved by e3_multi_seed.py.
"""
def stage_cost(x, u):
    # Penalize position deviation from the origin (p_x and p_z)
    # Use quadratic penalties: position errors are critical for regulation
    pos = 10.0 * (x[0] * x[0] + (x[1] - 0.0) * (x[1] - 0.0))
    
    # Penalize pitch deviation from level (theta), but allow moderate tilt
    # Use sin^2 or theta^2 — theta^2 is smooth and appropriate for small angles;
    # we use theta^2 to avoid singularities and match linearized cost design
    att = 5.0 * x[2] * x[2]
    
    # Penalize translational and angular velocity deviations (v_x, v_z, omega)
    # Weaker than position, stronger than control effort
    vel = 1.0 * (x[3] * x[3] + x[4] * x[4]) + 0.5 * x[5] * x[5]
    
    # Penalize control effort relative to hover-equilibrium thrust (u_hover = m*g/2 = 4.905)
    u_hover = 4.905
    ctrl = 0.01 * ((u[0] - u_hover) * (u[0] - u_hover) + (u[1] - u_hover) * (u[1] - u_hover))
    
    return pos + att + vel + ctrl

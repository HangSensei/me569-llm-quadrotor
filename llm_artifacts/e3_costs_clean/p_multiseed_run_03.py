"""P-condition stage cost from multi-seed run 03. Qwen-Plus call at temperature 0.3. Saved by e3_multi_seed.py.
"""
def stage_cost(x, u):
    # Penalize position deviation from the origin (p_x, p_z)
    pos = 10.0 * (x[0]**2 + x[1]**2)
    # Penalize pitch deviation from level (theta) — use sin^2 or theta^2; theta^2 is smooth and sufficient
    att = 5.0 * x[2]**2
    # Penalize translational and angular velocity (v_x, v_z, omega)
    vel = 2.0 * (x[3]**2 + x[4]**2) + 1.0 * x[5]**2
    # Penalize control effort relative to hover-equilibrium thrust (u_hover = m*g/2 = 4.905)
    u_hover = 4.905
    ctrl = 0.1 * ((u[0] - u_hover)**2 + (u[1] - u_hover)**2)
    return pos + att + vel + ctrl

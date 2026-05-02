"""Q-condition stage cost from Qwen3.5-4B-MLX-8bit run 01. Saved by scripts/run_e3_full.py.
"""
def stage_cost(x, u):
        m = 1.0
        g = 9.81
        u_hover = m * g / 2
        # Position
        pos_cost = 10.0 * (x[0]**2 + x[1]**2)
        # Attitude
        att_cost = 1.0 * x[2]**2
        # Velocity
        vel_cost = 0.1 * (x[3]**2 + x[4]**2 + x[5]**2)
        # Control
        ctrl_cost = 0.01 * ((u[0] - u_hover)**2 + (u[1] - u_hover)**2)
        return pos_cost + att_cost + vel_cost + ctrl_cost

"""Q-condition stage cost from Qwen3.5-4B-MLX-8bit run 01. Saved by scripts/run_e3_full.py.
"""
def stage_cost(x, u):
            # State weights
            pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
            vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
            att_cost = 5.0 * x[2] ** 2
            omega_cost = 0.5 * x[5] ** 2
            # Control weights
            u_hover = 4.905
            ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
            return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost

def stage_cost(x, u):
        # State weights
        # Position (p_x, p_z)
        pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
        # Velocity (v_x, v_z)
        vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
        # Attitude (theta)
        att_cost = 5.0 * x[2] ** 2
        # Angular velocity (omega)
        omega_cost = 0.5 * x[5] ** 2
        # Control effort (deviation from hover thrust)
        u_hover = 1.0 * 9.81 / 2.0
        ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
        return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost

def stage_cost(x, u):
    # State weights
    # Position (p_x, p_z)
    pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
    # Velocity (v_x, v_z)
    vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
    # Attitude (theta)
    att_cost = 5.0 * x[2] ** 2
    # Angular velocity (omega)
    omega_cost = 0.5 * x[5] ** 2
    # Control effort (deviation from hover thrust)
    u_hover = 4.905
    ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
    return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost

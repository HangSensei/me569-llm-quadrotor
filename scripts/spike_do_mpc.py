"""Spike test: verify do-mpc + casadi work on Apple Silicon M4."""
import sys

try:
    import casadi as ca
    import do_mpc
    import numpy as np
except ImportError as e:
    print(f"IMPORT FAIL: {e}")
    sys.exit(1)

# Minimal single-integrator MPC problem: x_dot = u, drive x to 0 from x0 = 1
model = do_mpc.model.Model("continuous")
x = model.set_variable(var_type="_x", var_name="x")
u = model.set_variable(var_type="_u", var_name="u")
model.set_rhs("x", u)
model.setup()

mpc = do_mpc.controller.MPC(model)
mpc.set_param(
    n_horizon=10,
    t_step=0.1,
    store_full_solution=False,
)
mpc.set_objective(mterm=x**2, lterm=x**2 + 0.01 * u**2)
mpc.set_rterm(u=0.01)
mpc.bounds["lower", "_u", "u"] = -1.0
mpc.bounds["upper", "_u", "u"] = 1.0
mpc.setup()

x0 = np.array([[1.0]])
mpc.x0 = x0
mpc.set_initial_guess()

u0 = mpc.make_step(x0)
print(f"do-mpc OK: first control action = {float(u0[0, 0]):.4f}")
print(f"  (expected negative to drive x from 1 toward 0)")
print(f"casadi version: {ca.__version__}")
print(f"do_mpc version: {do_mpc.__version__}")
print("SPIKE 1 (do-mpc) PASSED")

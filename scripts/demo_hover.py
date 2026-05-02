"""Demo: LQR stabilizes the Planar Quadrotor from a randomized initial state.

This script is the Week 1 sanity check for the full simulator stack:
    QuadrotorParams -> dynamics -> RK4 -> LQR -> viz

It runs one 10-second closed-loop rollout, prints the final-state summary,
and saves a 3-panel trajectory plot to ``results/hover_demo.png``.

Run it from the repo root with the project venv activated:

    python scripts/demo_hover.py

Expected output:
    - Console line showing final position error well under 5 cm
    - File results/hover_demo.png showing p_x, p_z, theta decaying to zero
"""
from __future__ import annotations

import numpy as np

from me569_project.controllers.lqr_hover import LQRHoverController
from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step
from me569_project.viz import plot_trajectory


def main(seed: int = 0) -> None:
    params = QuadrotorParams()
    ctrl = LQRHoverController(params)
    dt = 0.02
    n_steps = 500  # 10 seconds

    rng = np.random.default_rng(seed)
    state = rng.uniform(
        low=np.array([-1.0, -1.0, -0.3, -0.5, -0.5, -0.5]),
        high=np.array([1.0, 1.0, 0.3, 0.5, 0.5, 0.5]),
    )

    history = np.zeros((n_steps + 1, 6), dtype=np.float64)
    actions = np.zeros((n_steps, 2), dtype=np.float64)
    history[0] = state

    for i in range(n_steps):
        u = ctrl.act(state)
        actions[i] = u
        state = rk4_step(state, u, dt, params)
        history[i + 1] = state

    final_pos_err = float(np.linalg.norm(history[-1, :2]))
    final_att_err = float(abs(history[-1, 2]))
    final_vel_err = float(np.linalg.norm(history[-1, 3:]))

    print("=" * 60)
    print("LQR Hover Demo")
    print("=" * 60)
    print(f"Seed:             {seed}")
    print(f"Initial state:    {history[0]}")
    print(f"Simulation:       {n_steps} steps at dt={dt} s ({n_steps * dt} s total)")
    print("-" * 60)
    print(f"Final state:      {history[-1]}")
    print(f"Final pos error:  {final_pos_err:.6f} m")
    print(f"Final attitude:   {final_att_err:.6f} rad")
    print(f"Final vel norm:   {final_vel_err:.6f}")
    print("-" * 60)

    out_path = plot_trajectory(
        history,
        dt=dt,
        save_path="results/hover_demo.png",
        title=f"LQR Hover Demo (seed={seed})",
    )
    print(f"Saved trajectory plot to {out_path}")

    if final_pos_err < 0.05:
        print("\n[PASS] Converged to within 5 cm of origin.")
    else:
        print("\n[WARN] Final position error > 5 cm; controller may be untuned.")


if __name__ == "__main__":
    main()

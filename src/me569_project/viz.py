"""Matplotlib visualization utilities for the Planar Quadrotor.

Currently provides:
- plot_trajectory: 3-panel static plot (position, attitude, rates) saved to disk.
  Suitable for headless environments (no plt.show()).

Animation and rendering into the final report figures are planned but not yet
implemented — the static multi-panel plot is enough for the Week 1 LQR demo
and the Week 5 Project Proposal preliminary results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    states: np.ndarray,
    dt: float,
    save_path: str | Path,
    title: str = "Planar Quadrotor Trajectory",
) -> Path:
    """Render a 3-panel trajectory plot and save it to disk.

    Panels (top to bottom):
        1. Horizontal and vertical position vs. time.
        2. Pitch angle vs. time.
        3. Linear and angular velocities vs. time.

    Parameters
    ----------
    states : np.ndarray of shape (T, 6)
        State trajectory. Column order is
        [p_x, p_z, theta, v_x, v_z, omega].
    dt : float
        Integration step used to produce the trajectory. Used only for the
        time axis.
    save_path : str or Path
        Destination file path. Parent directory is created if missing.
    title : str
        Figure suptitle.

    Returns
    -------
    Path
        The absolute path to the saved figure.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(
            f"Expected states of shape (T, 6), got {states.shape}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    T = states.shape[0]
    t = np.arange(T) * dt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(title)

    axes[0].plot(t, states[:, 0], label="$p_x$", linewidth=1.5)
    axes[0].plot(t, states[:, 1], label="$p_z$", linewidth=1.5)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("position (m)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, states[:, 2], label=r"$\theta$", color="C2", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel("pitch (rad)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, states[:, 3], label="$v_x$", linewidth=1.5)
    axes[2].plot(t, states[:, 4], label="$v_z$", linewidth=1.5)
    axes[2].plot(t, states[:, 5], label=r"$\omega$", linewidth=1.5)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[2].set_ylabel("rates (m/s, rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path.resolve()

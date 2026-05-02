"""Reference trajectories for tracking MPC experiments."""
from __future__ import annotations

import math

import numpy as np


def step_reference(
    t: float,
    step_time: float = 2.0,
    target_x: float = 1.0,
    target_z: float = 0.5,
) -> np.ndarray:
    """Step reference from origin to a fixed hover offset."""
    if t < step_time:
        return np.zeros(6, dtype=np.float64)
    return np.array([target_x, target_z, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def figure_eight_reference(
    t: float,
    period: float = 8.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Figure-8 reference in the position plane with zero attitude target."""
    phase = 2.0 * math.pi * t / period
    p_x = amplitude * math.sin(phase)
    p_z = 0.5 * amplitude * math.sin(2.0 * phase)
    return np.array([p_x, p_z, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

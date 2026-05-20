"""Baseline reward (B condition) and reward-clip helper for E4.

The B condition uses the negative of the textbook E2 stage cost
(Bryson's-rule weights) so that PPO maximizing the reward solves the
same hover problem the MPC solves analytically. The crash-penalty /
survival-bonus / termination logic is handled by ``LLMRewardWrapper``,
not by this function — so the dense per-step reward shape is the only
thing the LLM (in P, Q conditions) needs to design.

``clip_reward`` is a numeric safety guard that wraps any reward
function with a `np.clip` to a configurable range. We apply it to
LLM-generated rewards before plugging them into PPO so that a single
catastrophic outlier does not blow up the value function.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


_HOVER_THRUST = 4.905  # m * g / 2 = 1.0 * 9.81 / 2


def baseline_reward(state: np.ndarray, action: np.ndarray) -> float:
    """Textbook negative-quadratic hover reward (Bryson's-rule weights).

    Returns 0.0 at the hover equilibrium ``state = 0``,
    ``action = [u_hover, u_hover]`` and a strictly negative value
    everywhere else. The magnitude grows with position, attitude,
    velocity, and control-effort errors.

    Uses Bryson's-rule weights close to
    ``mpc.baseline_cost.make_baseline_stage_cost`` with the sign
    flipped (PPO maximizes, MPC minimizes). The one deviation is the
    angular-rate weight: 1.0 here versus 0.5 in the MPC stage cost,
    because the velocity terms are lumped under a single weight.

    Parameters
    ----------
    state : np.ndarray of shape (6,)
        Indexable as state[0]..state[5] = [p_x, p_z, theta, v_x, v_z, omega].
    action : np.ndarray of shape (2,)
        Indexable as action[0], action[1] = [u_1, u_2].

    Returns
    -------
    float, in (-inf, 0].
    """
    state = np.asarray(state, dtype=np.float64).ravel()
    action = np.asarray(action, dtype=np.float64).ravel()

    pos_err_sq = state[0] ** 2 + state[1] ** 2
    att_err_sq = state[2] ** 2
    vel_err_sq = state[3] ** 2 + state[4] ** 2 + state[5] ** 2
    ctrl_err_sq = (action[0] - _HOVER_THRUST) ** 2 + (action[1] - _HOVER_THRUST) ** 2

    return float(
        -(
            10.0 * pos_err_sq
            + 5.0 * att_err_sq
            + 1.0 * vel_err_sq
            + 0.1 * ctrl_err_sq
        )
    )


def clip_reward(
    fn: Callable[[np.ndarray, np.ndarray], float],
    low: float = -1000.0,
    high: float = 1000.0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Wrap a reward function with a numeric safety clip.

    Used on LLM-generated rewards before plugging them into PPO. The
    clip range is wide enough not to interfere with reasonable
    quadratic rewards (the baseline ranges roughly [-100, 0]) but
    tight enough to prevent value-function blow-up from a malformed
    LLM reward producing 1e10 or NaN.

    Non-finite values from the inner function are mapped to ``low``.

    Parameters
    ----------
    fn : callable (state, action) -> float
        The reward function to wrap.
    low, high : float
        The clipped output range.
    """
    def clipped(state: np.ndarray, action: np.ndarray) -> float:
        try:
            r = float(fn(state, action))
        except Exception:
            return low
        if not np.isfinite(r):
            return low
        return float(np.clip(r, low, high))

    return clipped

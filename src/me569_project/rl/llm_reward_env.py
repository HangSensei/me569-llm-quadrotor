"""Gymnasium ``Wrapper`` that swaps in an LLM-generated reward function.

Used by E4 to inject any callable ``reward(state, action) -> float`` into
the existing ``PlanarQuadrotorEnv`` while keeping the workspace-bound
termination, the crash penalty, and the per-step survival bonus identical
across the B / P / Q conditions. The LLM-generated reward is responsible
*only* for the dense per-step shape — the discontinuous boundary signals
(crash, survival) live here.

This isolation matches the Eureka paper's convention: the LLM optimizes
the *shape* of the reward, not the boundary handling.

Convention: the wrapped env evaluates ``reward_fn(obs, action_clipped)``
on the **post-step state** (i.e. the state after the action has been
applied). This is the standard RL convention for `r(s_t, a_t)` where
the reward is delivered to the agent at step ``t`` after applying
``a_t``.

Companion ``HoverThrustOffsetWrapper`` provides an ``ActionWrapper`` that
shifts the agent's zero-mean action to hover-equilibrium thrust. PPO's
default Gaussian policy starts at ``N(0, 1)`` on each action component;
without the shift, that maps to roughly zero rotor thrust under the
[0, u_max] action space, the quadrotor falls under gravity, and the
agent crashes within ~50 steps long before learning anything. With the
shift, the initial policy hovers (or nearly so), giving PPO a usable
starting point.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np


class LLMRewardWrapper(gym.Wrapper):
    """Inject a custom reward function into a Gymnasium env.

    Parameters
    ----------
    env : gymnasium.Env
        The base env (typically ``PlanarQuadrotorEnv``).
    reward_fn : callable (state, action) -> float
        The LLM-generated or baseline reward shape. Called on the
        post-step state and the (clipped) action.
    crash_penalty : float
        Added to ``reward_fn`` output if the underlying env terminates
        the episode (out-of-workspace). Default -100.
    survival_bonus : float
        Added to ``reward_fn`` output every step that does NOT
        terminate. Default 0.1.
    action_clip_low, action_clip_high : float
        Clip action passed to ``reward_fn`` to this range. The
        underlying env also clips internally; we mirror that here so
        the LLM reward sees the actual applied action.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        crash_penalty: float = -100.0,
        survival_bonus: float = 0.1,
        action_clip_low: float | None = None,
        action_clip_high: float | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_fn = reward_fn
        self.crash_penalty = float(crash_penalty)
        self.survival_bonus = float(survival_bonus)

        if action_clip_low is None:
            action_clip_low = float(env.action_space.low.min())
        if action_clip_high is None:
            action_clip_high = float(env.action_space.high.max())
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float64).ravel()
        action_clipped = np.clip(
            action_arr, self.action_clip_low, self.action_clip_high
        )

        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        # Compute custom reward on post-step state and clipped action.
        # Call ``reward_fn`` exactly once per step — both the env reward
        # and the info-dict raw value share the same evaluation. Calling
        # twice would inflate counts when the reward has side effects
        # (e.g. logging the action it sees in tests).
        raw_reward = float(self.reward_fn(obs, action_clipped))
        if terminated:
            custom_reward = raw_reward + self.crash_penalty
        else:
            custom_reward = raw_reward + self.survival_bonus

        info = dict(info)
        info["llm_reward_raw"] = raw_reward
        info["llm_reward_total"] = custom_reward
        info["llm_reward_terminal"] = terminated

        return obs, custom_reward, terminated, truncated, info


class HoverThrustOffsetWrapper(gym.ActionWrapper):
    """Shift the agent's action so that 0 corresponds to hover-equilibrium thrust.

    The base env exposes an action space of ``[0, u_max]^2`` for the two
    rotors. PPO's default Gaussian policy starts roughly centered at 0,
    which under that space means *zero thrust* — the quadrotor falls and
    crashes before PPO can learn anything. This wrapper exposes a
    centered action space ``[-u_hover, u_max - u_hover]^2`` to the agent
    and shifts the action by ``+u_hover`` before forwarding to the base
    env. The agent's zero-mean initial policy then corresponds to
    hovering, which is a viable starting point.

    Mathematically: ``base_action = clip(agent_action + u_hover, 0, u_max)``.
    """

    def __init__(self, env: gym.Env, u_hover: float = 4.905) -> None:
        super().__init__(env)
        self._u_hover = float(u_hover)
        old_low = np.asarray(env.action_space.low, dtype=np.float64)
        old_high = np.asarray(env.action_space.high, dtype=np.float64)
        new_low = old_low - self._u_hover
        new_high = old_high - self._u_hover
        self.action_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=env.action_space.shape,
            dtype=np.float64,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float64).ravel()
        shifted = action_arr + self._u_hover
        return np.clip(
            shifted,
            self.env.action_space.low,
            self.env.action_space.high,
        )

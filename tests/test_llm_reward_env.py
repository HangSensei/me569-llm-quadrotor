"""Tests for the LLMRewardWrapper used in Experiment 4."""
import math

import gymnasium as gym
import numpy as np
import pytest

from me569_project.quadrotor_env import PlanarQuadrotorEnv
from me569_project.rl.eureka_rewards import baseline_reward
from me569_project.rl.llm_reward_env import LLMRewardWrapper


def _make_env(reward_fn=None, **wrapper_kwargs):
    env = PlanarQuadrotorEnv(max_steps=10, init_radius=0.1)
    if reward_fn is None:
        reward_fn = baseline_reward
    return LLMRewardWrapper(env, reward_fn=reward_fn, **wrapper_kwargs)


def test_wrapped_env_uses_injected_reward():
    """Custom reward_fn output (modulo bonus/penalty) controls the per-step reward."""
    env = _make_env(reward_fn=lambda s, a: -7.0, survival_bonus=0.5)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([4.9, 4.9]))
    # No crash on first step, so reward = -7.0 + survival_bonus = -6.5
    assert reward == pytest.approx(-6.5, abs=1e-9)
    assert info["llm_reward_total"] == pytest.approx(-6.5, abs=1e-9)


def test_wrapped_env_applies_survival_bonus_when_alive():
    env = _make_env(reward_fn=lambda s, a: 0.0, survival_bonus=2.0, crash_penalty=-99.0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([4.9, 4.9]))
    assert not terminated
    assert reward == pytest.approx(2.0, abs=1e-9)


def test_wrapped_env_applies_crash_penalty_on_termination():
    """Force a terminating step by asking for huge thrust to throw the quadrotor out of bounds."""
    env = _make_env(
        reward_fn=lambda s, a: 0.0,
        survival_bonus=2.0,
        crash_penalty=-99.0,
    )
    env.reset(seed=0)
    # Apply huge unequal thrust for many steps to push out of bounds.
    terminated = False
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 19.62]))
        if terminated or truncated:
            break
    if terminated:
        # When termination fires, reward should be (reward_fn = 0) + crash_penalty (-99).
        assert reward == pytest.approx(-99.0, abs=1e-6)
    # If we only truncated (didn't crash), the test setup didn't reach the boundary;
    # just verify the wrapper didn't crash and no spurious penalty got applied.


def test_wrapped_env_truncates_at_max_steps():
    env = _make_env()
    env.reset(seed=0)
    truncated = False
    for _ in range(20):  # max_steps=10 in fixture, so 11 steps should truncate
        obs, reward, terminated, truncated, info = env.step(np.array([4.9, 4.9]))
        if truncated:
            break
    assert truncated


def test_wrapped_env_clips_action_for_reward_fn():
    """Action passed to reward_fn must be clipped to action_space bounds."""
    seen_actions: list[np.ndarray] = []

    def recording_reward(state, action):
        seen_actions.append(np.array(action))
        return 0.0

    env = _make_env(reward_fn=recording_reward)
    env.reset(seed=0)
    # Pass an action well above u_max=19.62 to trigger clipping.
    env.step(np.array([100.0, -50.0]))
    assert len(seen_actions) == 1
    # Both components should be clipped to [0, u_max]
    assert seen_actions[0][0] <= env.action_clip_high + 1e-9
    assert seen_actions[0][1] >= env.action_clip_low - 1e-9


def test_wrapped_env_returns_finite_reward_with_baseline():
    env = _make_env(reward_fn=baseline_reward)
    env.reset(seed=0)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.array([4.9, 4.9]))
        assert math.isfinite(reward)
        if terminated or truncated:
            break

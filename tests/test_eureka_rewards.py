"""Tests for the E4 baseline reward and reward-clip helper."""
import math

import numpy as np
import pytest

from me569_project.rl.eureka_rewards import baseline_reward, clip_reward


_U_HOVER = 4.905


def test_baseline_reward_zero_at_hover_equilibrium():
    state = np.zeros(6)
    action = np.array([_U_HOVER, _U_HOVER])
    r = baseline_reward(state, action)
    assert r == pytest.approx(0.0, abs=1e-12)


def test_baseline_reward_higher_at_hover_than_disturbed():
    hover_state = np.zeros(6)
    disturbed_state = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05])
    action = np.array([_U_HOVER, _U_HOVER])
    r_hover = baseline_reward(hover_state, action)
    r_disturbed = baseline_reward(disturbed_state, action)
    assert r_hover > r_disturbed


def test_baseline_reward_lower_with_zero_thrust_at_hover():
    """Control penalty must fire when action != hover thrust, even at the hover state."""
    state = np.zeros(6)
    r_hover_action = baseline_reward(state, np.array([_U_HOVER, _U_HOVER]))
    r_zero_action = baseline_reward(state, np.array([0.0, 0.0]))
    assert r_hover_action > r_zero_action


def test_baseline_reward_smoke_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        state = rng.normal(size=6)
        action = rng.uniform(0.0, 10.0, size=2)
        r = baseline_reward(state, action)
        assert math.isfinite(r)
        assert r <= 0.0  # negative-quadratic must be non-positive


def test_clipped_reward_handles_inf():
    def bad_fn(state, action):
        return float("inf")
    clipped = clip_reward(bad_fn, low=-50.0, high=50.0)
    r = clipped(np.zeros(6), np.zeros(2))
    assert r == -50.0  # non-finite is mapped to low


def test_clipped_reward_handles_nan():
    def nan_fn(state, action):
        return float("nan")
    clipped = clip_reward(nan_fn, low=-50.0, high=50.0)
    r = clipped(np.zeros(6), np.zeros(2))
    assert r == -50.0


def test_clipped_reward_handles_exception():
    def raising_fn(state, action):
        raise ValueError("oops")
    clipped = clip_reward(raising_fn, low=-50.0, high=50.0)
    r = clipped(np.zeros(6), np.zeros(2))
    assert r == -50.0


def test_clipped_reward_passes_through_in_range():
    def good_fn(state, action):
        return -42.0
    clipped = clip_reward(good_fn, low=-1000.0, high=1000.0)
    r = clipped(np.zeros(6), np.zeros(2))
    assert r == -42.0


def test_clipped_reward_clips_high_outlier():
    def big_fn(state, action):
        return 1e9
    clipped = clip_reward(big_fn, low=-1000.0, high=1000.0)
    r = clipped(np.zeros(6), np.zeros(2))
    assert r == 1000.0

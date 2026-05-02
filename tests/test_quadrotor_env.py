"""Tests for the Planar Quadrotor Gymnasium environment.

The env wraps the continuous quadrotor dynamics and RK4 integrator into a
discrete-time Gymnasium step API for RL, MPC closed-loop, and SINDy data
generation.

Gymnasium step API returns a 5-tuple:
    (obs, reward, terminated, truncated, info)
"""
import numpy as np
import pytest

from me569_project.quadrotor_dynamics import QuadrotorParams
from me569_project.quadrotor_env import PlanarQuadrotorEnv


def test_env_reset_returns_correct_shape_and_dtype():
    env = PlanarQuadrotorEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (6,)
    assert obs.dtype == np.float64
    assert isinstance(info, dict)


def test_env_observation_and_action_space_shapes():
    env = PlanarQuadrotorEnv()

    assert env.observation_space.shape == (6,)
    assert env.action_space.shape == (2,)
    # Each rotor thrust is in [0, u_max]
    assert np.all(env.action_space.low == 0.0)
    assert np.all(env.action_space.high == env.params.u_max)


def test_env_step_returns_5_tuple_gymnasium_api():
    """Modern Gymnasium API returns (obs, reward, terminated, truncated, info)."""
    env = PlanarQuadrotorEnv()
    env.reset(seed=0)
    action = np.array([env.params.u_hover, env.params.u_hover])

    result = env.step(action)

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (6,)
    assert obs.dtype == np.float64
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_action_clipping_does_not_crash_on_out_of_bounds():
    """Huge or negative actions must be silently clipped to [0, u_max]."""
    env = PlanarQuadrotorEnv()
    env.reset(seed=0)
    wild_action = np.array([1e6, -1e6])  # one huge, one negative

    obs, _, _, _, _ = env.step(wild_action)

    assert np.all(np.isfinite(obs)), "State should stay finite after clipping"


def test_env_truncates_after_max_steps():
    """Truncation flag should fire exactly when step_count reaches max_steps."""
    env = PlanarQuadrotorEnv(max_steps=5)
    env.reset(seed=0)
    hover = np.array([env.params.u_hover, env.params.u_hover])

    truncated = False
    steps_taken = 0
    for _ in range(10):
        _, _, _, truncated, _ = env.step(hover)
        steps_taken += 1
        if truncated:
            break

    assert truncated
    assert steps_taken == 5, "Truncation should fire after exactly max_steps"


def test_env_reset_is_deterministic_given_seed():
    env = PlanarQuadrotorEnv()
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)


def test_env_reset_differs_across_seeds():
    """Different seeds should (with overwhelming probability) produce different states."""
    env = PlanarQuadrotorEnv()
    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=2)

    assert not np.allclose(obs1, obs2)


def test_env_terminates_on_flying_out_of_bounds():
    """Manually forcing the state out of the 5 m box should trigger termination."""
    env = PlanarQuadrotorEnv()
    env.reset(seed=0)
    env.state = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # way out of bounds

    _, _, terminated, _, _ = env.step(np.zeros(2))

    assert terminated


def test_env_reward_is_better_at_origin_than_off_origin():
    """Default hover reward should peak at the origin."""
    env = PlanarQuadrotorEnv()
    env.reset(seed=0)
    hover = np.array([env.params.u_hover, env.params.u_hover])

    env.state = np.zeros(6)
    _, reward_origin, _, _, _ = env.step(hover)

    env.reset(seed=0)
    env.state = np.array([1.0, 1.0, 0.3, 0.0, 0.0, 0.0])
    _, reward_off, _, _, _ = env.step(hover)

    assert reward_origin > reward_off


def test_env_reset_state_within_init_radius():
    """Initial state after reset should be inside the configured init ball."""
    env = PlanarQuadrotorEnv(init_radius=0.5)
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        # Positions should be inside the radius
        assert abs(obs[0]) <= 0.5
        assert abs(obs[1]) <= 0.5
        # Attitude should be small
        assert abs(obs[2]) <= 0.2


def test_env_step_count_resets_on_reset():
    env = PlanarQuadrotorEnv(max_steps=100)
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.zeros(2))
    assert env.step_count == 10
    env.reset(seed=1)
    assert env.step_count == 0

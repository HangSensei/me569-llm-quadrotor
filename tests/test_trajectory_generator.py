"""Tests for the trajectory generator used to produce SysID data.

The generator rolls out the quadrotor dynamics with random but bounded
control sequences (zero-order hold) starting from randomized initial
states. The resulting arrays feed SINDy (E1) and MPC ablation studies.
"""
import numpy as np
import pytest

from me569_project.data.trajectory_generator import (
    TrajectoryData,
    generate_trajectories,
)
from me569_project.quadrotor_dynamics import QuadrotorParams


def test_generate_returns_correct_shapes():
    data = generate_trajectories(
        num_trajectories=3,
        steps_per_trajectory=50,
        dt=0.02,
        seed=0,
    )

    assert isinstance(data, TrajectoryData)
    assert data.states.shape == (3, 51, 6)        # initial + 50 post-step
    assert data.actions.shape == (3, 50, 2)
    assert data.next_states.shape == (3, 50, 6)
    assert data.states.dtype == np.float64
    assert data.actions.dtype == np.float64


def test_generate_is_deterministic_given_seed():
    d1 = generate_trajectories(num_trajectories=2, steps_per_trajectory=30, seed=42)
    d2 = generate_trajectories(num_trajectories=2, steps_per_trajectory=30, seed=42)

    np.testing.assert_array_equal(d1.states, d2.states)
    np.testing.assert_array_equal(d1.actions, d2.actions)


def test_generate_differs_across_seeds():
    d1 = generate_trajectories(num_trajectories=2, steps_per_trajectory=30, seed=1)
    d2 = generate_trajectories(num_trajectories=2, steps_per_trajectory=30, seed=2)

    assert not np.allclose(d1.states, d2.states)


def test_generate_actions_respect_bounds():
    params = QuadrotorParams()
    data = generate_trajectories(
        num_trajectories=5,
        steps_per_trajectory=100,
        params=params,
        seed=0,
    )

    assert np.all(data.actions >= 0.0)
    assert np.all(data.actions <= params.u_max)


def test_generate_states_are_finite():
    """With bounded actions around hover, the quadrotor should not diverge to inf/nan."""
    data = generate_trajectories(
        num_trajectories=5,
        steps_per_trajectory=200,
        seed=0,
    )

    assert np.all(np.isfinite(data.states))


def test_next_states_equal_states_shifted():
    """next_states[n, t] must equal states[n, t+1] by construction."""
    data = generate_trajectories(
        num_trajectories=2,
        steps_per_trajectory=20,
        seed=0,
    )

    np.testing.assert_array_equal(data.next_states, data.states[:, 1:, :])


def test_zero_order_hold_keeps_action_constant_within_window():
    """With hold_steps=5, actions within each 5-step window should be identical."""
    data = generate_trajectories(
        num_trajectories=1,
        steps_per_trajectory=20,
        hold_steps=5,
        seed=0,
    )

    actions = data.actions[0]
    for start in range(0, 20, 5):
        window = actions[start : start + 5]
        np.testing.assert_array_equal(window, np.broadcast_to(window[0], window.shape))


def test_hold_steps_one_gives_independent_actions():
    """With hold_steps=1, consecutive actions should differ (with overwhelming probability)."""
    data = generate_trajectories(
        num_trajectories=1,
        steps_per_trajectory=50,
        hold_steps=1,
        seed=0,
    )

    actions = data.actions[0]
    # At least some consecutive pairs should differ
    diffs = np.any(actions[1:] != actions[:-1], axis=1)
    assert diffs.sum() > 40, "Expected most consecutive actions to differ at hold_steps=1"


def test_initial_states_within_init_radius():
    data = generate_trajectories(
        num_trajectories=20,
        steps_per_trajectory=10,
        init_radius=0.4,
        seed=0,
    )

    initial_positions = data.states[:, 0, :2]  # (20, 2) - p_x, p_z
    assert np.all(np.abs(initial_positions) <= 0.4 + 1e-12)


def test_trajectory_data_stores_metadata():
    params = QuadrotorParams()
    data = generate_trajectories(
        num_trajectories=2,
        steps_per_trajectory=10,
        dt=0.015,
        params=params,
        seed=0,
    )

    assert data.dt == 0.015
    assert data.params is params


def test_trajectory_data_num_transitions_helper():
    """Helper property: total number of (state, action, next_state) transitions."""
    data = generate_trajectories(
        num_trajectories=4,
        steps_per_trajectory=25,
        seed=0,
    )
    assert data.num_transitions == 4 * 25


def test_trajectory_data_flatten_for_sysid():
    """flatten() should stack all trajectories into (N*T, ...) arrays for SINDy input."""
    data = generate_trajectories(
        num_trajectories=3,
        steps_per_trajectory=10,
        seed=0,
    )
    flat_states, flat_actions, flat_next = data.flatten()

    assert flat_states.shape == (30, 6)
    assert flat_actions.shape == (30, 2)
    assert flat_next.shape == (30, 6)

    # First 10 entries should match the first trajectory
    np.testing.assert_array_equal(flat_states[:10], data.states[0, :-1, :])
    np.testing.assert_array_equal(flat_next[:10], data.next_states[0])

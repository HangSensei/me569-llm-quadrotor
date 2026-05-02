"""Gymnasium-compatible Planar Quadrotor environment.

This env wraps the continuous quadrotor dynamics and RK4 integrator into a
discrete-time Gymnasium step API. It is used for three purposes in the
project:

1. **Data generation for SINDy (Experiment 1).** Random rollouts produce
   state-action trajectories that feed PySINDy for system identification.
2. **Closed-loop MPC simulation (Experiment 3).** The env is the plant; the
   MPC controller computes an action and calls ``step`` to advance it.
3. **RL training (stretch Experiment 4).** Standard Gymnasium-compatible
   interface so that Stable-Baselines3 PPO can train directly against it.

The default ``step`` reward is a simple negative quadratic hover cost. This
is a *baseline* reward only; the stretch Experiment 4 will override it with
LLM-generated reward functions (the Eureka-style methodology).

Observations use the same state vector as ``quadrotor_dynamics`` — see its
module docstring for the convention.

Gymnasium step API (returned 5-tuple):
    (obs, reward, terminated, truncated, info)
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step


class PlanarQuadrotorEnv(gym.Env):
    """Planar Quadrotor hover environment.

    Parameters
    ----------
    params : QuadrotorParams, optional
        Physical parameters. Defaults to the values in ``QuadrotorParams()``.
    dt : float
        Integration step size in seconds. 0.02 = 50 Hz, matches the typical
        control rate for quadrotor stabilization.
    max_steps : int
        Maximum episode length in env steps. At dt=0.02 and max_steps=500
        the episode lasts 10 seconds of simulated time.
    init_radius : float
        Radius of the uniform-random ball used to sample the initial state on
        each ``reset``. Position components are sampled in [-init_radius,
        init_radius]; attitude and velocities get narrower ranges.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: QuadrotorParams | None = None,
        dt: float = 0.02,
        max_steps: int = 500,
        init_radius: float = 0.5,
    ) -> None:
        super().__init__()
        self.params = params if params is not None else QuadrotorParams()
        self.dt = dt
        self.max_steps = max_steps
        self.init_radius = init_radius

        # Observation bounds: generous box around the operating region. These
        # are NOT hard constraints on the dynamics; they are just the Box
        # advertised to RL libraries. The env will terminate the episode if
        # the quadrotor leaves a smaller 5 m workspace (see ``step``).
        high_obs = np.array([5.0, 5.0, np.pi, 10.0, 10.0, 10.0], dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            dtype=np.float64,
        )

        # Action bounds: each rotor thrust in [0, u_max]
        self.action_space = spaces.Box(
            low=0.0,
            high=self.params.u_max,
            shape=(2,),
            dtype=np.float64,
        )

        self.state: np.ndarray = np.zeros(6, dtype=np.float64)
        self.step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample a new random initial state and reset the step counter.

        Initial state is drawn uniformly from a box around the origin:
        positions in [-init_radius, init_radius], attitude in [-0.2, 0.2] rad,
        velocities in [-0.5, 0.5] (linear) and [-0.5, 0.5] (angular).
        """
        super().reset(seed=seed)
        r = self.init_radius
        low = np.array([-r, -r, -0.2, -0.5, -0.5, -0.5])
        high = np.array([r, r, 0.2, 0.5, 0.5, 0.5])
        self.state = self.np_random.uniform(low=low, high=high).astype(np.float64)
        self.step_count = 0
        return self.state.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one dt step using RK4 with the (clipped) action."""
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            0.0,
            self.params.u_max,
        )
        self.state = rk4_step(self.state, action, self.dt, self.params)
        self.step_count += 1

        reward = self._compute_reward(self.state, action)
        terminated = bool(
            np.abs(self.state[0]) > 5.0 or np.abs(self.state[1]) > 5.0
        )
        truncated = self.step_count >= self.max_steps

        return self.state.copy(), reward, terminated, truncated, {}

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Default hover reward: negative quadratic cost.

        Position error dominates (weight 1.0), attitude error is half as
        important (0.5), velocity error is a small regularizer (0.05).

        This is the *baseline* reward only. The stretch Experiment 4 will
        replace it with LLM-generated reward functions (Eureka-style).
        """
        pos_err = float(state[0] ** 2 + state[1] ** 2)
        att_err = float(state[2] ** 2)
        vel_err = float(np.sum(state[3:] ** 2))
        return -(pos_err + 0.5 * att_err + 0.05 * vel_err)

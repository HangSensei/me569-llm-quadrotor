"""Trajectory generator for SysID and MPC ablation data.

Produces (state, action, next_state) transitions by rolling out the
Planar Quadrotor dynamics with random piecewise-constant control
sequences (zero-order hold) starting from uniformly-random initial
states inside a small ball around the hover equilibrium.

Design choices
--------------
**Action sampling.** Each control interval samples a pair of rotor
thrusts uniformly in ``[0.7 * u_hover, 1.3 * u_hover]``. This keeps the
dataset concentrated around the hover operating region where SINDy
linearization and the MPC hover task are meaningful, while still
providing enough excitation for basis functions to separate out.
Pure uniform sampling in ``[0, u_max]`` was rejected because the
quadrotor would diverge to the 5 m workspace boundary within a handful
of steps, and the resulting trajectories would be dominated by the
near-free-fall regime rather than controlled flight.

**Zero-order hold.** Control actions are held constant for
``hold_steps`` simulation steps (default 5, i.e. 0.1 s at dt=0.02).
This matches the typical control rate of 10 Hz for hover tasks and
produces lower-frequency excitation that is more useful for SysID.
Sampling a new action every single dt step (``hold_steps=1``) is
supported for testing but produces nearly-white noise that is harder
to identify from.

**Initial state.** Sampled uniformly from a box around the origin:
positions in ``[-init_radius, init_radius]``, attitude in
``[-0.2, 0.2] rad``, and velocities in ``[-0.5, 0.5]``. These ranges
match the default ``PlanarQuadrotorEnv.reset`` configuration.

Outputs
-------
``generate_trajectories`` returns a ``TrajectoryData`` dataclass with:

- ``states``     shape (N, T+1, 6)  — full state sequence including
                                       initial state
- ``actions``    shape (N, T,   2)  — applied control at each step
- ``next_states`` shape (N, T,  6)  — ``states[:, 1:, :]`` for convenience
- ``dt``, ``params`` — metadata propagated for downstream use

plus helper methods ``num_transitions`` and ``flatten`` to reshape the
data into the ``(N*T, ...)`` layout expected by PySINDy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step


@dataclass(frozen=True)
class TrajectoryData:
    """Container for a batch of rollout trajectories plus metadata."""

    states: np.ndarray       # shape (N, T+1, 6)
    actions: np.ndarray      # shape (N, T,   2)
    next_states: np.ndarray  # shape (N, T,   6)
    dt: float
    params: QuadrotorParams

    @property
    def num_trajectories(self) -> int:
        return int(self.actions.shape[0])

    @property
    def steps_per_trajectory(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_transitions(self) -> int:
        return self.num_trajectories * self.steps_per_trajectory

    def flatten(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack all trajectories into ``(N*T, ...)`` arrays for SINDy input.

        Returns (flat_states, flat_actions, flat_next_states), each with
        the first dimension equal to ``num_transitions``.
        """
        N, T, _ = self.actions.shape
        flat_states = self.states[:, :-1, :].reshape(N * T, 6)
        flat_actions = self.actions.reshape(N * T, 2)
        flat_next = self.next_states.reshape(N * T, 6)
        return flat_states, flat_actions, flat_next


def generate_trajectories(
    num_trajectories: int,
    steps_per_trajectory: int = 400,
    dt: float = 0.02,
    hold_steps: int = 5,
    action_perturbation: float = 0.3,
    init_radius: float = 0.3,
    params: QuadrotorParams | None = None,
    seed: int = 0,
) -> TrajectoryData:
    """Generate a batch of rollouts for SysID.

    Parameters
    ----------
    num_trajectories : int
        Number of independent rollouts to generate.
    steps_per_trajectory : int
        Number of simulation steps per rollout. At dt=0.02 and
        steps=400, each rollout lasts 8 seconds of simulated time.
    dt : float
        Integration step in seconds.
    hold_steps : int
        Number of consecutive steps over which each sampled action is
        held constant (zero-order hold). Must be >= 1.
    action_perturbation : float
        Relative perturbation around hover thrust. Each rotor thrust is
        drawn uniformly from ``[(1 - p) * u_hover, (1 + p) * u_hover]``.
        Default 0.3 gives ~30% thrust variation, which stays within a
        controllable envelope for several hundred steps.
    init_radius : float
        Radius of the position ball used for initial state sampling.
    params : QuadrotorParams, optional
        Physical parameters; defaults to ``QuadrotorParams()``.
    seed : int
        PRNG seed for reproducibility.

    Returns
    -------
    TrajectoryData
        Structured container with states, actions, next_states, and metadata.
    """
    if hold_steps < 1:
        raise ValueError(f"hold_steps must be >= 1, got {hold_steps}")
    if num_trajectories < 1:
        raise ValueError(f"num_trajectories must be >= 1, got {num_trajectories}")
    if steps_per_trajectory < 1:
        raise ValueError(
            f"steps_per_trajectory must be >= 1, got {steps_per_trajectory}"
        )

    params = params if params is not None else QuadrotorParams()
    rng = np.random.default_rng(seed)

    u_hover = params.u_hover
    u_low = (1.0 - action_perturbation) * u_hover
    u_high = (1.0 + action_perturbation) * u_hover

    # Initial state sampling ranges (match PlanarQuadrotorEnv.reset defaults)
    init_low = np.array(
        [-init_radius, -init_radius, -0.2, -0.5, -0.5, -0.5],
        dtype=np.float64,
    )
    init_high = np.array(
        [init_radius, init_radius, 0.2, 0.5, 0.5, 0.5],
        dtype=np.float64,
    )

    N = num_trajectories
    T = steps_per_trajectory

    states = np.zeros((N, T + 1, 6), dtype=np.float64)
    actions = np.zeros((N, T, 2), dtype=np.float64)

    for n in range(N):
        state = rng.uniform(low=init_low, high=init_high)
        states[n, 0] = state

        for t in range(T):
            if t % hold_steps == 0:
                current_action = rng.uniform(
                    low=u_low, high=u_high, size=2
                ).astype(np.float64)
            actions[n, t] = current_action
            state = rk4_step(state, current_action, dt, params)
            states[n, t + 1] = state

    next_states = states[:, 1:, :]

    return TrajectoryData(
        states=states,
        actions=actions,
        next_states=next_states,
        dt=dt,
        params=params,
    )

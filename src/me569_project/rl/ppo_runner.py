"""PPO training and evaluation helpers for Experiment 4.

Each B / P / Q condition is trained with stable-baselines3 PPO using
the same hyperparameters; only the reward function differs.
``train_ppo`` returns a trained model plus a learning-curve array.
``evaluate_ppo`` runs the trained policy on a fixed set of initial
states and returns metrics structurally compatible with the E2
evaluation harness so the two experiments are directly comparable.

Convention: the LLM-generated reward function is wrapped via
``LLMRewardWrapper`` before training; PPO never sees the raw env.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from me569_project.quadrotor_dynamics import QuadrotorParams, rk4_step
from me569_project.quadrotor_env import PlanarQuadrotorEnv
from me569_project.rl.llm_reward_env import LLMRewardWrapper


@dataclass(frozen=True)
class TrainResult:
    """Training output: trained model + learning curve."""

    model: PPO
    learning_curve_steps: np.ndarray  # (N,)
    learning_curve_rewards: np.ndarray  # (N,)
    train_time_s: float


@dataclass(frozen=True)
class EvalResult:
    """Closed-loop evaluation metrics matching E2's EvaluationResult schema."""

    num_rollouts: int
    hover_success_rate: float
    num_successful: int
    mean_settling_time_s: float
    mean_control_energy: float
    mean_final_position_error: float
    mean_final_attitude_error: float
    failed_due_to_crash: int
    failed_due_to_timeout: int


class _RewardLogger(BaseCallback):
    """Tracks ``ep_rew_mean`` after each PPO rollout for a learning curve."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[int] = []
        self.rewards: list[float] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # The Monitor wrapper that PPO inserts populates ep_info_buffer
        # with the most recent episodes. We grab the rolling-mean reward.
        info_buf = list(self.model.ep_info_buffer)
        if info_buf:
            mean_rew = float(np.mean([ep["r"] for ep in info_buf]))
        else:
            mean_rew = float("nan")
        self.steps.append(int(self.num_timesteps))
        self.rewards.append(mean_rew)


def _make_wrapped_env_factory(
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    max_steps: int,
    init_radius: float,
    crash_penalty: float,
    survival_bonus: float,
):
    """Closure-friendly factory for SubprocVecEnv (must be picklable at top level).

    Wraps in this order (innermost first):
        PlanarQuadrotorEnv         (raw thrust action [0, u_max], default reward)
        -> LLMRewardWrapper        (sees the RAW thrust, computes custom reward)
        -> HoverThrustOffsetWrapper (presents centered action to the agent;
                                     adds u_hover before forwarding inward)
        -> Monitor                 (populates ep_info_buffer for learning curves)

    The order matters: LLMRewardWrapper must be inside the offset wrapper
    so that it observes the actual thrust applied to the dynamics, not
    the agent's centered action. Otherwise the reward function (which
    expects raw thrust around u_hover) would see action ≈ 0 at hover and
    apply a huge spurious control penalty.
    """
    from stable_baselines3.common.monitor import Monitor

    from me569_project.rl.llm_reward_env import HoverThrustOffsetWrapper

    def _factory():
        env = PlanarQuadrotorEnv(max_steps=max_steps, init_radius=init_radius)
        env = LLMRewardWrapper(
            env,
            reward_fn=reward_fn,
            crash_penalty=crash_penalty,
            survival_bonus=survival_bonus,
        )
        env = HoverThrustOffsetWrapper(env)
        return Monitor(env)
    return _factory


def train_ppo(
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    total_timesteps: int = 300_000,
    seed: int = 0,
    n_envs: int = 8,
    max_steps_per_episode: int = 500,
    init_radius: float = 0.3,
    crash_penalty: float = -100.0,
    survival_bonus: float = 0.1,
    device: str = "cpu",
    use_subproc: bool = False,
) -> TrainResult:
    """Train a PPO agent on the LLM-reward-wrapped Planar Quadrotor env.

    Single-process ``DummyVecEnv`` is the default — much more reliable on
    macOS than ``SubprocVecEnv`` (mlx-vlm + multiprocessing can deadlock
    on the resource tracker). On a 16-core M4 the two are roughly the
    same wall-clock time for these small envs anyway.
    """
    import time

    factory = _make_wrapped_env_factory(
        reward_fn=reward_fn,
        max_steps=max_steps_per_episode,
        init_radius=init_radius,
        crash_penalty=crash_penalty,
        survival_bonus=survival_bonus,
    )
    if use_subproc:
        vec_env = SubprocVecEnv([factory for _ in range(n_envs)])
    else:
        vec_env = DummyVecEnv([factory for _ in range(n_envs)])

    # Reward normalization only (not observations). Without it, PPO's
    # value-function targets are dominated by the dense -2 to -3
    # per-step cost (-1000 to -1500 per episode), the policy gradient
    # swings wildly, and learning regresses. We deliberately keep
    # ``norm_obs=False`` so the trained policy operates on raw 6-D
    # state at eval time without needing the running-stats blob.
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        device=device,
        verbose=0,
    )

    callback = _RewardLogger()
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    train_time_s = time.time() - t0
    vec_env.close()

    return TrainResult(
        model=model,
        learning_curve_steps=np.asarray(callback.steps, dtype=np.int64),
        learning_curve_rewards=np.asarray(callback.rewards, dtype=np.float64),
        train_time_s=train_time_s,
    )


def evaluate_ppo(
    model: PPO,
    num_initial_states: int = 20,
    max_steps: int = 500,
    init_radius: float = 0.3,
    pos_tol: float = 0.15,
    att_tol: float = 0.15,
    settle_window: int = 25,
    workspace_bound: float = 5.0,
    seed: int = 0,
    deterministic: bool = True,
) -> EvalResult:
    """Closed-loop evaluation matching E2's evaluation API.

    Reuses the seeded uniform-box initial-state sampler from
    ``mpc.evaluation._sample_initial_states`` to give the same
    initial-state distribution as the E2 hover task; this makes the
    PPO policy's metrics directly comparable to E2's MPC metrics.
    """
    rng = np.random.default_rng(seed)
    r = init_radius
    low = np.array([-r, -r, -0.2, -0.5, -0.5, -0.5])
    high = np.array([r, r, 0.2, 0.5, 0.5, 0.5])
    initial_states = rng.uniform(low=low, high=high, size=(num_initial_states, 6))

    params = QuadrotorParams()
    u_hover_vec = np.array([params.u_hover, params.u_hover])
    dt = 0.02

    settled_steps: list[int] = []
    final_pos_errors: list[float] = []
    final_att_errors: list[float] = []
    control_energies: list[float] = []
    crashes = 0
    timeouts = 0

    for x0 in initial_states:
        state = np.asarray(x0, dtype=np.float64).copy()
        in_box_streak = 0
        settled_step = -1
        crashed = False
        control_energy = 0.0

        for step in range(max_steps):
            obs = state.reshape(1, 6).astype(np.float64)
            action, _ = model.predict(obs, deterministic=deterministic)
            # The trained policy lives in the HoverThrustOffsetWrapper's
            # action space (delta-thrust around hover). Apply the offset
            # before passing the action to the underlying dynamics.
            agent_action = np.asarray(action, dtype=np.float64).reshape(2)
            u = np.clip(agent_action + params.u_hover, 0.0, params.u_max)
            control_energy += float(np.sum((u - u_hover_vec) ** 2)) * dt

            state = rk4_step(state, u, dt, params)
            if abs(state[0]) > workspace_bound or abs(state[1]) > workspace_bound:
                crashed = True
                break

            pos_err = float(np.linalg.norm(state[:2]))
            att_err = float(abs(state[2]))
            in_box = pos_err < pos_tol and att_err < att_tol
            if in_box:
                in_box_streak += 1
                if settled_step < 0 and in_box_streak >= settle_window:
                    settled_step = step - settle_window + 1
            else:
                in_box_streak = 0

        final_pos_errors.append(float(np.linalg.norm(state[:2])))
        final_att_errors.append(float(abs(state[2])))
        control_energies.append(control_energy)

        if crashed:
            crashes += 1
        elif settled_step >= 0:
            settled_steps.append(settled_step)
        else:
            timeouts += 1

    num_successful = len(settled_steps)
    success_rate = num_successful / num_initial_states
    mean_settle = (
        float(np.mean(settled_steps) * dt)
        if settled_steps
        else float("nan")
    )

    return EvalResult(
        num_rollouts=num_initial_states,
        hover_success_rate=float(success_rate),
        num_successful=num_successful,
        mean_settling_time_s=mean_settle,
        mean_control_energy=float(np.mean(control_energies)),
        mean_final_position_error=float(np.mean(final_pos_errors)),
        mean_final_attitude_error=float(np.mean(final_att_errors)),
        failed_due_to_crash=crashes,
        failed_due_to_timeout=timeouts,
    )

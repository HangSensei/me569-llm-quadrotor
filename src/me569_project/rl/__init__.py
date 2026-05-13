"""Reinforcement-learning modules for Experiment 4 (Eureka-style reward design).

Public surface:

- ``baseline_reward(state, action)`` — textbook negative-quadratic
  hover reward used as the B condition.
- ``clip_reward(fn, low, high)`` — wrap an arbitrary reward function
  with a numeric safety clip; used on LLM-generated rewards before
  they enter the PPO training loop to prevent value-function blow-up.
- ``LLMRewardWrapper`` — Gymnasium ``Wrapper`` that injects an
  arbitrary ``reward(state, action) -> float`` callable into the
  closed-loop env, plus a uniform crash-penalty / survival-bonus /
  workspace-bound termination policy applied identically across all
  conditions.
- ``train_ppo`` and ``evaluate_ppo`` — thin sb3 wrappers that
  encapsulate the multi-seed PPO training and the E2-compatible
  evaluation harness.
"""
from __future__ import annotations

from me569_project.rl.eureka_rewards import baseline_reward, clip_reward
from me569_project.rl.llm_reward_env import LLMRewardWrapper

__all__ = [
    "baseline_reward",
    "clip_reward",
    "LLMRewardWrapper",
]

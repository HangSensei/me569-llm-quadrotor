"""Tests for the E4 Eureka-style RL reward prompt template.

These tests enforce the anti-leak rules from the E4 plan
(`docs/superpowers/plans/2026-05-03-e4-eureka-rl-plan.md` §6 / §13.4)
plus the carry-over lessons from D033 (no Bryson-shape worked example)
and D036 (use a dedicated system description, do not reuse the SINDy /
Koopman ones).
"""
import re

import pytest

from me569_project.llm.prompts import (
    EUREKA_REWARD_EOM_APPENDIX,
    EUREKA_REWARD_TASK_INSTRUCTIONS,
    EUREKA_SYSTEM_DESCRIPTION,
    KOOPMAN_SYSTEM_DESCRIPTION,
    SINDY_BASIS_SYSTEM_DESCRIPTION,
    build_eureka_reward_prompt,
)


# ---------------------------------------------------------------------------
# Anti-leak: forbidden hints in the assembled prompt body.
# ---------------------------------------------------------------------------

def test_task_body_does_not_mention_bryson_or_specific_weight_words():
    body = EUREKA_REWARD_TASK_INSTRUCTIONS.lower()
    forbidden = ["bryson", "quadratic", "lqr"]
    for token in forbidden:
        pattern = r"\b" + re.escape(token) + r"\b"
        assert re.search(pattern, body) is None, (
            f"task body leaks forbidden hint word: {token!r}"
        )


def test_task_body_does_not_contain_specific_weight_numbers():
    """No 10.0, 5.0, 0.1 etc — the audited weights from D033 / D028."""
    body = EUREKA_REWARD_TASK_INSTRUCTIONS
    # The body may contain the constant ``4.905`` (u_hover) as a
    # physics fact, but not the specific Bryson weights.
    forbidden_numbers = ["10.0 *", "5.0 *", "0.5 *", "0.1 *", "diag(10",  "diag(5"]
    for token in forbidden_numbers:
        assert token not in body, (
            f"task body leaks specific weight number: {token!r}"
        )


def test_task_body_does_not_show_worked_example_with_concrete_numbers():
    body = EUREKA_REWARD_TASK_INSTRUCTIONS
    # Specific numerical terms like ``state[0]**2`` followed by a coefficient.
    assert "10 * state" not in body
    assert "5 * state" not in body
    assert "10.0 *" not in body


def test_task_body_specifies_function_signature():
    body = EUREKA_REWARD_TASK_INSTRUCTIONS
    assert "def reward(state, action):" in body


def test_task_body_specifies_higher_is_better():
    body = EUREKA_REWARD_TASK_INSTRUCTIONS.lower()
    assert "higher" in body and "better" in body


def test_task_body_warns_against_io_and_random():
    body = EUREKA_REWARD_TASK_INSTRUCTIONS.lower()
    assert "no i/o" in body
    assert "random" in body


# ---------------------------------------------------------------------------
# System description distinct from SINDy / Koopman ones (D036 lesson).
# ---------------------------------------------------------------------------

def test_eureka_system_description_is_distinct_from_sindy():
    assert EUREKA_SYSTEM_DESCRIPTION != SINDY_BASIS_SYSTEM_DESCRIPTION


def test_eureka_system_description_is_distinct_from_koopman():
    assert EUREKA_SYSTEM_DESCRIPTION != KOOPMAN_SYSTEM_DESCRIPTION


def test_eureka_system_description_does_not_mention_sindy_or_edmdc():
    body = EUREKA_SYSTEM_DESCRIPTION.lower()
    forbidden = ["sindy", "stlsq", "edmdc", "sparse identification", "koopman"]
    for token in forbidden:
        assert token not in body, (
            f"EUREKA_SYSTEM_DESCRIPTION must not mention {token!r}"
        )


def test_eureka_system_description_mentions_ppo_and_eureka():
    body = EUREKA_SYSTEM_DESCRIPTION
    assert "PPO" in body
    assert "Eureka" in body


# ---------------------------------------------------------------------------
# EOM appendix.
# ---------------------------------------------------------------------------

def test_eom_appendix_is_separate_constant():
    assert "Equations of motion" not in EUREKA_REWARD_TASK_INSTRUCTIONS
    assert "Equations of motion" in EUREKA_REWARD_EOM_APPENDIX


def test_eom_appendix_includes_dynamics():
    body = EUREKA_REWARD_EOM_APPENDIX
    assert "v_x_dot" in body
    assert "theta_dot = omega" in body


# ---------------------------------------------------------------------------
# Builder behavior.
# ---------------------------------------------------------------------------

def test_build_prompt_default_excludes_eom():
    assert "Equations of motion" not in build_eureka_reward_prompt()


def test_build_prompt_with_eom_includes_dynamics():
    prompt = build_eureka_reward_prompt(include_eom=True)
    assert "Equations of motion" in prompt
    assert "v_x_dot" in prompt


def test_build_prompt_does_not_leak_sindy_or_edmdc_anywhere():
    for include_eom in (False, True):
        prompt = build_eureka_reward_prompt(include_eom=include_eom).lower()
        for token in ["sindy", "stlsq", "edmdc", "sparse identification", "koopman"]:
            assert token not in prompt, (
                f"assembled E4 prompt (include_eom={include_eom}) leaks {token!r}"
            )

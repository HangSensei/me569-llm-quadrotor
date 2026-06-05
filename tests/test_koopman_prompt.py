"""Tests for the Koopman observable prompt template (E3, external numbering).

These tests enforce the prompt anti-leak rules:

1. The Koopman task instructions do not name specific trig observables
   (no ``sin``, ``cos``, ``bilinear``) outside the EOM appendix.
2. The state-recovery convention (first 6 entries = state) is stated.
3. The output dim range [6, 50] is stated.
4. The EOM appendix is included only when ``include_eom=True`` and
   then contains the full dynamics.
5. The task body shows only the function shell — no worked example
   with specific observables.
6. The system description is the dedicated ``KOOPMAN_SYSTEM_DESCRIPTION``,
   not the SINDy one. It does NOT mention SINDy / STLSQ. It states the
   observable is state-only and ``u`` enters separately through ``B``.
"""
import re

import pytest

from me569_project.llm.prompts import (
    KOOPMAN_OBSERVABLE_EOM_APPENDIX,
    KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS,
    KOOPMAN_SYSTEM_DESCRIPTION,
    SINDY_BASIS_SYSTEM_DESCRIPTION,
    build_koopman_observable_prompt,
)


# ---------------------------------------------------------------------------
# Anti-leak: forbidden hints in task instructions.
# ---------------------------------------------------------------------------

def test_task_instructions_do_not_mention_sin_cos_or_bilinear():
    """The task body itself should not give away the trig structure or
    the word 'bilinear'. The EOM appendix is allowed to (and is added
    only on demand for the Q condition)."""
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS.lower()
    forbidden = ["sin", "cos", "bilinear"]
    for token in forbidden:
        # Catch as a whole word (so we don't fire on words containing 'sin'
        # like 'using' or 'sing'... wait, 'using' contains 'sin'. We use a
        # word-boundary regex that matches sin/cos/bilinear only.)
        pattern = r"\b" + re.escape(token) + r"\b"
        assert re.search(pattern, body) is None, (
            f"task instructions leak forbidden hint word: {token!r}"
        )


def test_task_instructions_do_not_show_example_observables():
    """No literal numeric observable values, no example trig terms.

    The function shell shows the first six state slots and a
    placeholder ``# ... your additional observables here ...``. That
    placeholder is not a worked example.
    """
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    # No occurrences of 'np.sin' or 'np.cos' (we use those in the EOM
    # appendix, not in the task body).
    assert "np.sin" not in body
    assert "np.cos" not in body
    # No specific worked observable like 'x[0] * np.sin(x[2])'.
    assert "* np." not in body


# ---------------------------------------------------------------------------
# Required content in task instructions.
# ---------------------------------------------------------------------------

def test_task_instructions_state_recovery_convention():
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    assert "first" in body.lower()
    assert "6" in body
    # The exact convention sentence.
    assert "FIRST six components of the output MUST equal the input state" in body


def test_task_instructions_specifies_dim_range():
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    assert "6 <= N <= 50" in body or "6 ≤ N ≤ 50" in body


def test_task_instructions_specifies_function_signature():
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    assert "def observables(x):" in body


def test_task_instructions_forbids_extra_imports():
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS.lower()
    assert "no other imports" in body


# ---------------------------------------------------------------------------
# EOM appendix.
# ---------------------------------------------------------------------------

def test_eom_appendix_is_separate_constant():
    """Appendix lives outside the task instructions string."""
    assert "Equations of motion" not in KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    assert "Equations of motion" in KOOPMAN_OBSERVABLE_EOM_APPENDIX


def test_eom_appendix_includes_dynamics():
    body = KOOPMAN_OBSERVABLE_EOM_APPENDIX
    assert "p_x_dot" in body
    assert "v_x_dot" in body
    assert "theta_dot = omega" in body


def test_build_prompt_without_eom_excludes_dynamics():
    prompt = build_koopman_observable_prompt(include_eom=False)
    assert "Equations of motion" not in prompt
    # Sanity: the Sindy system description (which IS in the prompt) does
    # NOT name 'sin' or 'cos' as words. (It mentions 'pitch-dependent
    # thrust projection' as physics intuition, which is the same level
    # of hint as the existing E1 prompt.)


def test_build_prompt_with_eom_includes_dynamics():
    prompt = build_koopman_observable_prompt(include_eom=True)
    assert "Equations of motion" in prompt
    assert "v_x_dot" in prompt


def test_build_prompt_default_excludes_eom():
    prompt = build_koopman_observable_prompt()
    assert "Equations of motion" not in prompt


# ---------------------------------------------------------------------------
# Post-audit checks: dedicated KOOPMAN_SYSTEM_DESCRIPTION, no SINDy
# leakage in the assembled prompt, observable explicitly state-only.
# ---------------------------------------------------------------------------

def test_koopman_system_description_does_not_mention_sindy_or_stlsq():
    """The Koopman system description must not reference SINDy / STLSQ.

    Mixing E1's SINDy framing with E3's EDMDc task body confused Q on
    the original run — Q's first def attempted to consume
    ``x[6]`` and ``x[7]`` as controls before self-correcting in later
    def blocks. The clean E3 system description below removes the
    SINDy mentions and the ``(p_x..u_2)`` listing that suggested
    controls belonged in the observable input.
    """
    body = KOOPMAN_SYSTEM_DESCRIPTION.lower()
    forbidden = ["sindy", "stlsq", "sparse identification", "sparse regression"]
    for token in forbidden:
        assert token not in body, (
            f"KOOPMAN_SYSTEM_DESCRIPTION must not mention {token!r}"
        )


def test_koopman_system_description_is_distinct_from_sindy():
    """Sanity: the two system descriptions are independent strings."""
    assert KOOPMAN_SYSTEM_DESCRIPTION != SINDY_BASIS_SYSTEM_DESCRIPTION


def test_koopman_system_description_marks_observable_as_state_only():
    body = KOOPMAN_SYSTEM_DESCRIPTION
    # Explicit state-only language must be present.
    assert "state-only" in body or "state ONLY" in body or "ONLY" in body
    # Reference to the EDMDc lifted-space form.
    assert "EDMDc" in body or "Extended Dynamic Mode Decomposition" in body


def test_koopman_system_description_explains_u_enters_through_B():
    body = KOOPMAN_SYSTEM_DESCRIPTION
    assert "B matrix" in body or "B u_k" in body or "B u" in body


def test_koopman_system_description_warns_against_consuming_controls():
    """The audit fix specifically tells the LLM not to consume u_1, u_2."""
    body = KOOPMAN_SYSTEM_DESCRIPTION
    # We allow either explicit phrasing.
    assert "NOT inputs to your function" in body or "not inputs to your function" in body


def test_built_prompt_does_not_contain_sindy_or_stlsq_anywhere():
    """Final defense: even after concatenation with task body and the
    optional EOM appendix, the assembled prompt must not mention SINDy
    or STLSQ. Otherwise Q would inherit the same E1 framing again."""
    for include_eom in (False, True):
        prompt = build_koopman_observable_prompt(include_eom=include_eom).lower()
        for token in ["sindy", "stlsq", "sparse identification", "sparse regression"]:
            assert token not in prompt, (
                f"assembled Koopman prompt (include_eom={include_eom}) "
                f"leaks {token!r}"
            )


def test_task_instructions_no_longer_reference_sindy_task_above():
    """The 'Input convention (same as the SINDy task above)' phrase
    was a copy-paste leftover from the E1 prompt. It is removed in the
    audited E3 prompt; the convention is stated directly."""
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    assert "same as the SINDy task above" not in body
    assert "SINDy task above" not in body


def test_task_instructions_warns_against_x6_x7():
    body = KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS
    assert "x[6]" in body  # the warning sentence references x[6]
    assert "MUST NOT" in body

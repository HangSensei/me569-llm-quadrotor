"""Sanity tests for the SINDy basis prompt template (E1) and the MPC
stage cost prompt template (E3)."""
from me569_project.llm.prompts import (
    MPC_COST_TASK_INSTRUCTIONS,
    SINDY_BASIS_SYSTEM_DESCRIPTION,
    SINDY_BASIS_TASK_INSTRUCTIONS,
    build_mpc_cost_prompt,
    build_sindy_basis_prompt,
)


def test_system_description_mentions_state_vector():
    assert "p_x, p_z, theta, v_x, v_z, omega" in SINDY_BASIS_SYSTEM_DESCRIPTION


def test_system_description_mentions_control_vector():
    assert "u_1" in SINDY_BASIS_SYSTEM_DESCRIPTION
    assert "u_2" in SINDY_BASIS_SYSTEM_DESCRIPTION


def test_system_description_does_not_leak_ground_truth_dynamics():
    """The prompt must NOT include the equations of motion — the LLM is
    being asked to rely on its own physical knowledge, not to translate
    a cheat sheet.
    """
    assert "v_x_dot" not in SINDY_BASIS_SYSTEM_DESCRIPTION
    assert "v_z_dot" not in SINDY_BASIS_SYSTEM_DESCRIPTION
    assert "omega_dot" not in SINDY_BASIS_SYSTEM_DESCRIPTION
    assert "sin(theta)" not in SINDY_BASIS_SYSTEM_DESCRIPTION
    assert "cos(theta)" not in SINDY_BASIS_SYSTEM_DESCRIPTION


def test_system_description_references_physical_knowledge_without_equations():
    """The prompt should point the LLM at its own physics knowledge, in
    natural language, without spelling out the dynamics.
    """
    text = SINDY_BASIS_SYSTEM_DESCRIPTION.lower()
    assert "planar quadrotor" in text
    assert "gravity" in text
    assert "thrust" in text


def test_task_instructions_specify_function_name_and_signature():
    assert "basis(xu)" in SINDY_BASIS_TASK_INSTRUCTIONS
    assert "length 8" in SINDY_BASIS_TASK_INSTRUCTIONS


def test_task_instructions_request_fenced_code():
    assert "```python" in SINDY_BASIS_TASK_INSTRUCTIONS
    assert "code fence" in SINDY_BASIS_TASK_INSTRUCTIONS


def test_build_sindy_basis_prompt_combines_both_sections():
    prompt = build_sindy_basis_prompt()
    assert SINDY_BASIS_SYSTEM_DESCRIPTION in prompt
    assert SINDY_BASIS_TASK_INSTRUCTIONS in prompt
    # Sections are separated by a blank line
    assert "\n\n" in prompt


# ----------------------------------------------------------------------
# E3 MPC stage cost prompt tests
# ----------------------------------------------------------------------


def test_mpc_task_specifies_stage_cost_function_name():
    assert "stage_cost(x, u)" in MPC_COST_TASK_INSTRUCTIONS


def test_mpc_task_lists_state_indices():
    text = MPC_COST_TASK_INSTRUCTIONS
    for i, name in enumerate(["p_x", "p_z", "theta", "v_x", "v_z", "omega"]):
        assert f"x[{i}] = {name}" in text


def test_mpc_task_lists_control_indices():
    assert "u[0] = u_1" in MPC_COST_TASK_INSTRUCTIONS
    assert "u[1] = u_2" in MPC_COST_TASK_INSTRUCTIONS


def test_mpc_task_forbids_numpy_and_imports():
    text = MPC_COST_TASK_INSTRUCTIONS.lower()
    assert "no ``np.array``" in MPC_COST_TASK_INSTRUCTIONS or "no ``np.array``" in text
    assert "you may not import anything" in text or "may not import anything" in text


def test_mpc_task_lists_allowed_math_helpers():
    """The injected CasADi math names must be advertised verbatim."""
    text = MPC_COST_TASK_INSTRUCTIONS
    for name in ("sin", "cos", "exp", "sqrt", "log", "fabs"):
        assert f"``{name}``" in text


def test_mpc_task_explains_why_no_numpy():
    """LLM should be told this is for CasADi symbolic propagation, not arbitrary."""
    text = MPC_COST_TASK_INSTRUCTIONS.lower()
    assert "casadi" in text
    assert "symbolic" in text


def test_mpc_task_does_not_leak_specific_Q_R_weights():
    """The example structure must be a generic skeleton, not the
    Bryson-rule textbook weights that would amount to the answer.
    """
    text = MPC_COST_TASK_INSTRUCTIONS
    # No Bryson position-weight pair as a literal numerical example.
    assert "10.0 * x[0] ** 2 + 10.0 * x[1] ** 2" not in text
    assert "10 * x[0] ** 2 + 10 * x[1] ** 2" not in text
    # No Bryson attitude weight as a literal numerical example.
    assert "5.0 * x[2] ** 2" not in text
    # No Bryson R-weight (0.1) appearing next to the hover-thrust offset.
    assert "0.1 * (u[0] - 4.905)" not in text
    # The example must use generic placeholder ellipses, not concrete weights.
    assert "pos = ..." in text or "pos = ...," in text


def test_build_mpc_cost_prompt_combines_system_and_task():
    prompt = build_mpc_cost_prompt()
    assert SINDY_BASIS_SYSTEM_DESCRIPTION in prompt
    assert MPC_COST_TASK_INSTRUCTIONS in prompt


def test_build_mpc_cost_prompt_does_not_leak_dynamics():
    """E3 prompt also must not contain the equations of motion."""
    prompt = build_mpc_cost_prompt()
    assert "v_x_dot" not in prompt
    assert "v_z_dot" not in prompt
    assert "omega_dot" not in prompt


def test_mpc_task_mentions_hover_equilibrium_value():
    """The LLM needs to know the hover thrust value (m*g/2) to write
    a sensible control deviation penalty.
    """
    # 4.905 = 1.0 * 9.81 / 2
    assert "4.905" in MPC_COST_TASK_INSTRUCTIONS

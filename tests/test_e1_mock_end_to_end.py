"""End-to-end integration test: mocked LLM response -> E1 P-condition metrics.

This test wires up the W3 infrastructure without calling a real LLM:

    mock response string
        |
        v
    strip_thinking_mode + extract_python_code   (llm.code_extraction)
        |
        v
    load_callable                                 (llm.sandbox)
        |
        v
    fit_with_basis_fn                             (sysid.llm_sysid_model)
        |
        v
    one_step_mse / rollout_mse                    (same)

When the real Qwen-Plus and Qwen3.5-4B clients land in the next session,
they replace the hardcoded ``mock_llm_response`` string and the rest of
the pipeline is unchanged.
"""
import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.code_extraction import extract_python_code
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn


# Simulated Qwen-Plus response: thinking preamble, natural-language
# framing, and a fenced Python code block defining basis(xu).
MOCK_LLM_RESPONSE = """<think>
The user wants a basis function for a planar quadrotor. The state is
[p_x, p_z, theta, v_x, v_z, omega] and the control is [u_1, u_2].
Gravity and thrust couple through sin(theta) and cos(theta), so I
should include trigonometric terms and the total thrust u_1 + u_2.
</think>

Here is a physics-informed basis function for the planar quadrotor.
It includes linear state and control terms, trig of the pitch angle,
the sum and difference of the rotor thrusts (which drive total thrust
and torque), and the couplings of total thrust with sin/cos of theta
that appear in the v_x and v_z equations of motion.

```python
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u1, u2 = xu
    total_thrust = u1 + u2
    return [
        1.0,
        p_x,
        p_z,
        theta,
        v_x,
        v_z,
        omega,
        np.sin(theta),
        np.cos(theta),
        u1,
        u2,
        total_thrust,
        u2 - u1,
        total_thrust * np.sin(theta),
        total_thrust * np.cos(theta),
    ]
```
"""


@pytest.fixture(scope="module")
def small_train_data():
    return generate_trajectories(
        num_trajectories=15,
        steps_per_trajectory=200,
        dt=0.02,
        hold_steps=5,
        seed=0,
    )


@pytest.fixture(scope="module")
def small_val_data():
    return generate_trajectories(
        num_trajectories=5,
        steps_per_trajectory=200,
        dt=0.02,
        hold_steps=5,
        seed=42,
    )


def test_mock_response_extracts_runnable_basis_function():
    """The full parse -> load pipeline should yield a callable basis."""
    code = extract_python_code(MOCK_LLM_RESPONSE)
    assert "def basis(xu):" in code
    assert "<think>" not in code
    assert "Here is a physics-informed" not in code

    basis_fn = load_callable(code, function_name="basis")
    sample_xu = np.array([0.1, 0.2, 0.05, 0.0, 0.0, 0.0, 4.9, 4.9])
    features = basis_fn(sample_xu)

    assert len(features) == 15
    # First feature is the constant bias
    assert features[0] == 1.0
    # Trig of theta
    assert features[7] == pytest.approx(np.sin(0.05))
    assert features[8] == pytest.approx(np.cos(0.05))


def test_mock_basis_can_fit_sindy_model(small_train_data):
    code = extract_python_code(MOCK_LLM_RESPONSE)
    basis_fn = load_callable(code)
    result = fit_with_basis_fn(small_train_data, basis_fn=basis_fn)

    assert result.n_basis == 15
    assert result.active_terms > 0
    assert result.Xi.shape == (15, 6)


def test_mock_pipeline_produces_finite_metrics(small_train_data, small_val_data):
    code = extract_python_code(MOCK_LLM_RESPONSE)
    basis_fn = load_callable(code)
    result = fit_with_basis_fn(small_train_data, basis_fn=basis_fn)

    m1 = result.one_step_mse(small_val_data)
    m10 = result.rollout_mse(small_val_data, horizon=10)
    m50 = result.rollout_mse(small_val_data, horizon=50)

    for m in (m1, m10, m50):
        assert m >= 0.0
        assert np.isfinite(m)

    # Rollout error should compound with horizon (weak monotonicity)
    assert m50 >= m10 * 0.5


def test_mock_physics_informed_basis_beats_linear_basis(
    small_train_data, small_val_data
):
    """A physics-informed basis (trig + thrust couplings) should have lower
    one-step MSE than a plain linear basis on the nonlinear quadrotor.
    """
    # Physics-informed basis from the mock response
    code = extract_python_code(MOCK_LLM_RESPONSE)
    physics_basis = load_callable(code)
    physics_result = fit_with_basis_fn(small_train_data, basis_fn=physics_basis)
    m_physics = physics_result.one_step_mse(small_val_data)

    # Plain linear basis as a weak contender
    linear_basis = lambda xu: [1.0] + list(xu)
    linear_result = fit_with_basis_fn(small_train_data, basis_fn=linear_basis)
    m_linear = linear_result.one_step_mse(small_val_data)

    assert m_physics < m_linear, (
        f"Physics-informed basis ({m_physics:.3f}) should beat "
        f"linear-only basis ({m_linear:.3f}) on the nonlinear quadrotor"
    )


def test_mock_qwen_style_malformed_response_raises_cleanly():
    """An LLM response without a valid code block should fail predictably."""
    malformed = "<think>I don't know how to do this</think>\n\nSorry, I can't help."

    code = extract_python_code(malformed)
    # The cleaned text is just "Sorry, I can't help." which is not valid Python;
    # the sandbox should raise SandboxError on exec.
    from me569_project.llm.sandbox import SandboxError
    with pytest.raises(SandboxError):
        load_callable(code, function_name="basis")

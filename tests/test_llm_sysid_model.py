"""Tests for the basis-function-driven SysID model used in E1 P/Q conditions."""
import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.sysid.llm_sysid_model import (
    LLMBasisSysIDModel,
    fit_with_basis_fn,
)


def _linear_basis(xu):
    """Minimal test basis: constant + identity on all 8 features."""
    return [1.0] + list(xu)


def _nonlinear_basis(xu):
    """A slightly richer basis with trig on theta and interaction terms."""
    return [
        1.0,
        xu[0], xu[1], xu[2], xu[3], xu[4], xu[5],
        np.sin(xu[2]), np.cos(xu[2]),
        xu[0] * xu[5],
        xu[6], xu[7],
        xu[6] + xu[7],
        (xu[6] + xu[7]) * np.sin(xu[2]),
        (xu[6] + xu[7]) * np.cos(xu[2]),
        xu[7] - xu[6],
    ]


@pytest.fixture(scope="module")
def train_data():
    return generate_trajectories(
        num_trajectories=20,
        steps_per_trajectory=200,
        dt=0.02,
        hold_steps=5,
        seed=0,
    )


@pytest.fixture(scope="module")
def val_data():
    return generate_trajectories(
        num_trajectories=5,
        steps_per_trajectory=200,
        dt=0.02,
        hold_steps=5,
        seed=42,
    )


def test_fit_runs_on_linear_basis(train_data):
    result = fit_with_basis_fn(
        train_data,
        basis_fn=_linear_basis,
        feature_names=["1"] + [f"xu{i}" for i in range(8)],
    )
    assert isinstance(result, LLMBasisSysIDModel)


def test_Xi_has_expected_shape(train_data):
    """Xi should be (n_basis, n_state) = (9, 6) for the linear basis."""
    result = fit_with_basis_fn(train_data, basis_fn=_linear_basis)
    assert result.Xi.shape == (9, 6)


def test_active_terms_is_positive_for_nontrivial_basis(train_data):
    result = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis, threshold=0.01)
    assert result.active_terms > 0


def test_higher_threshold_gives_sparser_model(train_data):
    sparse = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis, threshold=1.0)
    dense = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis, threshold=0.001)
    assert sparse.active_terms <= dense.active_terms


def test_predict_returns_correct_shape_and_finite(train_data):
    result = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis)
    state = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = np.array([train_data.params.u_hover, train_data.params.u_hover])

    x_dot = result.predict(state, action)
    assert x_dot.shape == (6,)
    assert np.all(np.isfinite(x_dot))


def test_one_step_mse_is_nonneg(train_data, val_data):
    result = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis)
    mse = result.one_step_mse(val_data)
    assert mse >= 0.0
    assert np.isfinite(mse)


def test_rollout_mse_grows_with_horizon(train_data, val_data):
    result = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis)
    m10 = result.rollout_mse(val_data, horizon=10)
    m50 = result.rollout_mse(val_data, horizon=50)
    assert m50 >= m10 * 0.5  # weak monotonicity allowance


def test_basis_size_is_auto_detected(train_data):
    """We should not have to tell the fitter how many basis features there are."""
    def small_basis(xu):
        return [1.0, xu[0], xu[1]]

    result = fit_with_basis_fn(train_data, basis_fn=small_basis)
    assert result.Xi.shape == (3, 6)
    assert result.n_basis == 3


def test_nontrivial_basis_can_beat_pure_constant(train_data, val_data):
    """A nonlinear basis should outperform a constant-only basis on one-step MSE."""
    constant_result = fit_with_basis_fn(train_data, basis_fn=lambda xu: [1.0])
    nonlinear_result = fit_with_basis_fn(train_data, basis_fn=_nonlinear_basis)

    m_constant = constant_result.one_step_mse(val_data)
    m_nonlinear = nonlinear_result.one_step_mse(val_data)

    assert m_nonlinear < m_constant, (
        f"Nonlinear basis ({m_nonlinear:.3f}) should do better than "
        f"constant-only ({m_constant:.3f})"
    )

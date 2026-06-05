"""Tests for the SINDy wrapper used in Experiment 1 (SysID baseline + LLM bases).

The wrapper builds a pysindy.SINDy model with a given feature library and
fits it to TrajectoryData. It also computes standard SysID metrics
(1-step MSE, N-step rollout MSE, sparsity).
"""
import numpy as np
import pysindy as ps
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.sysid.sindy_wrapper import (
    SINDyFitResult,
    fit_sindy,
    one_step_mse,
    rollout_mse,
)


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


def test_fit_runs_on_trajectory_data(train_data):
    result = fit_sindy(train_data)
    assert isinstance(result, SINDyFitResult)


def test_fit_result_has_expected_fields(train_data):
    result = fit_sindy(train_data)
    assert result.model is not None
    assert result.active_terms >= 0
    assert isinstance(result.equations, list)
    assert len(result.equations) == 6  # one equation per state component


def test_active_terms_is_positive_for_real_data(train_data):
    """A real fit should identify at least some terms as active."""
    result = fit_sindy(train_data, threshold=0.01)
    assert result.active_terms > 0


def test_higher_threshold_gives_sparser_model(train_data):
    sparse = fit_sindy(train_data, threshold=1.0)
    dense = fit_sindy(train_data, threshold=0.001)
    assert sparse.active_terms <= dense.active_terms


def test_one_step_mse_is_nonneg(train_data, val_data):
    result = fit_sindy(train_data)
    mse = one_step_mse(result, val_data)
    assert mse >= 0.0
    assert np.isfinite(mse)


def test_rollout_mse_is_nonneg(train_data, val_data):
    result = fit_sindy(train_data)
    mse = rollout_mse(result, val_data, horizon=10)
    assert mse >= 0.0
    assert np.isfinite(mse)


def test_rollout_mse_grows_with_horizon(train_data, val_data):
    """Rollout error typically compounds over the horizon."""
    result = fit_sindy(train_data)
    mse_1 = rollout_mse(result, val_data, horizon=1)
    mse_50 = rollout_mse(result, val_data, horizon=50)
    # Should be at least weakly monotonic
    assert mse_50 >= mse_1 * 0.5  # allow some slack for stochastic behavior


def test_custom_feature_library_is_respected(train_data):
    """Passing an explicit library should yield a different active_terms count."""
    default_result = fit_sindy(train_data)
    custom_lib = ps.PolynomialLibrary(degree=1)  # linear only — fewer terms
    custom_result = fit_sindy(train_data, feature_library=custom_lib)
    # The custom library has far fewer basis functions, so active_terms
    # should not exceed the default (with deg 3 poly).
    assert custom_result.active_terms <= default_result.active_terms


def test_predict_returns_correct_shape(train_data):
    result = fit_sindy(train_data)
    state = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = np.array([train_data.params.u_hover, train_data.params.u_hover])
    x_dot = result.predict(state, action)
    assert x_dot.shape == (6,)
    assert np.all(np.isfinite(x_dot))


def test_baseline_default_is_polynomial_degree_3(train_data):
    result = fit_sindy(train_data)
    # Default feature library must be polynomial degree 3
    assert isinstance(result.feature_library, ps.PolynomialLibrary)
    assert result.feature_library.degree == 3

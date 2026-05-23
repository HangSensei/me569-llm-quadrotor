"""Unit tests for the data-efficiency sweep helpers."""
from __future__ import annotations

import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.sysid.llm_sysid_model import (
    build_design_matrix,
    fit_from_design_matrix,
)
from me569_project.sysid.polynomial_basis import make_polynomial_basis_fn

from scripts.run_data_efficiency import subsample_indices, sweep_condition_e1


def test_subsample_indices_deterministic_under_seed():
    a = subsample_indices(1000, 50, np.random.default_rng(7))
    b = subsample_indices(1000, 50, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert len(a) == 50
    assert len(np.unique(a)) == 50  # no replacement
    assert a.max() < 1000 and a.min() >= 0


def test_subsample_indices_rejects_oversized_request():
    with pytest.raises(ValueError):
        subsample_indices(10, 11, np.random.default_rng(0))


def test_fit_from_design_matrix_returns_valid_model():
    data = generate_trajectories(num_trajectories=4, steps_per_trajectory=50, seed=0)
    basis_fn, names = make_polynomial_basis_fn(degree=2, n_input=8)
    Phi, Y, n_basis = build_design_matrix(data, basis_fn)
    m = fit_from_design_matrix(Phi, Y, basis_fn, n_basis, threshold=0.1)
    assert m.n_basis == n_basis
    assert np.all(np.isfinite(m.Xi))


def test_sweep_more_data_does_not_blow_up_mse():
    """A structurally adequate basis should not get *worse* with more data."""
    train = generate_trajectories(num_trajectories=20, steps_per_trajectory=100, seed=0)
    val = generate_trajectories(num_trajectories=5, steps_per_trajectory=100, seed=42)
    basis_fn, names = make_polynomial_basis_fn(degree=3, n_input=8)
    rows = sweep_condition_e1(
        condition="B", basis_fn=basis_fn, feature_names=names,
        train_pool=train, val_data=val, ladder=[200, 2000], seeds=[0],
    )
    by_n = {r["n_samples"]: r["val_one_step_mse"] for r in rows}
    assert by_n[2000] <= by_n[200] * 5.0  # generous: more data isn't catastrophic


def test_sweep_val_set_is_disjoint_from_train_pool():
    train = generate_trajectories(num_trajectories=3, steps_per_trajectory=20, seed=0)
    val = generate_trajectories(num_trajectories=3, steps_per_trajectory=20, seed=42)
    # Different seeds -> different initial states -> disjoint trajectories.
    assert not np.allclose(train.states[:, 0, :], val.states[:, 0, :])


def test_sweep_condition_e3_does_not_blow_up_mse():
    from scripts.run_data_efficiency import sweep_condition_e3
    from me569_project.sysid.koopman_baselines import polynomial_observable

    train = generate_trajectories(num_trajectories=20, steps_per_trajectory=100, seed=0)
    val = generate_trajectories(num_trajectories=5, steps_per_trajectory=100, seed=42)
    rows = sweep_condition_e3(
        condition="B", observable_fn=polynomial_observable,
        train_pool=train, val_data=val, ladder=[200, 2000], seeds=[0],
    )
    by_n = {r["n_samples"]: r["val_one_step_mse"] for r in rows}
    assert by_n[2000] <= by_n[200] * 5.0

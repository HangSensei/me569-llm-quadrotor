"""Tests for the hand-built baseline observable functions used in E3 (Koopman/EDMDc).

The B condition of E3 uses ``polynomial_observable`` (which is expected
to underperform like the SINDy polynomial baseline did in E1). We also
provide a ``physics_aware_observable`` for testing the fitter on a
known-good observable set.

These tests verify:

1. Each observable returns the expected dimension.
2. Each observable satisfies the state-recovery convention
   (first 6 entries = input state).
3. Each observable runs without error on a random sample of inputs.
"""
import numpy as np
import pytest

from me569_project.sysid.koopman_baselines import (
    polynomial_observable,
    physics_aware_observable,
)


def test_polynomial_observable_returns_27_components():
    x = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05])
    psi = polynomial_observable(x)
    assert psi.shape == (27,), f"expected 27 dims, got {psi.shape}"


def test_polynomial_observable_first_six_are_state():
    x = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05])
    psi = polynomial_observable(x)
    np.testing.assert_allclose(psi[:6], x, atol=1e-12)


def test_polynomial_observable_smoke_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=6)
        psi = polynomial_observable(x)
        assert np.all(np.isfinite(psi))


def test_polynomial_observable_rejects_wrong_dim():
    with pytest.raises(ValueError):
        polynomial_observable(np.zeros(5))


def test_physics_aware_observable_returns_16_components():
    x = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05])
    psi = physics_aware_observable(x)
    assert psi.shape == (16,)


def test_physics_aware_observable_first_six_are_state():
    x = np.array([1.0, 1.0, 0.7, -0.5, 0.5, -0.3])
    psi = physics_aware_observable(x)
    np.testing.assert_allclose(psi[:6], x, atol=1e-12)


def test_physics_aware_observable_includes_sin_cos_theta():
    th = 0.123
    x = np.array([0.0, 0.0, th, 0.0, 0.0, 0.0])
    psi = physics_aware_observable(x)
    # By layout, sin(theta) is at index 6, cos(theta) at index 7.
    assert np.isclose(psi[6], np.sin(th))
    assert np.isclose(psi[7], np.cos(th))

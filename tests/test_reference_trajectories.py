"""Tests for time-varying reference trajectories used by tracking MPC."""
import numpy as np

from me569_project.mpc.reference_trajectories import (
    figure_eight_reference,
    step_reference,
)


def test_step_reference_before_step_is_origin():
    ref = step_reference(1.9, step_time=2.0, target_x=1.0, target_z=0.5)
    np.testing.assert_allclose(ref, np.zeros(6))


def test_step_reference_after_step_is_target():
    ref = step_reference(2.1, step_time=2.0, target_x=1.0, target_z=0.5)
    np.testing.assert_allclose(ref, np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]))


def test_figure_eight_is_periodic():
    ref_0 = figure_eight_reference(1.25, period=8.0, amplitude=1.0)
    ref_1 = figure_eight_reference(9.25, period=8.0, amplitude=1.0)
    np.testing.assert_allclose(ref_0, ref_1, atol=1e-12)


def test_figure_eight_passes_through_origin():
    ref = figure_eight_reference(0.0, period=8.0, amplitude=1.0)
    np.testing.assert_allclose(ref, np.zeros(6), atol=1e-12)

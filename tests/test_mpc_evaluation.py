"""Tests for the closed-loop MPC evaluation harness.

These tests are slower than the rest of the suite (each rollout
involves multiple Ipopt solves) so they use a small number of
initial states and a short max_steps. The full E3 sweep in
scripts/run_e3_full.py uses the production defaults.
"""
import math

import numpy as np
import pytest

from me569_project.mpc.baseline_cost import make_baseline_stage_cost
from me569_project.mpc.evaluation import (
    EvaluationResult,
    evaluate_controller,
)
from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
from me569_project.quadrotor_dynamics import QuadrotorParams


@pytest.fixture(scope="module")
def baseline_mpc():
    params = QuadrotorParams()
    cost_fn = make_baseline_stage_cost(params=params)
    return PlanarQuadrotorMPC(
        params=params,
        stage_cost=cost_fn,
        config=MPCConfig(horizon=15),
    )


def test_evaluate_returns_evaluation_result(baseline_mpc):
    result = evaluate_controller(
        baseline_mpc,
        num_initial_states=2,
        max_steps=200,
        seed=0,
    )
    assert isinstance(result, EvaluationResult)


def test_evaluate_records_correct_rollout_count(baseline_mpc):
    result = evaluate_controller(
        baseline_mpc,
        num_initial_states=3,
        max_steps=200,
        seed=0,
    )
    assert result.num_rollouts == 3


def test_baseline_mpc_succeeds_on_small_perturbations(baseline_mpc):
    """The textbook quadratic baseline should achieve high success on
    small initial perturbations.
    """
    result = evaluate_controller(
        baseline_mpc,
        num_initial_states=4,
        max_steps=400,
        init_radius=0.2,
        seed=7,
    )
    assert result.hover_success_rate >= 0.5, (
        f"Baseline expected to converge on most rollouts, got {result.hover_success_rate:.2f}"
    )


def test_evaluate_metrics_have_expected_types(baseline_mpc):
    result = evaluate_controller(
        baseline_mpc,
        num_initial_states=2,
        max_steps=100,
        seed=0,
    )
    assert isinstance(result.hover_success_rate, float)
    assert isinstance(result.num_successful, int)
    assert isinstance(result.mean_control_energy, float)
    assert isinstance(result.mean_final_position_error, float)
    # mean settling time may be NaN if no rollout converged in 100 steps
    assert isinstance(result.mean_settling_time_s, float)


def test_evaluate_seed_determinism(baseline_mpc):
    """Same seed -> same initial states -> reproducible metrics.

    Tolerance is loose (rel=1e-4) because Ipopt's internal heuristics
    introduce ~1e-7 relative noise across calls even when reusing the
    same MPC instance — that is well below anything we care about for
    the E3 condition comparison.
    """
    r1 = evaluate_controller(
        baseline_mpc, num_initial_states=2, max_steps=100, seed=42
    )
    r2 = evaluate_controller(
        baseline_mpc, num_initial_states=2, max_steps=100, seed=42
    )
    assert r1.hover_success_rate == r2.hover_success_rate
    assert r1.mean_control_energy == pytest.approx(r2.mean_control_energy, rel=1e-4)
    assert r1.mean_final_position_error == pytest.approx(
        r2.mean_final_position_error, rel=1e-4
    )


def test_evaluate_control_energy_is_non_negative(baseline_mpc):
    result = evaluate_controller(
        baseline_mpc, num_initial_states=2, max_steps=100, seed=0
    )
    assert result.mean_control_energy >= 0.0


def test_evaluate_failed_counters_sum_to_failures(baseline_mpc):
    result = evaluate_controller(
        baseline_mpc, num_initial_states=4, max_steps=100, seed=1
    )
    failures = result.num_rollouts - result.num_successful
    counted = result.failed_due_to_crash + result.failed_due_to_timeout
    assert counted == failures


def test_evaluate_no_nan_in_finite_metrics(baseline_mpc):
    result = evaluate_controller(
        baseline_mpc, num_initial_states=2, max_steps=200, seed=0
    )
    assert math.isfinite(result.hover_success_rate)
    assert math.isfinite(result.mean_control_energy)
    assert math.isfinite(result.mean_final_position_error)
    assert math.isfinite(result.mean_final_attitude_error)

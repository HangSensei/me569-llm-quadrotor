"""Integration tests for tracking MPC and tracking evaluation."""
import math

import numpy as np

from me569_project.mpc.evaluation import TrackingResult, evaluate_tracking
from me569_project.mpc.reference_trajectories import step_reference
from me569_project.mpc.tracking_cost import make_tracking_stage_cost
from me569_project.mpc.tracking_mpc import TrackingMPCConfig, PlanarQuadrotorTrackingMPC
from me569_project.quadrotor_dynamics import QuadrotorParams


def test_tracking_mpc_constructor_runs_without_error():
    params = QuadrotorParams()
    reference_fn = lambda t: step_reference(t, step_time=0.2, target_x=0.2, target_z=0.1)
    stage_cost = make_tracking_stage_cost(reference_fn=reference_fn, params=params)

    mpc = PlanarQuadrotorTrackingMPC(
        params=params,
        reference_fn=reference_fn,
        stage_cost=stage_cost,
        config=TrackingMPCConfig(horizon=10),
    )

    assert mpc is not None


def test_evaluate_tracking_reports_finite_metrics():
    params = QuadrotorParams()
    reference_fn = lambda t: step_reference(t, step_time=0.2, target_x=0.2, target_z=0.1)
    stage_cost = make_tracking_stage_cost(reference_fn=reference_fn, params=params)
    mpc = PlanarQuadrotorTrackingMPC(
        params=params,
        reference_fn=reference_fn,
        stage_cost=stage_cost,
        config=TrackingMPCConfig(horizon=10),
    )

    result = evaluate_tracking(
        mpc=mpc,
        reference_fn=reference_fn,
        params=params,
        num_initial_states=1,
        max_steps=80,
        seed=0,
    )

    assert isinstance(result, TrackingResult)
    assert math.isfinite(result.mean_tracking_error)
    assert math.isfinite(result.mean_control_energy)
    assert result.num_rollouts == 1
    assert 0.0 <= result.tracking_success_rate <= 1.0

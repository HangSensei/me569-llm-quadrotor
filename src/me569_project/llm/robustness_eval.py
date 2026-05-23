"""Per-task evaluators for prompt-robustness.

Each takes a raw LLM response string and returns a ParaphraseResult:
  emitted  -> did it yield a usable, task-validated artifact?
  quality  -> the task's primary quality metric (NaN if not emitted /
              not evaluated):
                E1, E3 -> validation one-step MSE (lower better)
                E2, E4 -> mean control energy (lower better)
Heavy deps (CasADi MPC, SB3 PPO) are imported lazily inside the
functions that need them so the module imports cheaply for tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from me569_project.data.trajectory_generator import TrajectoryData
from me569_project.llm.code_extraction import CodeExtractionError, extract_python_code
from me569_project.llm.sandbox import SandboxError, load_callable


@dataclass(frozen=True)
class ParaphraseResult:
    task: str
    emitted: bool
    quality: float  # NaN when not emitted / not evaluated
    status: str     # "ok", "extract_failed", "sandbox_failed", "validate_failed",
                    #  "build_failed", "eval_failed", "emitted_no_eval"
    code: str | None
    error: str


def _fail(task: str, status: str, error: str, code: str | None = None) -> ParaphraseResult:
    return ParaphraseResult(task, False, float("nan"), status, code, error)


def _extract(task: str, response: str) -> tuple[str | None, ParaphraseResult | None]:
    try:
        return extract_python_code(response), None
    except CodeExtractionError as e:
        return None, _fail(task, "extract_failed", str(e))


def evaluate_e1_response(
    response: str, train_data: TrajectoryData, val_data: TrajectoryData
) -> ParaphraseResult:
    from me569_project.sysid.llm_sysid_model import fit_with_basis_fn

    code, fail = _extract("E1", response)
    if fail:
        return fail
    try:
        basis_fn = load_callable(code, function_name="basis")
    except SandboxError as e:
        return _fail("E1", "sandbox_failed", str(e), code)
    try:
        sample = np.array([0.1, 0.2, 0.05, 0.1, 0.0, 0.0, 4.9, 4.9])
        _ = list(basis_fn(sample))  # probe it runs
    except Exception as e:  # noqa: BLE001
        return _fail("E1", "validate_failed", f"{type(e).__name__}: {e}", code)
    try:
        model = fit_with_basis_fn(train_data, basis_fn=basis_fn, threshold=0.1)
        mse = model.one_step_mse(val_data)
    except Exception as e:  # noqa: BLE001
        return _fail("E1", "eval_failed", f"{type(e).__name__}: {e}", code)
    return ParaphraseResult("E1", True, float(mse), "ok", code, "")


def evaluate_e3_response(
    response: str, train_data: TrajectoryData, val_data: TrajectoryData
) -> ParaphraseResult:
    from me569_project.sysid.edmdc import edmdc_one_step_mse, fit_edmdc

    code, fail = _extract("E3", response)
    if fail:
        return fail
    try:
        obs_fn = load_callable(code, function_name="observables")
    except SandboxError as e:
        return _fail("E3", "sandbox_failed", str(e), code)
    # State-recovery convention probe (mirrors run_koopman acceptance gate).
    try:
        sentinel = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05])
        out = np.asarray(obs_fn(sentinel), dtype=np.float64).ravel()
        ok = (np.all(np.isfinite(out)) and 6 <= out.shape[0] <= 50
              and np.allclose(out[:6], sentinel, atol=1e-10))
        if not ok:
            return _fail("E3", "validate_failed", "observable failed sentinel checks", code)
    except Exception as e:  # noqa: BLE001
        return _fail("E3", "validate_failed", f"{type(e).__name__}: {e}", code)
    try:
        X, U, Xn = train_data.flatten()
        Xv, Uv, Xvn = val_data.flatten()
        res = fit_edmdc(X, U, Xn, obs_fn, observable_name="robustness")
        mse = edmdc_one_step_mse(res, Xv, Uv, Xvn)
    except Exception as e:  # noqa: BLE001
        return _fail("E3", "eval_failed", f"{type(e).__name__}: {e}", code)
    return ParaphraseResult("E3", True, float(mse), "ok", code, "")


def evaluate_e2_response(response: str, params=None) -> ParaphraseResult:
    from me569_project.mpc.evaluation import evaluate_controller
    from me569_project.mpc.mpc_math import load_stage_cost_callable
    from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
    from me569_project.quadrotor_dynamics import QuadrotorParams

    if params is None:
        params = QuadrotorParams()
    code, fail = _extract("E2", response)
    if fail:
        return fail
    try:
        stage_cost = load_stage_cost_callable(code, function_name="stage_cost")
    except SandboxError as e:
        return _fail("E2", "sandbox_failed", str(e), code)
    try:
        mpc = PlanarQuadrotorMPC(stage_cost=stage_cost, params=params,
                                 config=MPCConfig(horizon=20))
    except Exception as e:  # noqa: BLE001
        return _fail("E2", "build_failed", f"{type(e).__name__}: {e}", code)
    try:
        result = evaluate_controller(mpc=mpc, params=params,
                                     num_initial_states=20, max_steps=500, seed=0)
    except Exception as e:  # noqa: BLE001
        return _fail("E2", "eval_failed", f"{type(e).__name__}: {e}", code)
    return ParaphraseResult("E2", True, float(result.mean_control_energy), "ok", code, "")


def evaluate_e4_response(
    response: str, total_timesteps: int = 300_000, train: bool = True, seed: int = 0
) -> ParaphraseResult:
    from me569_project.rl.eureka_rewards import clip_reward

    code, fail = _extract("E4", response)
    if fail:
        return fail
    try:
        raw_fn = load_callable(code, function_name="reward")
    except SandboxError as e:
        return _fail("E4", "sandbox_failed", str(e), code)
    # Sentinel validation: finite, and higher at hover than at disturbed states
    # (mirrors run_eureka._validate_reward_function).
    states = [np.zeros(6), np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05]),
              np.array([1.0, 1.0, 0.5, -0.5, 0.5, -0.5])]
    a = np.array([4.905, 4.905])
    try:
        rewards = [float(raw_fn(s, a)) for s in states]
    except Exception as e:  # noqa: BLE001
        return _fail("E4", "validate_failed", f"{type(e).__name__}: {e}", code)
    if not all(math.isfinite(r) for r in rewards):
        return _fail("E4", "validate_failed", "reward non-finite on sentinel", code)
    if not (rewards[0] > rewards[1] and rewards[0] > rewards[2]):
        return _fail("E4", "validate_failed", "reward not maximal at hover", code)

    if not train:
        return ParaphraseResult("E4", True, float("nan"), "emitted_no_eval", code, "")

    from me569_project.rl.ppo_runner import evaluate_ppo, train_ppo
    try:
        fn = clip_reward(raw_fn, low=-1000.0, high=1000.0)
        tr = train_ppo(reward_fn=fn, total_timesteps=total_timesteps, seed=seed,
                       n_envs=8, init_radius=0.15, device="cpu")
        metrics = evaluate_ppo(tr.model, num_initial_states=20, init_radius=0.30, seed=0)
    except Exception as e:  # noqa: BLE001
        return _fail("E4", "eval_failed", f"{type(e).__name__}: {e}", code)
    return ParaphraseResult("E4", True, float(metrics.mean_control_energy), "ok", code, "")

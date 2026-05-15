"""Multi-reward gap-fill for Experiment 4 (Eureka RL): isolate LLM
variance from PPO seed variance.

The single-seed E4 (in ``eureka_results.csv``) trained 3 PPO seeds on
the SAME P reward (one Qwen 3.6-Plus call). That gives us PPO seed
variance but not LLM-call variance. To separate them, this script
calls Qwen 3.6-Plus 3 times under the same clean prompt and trains
ONE PPO seed (seed=0) on each of the 3 distinct rewards.

Output:
- ``results/eureka_multi_reward.csv`` — 3 rows (one per LLM call)
- ``llm_artifacts/eureka_rewards_multireward/p_qwen_plus_run_{01,02,03}.py``

Q is intentionally skipped — D038 records 1 Q failure under the clean
prompt, plus 3 prior E3 failures. Combined fragility narrative is
preserved without further retries.

Env vars:
- ``DASHSCOPE_API_KEY`` required.
- ``EUREKA_MULTI_QUICK=1`` reduces total_timesteps to 50k for smoke testing.

Usage::

    DASHSCOPE_API_KEY=sk-... python scripts/eureka_multi_reward.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from me569_project.llm.code_extraction import (
    CodeExtractionError,
    extract_python_code,
)
from me569_project.llm.prompts import build_eureka_reward_prompt
from me569_project.llm.qwen_plus_client import QwenPlusClient, QwenPlusError
from me569_project.llm.sandbox import SandboxError, load_callable
from me569_project.rl.eureka_rewards import clip_reward
from me569_project.rl.ppo_runner import EvalResult, evaluate_ppo, train_ppo


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "eureka_multi_reward.csv"
ARTIFACTS_DIR = REPO_ROOT / "llm_artifacts" / "eureka_rewards_multireward"

NUM_LLM_CALLS = 3
PPO_SEED = 0
TRAIN_INIT_RADIUS = 0.15
EVAL_RADII = [0.15, 0.30]
QUICK_TIMESTEPS = 50_000
FULL_TIMESTEPS = 300_000


def _validate_reward_function(fn) -> bool:
    """Same sentinel checks as scripts/run_eureka.py: hover > disturbed states."""
    sentinels_state = [
        np.zeros(6),
        np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05]),
        np.array([1.0, 1.0, 0.5, -0.5, 0.5, -0.5]),
    ]
    sentinels_action = [np.array([4.905, 4.905])] * 3
    rewards = []
    for s, a in zip(sentinels_state, sentinels_action):
        try:
            r = float(fn(s, a))
        except Exception as e:
            print(f"      reward function crashed on sentinel: {e}")
            return False
        if not np.isfinite(r):
            print(f"      reward returned non-finite value: {r!r}")
            return False
        rewards.append(r)
    if not (rewards[0] > rewards[1] and rewards[0] > rewards[2]):
        print(
            f"      reward at hover ({rewards[0]:.3f}) not strictly greater "
            f"than at disturbed states ({rewards[1]:.3f}, {rewards[2]:.3f})"
        )
        return False
    return True


def _eval_to_dict(metrics: EvalResult, init_radius: float) -> dict:
    return {
        f"hover_success_rate@{init_radius}": metrics.hover_success_rate,
        f"mean_settling_time_s@{init_radius}": metrics.mean_settling_time_s,
        f"mean_control_energy@{init_radius}": metrics.mean_control_energy,
        f"mean_final_position_error@{init_radius}": metrics.mean_final_position_error,
        f"mean_final_attitude_error@{init_radius}": metrics.mean_final_attitude_error,
        f"failed_due_to_crash@{init_radius}": metrics.failed_due_to_crash,
        f"failed_due_to_timeout@{init_radius}": metrics.failed_due_to_timeout,
    }


def call_p_and_train(
    run_idx: int, total_timesteps: int, client: QwenPlusClient,
) -> dict | None:
    prompt = build_eureka_reward_prompt(include_eom=False)
    print(f"\n[P|run {run_idx}] Calling Qwen 3.6-Plus ...")
    t0 = time.time()
    try:
        response = client.call(prompt)
    except QwenPlusError as e:
        print(f"[P|run {run_idx}] Qwen call failed: {e}")
        return None
    call_time = time.time() - t0
    tokens = (
        client.last_usage.get("total_tokens", "?") if client.last_usage else "?"
    )
    print(f"[P|run {run_idx}]   response in {call_time:.1f}s, tokens={tokens}")

    try:
        code = extract_python_code(response)
    except CodeExtractionError as e:
        print(f"[P|run {run_idx}] Code extract failed: {e}")
        return None

    try:
        raw_fn = load_callable(code, function_name="reward")
    except SandboxError as e:
        print(f"[P|run {run_idx}] Sandbox load failed: {e}")
        return None

    if not _validate_reward_function(raw_fn):
        return None

    fn = clip_reward(raw_fn, low=-1000.0, high=1000.0)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS_DIR / f"p_qwen_plus_run_{run_idx:02d}.py"
    artifact_path.write_text(code)
    print(f"[P|run {run_idx}]   saved {artifact_path}")

    print(
        f"[P|run {run_idx}] Training PPO ({total_timesteps:,} steps, "
        f"seed={PPO_SEED}) ..."
    )
    t1 = time.time()
    train = train_ppo(
        reward_fn=fn,
        total_timesteps=total_timesteps,
        seed=PPO_SEED,
        n_envs=8,
        init_radius=TRAIN_INIT_RADIUS,
        device="cpu",
    )
    train_time = time.time() - t1
    last_curve_reward = (
        float(train.learning_curve_rewards[-1])
        if len(train.learning_curve_rewards)
        else float("nan")
    )
    print(
        f"[P|run {run_idx}]   trained in {train_time:.1f}s, "
        f"final_ep_rew={last_curve_reward:.1f}"
    )

    row: dict = {
        "condition": "P",
        "run_index": run_idx,
        "ppo_seed": PPO_SEED,
        "total_timesteps": total_timesteps,
        "train_time_s": train_time,
        "final_ep_rew_mean": last_curve_reward,
        "reward_description": "Qwen 3.6-Plus generated (multi-reward, seed=0)",
        "llm_call_time_s": call_time,
        "llm_model": "qwen-plus",
        "llm_total_tokens": (
            client.last_usage.get("total_tokens", 0)
            if client.last_usage else 0
        ),
        "llm_request_id": client.last_request_id or "",
    }

    for init_radius in EVAL_RADII:
        metrics = evaluate_ppo(
            train.model,
            num_initial_states=20,
            init_radius=init_radius,
            seed=0,
        )
        row.update(_eval_to_dict(metrics, init_radius))
        print(
            f"[P|run {run_idx}]   eval@{init_radius}: "
            f"success={metrics.hover_success_rate:.2f}, "
            f"settle={metrics.mean_settling_time_s:.2f}, "
            f"energy={metrics.mean_control_energy:.4f}"
        )

    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"\nWrote {len(rows)} row(s) to {path}")


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY not set. Aborting.")
        return 1

    quick = bool(os.environ.get("EUREKA_MULTI_QUICK"))
    total_timesteps = QUICK_TIMESTEPS if quick else FULL_TIMESTEPS

    print("=" * 76)
    print(
        f"E4 multi-reward gap-fill: P × {NUM_LLM_CALLS} LLM calls, "
        f"PPO seed={PPO_SEED}, {total_timesteps:,} timesteps each"
    )
    print("=" * 76)

    client = QwenPlusClient(api_key=api_key, max_tokens=4096)

    rows: list[dict] = []
    for i in range(1, NUM_LLM_CALLS + 1):
        row = call_p_and_train(i, total_timesteps, client)
        if row is not None:
            rows.append(row)

    print()
    print("=" * 76)
    print(
        f"E4 Multi-Reward Summary (P × {len(rows)} successful runs, "
        f"PPO seed={PPO_SEED}, eval@0.30)"
    )
    print("=" * 76)
    print(
        f"{'run':>3s} {'success':>8s} {'settle':>8s} {'energy':>8s} "
        f"{'tokens':>8s}"
    )
    for r in rows:
        print(
            f"{r['run_index']:3d} "
            f"{r['hover_success_rate@0.3']:8.2f} "
            f"{r['mean_settling_time_s@0.3']:8.2f} "
            f"{r['mean_control_energy@0.3']:8.4f} "
            f"{r['llm_total_tokens']:8d}"
        )
    if rows:
        succ = np.array([r["hover_success_rate@0.3"] for r in rows])
        settle_vals = np.array(
            [r["mean_settling_time_s@0.3"] for r in rows]
        )
        energy = np.array([r["mean_control_energy@0.3"] for r in rows])
        valid_settle = settle_vals[~np.isnan(settle_vals)]
        print(
            f"{'mean':>3s} {succ.mean():8.2f} "
            f"{(valid_settle.mean() if len(valid_settle) else float('nan')):8.2f} "
            f"{energy.mean():8.4f}"
        )
        print(
            f"{'std':>3s} {succ.std(ddof=1):8.2f} "
            f"{(valid_settle.std(ddof=1) if len(valid_settle) > 1 else 0.0):8.2f} "
            f"{energy.std(ddof=1):8.4f}"
        )
    print("=" * 76)

    write_csv(rows, RESULTS_CSV)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

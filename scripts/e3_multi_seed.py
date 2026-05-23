"""E3 multi-LLM-call variance study.

Answer the primary question left open after commit 23d13d1: is
Qwen-Plus really creative-stochastic on E3 (different weights per
call) and is Qwen3.5-4B really memorization-deterministic (identical
weights per call)? D029 logs the hypothesis; this script collects
the evidence.

For each condition (P, Q) the script makes 3 fresh real LLM calls
with identical prompts and temperature 0.3 (P) or temperature 0 /
no_think (Q), saves each response to
``llm_artifacts/e3_costs/{p,q}_run_NN.py``, builds a
``PlanarQuadrotorMPC`` with it, and evaluates on the same 20
seeded initial states (seed 0). Variance is reported as mean ± std
of hover success rate, settling time, and control energy across
the 3 runs. The baseline B condition is included as the first row
for comparison.

Usage::

    DASHSCOPE_API_KEY=sk-... python scripts/e3_multi_seed.py
    E3_SKIP_Q=1 DASHSCOPE_API_KEY=sk-... python scripts/e3_multi_seed.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from me569_project.llm.code_extraction import (
    CodeExtractionError,
    extract_python_code,
)
from me569_project.llm.prompts import build_mpc_cost_prompt
from me569_project.llm.qwen_local_client import QwenLocalClient, QwenLocalError
from me569_project.llm.qwen_plus_client import QwenPlusClient, QwenPlusError
from me569_project.llm.sandbox import SandboxError
from me569_project.mpc.baseline_cost import make_baseline_stage_cost
from me569_project.mpc.evaluation import evaluate_controller
from me569_project.mpc.mpc_math import load_stage_cost_callable
from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
from me569_project.quadrotor_dynamics import QuadrotorParams


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_SUFFIX = os.environ.get("E3_OUTPUT_SUFFIX", "")
ARTIFACTS_DIR = REPO_ROOT / "llm_artifacts" / f"e3_costs{OUTPUT_SUFFIX}"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = REPO_ROOT / "results" / f"e3_multi_seed{OUTPUT_SUFFIX}.csv"

NUM_INITIAL_STATES = 20
MAX_STEPS = 500
HORIZON = 20
INIT_SEED = 0


@dataclass
class E3Row:
    condition: str  # "B", "P", or "Q"
    run_index: int  # 1..N, 0 for baseline (which is deterministic)
    llm_call_time_s: float
    build_time_s: float
    eval_time_s: float
    success_rate: float
    mean_settling_time_s: float
    mean_control_energy: float
    mean_final_position_error: float
    mean_final_attitude_error: float
    failed_crash: int
    failed_timeout: int
    cost_file: str


def _save_code(stem: str, header: str, code: str) -> Path:
    path = ARTIFACTS_DIR / f"{stem}.py"
    wrapped = f'"""{header}\n"""\n{code}\n'
    path.write_text(wrapped)
    return path


def _build_and_eval(
    condition: str,
    run_index: int,
    stage_cost: Callable,
    llm_call_time_s: float,
    cost_file: str,
    params: QuadrotorParams,
) -> E3Row | None:
    print(f"  [{condition}/{run_index}] Building MPC controller ...")
    t0 = time.time()
    try:
        mpc = PlanarQuadrotorMPC(
            stage_cost=stage_cost,
            params=params,
            config=MPCConfig(horizon=HORIZON),
        )
    except Exception as e:
        print(f"  [{condition}/{run_index}] build failed: {type(e).__name__}: {e}")
        return None
    build_time = time.time() - t0

    print(f"  [{condition}/{run_index}] Evaluating ({NUM_INITIAL_STATES} rollouts) ...")
    t0 = time.time()
    try:
        result = evaluate_controller(
            mpc=mpc,
            params=params,
            num_initial_states=NUM_INITIAL_STATES,
            max_steps=MAX_STEPS,
            seed=INIT_SEED,
        )
    except Exception as e:
        print(f"  [{condition}/{run_index}] eval failed: {type(e).__name__}: {e}")
        return None
    eval_time = time.time() - t0

    print(
        f"  [{condition}/{run_index}] success={result.hover_success_rate * 100:.1f}% "
        f"settle={result.mean_settling_time_s:.3f}s "
        f"energy={result.mean_control_energy:.4f} "
        f"({build_time:.1f}s build, {eval_time:.1f}s eval)"
    )

    settling = result.mean_settling_time_s
    if not np.isfinite(settling):
        settling = float("nan")

    return E3Row(
        condition=condition,
        run_index=run_index,
        llm_call_time_s=llm_call_time_s,
        build_time_s=build_time,
        eval_time_s=eval_time,
        success_rate=result.hover_success_rate,
        mean_settling_time_s=settling,
        mean_control_energy=result.mean_control_energy,
        mean_final_position_error=result.mean_final_position_error,
        mean_final_attitude_error=result.mean_final_attitude_error,
        failed_crash=result.failed_due_to_crash,
        failed_timeout=result.failed_due_to_timeout,
        cost_file=cost_file,
    )


def run_baseline(params: QuadrotorParams) -> E3Row | None:
    print()
    print("[B] Hand-written baseline (D011 Bryson's rule) ...")
    stage_cost = make_baseline_stage_cost(params)
    return _build_and_eval(
        condition="B",
        run_index=0,
        stage_cost=stage_cost,
        llm_call_time_s=0.0,
        cost_file="<baseline>",
        params=params,
    )


def run_p_condition(params: QuadrotorParams, n_runs: int = 3) -> list[E3Row]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[P] DASHSCOPE_API_KEY not set, skipping Qwen-Plus multi-seed.")
        return []

    client = QwenPlusClient(api_key=api_key, max_tokens=1500)
    prompt = build_mpc_cost_prompt()
    rows: list[E3Row] = []

    for run_index in range(1, n_runs + 1):
        stem = f"p_multiseed_run_{run_index:02d}"
        print()
        print(f"[P/{run_index}] Calling Qwen-Plus ...")
        t0 = time.time()
        try:
            response = client.call(prompt)
        except QwenPlusError as e:
            print(f"  Qwen-Plus call failed: {e}")
            continue
        call_time = time.time() - t0
        tokens = (
            client.last_usage.get("total_tokens", 0) if client.last_usage else 0
        )
        print(f"  response in {call_time:.1f}s, {tokens} tokens")

        try:
            code = extract_python_code(response)
        except CodeExtractionError as e:
            print(f"  extract failed: {e}")
            continue

        path = _save_code(
            stem,
            header=(
                f"P-condition stage cost from multi-seed run {run_index:02d}. "
                f"Qwen-Plus call at temperature 0.3. Saved by e3_multi_seed.py."
            ),
            code=code,
        )
        print(f"  saved {path.name} ({len(code)} chars)")

        try:
            stage_cost = load_stage_cost_callable(code)
        except SandboxError as e:
            print(f"  sandbox load failed: {e}")
            continue

        row = _build_and_eval(
            condition="P",
            run_index=run_index,
            stage_cost=stage_cost,
            llm_call_time_s=call_time,
            cost_file=path.name,
            params=params,
        )
        if row is not None:
            rows.append(row)

    return rows


def run_q_condition(params: QuadrotorParams, n_runs: int = 3) -> list[E3Row]:
    if os.environ.get("E3_SKIP_Q"):
        print("[Q] E3_SKIP_Q set, skipping Qwen3.5-4B multi-seed.")
        return []

    client = QwenLocalClient(max_tokens=4000)
    base_prompt = build_mpc_cost_prompt()
    prompt = f"/no_think\n\n{base_prompt}"
    rows: list[E3Row] = []

    for run_index in range(1, n_runs + 1):
        stem = f"q_multiseed_run_{run_index:02d}"
        print()
        print(f"[Q/{run_index}] Calling Qwen3.5-4B (/no_think) ...")
        print(
            "  (first call loads 4.8 GB model, subsequent calls reuse cache)"
        )
        t0 = time.time()
        try:
            response = client.call(prompt)
        except QwenLocalError as e:
            print(f"  Qwen3.5-4B call failed: {e}")
            continue
        call_time = time.time() - t0
        load_time = client.last_load_time_s or 0.0
        gen_time = client.last_call_time_s or 0.0
        print(
            f"  response in {call_time:.1f}s (load {load_time:.1f}s, gen {gen_time:.1f}s)"
        )

        try:
            code = extract_python_code(response)
        except CodeExtractionError as e:
            print(f"  extract failed: {e}")
            continue

        path = _save_code(
            stem,
            header=(
                f"Q-condition stage cost from multi-seed run {run_index:02d}. "
                f"Qwen3.5-4B via mlx-vlm with /no_think. "
                f"Saved by e3_multi_seed.py."
            ),
            code=code,
        )
        print(f"  saved {path.name} ({len(code)} chars)")

        try:
            stage_cost = load_stage_cost_callable(code)
        except SandboxError as e:
            print(f"  sandbox load failed: {e}")
            continue

        row = _build_and_eval(
            condition="Q",
            run_index=run_index,
            stage_cost=stage_cost,
            llm_call_time_s=call_time,
            cost_file=path.name,
            params=params,
        )
        if row is not None:
            rows.append(row)

    return rows


def print_variance_summary(rows: list[E3Row]) -> None:
    print()
    print("=" * 96)
    print("E3 multi-LLM-call variance summary")
    print("=" * 96)

    by_condition: dict[str, list[E3Row]] = {}
    for r in rows:
        by_condition.setdefault(r.condition, []).append(r)

    for cond in ("B", "P", "Q"):
        runs = by_condition.get(cond, [])
        if not runs:
            print(f"{cond}  (no runs)")
            continue

        success = np.array([r.success_rate for r in runs])
        settle = np.array(
            [r.mean_settling_time_s for r in runs if np.isfinite(r.mean_settling_time_s)]
        )
        energy = np.array([r.mean_control_energy for r in runs])

        print(f"{cond}  (N = {len(runs)} runs)")
        print(
            f"  success_rate: mean {success.mean() * 100:.2f}%  "
            f"std {success.std() * 100:.2f}pp"
        )
        if settle.size > 0:
            settle_cv = (
                settle.std() / settle.mean() * 100 if settle.mean() > 0 else 0.0
            )
            print(
                f"  settling_s  : mean {settle.mean():.4f}s  "
                f"std {settle.std():.4e}s  CV {settle_cv:.2f}%"
            )
        energy_cv = (
            energy.std() / energy.mean() * 100 if energy.mean() > 0 else 0.0
        )
        print(
            f"  energy      : mean {energy.mean():.4e}  "
            f"std {energy.std():.4e}  CV {energy_cv:.2f}%"
        )
    print("=" * 96)


def write_csv(rows: list[E3Row], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"\nWrote {len(rows)} rows to {path}")


def main() -> int:
    params = QuadrotorParams()
    rows: list[E3Row] = []

    print("=" * 96)
    print("Experiment 3 multi-LLM-call variance study")
    print("  NUM_INITIAL_STATES =", NUM_INITIAL_STATES)
    print("  MAX_STEPS          =", MAX_STEPS)
    print("  HORIZON            =", HORIZON)
    print("  INIT_SEED          =", INIT_SEED)
    print("=" * 96)

    b_row = run_baseline(params)
    if b_row is not None:
        rows.append(b_row)

    rows.extend(run_p_condition(params, n_runs=3))
    rows.extend(run_q_condition(params, n_runs=3))

    print_variance_summary(rows)
    write_csv(rows, RESULTS_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())

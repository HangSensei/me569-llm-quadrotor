"""E3 prompt ablation: does prompt verbosity change MPC cost design?

Compares three prompt variants on the same closed-loop evaluation
harness and the same two LLMs (Qwen-Plus and Qwen3.5-4B), producing
one row per (variant, condition) combination:

- **MINIMAL**: a short prompt that names the state, control, and hover
  equilibrium, then asks for ``stage_cost(x, u)``.
- **CURRENT**: the committed honest prompt from
  ``me569_project.llm.prompts.build_mpc_cost_prompt()``.
- **CHEAT**: CURRENT plus explicit equations of motion appended.

Each prompt is sent to:

- **P**: Qwen-Plus via DashScope
- **Q**: Qwen3.5-4B via mlx-vlm, with ``/no_think`` prepended

The extracted code is loaded with
``me569_project.mpc.mpc_math.load_stage_cost_callable`` and plugged
into ``PlanarQuadrotorMPC`` with horizon 20. Results are evaluated on
20 initial states with seed 0 and written to
``results/e3_prompt_ablation.csv``.

Usage::

    DASHSCOPE_API_KEY=sk-... python scripts/e3_prompt_ablation.py
    E3_SKIP_Q=1 DASHSCOPE_API_KEY=sk-... python scripts/e3_prompt_ablation.py
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from me569_project.llm.code_extraction import (
    CodeExtractionError,
    extract_python_code,
)
from me569_project.llm.prompts import build_mpc_cost_prompt
from me569_project.llm.qwen_local_client import QwenLocalClient, QwenLocalError
from me569_project.llm.qwen_plus_client import QwenPlusClient, QwenPlusError
from me569_project.llm.sandbox import SandboxError
from me569_project.mpc.evaluation import evaluate_controller
from me569_project.mpc.mpc_math import load_stage_cost_callable
from me569_project.mpc.mpc_wrapper import MPCConfig, PlanarQuadrotorMPC
from me569_project.quadrotor_dynamics import QuadrotorParams


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_SUFFIX = os.environ.get("E3_OUTPUT_SUFFIX", "")
ARTIFACTS_DIR = REPO_ROOT / "llm_artifacts" / f"e3_costs{OUTPUT_SUFFIX}"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = REPO_ROOT / "results" / f"e3_prompt_ablation{OUTPUT_SUFFIX}.csv"

NUM_INITIAL_STATES = 20
MAX_STEPS = 500
HORIZON = 20
INIT_SEED = 0


MINIMAL_PROMPT = """\
Planar Quadrotor MPC stage cost.
State x = [p_x, p_z, theta, v_x, v_z, omega] (index 0..5)
Control u = [u_1, u_2] (index 0..1)
Hover: x = 0, u_1 = u_2 = 4.905 N

Write stage_cost(x, u) returning scalar cost (smaller=better, zero at hover).
Available math: sin, cos, exp, sqrt, log, fabs. No imports. No numpy. Use fabs not abs.
Return ONLY the function in a python code fence.
"""


CHEAT_SUFFIX = """

Hint: the ground-truth continuous-time dynamics are:
    p_x_dot   = v_x
    p_z_dot   = v_z
    theta_dot = omega
    v_x_dot   = -(u_1 + u_2) * sin(theta) / m
    v_z_dot   =  (u_1 + u_2) * cos(theta) / m - g
    omega_dot = (u_2 - u_1) * L / I_yy

Design a cost that penalizes deviation from hover, with awareness of these dynamics.
"""


CURRENT_PROMPT = build_mpc_cost_prompt()


VARIANTS: dict[str, str] = {
    "minimal": MINIMAL_PROMPT,
    "current": CURRENT_PROMPT,
    "cheat": CURRENT_PROMPT + CHEAT_SUFFIX,
}


@dataclass
class E3AblationRow:
    condition: str
    variant: str
    prompt_chars: int
    response_chars: int
    llm_call_time_s: float
    llm_total_tokens: int
    extract_ok: bool
    sandbox_ok: bool
    build_ok: bool
    eval_ok: bool
    status: str
    failure_stage: str
    error_message: str
    success_rate: float
    mean_settling_time_s: float
    mean_control_energy: float
    num_successful: int
    failed_due_to_crash: int
    failed_due_to_timeout: int
    cost_file: str


def _comment_block(text: str) -> str:
    lines = text.splitlines() or [""]
    return "\n".join(f"# {line}" for line in lines)


def _save_response_artifact(
    stem: str,
    header: str,
    response: str,
    code: str | None,
) -> Path:
    path = ARTIFACTS_DIR / f"{stem}.py"
    parts = [
        f'"""{header}\n"""',
        "# Raw LLM response follows.",
        _comment_block(response),
    ]
    if code is not None:
        parts.append("# Extracted stage_cost implementation follows.")
        parts.append(code.rstrip())
    else:
        parts.append("# No executable code could be extracted for this variant.")
    path.write_text("\n".join(parts) + "\n")
    return path


def _build_mpc(stage_cost: Callable, params: QuadrotorParams) -> PlanarQuadrotorMPC:
    return PlanarQuadrotorMPC(
        stage_cost=stage_cost,
        params=params,
        config=MPCConfig(horizon=HORIZON),
    )


def _make_failure_row(
    condition: str,
    variant: str,
    prompt: str,
    response: str,
    llm_call_time_s: float,
    llm_total_tokens: int,
    status: str,
    failure_stage: str,
    error_message: str,
    cost_file: str,
) -> E3AblationRow:
    return E3AblationRow(
        condition=condition,
        variant=variant,
        prompt_chars=len(prompt),
        response_chars=len(response),
        llm_call_time_s=llm_call_time_s,
        llm_total_tokens=llm_total_tokens,
        extract_ok=False,
        sandbox_ok=False,
        build_ok=False,
        eval_ok=False,
        status=status,
        failure_stage=failure_stage,
        error_message=error_message,
        success_rate=0.0,
        mean_settling_time_s=float("nan"),
        mean_control_energy=float("nan"),
        num_successful=0,
        failed_due_to_crash=0,
        failed_due_to_timeout=0,
        cost_file=cost_file,
    )


def _evaluate_variant(
    condition: str,
    variant: str,
    prompt: str,
    response: str,
    llm_call_time_s: float,
    llm_total_tokens: int,
    params: QuadrotorParams,
) -> E3AblationRow:
    stem = f"{condition.lower()}_ablation_{variant}"
    artifact_path = ARTIFACTS_DIR / f"{stem}.py"

    try:
        code = extract_python_code(response)
    except CodeExtractionError as e:
        _save_response_artifact(
            stem,
            header=(
                f"{condition}-condition prompt ablation variant '{variant}'. "
                "Extraction failed."
            ),
            response=response,
            code=None,
        )
        print(f"  [{condition}/{variant}] extract failed: {e}")
        return _make_failure_row(
            condition=condition,
            variant=variant,
            prompt=prompt,
            response=response,
            llm_call_time_s=llm_call_time_s,
            llm_total_tokens=llm_total_tokens,
            status="extract_failed",
            failure_stage="extract_python_code",
            error_message=str(e),
            cost_file=artifact_path.name,
        )

    _save_response_artifact(
        stem,
        header=(
            f"{condition}-condition prompt ablation variant '{variant}'. "
            "Saved by scripts/e3_prompt_ablation.py."
        ),
        response=response,
        code=code,
    )
    print(f"  [{condition}/{variant}] saved {artifact_path.name} ({len(code)} code chars)")

    try:
        stage_cost = load_stage_cost_callable(code, function_name="stage_cost")
    except SandboxError as e:
        print(f"  [{condition}/{variant}] sandbox load failed: {e}")
        row = _make_failure_row(
            condition=condition,
            variant=variant,
            prompt=prompt,
            response=response,
            llm_call_time_s=llm_call_time_s,
            llm_total_tokens=llm_total_tokens,
            status="sandbox_failed",
            failure_stage="load_stage_cost_callable",
            error_message=str(e),
            cost_file=artifact_path.name,
        )
        return row

    try:
        t0 = time.time()
        mpc = _build_mpc(stage_cost, params)
        build_time = time.time() - t0
        print(f"  [{condition}/{variant}] MPC built in {build_time:.1f}s")
    except Exception as e:
        print(f"  [{condition}/{variant}] build failed: {type(e).__name__}: {e}")
        return E3AblationRow(
            condition=condition,
            variant=variant,
            prompt_chars=len(prompt),
            response_chars=len(response),
            llm_call_time_s=llm_call_time_s,
            llm_total_tokens=llm_total_tokens,
            extract_ok=True,
            sandbox_ok=True,
            build_ok=False,
            eval_ok=False,
            status="build_failed",
            failure_stage="PlanarQuadrotorMPC",
            error_message=f"{type(e).__name__}: {e}",
            success_rate=0.0,
            mean_settling_time_s=float("nan"),
            mean_control_energy=float("nan"),
            num_successful=0,
            failed_due_to_crash=0,
            failed_due_to_timeout=0,
            cost_file=artifact_path.name,
        )

    try:
        t0 = time.time()
        result = evaluate_controller(
            mpc=mpc,
            params=params,
            num_initial_states=NUM_INITIAL_STATES,
            max_steps=MAX_STEPS,
            seed=INIT_SEED,
        )
        eval_time = time.time() - t0
    except Exception as e:
        print(f"  [{condition}/{variant}] eval failed: {type(e).__name__}: {e}")
        return E3AblationRow(
            condition=condition,
            variant=variant,
            prompt_chars=len(prompt),
            response_chars=len(response),
            llm_call_time_s=llm_call_time_s,
            llm_total_tokens=llm_total_tokens,
            extract_ok=True,
            sandbox_ok=True,
            build_ok=True,
            eval_ok=False,
            status="eval_failed",
            failure_stage="evaluate_controller",
            error_message=f"{type(e).__name__}: {e}",
            success_rate=0.0,
            mean_settling_time_s=float("nan"),
            mean_control_energy=float("nan"),
            num_successful=0,
            failed_due_to_crash=0,
            failed_due_to_timeout=0,
            cost_file=artifact_path.name,
        )

    print(
        f"  [{condition}/{variant}] success={result.hover_success_rate * 100:.1f}% "
        f"settle={result.mean_settling_time_s:.3f}s "
        f"energy={result.mean_control_energy:.4f} "
        f"({eval_time:.1f}s eval)"
    )
    return E3AblationRow(
        condition=condition,
        variant=variant,
        prompt_chars=len(prompt),
        response_chars=len(response),
        llm_call_time_s=llm_call_time_s,
        llm_total_tokens=llm_total_tokens,
        extract_ok=True,
        sandbox_ok=True,
        build_ok=True,
        eval_ok=True,
        status="ok",
        failure_stage="",
        error_message="",
        success_rate=result.hover_success_rate,
        mean_settling_time_s=result.mean_settling_time_s,
        mean_control_energy=result.mean_control_energy,
        num_successful=result.num_successful,
        failed_due_to_crash=result.failed_due_to_crash,
        failed_due_to_timeout=result.failed_due_to_timeout,
        cost_file=artifact_path.name,
    )


def run_p_variants(params: QuadrotorParams) -> list[E3AblationRow]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[P] DASHSCOPE_API_KEY not set, skipping Qwen-Plus ablation.")
        return []

    client = QwenPlusClient(api_key=api_key, max_tokens=1500)
    rows: list[E3AblationRow] = []

    for variant_name, prompt in VARIANTS.items():
        print(f"\n[P/{variant_name}] prompt length: {len(prompt)} chars")
        t0 = time.time()
        try:
            response = client.call(prompt)
        except QwenPlusError as e:
            call_time = time.time() - t0
            print(f"  Qwen-Plus call failed: {e}")
            _save_response_artifact(
                f"p_ablation_{variant_name}",
                header=(
                    f"P-condition prompt ablation variant '{variant_name}'. "
                    "Qwen-Plus call failed."
                ),
                response="",
                code=None,
            )
            rows.append(
                _make_failure_row(
                    condition="P",
                    variant=variant_name,
                    prompt=prompt,
                    response="",
                    llm_call_time_s=call_time,
                    llm_total_tokens=0,
                    status="llm_failed",
                    failure_stage="QwenPlusClient.call",
                    error_message=str(e),
                    cost_file=f"p_ablation_{variant_name}.py",
                )
            )
            continue
        call_time = time.time() - t0
        tokens = (
            client.last_usage.get("total_tokens", 0) if client.last_usage else 0
        )
        print(f"  received in {call_time:.1f}s ({tokens} tokens)")
        rows.append(
            _evaluate_variant(
                condition="P",
                variant=variant_name,
                prompt=prompt,
                response=response,
                llm_call_time_s=call_time,
                llm_total_tokens=tokens,
                params=params,
            )
        )

    return rows


def run_q_variants(params: QuadrotorParams) -> list[E3AblationRow]:
    if os.environ.get("E3_SKIP_Q"):
        print("[Q] E3_SKIP_Q is set, skipping Qwen3.5-4B ablation.")
        return []

    client = QwenLocalClient(max_tokens=4000)
    rows: list[E3AblationRow] = []

    for variant_name, base_prompt in VARIANTS.items():
        prompt = f"/no_think\n\n{base_prompt}"
        print(f"\n[Q/{variant_name}] prompt length: {len(prompt)} chars")
        t0 = time.time()
        try:
            response = client.call(prompt)
        except QwenLocalError as e:
            call_time = time.time() - t0
            print(f"  Qwen3.5-4B call failed: {e}")
            _save_response_artifact(
                f"q_ablation_{variant_name}",
                header=(
                    f"Q-condition prompt ablation variant '{variant_name}'. "
                    "Qwen3.5-4B call failed."
                ),
                response="",
                code=None,
            )
            rows.append(
                _make_failure_row(
                    condition="Q",
                    variant=variant_name,
                    prompt=prompt,
                    response="",
                    llm_call_time_s=call_time,
                    llm_total_tokens=0,
                    status="llm_failed",
                    failure_stage="QwenLocalClient.call",
                    error_message=str(e),
                    cost_file=f"q_ablation_{variant_name}.py",
                )
            )
            continue
        call_time = time.time() - t0
        print(f"  received in {call_time:.1f}s")
        rows.append(
            _evaluate_variant(
                condition="Q",
                variant=variant_name,
                prompt=prompt,
                response=response,
                llm_call_time_s=call_time,
                llm_total_tokens=0,
                params=params,
            )
        )

    return rows


def summarize(rows: list[E3AblationRow]) -> None:
    print()
    print("=" * 120)
    print("E3 prompt ablation summary")
    print("=" * 120)
    print(
        f"{'cond':<4s} {'variant':<10s} {'status':<14s} {'succ%':>7s} "
        f"{'settle_s':>10s} {'energy':>10s} {'extract':>8s} {'sandbox':>8s} "
        f"{'build':>8s} {'eval':>8s}"
    )
    print("-" * 120)
    for r in rows:
        settle = "nan" if not math.isfinite(r.mean_settling_time_s) else f"{r.mean_settling_time_s:.3f}"
        energy = "nan" if not math.isfinite(r.mean_control_energy) else f"{r.mean_control_energy:.4f}"
        print(
            f"{r.condition:<4s} {r.variant:<10s} {r.status:<14s} "
            f"{r.success_rate * 100:>6.1f}% {settle:>10s} {energy:>10s} "
            f"{str(r.extract_ok):>8s} {str(r.sandbox_ok):>8s} "
            f"{str(r.build_ok):>8s} {str(r.eval_ok):>8s}"
        )
    print("=" * 120)


def write_csv(rows: list[E3AblationRow], path: Path) -> None:
    if not rows:
        print(f"\nNo rows to write for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(rows)} rows to {path}")


def main() -> int:
    params = QuadrotorParams()
    rows: list[E3AblationRow] = []

    print("=" * 120)
    print("Experiment 3 prompt ablation")
    print("  NUM_INITIAL_STATES =", NUM_INITIAL_STATES)
    print("  MAX_STEPS          =", MAX_STEPS)
    print("  HORIZON            =", HORIZON)
    print("  INIT_SEED          =", INIT_SEED)
    print("=" * 120)

    print()
    print("=" * 120)
    print("P condition (Qwen-Plus via DashScope)")
    print("=" * 120)
    rows.extend(run_p_variants(params))

    print()
    print("=" * 120)
    print("Q condition (Qwen3.5-4B via mlx-vlm)")
    print("=" * 120)
    rows.extend(run_q_variants(params))

    summarize(rows)
    write_csv(rows, RESULTS_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())

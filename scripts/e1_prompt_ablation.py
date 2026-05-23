"""E1 prompt ablation: does physical hinting level change LLM basis quality?

Compares three prompt variants on the same train/val split and the same
two LLMs (Qwen-Plus and Qwen3.5-4B), producing one row per
(variant, condition) combination:

- **MINIMAL**: just the state vector, control vector, and the task
  ("propose a SINDy basis library as a Python function"). No physical
  parameters. No hints about pitch-dependent thrust or differential
  thrust torque.

- **CURRENT** (honest baseline): the committed SINDY_BASIS prompt from
  ``llm.prompts``. State vector, control vector, physical parameters
  (m, I_yy, L, g), and a natural-language paragraph pointing the LLM
  at its own quadrotor knowledge (gravity, thrust projection, differential
  thrust for torque). Does NOT include the equations of motion.

- **CHEAT**: CURRENT plus the full ground-truth equations of motion.
  Directly tests whether leaking the equations helps the LLM (Phase 1
  diagnostic already suggests it does not; this measures the gap
  quantitatively on the same data split as the other variants).

The responses are saved to ``llm_artifacts/e1_bases/`` with names
``{p,q}_variant_<variant_name>.py`` so they can be reused later
without re-calling the LLM.

Usage::

    DASHSCOPE_API_KEY=sk-... python scripts/e1_prompt_ablation.py
    E1_SKIP_Q=1 DASHSCOPE_API_KEY=sk-... python scripts/e1_prompt_ablation.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.code_extraction import extract_python_code
from me569_project.llm.qwen_local_client import QwenLocalClient, QwenLocalError
from me569_project.llm.qwen_plus_client import QwenPlusClient, QwenPlusError
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
BASES_DIR = REPO_ROOT / "llm_artifacts" / "e1_bases"
BASES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------

PROMPT_MINIMAL = """\
System: Planar Quadrotor.

State vector (length 6):
    x = [p_x, p_z, theta, v_x, v_z, omega]

Control vector (length 2):
    u = [u_1, u_2]   (each rotor thrust in Newtons)

Task: write a Python function ``basis(xu)`` that takes a single
concatenated state+control vector of length 8 and returns a list of
scalar basis feature values. The function will be passed into a SINDy
sparse regression fit. Return ONLY the function definition inside a
single ``python`` code fence.

Input convention: xu[0..5] = state, xu[6..7] = control.
Function name: ``basis``. Argument name: ``xu``. Return: list of floats.
Allowed imports: numpy only.
"""


# CURRENT comes from llm.prompts — import and reuse.
from me569_project.llm.prompts import build_sindy_basis_prompt

PROMPT_CURRENT = build_sindy_basis_prompt()


PROMPT_CHEAT = PROMPT_CURRENT + """

Hint: the ground-truth continuous-time dynamics are

    p_x_dot   = v_x
    p_z_dot   = v_z
    theta_dot = omega
    v_x_dot   = -(u_1 + u_2) * sin(theta) / m
    v_z_dot   =  (u_1 + u_2) * cos(theta) / m - g
    omega_dot = (u_2 - u_1) * L / I_yy

You should build a basis that spans the right-hand sides of these
equations after the sparse regression step.
"""


VARIANTS: dict[str, str] = {
    "minimal": PROMPT_MINIMAL,
    "current": PROMPT_CURRENT,
    "cheat": PROMPT_CHEAT,
}


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------


@dataclass
class AblationRow:
    condition: str
    variant: str
    prompt_chars: int
    response_chars: int
    llm_call_time_s: float
    llm_total_tokens: int
    n_basis: int
    active_terms: int
    one_step_mse: float
    rollout_mse_10: float
    rollout_mse_50: float
    basis_file: str


def _save_code(stem: str, header: str, code: str) -> Path:
    path = BASES_DIR / f"{stem}.py"
    wrapped = f'"""{header}\n"""\n{code}\n'
    path.write_text(wrapped)
    return path


def _fit_and_score(
    condition: str,
    variant: str,
    prompt: str,
    response: str,
    llm_call_time_s: float,
    llm_total_tokens: int,
    train_data,
    val_data,
) -> AblationRow | None:
    try:
        code = extract_python_code(response)
    except Exception as e:
        print(f"  [{condition}/{variant}] extract_python_code failed: {e}")
        return None

    stem = f"{condition.lower()}_variant_{variant}"
    path = _save_code(
        stem,
        header=(
            f"{condition}-condition basis from prompt variant '{variant}'. "
            "Saved by scripts/e1_prompt_ablation.py."
        ),
        code=code,
    )
    print(f"  [{condition}/{variant}] saved {path.name} ({len(code)} chars)")

    try:
        basis_fn = load_callable(code, function_name="basis")
    except Exception as e:
        print(f"  [{condition}/{variant}] sandbox load failed: {e}")
        return None

    try:
        model = fit_with_basis_fn(train_data, basis_fn=basis_fn, threshold=0.1)
    except Exception as e:
        print(f"  [{condition}/{variant}] fit failed: {e}")
        return None

    m1 = model.one_step_mse(val_data)
    m10 = model.rollout_mse(val_data, horizon=10)
    m50 = model.rollout_mse(val_data, horizon=50)
    print(
        f"  [{condition}/{variant}] n_basis={model.n_basis} "
        f"active={model.active_terms} "
        f"one_step={m1:.4e} roll10={m10:.4e} roll50={m50:.4e}"
    )

    return AblationRow(
        condition=condition,
        variant=variant,
        prompt_chars=len(prompt),
        response_chars=len(response),
        llm_call_time_s=llm_call_time_s,
        llm_total_tokens=llm_total_tokens,
        n_basis=model.n_basis,
        active_terms=model.active_terms,
        one_step_mse=m1,
        rollout_mse_10=m10,
        rollout_mse_50=m50,
        basis_file=path.name,
    )


def run_p_variants(train_data, val_data) -> list[AblationRow]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[P] DASHSCOPE_API_KEY not set, skipping Qwen-Plus ablation.")
        return []

    client = QwenPlusClient(api_key=api_key, max_tokens=1500)
    rows: list[AblationRow] = []

    for variant_name, prompt in VARIANTS.items():
        print(f"\n[P/{variant_name}] prompt length: {len(prompt)} chars")
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
        print(f"  received in {call_time:.1f}s ({tokens} tokens)")

        row = _fit_and_score(
            condition="P",
            variant=variant_name,
            prompt=prompt,
            response=response,
            llm_call_time_s=call_time,
            llm_total_tokens=tokens,
            train_data=train_data,
            val_data=val_data,
        )
        if row is not None:
            rows.append(row)

    return rows


def run_q_variants(train_data, val_data) -> list[AblationRow]:
    if os.environ.get("E1_SKIP_Q"):
        print("[Q] E1_SKIP_Q is set, skipping Qwen3.5-4B ablation.")
        return []

    client = QwenLocalClient(max_tokens=2000)
    rows: list[AblationRow] = []

    for variant_name, prompt in VARIANTS.items():
        print(f"\n[Q/{variant_name}] prompt length: {len(prompt)} chars")
        t0 = time.time()
        try:
            response = client.call(prompt)
        except QwenLocalError as e:
            print(f"  Qwen3.5-4B call failed: {e}")
            continue
        call_time = time.time() - t0
        print(f"  received in {call_time:.1f}s")

        row = _fit_and_score(
            condition="Q",
            variant=variant_name,
            prompt=prompt,
            response=response,
            llm_call_time_s=call_time,
            llm_total_tokens=0,
            train_data=train_data,
            val_data=val_data,
        )
        if row is not None:
            rows.append(row)

    return rows


def summarize(rows: list[AblationRow]) -> None:
    print()
    print("=" * 100)
    print("E1 prompt ablation summary")
    print("=" * 100)
    print(
        f"{'cond':<4s} {'variant':<10s} {'prompt':>8s} {'resp':>6s} "
        f"{'n_basis':>8s} {'active':>7s} {'one_step':>14s} "
        f"{'roll_10':>12s} {'roll_50':>12s}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r.condition:<4s} {r.variant:<10s} "
            f"{r.prompt_chars:>8d} {r.response_chars:>6d} "
            f"{r.n_basis:>8d} {r.active_terms:>7d} "
            f"{r.one_step_mse:>14.4e} {r.rollout_mse_10:>12.4e} "
            f"{r.rollout_mse_50:>12.4e}"
        )
    print("=" * 100)


def write_csv(rows: list[AblationRow], path: Path) -> None:
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
    num_train = 200
    num_val = 50
    steps_per_traj = 400

    print(
        f"Generating {num_train} train + {num_val} val trajectories "
        f"at {steps_per_traj} steps ..."
    )
    t0 = time.time()
    train = generate_trajectories(
        num_trajectories=num_train,
        steps_per_trajectory=steps_per_traj,
        seed=0,
    )
    val = generate_trajectories(
        num_trajectories=num_val,
        steps_per_trajectory=steps_per_traj,
        seed=42,
    )
    print(
        f"Data ready in {time.time() - t0:.1f}s "
        f"({train.num_transitions} train transitions)"
    )

    rows: list[AblationRow] = []
    print()
    print("=" * 100)
    print("P condition (Qwen-Plus via DashScope)")
    print("=" * 100)
    rows.extend(run_p_variants(train, val))

    print()
    print("=" * 100)
    print("Q condition (Qwen3.5-4B via mlx-vlm)")
    print("=" * 100)
    rows.extend(run_q_variants(train, val))

    summarize(rows)
    write_csv(rows, REPO_ROOT / "results" / "e1_prompt_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

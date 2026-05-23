"""E1 multi-LLM-call variance study.

Answer the question: are the single-shot P and Q results from commits
711a681 and ff0bb3b stable across LLM calls, or did we get lucky?

Procedure:
1. Reuse the existing run_01 bases already saved under
   ``llm_artifacts/e1_bases/`` (no new LLM call).
2. Fire N-1 additional Qwen-Plus calls with the same prompt and fixed
   temperature, save each raw response to ``run_0{i}.py``, extract
   the basis, fit on fixed train data, evaluate on fixed val data.
3. Same for Qwen3.5-4B locally via mlx-vlm.
4. Print a variance report and write a CSV with one row per
   (condition, run_index) pair.

Fixed data split across all runs:
    train seed = 0, val seed = 42
    num_train = 200 trajectories  (smaller than the 500 used for the
                                   commit-hash runs, to keep the total
                                   wall time under ~25 minutes)
    num_val   = 50 trajectories

Usage::

    DASHSCOPE_API_KEY=sk-... python scripts/e1_multi_seed.py
    E1_SKIP_Q=1 DASHSCOPE_API_KEY=sk-... python scripts/e1_multi_seed.py
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

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.code_extraction import extract_python_code
from me569_project.llm.prompts import build_sindy_basis_prompt
from me569_project.llm.qwen_local_client import QwenLocalClient, QwenLocalError
from me569_project.llm.qwen_plus_client import QwenPlusClient, QwenPlusError
from me569_project.llm.sandbox import load_callable
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
BASES_DIR = REPO_ROOT / "llm_artifacts" / "e1_bases"
BASES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunResult:
    condition: str
    run_index: int
    n_basis: int
    active_terms: int
    one_step_mse: float
    rollout_mse_10: float
    rollout_mse_50: float
    llm_call_time_s: float
    fit_time_s: float
    basis_file: str


def _save_basis_code(stem: str, header: str, code: str) -> Path:
    path = BASES_DIR / f"{stem}.py"
    if path.exists():
        return path
    wrapped = f'"""{header}\n"""\n{code}\n'
    path.write_text(wrapped)
    return path


def _load_saved_basis(stem: str) -> Callable:
    code = (BASES_DIR / f"{stem}.py").read_text()
    return load_callable(code, function_name="basis")


def _verify_basis_on_sample(label: str, basis_fn: Callable) -> int | None:
    sample_xu = np.array([0.1, 0.2, 0.05, 0.1, 0.0, 0.0, 4.9, 4.9])
    try:
        out = basis_fn(sample_xu)
        n = len(list(out))
        print(f"  {label} sample output length: {n}")
        return n
    except Exception as e:
        print(f"  {label} basis_fn crashed on sample input: {e}")
        return None


def fit_and_eval(
    condition: str,
    run_index: int,
    basis_fn: Callable,
    train_data,
    val_data,
    llm_call_time_s: float,
    basis_file: str,
) -> RunResult | None:
    print(f"  Fitting {condition} run {run_index} ...")
    t0 = time.time()
    try:
        model = fit_with_basis_fn(train_data, basis_fn=basis_fn, threshold=0.1)
    except Exception as e:
        print(f"  Fit failed: {type(e).__name__}: {e}")
        return None
    fit_time = time.time() - t0

    m1 = model.one_step_mse(val_data)
    m10 = model.rollout_mse(val_data, horizon=10)
    m50 = model.rollout_mse(val_data, horizon=50)

    print(
        f"  {condition} run {run_index}: n_basis={model.n_basis} "
        f"active={model.active_terms} "
        f"one_step={m1:.4e} roll10={m10:.4e} roll50={m50:.4e} "
        f"({fit_time:.1f}s fit, {llm_call_time_s:.1f}s LLM)"
    )
    return RunResult(
        condition=condition,
        run_index=run_index,
        n_basis=model.n_basis,
        active_terms=model.active_terms,
        one_step_mse=m1,
        rollout_mse_10=m10,
        rollout_mse_50=m50,
        llm_call_time_s=llm_call_time_s,
        fit_time_s=fit_time,
        basis_file=basis_file,
    )


def run_p_condition(n_runs: int, train_data, val_data) -> list[RunResult]:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[P] DASHSCOPE_API_KEY not set, skipping Qwen-Plus runs.")
        return []

    results: list[RunResult] = []
    prompt = build_sindy_basis_prompt()
    client = QwenPlusClient(api_key=api_key, max_tokens=1500)

    # Run 01 is already saved; load it from disk rather than re-calling.
    print("[P] Run 01 (loading saved basis from run_01)...")
    try:
        fn_01 = _load_saved_basis("p_qwen_plus_run_01")
        _verify_basis_on_sample("P run 01", fn_01)
        r01 = fit_and_eval(
            "P", 1, fn_01, train_data, val_data, 0.0, "p_qwen_plus_run_01.py"
        )
        if r01 is not None:
            results.append(r01)
    except Exception as e:
        print(f"  Could not load saved P run 01: {e}")

    # Runs 02 and 03: fresh LLM calls.
    for run_index in range(2, n_runs + 1):
        stem = f"p_qwen_plus_run_{run_index:02d}"
        print(f"\n[P] Run {run_index:02d} (fresh Qwen-Plus call)...")
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
        print(f"  Response received in {call_time:.1f}s ({tokens} tokens)")

        try:
            code = extract_python_code(response)
        except Exception as e:
            print(f"  extract_python_code failed: {e}")
            continue

        path = _save_basis_code(
            stem,
            header=(
                f"P-condition basis from Qwen-Plus run {run_index:02d}. "
                "Temperature 0.3, max_tokens 1500. Saved by scripts/e1_multi_seed.py."
            ),
            code=code,
        )
        print(f"  Saved raw code to {path.name} ({len(code)} chars)")

        try:
            basis_fn = load_callable(code, function_name="basis")
        except Exception as e:
            print(f"  Sandbox load failed: {e}")
            continue

        if _verify_basis_on_sample(f"P run {run_index:02d}", basis_fn) is None:
            continue

        result = fit_and_eval(
            "P", run_index, basis_fn, train_data, val_data, call_time, path.name
        )
        if result is not None:
            results.append(result)

    return results


def run_q_condition(n_runs: int, train_data, val_data) -> list[RunResult]:
    if os.environ.get("E1_SKIP_Q"):
        print("[Q] E1_SKIP_Q is set, skipping Qwen3.5-4B runs.")
        return []

    results: list[RunResult] = []
    prompt = build_sindy_basis_prompt()
    client = QwenLocalClient(max_tokens=2000)

    # Run 01 is already saved.
    print("\n[Q] Run 01 (loading saved basis from run_01)...")
    try:
        fn_01 = _load_saved_basis("q_qwen_local_run_01")
        _verify_basis_on_sample("Q run 01", fn_01)
        r01 = fit_and_eval(
            "Q", 1, fn_01, train_data, val_data, 0.0, "q_qwen_local_run_01.py"
        )
        if r01 is not None:
            results.append(r01)
    except Exception as e:
        print(f"  Could not load saved Q run 01: {e}")

    for run_index in range(2, n_runs + 1):
        stem = f"q_qwen_local_run_{run_index:02d}"
        print(f"\n[Q] Run {run_index:02d} (fresh Qwen3.5-4B call — may take minutes)...")
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
            f"  Response received in {call_time:.1f}s "
            f"(load {load_time:.1f}s, generate {gen_time:.1f}s)"
        )

        try:
            code = extract_python_code(response)
        except Exception as e:
            print(f"  extract_python_code failed: {e}")
            continue

        path = _save_basis_code(
            stem,
            header=(
                f"Q-condition basis from Qwen3.5-4B run {run_index:02d}. "
                "max_tokens 2000. Saved by scripts/e1_multi_seed.py."
            ),
            code=code,
        )
        print(f"  Saved raw code to {path.name} ({len(code)} chars)")

        try:
            basis_fn = load_callable(code, function_name="basis")
        except Exception as e:
            print(f"  Sandbox load failed: {e}")
            continue

        if _verify_basis_on_sample(f"Q run {run_index:02d}", basis_fn) is None:
            continue

        result = fit_and_eval(
            "Q", run_index, basis_fn, train_data, val_data, call_time, path.name
        )
        if result is not None:
            results.append(result)

    return results


def variance_summary(results: list[RunResult]) -> None:
    print()
    print("=" * 90)
    print("E1 multi-LLM-call variance summary")
    print("=" * 90)
    for condition in ("P", "Q"):
        rows = [r for r in results if r.condition == condition]
        if not rows:
            print(f"{condition}: no runs")
            continue

        one_step = np.array([r.one_step_mse for r in rows])
        roll10 = np.array([r.rollout_mse_10 for r in rows])
        roll50 = np.array([r.rollout_mse_50 for r in rows])
        n_basis = np.array([r.n_basis for r in rows])
        active = np.array([r.active_terms for r in rows])

        print(f"{condition}  (N={len(rows)} runs)")
        print(f"  n_basis     : {n_basis.min():>4d}..{n_basis.max():<4d}  "
              f"mean {n_basis.mean():.1f}")
        print(f"  active_terms: {active.min():>4d}..{active.max():<4d}  "
              f"mean {active.mean():.1f}")
        print(f"  one_step_mse: mean {one_step.mean():.4e}  "
              f"std {one_step.std():.2e}  "
              f"CV {one_step.std() / one_step.mean() * 100:.1f}%")
        print(f"  roll_mse_10 : mean {roll10.mean():.4e}  "
              f"std {roll10.std():.2e}  "
              f"CV {roll10.std() / roll10.mean() * 100:.1f}%")
        print(f"  roll_mse_50 : mean {roll50.mean():.4e}  "
              f"std {roll50.std():.2e}  "
              f"CV {roll50.std() / roll50.mean() * 100:.1f}%")
        print()
    print("=" * 90)


def write_csv(results: list[RunResult], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nWrote {len(results)} rows to {path}")


def main() -> int:
    num_train = 200
    num_val = 50
    steps_per_traj = 400
    n_runs = 3

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

    print()
    print("=" * 90)
    print("P condition (Qwen-Plus via DashScope)")
    print("=" * 90)
    p_results = run_p_condition(n_runs, train, val)

    print()
    print("=" * 90)
    print("Q condition (Qwen3.5-4B via mlx-vlm)")
    print("=" * 90)
    q_results = run_q_condition(n_runs, train, val)

    all_results = p_results + q_results
    variance_summary(all_results)
    write_csv(all_results, REPO_ROOT / "results" / "e1_multi_seed.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())

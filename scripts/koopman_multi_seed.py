"""Multi-seed (P × 3) gap-fill for Experiment 3 (Koopman / EDMDc).

Single-seed E3 (in ``koopman_results_clean.csv``) has only one P call
and one Q attempt. To get an error bar on the P headline number, we
re-call Qwen 3.6-Plus three times under the same clean prompt and
fit EDMDc each time. Q is intentionally skipped — D036 already
records 3 independent Q failures under the clean prompt; more
attempts dilute the fragility narrative without adding info.

Output:
- ``results/koopman_multi_seed.csv`` — 3 P rows
- ``llm_artifacts/koopman_observables_multiseed/p_qwen_plus_run_{01,02,03}.py``

Env vars:
- ``DASHSCOPE_API_KEY`` required.

Usage::

    DASHSCOPE_API_KEY=sk-... python scripts/koopman_multi_seed.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.code_extraction import (
    CodeExtractionError,
    extract_python_code,
)
from me569_project.llm.prompts import build_koopman_observable_prompt
from me569_project.llm.qwen_plus_client import QwenPlusClient, QwenPlusError
from me569_project.llm.sandbox import SandboxError, load_callable
from me569_project.sysid.edmdc import (
    edmdc_one_step_mse,
    edmdc_rollout_mse,
    fit_edmdc,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "koopman_multi_seed.csv"
ARTIFACTS_DIR = REPO_ROOT / "llm_artifacts" / "koopman_observables_multiseed"

NUM_RUNS = 3


def _flatten(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten TrajectoryData into (X, U, X_next) flat arrays.

    Duplicated from scripts/run_koopman.py because scripts/ is not a package.
    """
    X = data.states[:, :-1, :].reshape(-1, 6)
    U = data.actions.reshape(-1, 2)
    X_next = data.next_states.reshape(-1, 6)
    return X, U, X_next


def _validate_and_load_observable(
    code: str, function_name: str = "observables",
) -> tuple[callable, int] | None:
    """Sandbox-load + sentinel-check + state-recovery check.

    Returns (callable, n_obs) or None on failure (with a printed reason).
    Duplicated from scripts/run_koopman.py.
    """
    try:
        fn = load_callable(code, function_name=function_name)
    except SandboxError as e:
        print(f"      sandbox failed to load function: {e}")
        return None

    sentinel = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.05])
    try:
        out = np.asarray(fn(sentinel), dtype=np.float64).ravel()
    except Exception as e:
        print(f"      observable crashed on sentinel: {e}")
        return None

    if not np.all(np.isfinite(out)):
        print("      observable produced non-finite output on sentinel")
        return None
    n_obs = out.shape[0]
    if n_obs < 6 or n_obs > 50:
        print(f"      observable dim {n_obs} out of allowed range [6, 50]")
        return None
    if not np.allclose(out[:6], sentinel, atol=1e-10):
        print(
            f"      observable violates state-recovery convention "
            f"(first 6 != input). Got {out[:6]}, expected {sentinel}"
        )
        return None

    return fn, n_obs


def call_p_and_fit(
    run_idx: int, train, val, client: QwenPlusClient
) -> dict | None:
    prompt = build_koopman_observable_prompt(include_eom=False)
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

    loaded = _validate_and_load_observable(code)
    if loaded is None:
        return None
    obs_fn, _ = loaded

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS_DIR / f"p_qwen_plus_run_{run_idx:02d}.py"
    artifact_path.write_text(code)
    print(f"[P|run {run_idx}]   saved {artifact_path}")

    print(f"[P|run {run_idx}] Fitting EDMDc ...")
    t1 = time.time()
    res = fit_edmdc(
        *_flatten(train), obs_fn,
        observable_name=f"P-Qwen-Plus-multi-seed-run-{run_idx:02d}",
    )
    fit_time = time.time() - t1
    print(
        f"[P|run {run_idx}]   fit in {fit_time:.2f}s, n_obs={res.n_obs}, "
        f"train_one_step_mse={res.train_one_step_mse:.4e}"
    )

    X_va, U_va, X_va_next = _flatten(val)
    one_step = edmdc_one_step_mse(res, X_va, U_va, X_va_next)
    rollout = edmdc_rollout_mse(
        res, val.states[:, 0, :], val.actions, val.states[:, 1:, :],
        horizon=50,
    )
    print(
        f"[P|run {run_idx}]   val one_step={one_step:.4e}, rollout_50={rollout:.4e}"
    )

    return {
        "condition": "P",
        "run_index": run_idx,
        "observable_description": "Qwen 3.6-Plus generated (multi-seed)",
        "n_obs": res.n_obs,
        "fit_time_s": fit_time,
        "train_one_step_mse": res.train_one_step_mse,
        "one_step_mse": one_step,
        "rollout_mse_50": rollout,
        "llm_call_time_s": call_time,
        "llm_model": "qwen-plus",
        "llm_total_tokens": (
            client.last_usage.get("total_tokens", 0)
            if client.last_usage else 0
        ),
        "llm_request_id": client.last_request_id or "",
    }


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

    print("Generating 500 train + 100 val trajectories ...")
    t0 = time.time()
    train = generate_trajectories(
        num_trajectories=500, steps_per_trajectory=400, seed=0,
    )
    val = generate_trajectories(
        num_trajectories=100, steps_per_trajectory=400, seed=42,
    )
    print(
        f"Data done in {time.time() - t0:.1f}s "
        f"({train.num_transitions} train, {val.num_transitions} val transitions)"
    )

    client = QwenPlusClient(api_key=api_key, max_tokens=4096)

    rows: list[dict] = []
    for i in range(1, NUM_RUNS + 1):
        row = call_p_and_fit(i, train, val, client)
        if row is not None:
            rows.append(row)

    print()
    print("=" * 76)
    print(f"E3 Multi-Seed Summary (P × {len(rows)} successful runs)")
    print("=" * 76)
    print(
        f"{'run':>3s} {'n_obs':>6s} {'val_1step':>14s} {'roll_50':>14s} "
        f"{'tokens':>8s}"
    )
    for r in rows:
        print(
            f"{r['run_index']:3d} {r['n_obs']:6d} "
            f"{r['one_step_mse']:14.4e} {r['rollout_mse_50']:14.4e} "
            f"{r['llm_total_tokens']:8d}"
        )
    if rows:
        ones = np.array([r["one_step_mse"] for r in rows])
        rolls = np.array([r["rollout_mse_50"] for r in rows])
        print(
            f"{'mean':>3s} {'':>6s} {ones.mean():14.4e} {rolls.mean():14.4e}"
        )
        print(
            f"{'std':>3s} {'':>6s} {ones.std(ddof=1):14.4e} "
            f"{rolls.std(ddof=1):14.4e}"
        )
    print("=" * 76)

    write_csv(rows, RESULTS_CSV)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

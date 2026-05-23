"""Prompt-robustness harness (grader 4): K=5 paraphrases per task x P/Q.

For each (task, condition, paraphrase): call the LLM once with the
task/condition-appropriate config, evaluate the response, save the
artifact (including failures, for the audit trail), and record one CSV
row. Aggregate metrics are computed downstream by
scripts/figure_prompt_robustness.py.

Examples:
    export DASHSCOPE_API_KEY=sk-...
    uv run python scripts/run_prompt_robustness.py                  # all tasks, P+Q
    uv run python scripts/run_prompt_robustness.py --condition P    # cloud only
    uv run python scripts/run_prompt_robustness.py --task E1,E2     # subset
    uv run python scripts/run_prompt_robustness.py --e4-timesteps 300000
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.prompt_paraphrases import PARAPHRASES, TASK_IDS
from me569_project.llm.prompts import (
    EUREKA_REWARD_EOM_APPENDIX,
    KOOPMAN_OBSERVABLE_EOM_APPENDIX,
)
from me569_project.llm.robustness_eval import (
    evaluate_e1_response,
    evaluate_e2_response,
    evaluate_e3_response,
    evaluate_e4_response,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "prompt_robustness.csv"
ARTIFACTS_ROOT = REPO_ROOT / "llm_artifacts" / "prompt_robustness"

# Per-task client config: (P max_tokens, Q max_tokens, Q uses /no_think,
# Q appends EOM appendix). Mirrors the existing per-experiment runners.
TASK_CFG = {
    "E1": dict(p_max=1500, q_max=2000, q_nothink=False, q_eom=None),
    "E2": dict(p_max=1500, q_max=4000, q_nothink=True,  q_eom=None),
    "E3": dict(p_max=4096, q_max=4096, q_nothink=True,  q_eom=KOOPMAN_OBSERVABLE_EOM_APPENDIX),
    "E4": dict(p_max=4096, q_max=4096, q_nothink=True,  q_eom=EUREKA_REWARD_EOM_APPENDIX),
}

QUALITY_NAME = {"E1": "val_one_step_mse", "E2": "control_energy",
                "E3": "val_one_step_mse", "E4": "control_energy"}


def _build_q_prompt(task: str, base: str) -> str:
    cfg = TASK_CFG[task]
    prompt = base
    if cfg["q_eom"]:
        prompt = f"{prompt}\n\n{cfg['q_eom']}"
    if cfg["q_nothink"]:
        prompt = f"/no_think\n\n{prompt}"
    return prompt


def _evaluate(task: str, response: str, train, val, e4_timesteps: int):
    if task == "E1":
        return evaluate_e1_response(response, train, val)
    if task == "E2":
        return evaluate_e2_response(response)
    if task == "E3":
        return evaluate_e3_response(response, train, val)
    if task == "E4":
        return evaluate_e4_response(response, total_timesteps=e4_timesteps, train=True)
    raise ValueError(task)


def _save_artifact(task: str, condition: str, k: int, response: str, code: str | None) -> Path:
    d = ARTIFACTS_ROOT / task
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{condition}_paraphrase_{k}.py"
    parts = [f'"""{condition} {task} paraphrase {k}. Saved by run_prompt_robustness.py."""',
             "# --- raw LLM response ---"]
    parts += [f"# {line}" for line in response.splitlines()]
    if code is not None:
        parts.append("# --- extracted code ---")
        parts.append(code.rstrip())
    else:
        parts.append("# (no code extracted)")
    path.write_text("\n".join(parts) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--condition", default="all", choices=["all", "P", "Q"])
    ap.add_argument("--e4-timesteps", type=int, default=300_000)
    args = ap.parse_args()

    tasks = TASK_IDS if args.task == "all" else args.task.split(",")
    conditions = ["P", "Q"] if args.condition == "all" else [args.condition]

    # Shared sysid data for E1/E3 quality (same as published runs).
    train = generate_trajectories(500, steps_per_trajectory=400, seed=0)
    val = generate_trajectories(100, steps_per_trajectory=400, seed=42)

    # Lazy client construction so a P-only or Q-only run needs only one backend.
    p_client = q_client = None
    rows: list[dict] = []

    for task in tasks:
        cfg = TASK_CFG[task]
        for k, base_prompt in enumerate(PARAPHRASES[task]):
            for condition in conditions:
                if condition == "P":
                    if p_client is None:
                        from me569_project.llm.qwen_plus_client import QwenPlusClient
                        p_client = QwenPlusClient(max_tokens=4096)
                    p_client.max_tokens = cfg["p_max"]
                    prompt = base_prompt
                    client = p_client
                else:
                    if q_client is None:
                        from me569_project.llm.qwen_local_client import QwenLocalClient
                        q_client = QwenLocalClient(max_tokens=4096)
                    q_client.max_tokens = cfg["q_max"]
                    prompt = _build_q_prompt(task, base_prompt)
                    client = q_client

                print(f"[{task}/{condition}/p{k}] calling ({len(prompt)} chars) ...")
                t0 = time.time()
                try:
                    response = client.call(prompt)
                except Exception as e:  # noqa: BLE001
                    response = ""
                    print(f"   call failed: {type(e).__name__}: {e}")
                call_time = time.time() - t0
                tokens = 0
                if condition == "P" and getattr(client, "last_usage", None):
                    tokens = int(client.last_usage.get("total_tokens", 0))

                res = _evaluate(task, response, train, val, args.e4_timesteps)
                _save_artifact(task, condition, k, response, res.code)
                qval = "" if (res.quality != res.quality) else res.quality  # NaN -> ""
                rows.append({
                    "task": task,
                    "condition": condition,
                    "paraphrase_id": k,
                    "emitted_code": int(res.emitted),
                    "quality_metric_name": QUALITY_NAME[task],
                    "primary_metric_value": qval,
                    "status": res.status,
                    "llm_call_time_s": round(call_time, 2),
                    "llm_total_tokens": tokens,
                })
                print(f"   emitted={res.emitted} status={res.status} "
                      f"quality={qval} ({call_time:.1f}s)")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "condition", "paraphrase_id", "emitted_code",
                  "quality_metric_name", "primary_metric_value", "status",
                  "llm_call_time_s", "llm_total_tokens"]
    # Merge with any existing rows so P-only then Q-only runs accumulate.
    existing: list[dict] = []
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open() as f:
            existing = [r for r in csv.DictReader(f)
                        if not any(r["task"] == n["task"] and r["condition"] == n["condition"]
                                   and r["paraphrase_id"] == str(n["paraphrase_id"]) for n in rows)]
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing + rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} new rows ({len(existing)} preserved) to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

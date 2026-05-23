"""Probe three strategies for fixing the Qwen3.5-4B E3 thinking-mode loop.

Background
----------
The first two attempts at the Q condition for E3 (commit 8251801) hit
a degenerate thinking-mode loop on Qwen3.5-4B: the model kept repeating
``Wait, I need to ensure I don't use np.sin. I will use sin if needed.``
and never emitted any code, burning through 2000 and then 4000 tokens
in the preamble alone. The same model works fine on the E1 SINDy prompt,
so the bug is specific to E3's constraint-heavy wording.

This script tries three candidate fixes in isolation and prints whether
each produced parseable Python code:

    Strategy A: Prepend the Qwen3 ``/no_think`` directive to the existing
                (verbose) E3 prompt. If the model honors the directive
                it will skip thinking entirely and emit only the answer.

    Strategy B: Use a new, concise E3 prompt (~500 chars vs 3300 chars)
                that lists constraints exactly once, at the bottom,
                instead of interleaving them throughout.

    Strategy C: Both A and B combined.

For each strategy this script prints:
- token / time cost
- whether extract_python_code succeeded
- whether load_stage_cost_callable succeeded
- the first few lines of the extracted code

Usage
-----
    python scripts/probe_q_e3_strategies.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from me569_project.llm.code_extraction import (
    CodeExtractionError,
    extract_python_code,
)
from me569_project.llm.prompts import build_mpc_cost_prompt
from me569_project.llm.qwen_local_client import QwenLocalClient, QwenLocalError
from me569_project.llm.sandbox import SandboxError
from me569_project.mpc.mpc_math import load_stage_cost_callable


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "llm_artifacts" / "e3_costs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CONCISE_E3_PROMPT = """\
You are writing an MPC stage cost for a Planar Quadrotor hovering at the origin.

State   x = [p_x, p_z, theta, v_x, v_z, omega]  (index 0..5)
Control u = [u_1, u_2]                           (index 0..1)
Hover equilibrium: x = 0, u_1 = u_2 = 4.905 N

Write a function ``stage_cost(x, u)`` that returns one scalar cost
(smaller = better). It must be zero at the hover equilibrium and
positive off-equilibrium. The cost is evaluated symbolically inside
a CasADi-based MPC solver.

Available math helpers are already bound as bare names:
    sin, cos, exp, sqrt, log, fabs
Allowed operators: + - * / ** parentheses and numeric literals.
No imports. No numpy. Use ``fabs`` instead of Python's ``abs``.

Respond with ONLY the function inside a single ``python`` code fence.
Do not output any reasoning, commentary, or analysis before or after
the code fence.
"""


def _try_strategy(
    label: str,
    prompt: str,
    client: QwenLocalClient,
    save_stem: str,
) -> dict:
    print()
    print("=" * 70)
    print(f"Strategy: {label}")
    print(f"  prompt length: {len(prompt)} chars")
    print("=" * 70)

    t0 = time.time()
    try:
        response = client.call(prompt)
    except QwenLocalError as e:
        print(f"  ERROR calling model: {e}")
        return {"label": label, "ok": False, "reason": "call_error"}
    call_time = time.time() - t0
    print(f"  response in {call_time:.1f}s, {len(response)} chars")

    # Save raw response for forensic analysis.
    raw_path = OUTPUT_DIR / f"probe_{save_stem}.raw.txt"
    raw_path.write_text(response)

    try:
        code = extract_python_code(response)
    except CodeExtractionError as e:
        print(f"  extract_python_code FAILED: {e}")
        print(f"  (raw response saved to {raw_path.name})")
        # Print the last 500 chars of the response so we can see
        # what the model was producing at the end.
        tail = response[-500:]
        print("  --- last 500 chars of raw response ---")
        print(tail)
        return {
            "label": label,
            "ok": False,
            "reason": "extract_failed",
            "call_time": call_time,
        }

    code_path = OUTPUT_DIR / f"probe_{save_stem}.py"
    code_path.write_text(code)
    print(f"  extracted code: {len(code)} chars -> {code_path.name}")

    try:
        fn = load_stage_cost_callable(code)
    except SandboxError as e:
        print(f"  sandbox load FAILED: {e}")
        print("  --- first 20 code lines ---")
        for line in code.splitlines()[:20]:
            print(f"    {line}")
        return {
            "label": label,
            "ok": False,
            "reason": "sandbox_failed",
            "call_time": call_time,
            "code_chars": len(code),
        }

    # Try a quick numerical call to make sure the function actually
    # works, not just that it parses.
    try:
        import numpy as np

        sample_x = np.zeros(6)
        sample_u = np.array([4.905, 4.905])
        val = float(fn(sample_x, sample_u))
        print(f"  stage_cost at hover = {val:.6e}")
    except Exception as e:
        print(f"  call-at-hover FAILED: {type(e).__name__}: {e}")
        return {
            "label": label,
            "ok": False,
            "reason": "runtime_error",
            "call_time": call_time,
            "code_chars": len(code),
        }

    print("  ---- FIRST LINES OF THE EXTRACTED FUNCTION ----")
    for line in code.splitlines()[:15]:
        print(f"    {line}")
    print("  --------")
    print("  RESULT: OK")
    return {
        "label": label,
        "ok": True,
        "call_time": call_time,
        "code_chars": len(code),
        "code_path": str(code_path),
    }


def main() -> int:
    # Use 4000 max_tokens across the board so the test is about the
    # prompt, not about the budget.
    client = QwenLocalClient(max_tokens=4000)

    verbose_prompt = build_mpc_cost_prompt()
    no_think_verbose = f"/no_think\n\n{verbose_prompt}"
    no_think_concise = f"/no_think\n\n{CONCISE_E3_PROMPT}"

    strategies = [
        ("A: /no_think + verbose prompt", no_think_verbose, "a_nothink_verbose"),
        ("B: concise prompt (no /no_think)", CONCISE_E3_PROMPT, "b_concise"),
        ("C: /no_think + concise prompt", no_think_concise, "c_nothink_concise"),
    ]

    results = []
    for label, prompt, stem in strategies:
        try:
            results.append(_try_strategy(label, prompt, client, stem))
        except Exception as e:
            print(f"  strategy crashed: {type(e).__name__}: {e}")
            results.append({"label": label, "ok": False, "reason": "probe_crash"})

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        status = "OK" if r.get("ok") else f"FAIL ({r.get('reason', '?')})"
        time_str = f"{r.get('call_time', 0):.1f}s"
        print(f"  {r['label']:<48s}  {status:<20s}  {time_str}")

    # Return exit code 0 if at least one strategy worked
    return 0 if any(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

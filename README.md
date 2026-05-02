# me569-llm-quadrotor

**LLM-Augmented Data-Driven Control on the Planar Quadrotor**

A course project for **ME569 Machine Learning Control** (University of Washington, Spring 2026, Prof. Steven Brunton) investigating whether large language models can serve as automated *physics prior providers* inside classical data-driven control pipelines.

## Overview

This project evaluates LLM-generated code at two stages of the data-driven control pipeline on a Planar Quadrotor hover task:

1. **System identification** — generating a candidate basis library Φ(x,u) for Sparse Identification of Nonlinear Dynamics (SINDy).
2. **Optimal control** — generating a stage cost function ℓ(x,u) for Model Predictive Control (MPC).

Each stage is compared in a three-way study:

- **Baseline (B):** textbook polynomial basis and quadratic cost — no LLM.
- **Qwen 3.6-Plus (P):** cloud frontier LLM via Alibaba DashScope API.
- **Qwen3.5-4B (Q):** locally-runnable edge LLM via mlx-vlm on Apple Silicon.

The deliverable is a cost-vs-quality trade-off table for engineers deciding between cloud and edge LLMs when automating control design tasks.

## Project status

Primary scope (Experiment 1: SINDy basis selection, Experiment 2: MPC stage cost design) complete with full B/P/Q coverage. Course timeline: **2026-04-08 → 2026-06-05**.

Headline results (single-seed, 20 initial states):

| Stage | Metric | B | P (Qwen 3.6-Plus) | Q (Qwen3.5-4B) |
|---|---|---|---|---|
| E1 SINDy | one-step MSE | 16.06 | 0.556 | 0.787 |
| E1 SINDy | 50-step rollout MSE | 7.32 | 0.009 | 0.045 |
| E2 MPC | hover success rate | 100% | 100% | 100% |
| E2 MPC | mean settling time (s) | 0.82 | 1.11 | 0.82 |
| E2 MPC | mean control energy | 0.475 | 1.108 | 0.475 |

A second set of E2 results from a clean-prompt rerun (sanitized prompt template that no longer carries a Bryson-shape worked example) is in `results/e3_results_clean.csv`, alongside the original `e3_results.csv`. Multi-seed variance studies and prompt ablations are in progress and will be folded into the final report.

## Repository layout

```
src/me569_project/      Python package (dynamics, env, controllers, SysID, LLM clients)
tests/                  pytest test suite
scripts/                runnable experiments and demos
data/                   generated trajectory data (gitignored)
results/                experiment outputs (CSVs + figures)
llm_artifacts/          LLM-generated basis libraries (E1) and stage costs (E2)
```

## Getting started

Set up the environment (Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"            # tests only
pip install -e ".[all]"            # tests + experiment runners
pytest
```

### Running the experiments

Each experiment runs the baseline (B) unconditionally. The two LLM conditions need extra setup:

- **B (baseline):** nothing extra.
- **P (Qwen 3.6-Plus):** export `DASHSCOPE_API_KEY=sk-...` (Alibaba Cloud DashScope account required). Scripts skip P silently if the variable is unset.
- **Q (Qwen3.5-4B):** runs locally via `mlx-vlm` on Apple Silicon. Weights (`mlx-community/Qwen3.5-4B-MLX-8bit`, ~4.8 GB) auto-download from HuggingFace on first use and cache in `~/.cache/huggingface/`. Set `E1_SKIP_Q=1` or `E3_SKIP_Q=1` to skip Q on non-Apple-Silicon machines.

Headline runs:

```bash
DASHSCOPE_API_KEY=sk-... python scripts/run_e1_full.py     # E1 single-seed B/P/Q
DASHSCOPE_API_KEY=sk-... python scripts/run_e3_full.py     # E2 single-seed B/P/Q
```

CSV outputs land in `results/`; the LLM-generated Python code (basis libraries, stage costs) lands in `llm_artifacts/`.

## Author

**Suhang Xu** — solo project
GitHub: [@HangSensei](https://github.com/HangSensei)

## Attribution

- Parts of the system identification framework and data-generation patterns are adapted from the author's own prior project for **ME571** (University of Washington, Winter 2026), which studied DMDc/EDMDc/MLP on an actuated pendulum. That earlier work is not hosted publicly; the reused ideas are cited in the final report.
- This project is **inspired by** *"Eureka: Human-Level Reward Design via Coding Large Language Models"* (Ma et al., **ICLR 2024**, arXiv:2310.12931), which introduced LLM-generated reward functions for reinforcement learning.
- LLMs used as part of the project methodology: **Qwen 3.6-Plus** (Alibaba Cloud DashScope) and **Qwen3.5-4B** (local inference via mlx-vlm on Apple Silicon).
- AI-assisted development: code authoring, code review, and bug-finding were assisted by **Claude** (Anthropic, Opus 4.7, 1M-context). All project planning, scientific decisions, written prose, and test design were performed by the author.

## License

MIT — see [LICENSE](LICENSE).

# me569-llm-quadrotor

**LLM-Augmented Data-Driven Control on the Planar Quadrotor**

A course project for **ME569 Machine Learning Control** (University of Washington, Spring 2026, Prof. Steven Brunton) investigating whether large language models can serve as automated *physics prior providers* inside classical data-driven control pipelines.

## Overview

This project evaluates LLM-generated code across the stages of a data-driven control pipeline on a Planar Quadrotor (6D state, 2D control), comparing a frontier cloud model against a small edge model that runs on a laptop.

Four LLM-augmented experiments plus two closed-loop extensions:

1. **E1 — SINDy basis** — generate a basis library Φ(x,u) for Sparse Identification of Nonlinear Dynamics.
2. **E2 — MPC stage cost** — generate a stage cost ℓ(x,u) for Model Predictive Control.
3. **E3 — Koopman/EDMDc observable** — generate an observable ψ(x) for an EDMDc linear-predictor fit.
4. **E4 — Eureka-style PPO reward** — generate a reward r(s,a) for a Stable-Baselines3 PPO agent.
5. **Cascade** — drop the E1-fitted SINDy model into the E2 MPC as the prediction model.
6. **Disturbance sweep** — replay the E2 evaluation under additive process noise.

A non-LLM trajectory-tracking demo (step + figure-8 references) exercises the controller stack beyond hover.

Each experiment is a three-way study:

- **Baseline (B):** textbook polynomial basis / quadratic cost / negative-quadratic reward — no LLM.
- **Qwen 3.6-Plus (P):** cloud frontier LLM via Alibaba DashScope API.
- **Qwen3.5-4B (Q):** locally-runnable edge LLM via mlx-vlm on Apple Silicon.

The deliverable is a cost-vs-quality trade-off analysis for engineers deciding between cloud and edge LLMs when automating control-design tasks.

## Project status

All four experiments and both extensions are complete with B/P/Q coverage. Course timeline: **2026-04-08 → 2026-06-05**.

Headline results:

| Stage | Metric | B | P (Qwen 3.6-Plus) | Q (Qwen3.5-4B) |
|---|---|---|---|---|
| E1 SINDy | one-step MSE | 16.06 | 0.556 | 0.787 |
| E1 SINDy | 50-step rollout MSE | 7.32 | 0.009 | 0.045 |
| E2 MPC | hover success rate | 100% | 100% | 100% |
| E2 MPC | mean settling time (s) | 0.82 | 1.11 | 0.82 |
| E2 MPC | mean control energy | 0.475 | 1.108 | 0.475 |
| E3 Koopman | validation one-step MSE | 6.0e-3 | 2.3e-4 (≈26× lower) | failed (no code) |
| E4 PPO | hover success rate @ r=0.30 | 100% | 100% | failed (parse) |
| E4 PPO | mean control energy @ r=0.30 | 0.186 | 0.152 (18% lower) | failed (parse) |

The headline is bifurcated: LLMs deliver large **structural** gains at system-identification stages (E1, E3) where the right basis or observable unlocks representational capacity the polynomial baseline cannot reach, but they do not improve **numerical** tuning at the cost/reward stages (E2, E4) under nominal evaluation. Two honest caveats run the other way:

- **Cascade.** P's E1-fitted SINDy model dropped into MPC recovers ground-truth-model quality (100% hover, within 5% on all metrics); Q's basis collapses to 20% closed-loop success.
- **Disturbance.** Under additive process noise the nominal order reverses: at σ=0.10 the aggressively-tuned Q cost reaches 95% hover success vs B's 75% and P's 55%.

Q matches P only when the prompt scaffolds the answer (explicit equations or a worked example). Under sanitized prompts on novel tasks it produced no usable code across four independent attempts (three on E3, one on E4) — emitting reasoning text or markdown-contaminated indentation instead of a loadable function. Both the original (prompt-leaked) and sanitized reruns are kept under `results/`, because the prompt-leak audit is itself part of the contribution.

## Repository layout

```
src/me569_project/      Python package (dynamics, env, controllers, SysID, MPC, RL, LLM clients)
tests/                  pytest test suite (260 tests)
scripts/                runnable experiments, figure generators, and demos
data/                   generated trajectory data (gitignored)
results/                experiment outputs (CSVs + figures)
llm_artifacts/          LLM-generated code: SINDy bases (E1), MPC costs (E2),
                        Koopman observables (E3), Eureka rewards (E4)
```

## Getting started

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (Python ≥ 3.11; the committed `uv.lock` pins the full environment).

```bash
uv sync --extra all      # install package + all experiment runners
uv run pytest            # run the test suite
```

(A plain `pip install -e ".[all]"` into a virtualenv also works if you prefer.)

### Running the experiments

Each experiment runs the baseline (B) unconditionally. The two LLM conditions need extra setup:

- **B (baseline):** nothing extra.
- **P (Qwen 3.6-Plus):** export `DASHSCOPE_API_KEY=sk-...` (Alibaba Cloud DashScope account required). Scripts skip P silently if the variable is unset.
- **Q (Qwen3.5-4B):** runs locally via `mlx-vlm` on Apple Silicon. Weights (`mlx-community/Qwen3.5-4B-MLX-8bit`, ~4.8 GB) auto-download from HuggingFace on first use and cache in `~/.cache/huggingface/`. Skip flags (e.g. `E1_SKIP_Q=1`, `KOOPMAN_SKIP_Q=1`) let the runners proceed on non-Apple-Silicon machines.

```bash
DASHSCOPE_API_KEY=sk-... uv run python scripts/run_e1_full.py      # E1 SINDy basis (B/P/Q)
DASHSCOPE_API_KEY=sk-... uv run python scripts/run_e3_full.py      # E2 MPC cost (B/P/Q)
DASHSCOPE_API_KEY=sk-... uv run python scripts/run_koopman.py      # E3 Koopman/EDMDc (B/P/Q)
DASHSCOPE_API_KEY=sk-... uv run python scripts/run_eureka.py       # E4 Eureka PPO reward (B/P/Q)
uv run python scripts/run_tracking.py                              # trajectory tracking demo (no LLM)
uv run python scripts/run_cascade.py                               # SINDy-MPC cascade
uv run python scripts/run_disturbance.py                           # disturbance-robustness sweep
```

Figure generators live alongside the runners (`scripts/figure_*.py`, `scripts/*_figure.py`). CSV outputs land in `results/`; the LLM-generated Python code lands in `llm_artifacts/`.

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

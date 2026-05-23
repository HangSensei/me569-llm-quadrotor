"""Tests for prompt-robustness metrics, paraphrases, and evaluators."""
from __future__ import annotations

import math

import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.llm.prompt_paraphrases import PARAPHRASES, TASK_IDS
from me569_project.llm.robustness_eval import (
    ParaphraseResult,
    evaluate_e1_response,
    evaluate_e3_response,
    evaluate_e4_response,
)
from me569_project.llm.robustness_metrics import code_emission_rate, quality_cv


def test_emission_rate_basic():
    assert code_emission_rate([True, True, False, False, False]) == pytest.approx(0.4)
    assert code_emission_rate([True] * 5) == 1.0
    assert code_emission_rate([False] * 5) == 0.0


def test_emission_rate_empty_is_zero():
    assert code_emission_rate([]) == 0.0


def test_quality_cv_uses_sample_std_ddof1():
    vals = [1.0, 2.0, 3.0]
    expected = float(np.std(vals, ddof=1) / abs(np.mean(vals)))
    assert quality_cv(vals) == pytest.approx(expected)


def test_quality_cv_undefined_below_two_successes():
    assert quality_cv([]) is None
    assert quality_cv([1.0]) is None


def test_quality_cv_zero_mean_returns_none():
    assert quality_cv([-1.0, 1.0]) is None


# --- paraphrase fixtures ---

_REQUIRED_FN = {
    "E1": "basis",
    "E2": "stage_cost",
    "E3": "observables",
    "E4": "reward",
}


def test_paraphrase_taskset_complete():
    assert set(PARAPHRASES.keys()) == set(TASK_IDS) == {"E1", "E2", "E3", "E4"}


@pytest.mark.parametrize("task", ["E1", "E2", "E3", "E4"])
def test_five_distinct_paraphrases_per_task(task):
    variants = PARAPHRASES[task]
    assert len(variants) == 5
    assert len(set(variants)) == 5  # mutually distinct


@pytest.mark.parametrize("task", ["E1", "E2", "E3", "E4"])
def test_paraphrases_preserve_spec_invariants(task):
    fn = _REQUIRED_FN[task]
    for p in PARAPHRASES[task]:
        assert fn in p, f"{task} paraphrase missing function name {fn!r}"
        # Physical constants preserved
        assert "0.01" in p and "9.81" in p and "0.25" in p
        # Response-format contract preserved (single python code fence)
        assert "python" in p.lower() and "code fence" in p.lower()
        # Reasonable size (a full reworded prompt, not a stub)
        assert len(p) > 400


# --- evaluators ---

_GOOD_BASIS = """```python
def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    import numpy as np
    return [1.0, p_x, p_z, theta, v_x, v_z, omega,
            np.sin(theta)*(u_1+u_2), np.cos(theta)*(u_1+u_2), (u_2-u_1)]
```"""

_GOOD_OBS = """```python
def observables(x):
    import numpy as np
    px, pz, th, vx, vz, om = x
    return np.array([px, pz, th, vx, vz, om, np.sin(th), np.cos(th)])
```"""

_GOOD_REWARD = """```python
def reward(state, action):
    import numpy as np
    pos = -(state[0]**2 + state[1]**2)
    att = -state[2]**2
    vel = -0.1*(state[3]**2 + state[4]**2 + state[5]**2)
    ctrl = -0.01*((action[0]-4.905)**2 + (action[1]-4.905)**2)
    return float(pos + att + vel + ctrl)
```"""

_GARBAGE = "Here is some prose with no code fence and no function at all."


def test_e1_evaluator_accepts_good_basis():
    train = generate_trajectories(8, steps_per_trajectory=40, seed=0)
    val = generate_trajectories(3, steps_per_trajectory=40, seed=42)
    res = evaluate_e1_response(_GOOD_BASIS, train, val)
    assert isinstance(res, ParaphraseResult)
    assert res.emitted is True
    assert math.isfinite(res.quality)


def test_e1_evaluator_rejects_garbage():
    train = generate_trajectories(8, steps_per_trajectory=40, seed=0)
    val = generate_trajectories(3, steps_per_trajectory=40, seed=42)
    res = evaluate_e1_response(_GARBAGE, train, val)
    assert res.emitted is False
    assert math.isnan(res.quality)


def test_e3_evaluator_accepts_good_observable():
    train = generate_trajectories(8, steps_per_trajectory=40, seed=0)
    val = generate_trajectories(3, steps_per_trajectory=40, seed=42)
    res = evaluate_e3_response(_GOOD_OBS, train, val)
    assert res.emitted is True
    assert math.isfinite(res.quality)


def test_e4_evaluator_emission_without_training():
    # train=False => decide emission via sentinel validation, skip PPO.
    res = evaluate_e4_response(_GOOD_REWARD, total_timesteps=0, train=False)
    assert res.emitted is True
    assert math.isnan(res.quality)  # quality not computed when train=False
    bad = evaluate_e4_response(_GARBAGE, total_timesteps=0, train=False)
    assert bad.emitted is False

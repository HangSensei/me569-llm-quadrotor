"""Two-number prompt-robustness profile.

A robustness profile per (task, condition) is:
  1. code-emission rate: fraction of K paraphrases that yield a usable
     (loadable, sandbox-passing, task-validated) artifact, in [0, 1].
  2. quality CV: coefficient of variation (sample std / |mean|, ddof=1)
     of the task's primary quality metric among the *successful*
     paraphrases. Undefined (None -> report "n/a") with < 2 successes
     or a zero mean.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def code_emission_rate(emitted_flags: Sequence[bool]) -> float:
    """Fraction of paraphrases that emitted a usable artifact (0.0 if empty)."""
    flags = list(emitted_flags)
    if not flags:
        return 0.0
    return float(sum(1 for f in flags if f)) / len(flags)


def quality_cv(values: Sequence[float]) -> float | None:
    """Coefficient of variation (ddof=1) of the successful-paraphrase metric.

    Returns None when undefined: fewer than two values, or a mean of zero.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    mean = float(np.mean(vals))
    if mean == 0.0:
        return None
    return float(np.std(vals, ddof=1) / abs(mean))

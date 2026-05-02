"""Polynomial basis function used by the Experiment 1 B (baseline) condition.

This replaces the PySINDy ``PolynomialLibrary`` baseline with a plain
Python callable so that both baseline and LLM conditions feed through
the same ``fit_with_basis_fn`` path in ``llm_sysid_model.py``. Using a
single sparse-regression code path for all three B/P/Q conditions
guarantees the only difference between them is the basis library
itself — which is exactly the comparison Experiment 1 is trying to
make.

The library is the standard monomial basis of total degree ``<=
degree`` over ``n_input`` variables (states concatenated with
controls). For degree 3 over 8 variables this produces 165 features,
matching ``pysindy.PolynomialLibrary(degree=3)`` up to column order.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable

import numpy as np


def make_polynomial_basis_fn(
    degree: int = 3,
    n_input: int = 8,
) -> tuple[Callable[[np.ndarray], list[float]], list[str]]:
    """Build a polynomial basis function and its feature-name list.

    Parameters
    ----------
    degree : int
        Maximum total degree. degree=3 reproduces the default
        ``PolynomialLibrary`` used by the Week 2 baseline.
    n_input : int
        Number of input variables (state + control). Defaults to 8
        for the Planar Quadrotor.

    Returns
    -------
    basis_fn : Callable[[np.ndarray], list[float]]
        Function mapping an (n_input,) array to a list of monomial
        values.
    feature_names : list[str]
        Human-readable names for each output feature, in the same
        order the basis function produces them.
    """
    # Pre-compute the index tuples for every monomial up to the given
    # degree so the hot-path basis function just does products.
    index_tuples: list[tuple[int, ...]] = [()]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_input), d):
            index_tuples.append(combo)

    var_names = [f"xu{i}" for i in range(n_input)]
    feature_names: list[str] = []
    for combo in index_tuples:
        if not combo:
            feature_names.append("1")
        else:
            feature_names.append(" ".join(var_names[i] for i in combo))

    def basis(xu: np.ndarray) -> list[float]:
        xu_arr = np.asarray(xu, dtype=np.float64).reshape(-1)
        features = [1.0]
        for combo in index_tuples[1:]:
            term = 1.0
            for i in combo:
                term *= xu_arr[i]
            features.append(float(term))
        return features

    return basis, feature_names

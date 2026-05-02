"""Tests for the polynomial basis function used in the E1 baseline."""
import numpy as np
import pytest

from me569_project.sysid.polynomial_basis import make_polynomial_basis_fn


def test_degree_3_over_8_variables_has_165_features():
    """Matches ``PolynomialLibrary(degree=3)`` column count: C(8+3, 3) = 165."""
    basis, names = make_polynomial_basis_fn(degree=3, n_input=8)
    xu = np.zeros(8)
    out = basis(xu)
    assert len(out) == 165
    assert len(names) == 165


def test_degree_1_produces_constant_plus_linear_terms():
    basis, names = make_polynomial_basis_fn(degree=1, n_input=4)
    xu = np.array([1.0, 2.0, 3.0, 4.0])
    out = basis(xu)
    assert out == [1.0, 1.0, 2.0, 3.0, 4.0]
    assert names == ["1", "xu0", "xu1", "xu2", "xu3"]


def test_degree_2_over_3_variables_has_10_features():
    """C(3+2, 2) = 10: 1 + 3 linear + 6 quadratic (including squares)."""
    basis, names = make_polynomial_basis_fn(degree=2, n_input=3)
    xu = np.ones(3)
    out = basis(xu)
    assert len(out) == 10
    assert out[0] == 1.0  # constant


def test_quadratic_terms_are_correct():
    basis, names = make_polynomial_basis_fn(degree=2, n_input=2)
    # Features: [1, xu0, xu1, xu0*xu0, xu0*xu1, xu1*xu1]
    xu = np.array([2.0, 3.0])
    out = basis(xu)
    assert out == [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]


def test_first_feature_is_always_constant_one():
    basis, _ = make_polynomial_basis_fn(degree=3)
    xu = np.random.randn(8)
    out = basis(xu)
    assert out[0] == 1.0


def test_basis_handles_list_input():
    basis, _ = make_polynomial_basis_fn(degree=2, n_input=3)
    out = basis([1.0, 2.0, 3.0])
    assert len(out) == 10

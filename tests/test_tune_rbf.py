# tests/test_tune_rbf.py
import pytest

from scripts.tune_rbf import select_best


def test_select_best_picks_min_mse():
    records = [
        {"M": 50, "sigma": 1.0, "mse": 0.9},
        {"M": 100, "sigma": 2.0, "mse": 0.3},
        {"M": 400, "sigma": 0.5, "mse": 0.1},
    ]
    assert select_best(records)["M"] == 400


def test_select_best_respects_m_cap():
    records = [
        {"M": 50, "sigma": 1.0, "mse": 0.9},
        {"M": 100, "sigma": 2.0, "mse": 0.3},
        {"M": 400, "sigma": 0.5, "mse": 0.1},
    ]
    assert select_best(records, m_cap=100)["M"] == 100


def test_select_best_empty_under_cap_returns_none():
    records = [{"M": 400, "sigma": 0.5, "mse": 0.1}]
    assert select_best(records, m_cap=100) is None

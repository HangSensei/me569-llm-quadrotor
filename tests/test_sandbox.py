"""Tests for the LLM-code sandbox."""
import numpy as np
import pytest

from me569_project.llm.sandbox import (
    SandboxError,
    load_callable,
    run_code_in_sandbox,
)


def test_run_valid_code_returns_namespace():
    code = "x = 42\ndef f(y):\n    return y + 1\n"
    ns = run_code_in_sandbox(code)
    assert ns["x"] == 42
    assert ns["f"](1) == 2


def test_run_uses_injected_numpy():
    code = "import numpy as np\nresult = np.sin(0)"
    ns = run_code_in_sandbox(code)
    assert ns["result"] == 0.0


def test_run_code_with_implicit_numpy():
    """``np`` should be available without an explicit import."""
    code = "result = np.cos(0)"
    ns = run_code_in_sandbox(code)
    assert ns["result"] == 1.0


def test_run_raises_on_syntax_error():
    with pytest.raises(SandboxError, match="SyntaxError"):
        run_code_in_sandbox("def basis(x:\n    return x")


def test_run_raises_on_runtime_error():
    with pytest.raises(SandboxError, match="ZeroDivisionError"):
        run_code_in_sandbox("x = 1 / 0")


def test_run_blocks_disallowed_imports():
    with pytest.raises(SandboxError, match="disallowed"):
        run_code_in_sandbox("import os\nos.listdir('/')")


def test_run_blocks_unsafe_builtins_exec():
    """Even if the sandbox cannot statically stop all of them, ``exec`` should be
    missing from the namespace's __builtins__.
    """
    code = "result = 'exec' in dir(__builtins__)"
    ns = run_code_in_sandbox(code)
    assert ns["result"] is False


def test_run_blocks_unsafe_builtins_open():
    code = "result = 'open' in dir(__builtins__)"
    ns = run_code_in_sandbox(code)
    assert ns["result"] is False


def test_run_allows_custom_import_list():
    """Callers can expand the allowlist for advanced use cases."""
    code = "import itertools\nresult = list(itertools.islice(range(10), 3))"
    ns = run_code_in_sandbox(code, allowed_imports={"numpy", "math", "itertools"})
    assert ns["result"] == [0, 1, 2]


def test_load_callable_returns_the_function():
    code = """
def basis(x):
    return [1.0, x[0], np.sin(x[2])]
"""
    fn = load_callable(code, function_name="basis")
    result = fn(np.array([0.5, 0.0, 0.7, 0.0, 0.0, 0.0]))
    assert len(result) == 3
    assert result[0] == 1.0
    assert result[1] == 0.5
    assert result[2] == pytest.approx(np.sin(0.7))


def test_load_callable_raises_if_function_missing():
    code = "x = 42"
    with pytest.raises(SandboxError, match="not defined"):
        load_callable(code, function_name="basis")


def test_load_callable_raises_if_name_is_not_callable():
    code = "basis = 42"
    with pytest.raises(SandboxError, match="not callable"):
        load_callable(code, function_name="basis")


def test_load_callable_works_with_realistic_qwen_output():
    """End-to-end: a realistic LLM code string, loaded as a basis function."""
    code = """
import numpy as np

def basis(x):
    return [
        1.0,
        x[0],
        x[1],
        np.sin(x[2]),
        np.cos(x[2]),
        x[3],
        x[4],
        x[5],
    ]
"""
    fn = load_callable(code)
    state_and_ctrl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    out = fn(state_and_ctrl)
    assert len(out) == 8
    assert out[0] == 1.0
    assert out[3] == pytest.approx(np.sin(0.3))

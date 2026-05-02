"""Restricted exec sandbox for LLM-generated Python code.

The sandbox is **defensive, not adversarial**. Its job is to catch the
common failure modes of LLM code generation (syntax errors, name
errors, accidental filesystem access, runaway imports) and surface
them as a single ``SandboxError`` so the caller can retry or count
the failure. It is not designed to resist a genuinely malicious
author.

What the sandbox does
---------------------
1. **Pre-scan imports.** A regex scan of the source rejects any
   ``import`` or ``from ... import`` line that references a module
   outside the allowlist (default: ``numpy``, ``math``).
2. **Restrict builtins.** The namespace passed to ``exec`` only
   contains a curated subset of builtins — no ``open``, ``exec``,
   ``eval``, ``__import__``, ``compile``, ``input``, ``exit``, or
   ``quit``.
3. **Inject numpy and math as globals.** The LLM doesn't have to
   bother with ``import numpy as np`` in every response; the
   sandbox pre-populates the namespace with both names.
4. **Translate every exception into ``SandboxError``.** Callers see
   a single error type and can decide on retry / abandon / record.

What it does NOT do
-------------------
- Timeouts. LLM-generated basis functions for SysID are pure numpy
  and finish in microseconds; a hang would be a wild LLM bug and
  should be caught at a higher layer (e.g. asyncio.wait_for in the
  E1 runner). Adding signal-based timeouts here would prevent use
  off the main thread and would complicate pytest.
- Resource limits (memory, file descriptors). Not applicable to the
  expected workload.
- Bytecode analysis or AST inspection. Regex-level safety is enough
  for this project's threat model.

Usage
-----
>>> code = "def basis(x):\\n    return [1.0, x[0], np.sin(x[2])]"
>>> fn = load_callable(code, function_name="basis")
>>> fn(np.zeros(6))
[1.0, 0.0, 0.0]
"""
from __future__ import annotations

import builtins as _builtins
import math
import re
from typing import Callable

import numpy as np


class SandboxError(Exception):
    """Raised for any failure inside the LLM-code sandbox."""


DEFAULT_ALLOWED_IMPORTS: set[str] = {"numpy", "np", "math"}

_UNSAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "exit",
        "quit",
        "help",
        "breakpoint",
    }
)

_SAFE_BUILTINS_BASE: dict[str, object] = {
    name: getattr(_builtins, name)
    for name in dir(_builtins)
    if not name.startswith("_") and name not in _UNSAFE_BUILTINS
}

_IMPORT_LINE_RE = re.compile(
    r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.MULTILINE,
)


def _check_imports(code: str, allowed: set[str]) -> None:
    for match in _IMPORT_LINE_RE.finditer(code):
        module = match.group(1)
        if module not in allowed:
            raise SandboxError(
                f"Import of disallowed module: '{module}'. "
                f"Allowed modules: {sorted(allowed)}"
            )


def _make_safe_import(allowed: set[str]):
    """Return a shim ``__import__`` that enforces the module allowlist at runtime.

    This is a second layer of defense beyond ``_check_imports``: even if an
    LLM response slips an ``import`` line past the regex (e.g. via a lambda
    or ``getattr(__builtins__, 'eval')``), the runtime shim will still raise.
    """
    real_import = _builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in this sandbox "
                f"(allowed: {sorted(allowed)})"
            )
        return real_import(name, globals, locals, fromlist, level)

    return safe_import


def _build_namespace(
    allowed_imports: set[str],
    extra_globals: dict | None = None,
    inject_numpy: bool = True,
) -> dict:
    safe_builtins = dict(_SAFE_BUILTINS_BASE)
    safe_builtins["__import__"] = _make_safe_import(allowed_imports)
    namespace: dict = {"__builtins__": safe_builtins}
    if inject_numpy:
        namespace["np"] = np
        namespace["numpy"] = np
        namespace["math"] = math
    if extra_globals:
        namespace.update(extra_globals)
    return namespace


def run_code_in_sandbox(
    code: str,
    allowed_imports: set[str] | None = None,
    extra_globals: dict | None = None,
    inject_numpy: bool = True,
) -> dict:
    """Execute ``code`` in a restricted namespace and return the resulting dict.

    Parameters
    ----------
    code : str
        Python source string, typically produced by
        ``code_extraction.extract_python_code``.
    allowed_imports : set[str], optional
        Module names permitted inside the code. Defaults to
        ``DEFAULT_ALLOWED_IMPORTS`` (numpy, np, math).
    extra_globals : dict, optional
        Additional names to inject into the exec namespace. Use this to
        give the LLM access to a curated set of helpers (e.g. CasADi
        ``sin``, ``cos`` for the E3 MPC stage cost path) without
        having to allow a full ``import casadi``. Default ``None``.
    inject_numpy : bool, optional
        If ``True`` (default), the namespace pre-binds ``np``, ``numpy``,
        and ``math``. Set to ``False`` for sandboxes where the LLM
        output must be evaluable in a non-numpy environment (e.g. CasADi
        symbolic), so that ``np.sin`` etc. would silently break.

    Returns
    -------
    dict
        The globals dictionary produced by ``exec``, containing any
        functions, classes, or variables the code defined.

    Raises
    ------
    SandboxError
        On import-safety violation, syntax error, or any exception
        raised by the code during execution.
    """
    if allowed_imports is None:
        allowed_imports = DEFAULT_ALLOWED_IMPORTS

    _check_imports(code, allowed_imports)

    namespace = _build_namespace(
        allowed_imports,
        extra_globals=extra_globals,
        inject_numpy=inject_numpy,
    )
    try:
        exec(code, namespace)  # noqa: S102 - intentional, this is the sandbox
    except SyntaxError as e:
        raise SandboxError(f"SyntaxError: {e}") from e
    except Exception as e:
        raise SandboxError(f"{type(e).__name__}: {e}") from e

    return namespace


def load_callable(
    code: str,
    function_name: str = "basis",
    allowed_imports: set[str] | None = None,
    extra_globals: dict | None = None,
    inject_numpy: bool = True,
) -> Callable:
    """Exec ``code`` and return the callable named ``function_name``.

    This is the ergonomic wrapper used by the E1 runner: the LLM is
    prompted to define a single function (e.g. ``basis(x)``), and
    this helper hands back that function ready to call on numpy
    arrays. The E3 path uses the same helper with ``inject_numpy=False``
    and an ``extra_globals`` dict of CasADi math functions.
    """
    namespace = run_code_in_sandbox(
        code,
        allowed_imports=allowed_imports,
        extra_globals=extra_globals,
        inject_numpy=inject_numpy,
    )

    if function_name not in namespace:
        raise SandboxError(
            f"Function '{function_name}' is not defined in the code. "
            f"Defined names: {sorted(k for k in namespace if not k.startswith('__'))}"
        )

    obj = namespace[function_name]
    if not callable(obj):
        raise SandboxError(
            f"Name '{function_name}' is not callable (type={type(obj).__name__})"
        )

    return obj

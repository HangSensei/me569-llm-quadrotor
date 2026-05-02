"""Math functions safe for E3 LLM-generated stage cost code.

Why this module exists
----------------------
E3 stage cost functions are evaluated in two regimes:

1. **Symbolic** (during ``do_mpc.controller.MPC.set_objective``): the
   stage cost is called with CasADi ``SX`` / ``MX`` symbolic variables
   and the result is a symbolic expression that gets compiled into the
   MPC optimization problem.

2. **Numerical** (during testing or visualization): the stage cost is
   called with plain ``numpy.ndarray`` values and returns a Python
   float.

The same callable must work in **both** regimes. ``numpy.sin(x)`` does
not work on a CasADi symbol — it raises ``TypeError``. ``casadi.sin(x)``
works on both CasADi symbols and numpy arrays (returning a CasADi DM
matrix in the latter case, which Python can still treat as a number).

So the rule for E3 LLM code is: **use the names from this module
(``sin``, ``cos``, ``exp``, ``sqrt``, ``log``, ``fabs``)**, not the
``numpy`` or ``math`` versions. The sandbox configuration in
``load_stage_cost_callable`` injects these names directly into the
LLM's exec namespace and explicitly disables ``numpy``, so an LLM that
mistakenly writes ``np.sin`` will get a clean ``NameError`` instead of
silently producing a non-symbolic basis.

Public surface
--------------
- ``sin``, ``cos``, ``exp``, ``sqrt``, ``log``, ``fabs``: re-exports of
  ``casadi.sin``, ``casadi.cos``, ``casadi.exp``, ``casadi.sqrt``,
  ``casadi.log``, ``casadi.fabs``. ``fabs`` is named after the C math
  convention to avoid shadowing Python's built-in ``abs`` (which would
  silently fall through to the unsafe Python implementation).
- ``MPC_MATH_NAMESPACE``: the dict of name -> function used by
  ``load_stage_cost_callable`` to populate the sandbox.
- ``load_stage_cost_callable(code, function_name='stage_cost')``:
  convenience wrapper that loads an LLM-generated stage cost function
  with the right sandbox configuration (CasADi math, no numpy, no
  imports allowed).
"""
from __future__ import annotations

from typing import Callable

import casadi


# CasADi math functions. These all work on CasADi SX/MX symbols AND on
# numpy arrays / Python scalars. Use these in E3 LLM-generated code.
sin = casadi.sin
cos = casadi.cos
exp = casadi.exp
sqrt = casadi.sqrt
log = casadi.log
fabs = casadi.fabs


# Namespace dict consumed by ``load_stage_cost_callable`` to inject the
# math functions into the sandbox. Listed explicitly so adding a new
# allowed function is a one-line change here.
MPC_MATH_NAMESPACE: dict[str, Callable] = {
    "sin": sin,
    "cos": cos,
    "exp": exp,
    "sqrt": sqrt,
    "log": log,
    "fabs": fabs,
}


def load_stage_cost_callable(
    code: str,
    function_name: str = "stage_cost",
) -> Callable:
    """Load an LLM-generated stage cost function with E3-safe sandbox config.

    The returned callable is expected to have signature
    ``stage_cost(x, u) -> scalar`` where ``x`` is a 6-element state
    vector and ``u`` is a 2-element control vector. It should work on
    both numpy arrays (for testing) and CasADi symbolic variables (when
    plugged into ``do_mpc.controller.MPC.set_objective``).

    Sandbox configuration applied:
    - ``allowed_imports = set()`` — the LLM cannot ``import numpy`` or
      anything else. The math helpers it needs are pre-injected.
    - ``inject_numpy = False`` — ``np`` and ``numpy`` are NOT in the
      namespace. An LLM that types ``np.sin(x[2])`` gets a clean
      ``NameError`` rather than producing a function that silently
      fails to work in CasADi symbolic mode.
    - ``extra_globals = MPC_MATH_NAMESPACE`` — the CasADi math
      functions are bound as bare names (``sin``, ``cos``, ...), the
      same way they appear in textbook MPC cost expressions.

    Parameters
    ----------
    code : str
        Python source code defining the stage cost function. Typically
        produced by ``llm.code_extraction.extract_python_code`` from a
        Qwen response.
    function_name : str, optional
        Name of the function to extract from the namespace. Defaults to
        ``"stage_cost"``.

    Returns
    -------
    Callable
        The loaded stage cost function, ready to call on either
        numpy arrays or CasADi symbols.

    Raises
    ------
    SandboxError
        If the code fails to parse, contains a disallowed import, or
        defines no function named ``function_name``.
    """
    # Imported here to avoid a circular dependency at module load time.
    from me569_project.llm.sandbox import load_callable

    return load_callable(
        code,
        function_name=function_name,
        allowed_imports=set(),
        extra_globals=MPC_MATH_NAMESPACE,
        inject_numpy=False,
    )

"""Prompt templates for the LLM conditions of the ME569 project.

Currently provides:

- ``SINDY_BASIS_SYSTEM_DESCRIPTION``: a precise natural-language
  description of the Planar Quadrotor (state vector, control vector,
  ground-truth continuous dynamics). Shared by Experiment 1 prompts
  and by any stretch experiment that asks the LLM to reason about
  the physical system.

- ``SINDY_BASIS_TASK_INSTRUCTIONS``: the task-specific instructions
  for Experiment 1: tell the LLM exactly what ``basis(xu)`` must
  look like, what types to return, which imports are allowed, and
  how to format the response.

- ``build_sindy_basis_prompt()``: joins the two sections into a
  single prompt string suitable for
  ``QwenPlusClient.call()`` or the corresponding local client.

Both template constants are kept as Python strings (not Markdown
files on disk) so that they version-control cleanly, import into
tests without filesystem I/O, and can be templated in future via
f-string interpolation if we need condition-specific variants.
"""
from __future__ import annotations


SINDY_BASIS_SYSTEM_DESCRIPTION = """\
You are helping a control engineer identify the dynamics of a Planar Quadrotor
(a 2D simplification of a standard quadcopter, as covered in standard robotics
and data-driven control textbooks such as Brunton & Kutz 2019).

The system has a 6-dimensional state vector and a 2-dimensional control vector:

    x = [p_x, p_z, theta, v_x, v_z, omega] in R^6
    u = [u_1, u_2]                           in R^2

Interpretation:
- p_x, p_z : horizontal and vertical position in meters, p_z positive = up.
- theta    : pitch angle in radians, 0 = level, positive = nose up on the right.
- v_x, v_z : linear velocities in m/s.
- omega    : angular velocity in rad/s.
- u_1, u_2 : left and right rotor thrusts in Newtons. Each rotor is in
             [0, u_max] where u_max = 2*m*g. The hover-equilibrium thrust
             per rotor is u_hover = m*g/2.

Physical parameters: m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m (rotor arm
length), g = 9.81 m/s^2.

The engineer does NOT tell you the equations of motion. The engineer is
relying on your physical and control-theoretic knowledge of planar
quadrotors to propose a basis library. The basis will then be passed into
a SINDy (Sparse Identification of Nonlinear Dynamics) fit with STLSQ
sparse regression, which will figure out the correct sparse linear
combination of your proposed features for each of the six state
derivatives.

Your job is therefore to draw on your own knowledge of quadrotor physics
(gravity, rotor thrust, pitch-dependent thrust projection, differential
thrust producing torque, etc.) and decide which nonlinear functions of
(p_x, p_z, theta, v_x, v_z, omega, u_1, u_2) are most likely to appear
in the dynamics. Do NOT limit yourself to terms that look like brute-force
polynomials unless you believe they are physically meaningful.
"""


SINDY_BASIS_TASK_INSTRUCTIONS = """\
Your task: write a Python function ``basis(xu)`` that takes a single
concatenated state+control vector of length 8 and returns a list (or
numpy array) of scalar basis feature values.

Input convention:
    xu[0] = p_x
    xu[1] = p_z
    xu[2] = theta
    xu[3] = v_x
    xu[4] = v_z
    xu[5] = omega
    xu[6] = u_1
    xu[7] = u_2

Requirements:
- The function MUST be named ``basis`` and MUST accept exactly one positional
  argument named ``xu``.
- ``xu`` is a numpy 1D array of length 8. You may also assume indexing
  with integers works like a Python sequence.
- Return a Python list of floats, or a numpy array of shape (n_basis,).
- ``numpy`` is available as ``np`` (already imported in the sandbox).
  You may also use ``math`` or explicitly ``import numpy as np``.
- Do NOT import anything else. No file I/O, no prints, no global state.
- Choose 10–30 basis features that give SINDy enough flexibility to
  recover each of the six state-derivative equations, including the
  sin/cos couplings in v_x_dot and v_z_dot.

Response format:
- Return ONLY the function definition inside a single ``python`` code fence.
- Do not explain your reasoning outside the code fence. Any explanation
  belongs in comments inside the function body.

Example of the expected format (do NOT copy the exact contents):

```python
import numpy as np

def basis(xu):
    p_x, p_z, theta, v_x, v_z, omega, u_1, u_2 = xu
    # ... compute features ...
    return [
        1.0,
        p_x,
        # ... more features ...
    ]
```
"""


def build_sindy_basis_prompt() -> str:
    """Return the full SINDy basis prompt (system description + task)."""
    return f"{SINDY_BASIS_SYSTEM_DESCRIPTION}\n\n{SINDY_BASIS_TASK_INSTRUCTIONS}"


# ----------------------------------------------------------------------
# Experiment 3 — MPC stage cost prompt
# ----------------------------------------------------------------------

MPC_COST_TASK_INSTRUCTIONS = """\
Your task: write a Python function ``stage_cost(x, u)`` that returns
the per-step cost the MPC controller will minimize to drive the
Planar Quadrotor from a perturbed initial state back to the hover
equilibrium at the origin.

Function signature:

    def stage_cost(x, u):
        # x is a length-6 vector indexable as x[0]..x[5]
        # u is a length-2 vector indexable as u[0], u[1]
        # return a single scalar cost (smaller = better)

Input convention (same as the SINDy task above):
    x[0] = p_x       (horizontal position, m)
    x[1] = p_z       (vertical position, m)
    x[2] = theta     (pitch angle, rad)
    x[3] = v_x       (horizontal velocity, m/s)
    x[4] = v_z       (vertical velocity, m/s)
    x[5] = omega     (angular velocity, rad/s)
    u[0] = u_1       (left rotor thrust, N, in [0, u_max])
    u[1] = u_2       (right rotor thrust, N, in [0, u_max])

Hard requirements (the sandbox will reject violations):
- The function MUST be named ``stage_cost`` and take exactly two
  arguments named ``x`` and ``u`` in that order.
- It MUST return a single scalar value (not a tuple, not a list).
- You may NOT import anything. ``numpy`` is intentionally NOT
  available. The math helpers ``sin``, ``cos``, ``exp``, ``sqrt``,
  ``log``, ``fabs`` are pre-injected as bare names — use them
  directly without any import.
- You may use the standard Python operators ``+``, ``-``, ``*``,
  ``/``, ``**``, parentheses, and integer/float literals.
- You MUST NOT use any other helpers (no ``np.array``, no
  ``math.sqrt``, no ``abs`` builtin — use ``fabs`` instead).

Why these restrictions: the function is evaluated symbolically inside
a CasADi-based MPC solver, so anything outside the allowlist either
fails immediately (unknown name) or silently breaks symbolic
propagation. The helpers above are the CasADi-native versions and
they work both on Python floats and on CasADi symbolic variables.

Design guidance (use your own knowledge of MPC cost design):
- The hover equilibrium is x = 0, u_1 = u_2 = m * g / 2 ~= 4.905 N.
  A good cost is zero (or near zero) at this state and grows when
  the state or control deviates.
- Position errors should typically be penalized more strongly than
  velocity errors.
- Attitude error (theta) should be penalized but not so hard that
  the controller refuses to tilt at all (the quadrotor must tilt to
  translate horizontally).
- Control effort should be penalized lightly so the optimizer
  prefers low-energy solutions but is still allowed to use enough
  thrust to recover from disturbances.

Response format:
- Return ONLY the function definition inside a single ``python``
  code fence. No prose explanation outside the fence.

Example structure (a generic shape — choose your own weights and
the form of each term, including whether to use sin / cos / fabs / etc.):

```python
def stage_cost(x, u):
    # Penalize position deviation from the origin
    pos = ...
    # Penalize pitch deviation from level
    att = ...
    # Penalize translational and angular velocity
    vel = ...
    # Penalize control effort relative to the hover-equilibrium thrust
    ctrl = ...
    return pos + att + vel + ctrl
```
"""


def build_mpc_cost_prompt() -> str:
    """Return the full MPC stage cost prompt (system description + task).

    Reuses ``SINDY_BASIS_SYSTEM_DESCRIPTION`` for the physical
    description because the system, state, and control conventions
    are identical between Experiments 1 and 3. The task instructions
    are E3-specific (different function signature, no numpy, CasADi
    math helpers only).
    """
    return f"{SINDY_BASIS_SYSTEM_DESCRIPTION}\n\n{MPC_COST_TASK_INSTRUCTIONS}"

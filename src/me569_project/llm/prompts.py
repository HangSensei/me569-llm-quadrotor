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


# ----------------------------------------------------------------------
# Experiment 3 (external numbering) — Koopman / EDMDc observable prompt
#
# Externally we call this "E3 Koopman/EDMDc". Internally the file/CSV
# names use the unambiguous label ``koopman_*`` to avoid collision with
# the historical e2/e3 labelling for the MPC cost experiment.
#
# The system description below is intentionally distinct from
# ``SINDY_BASIS_SYSTEM_DESCRIPTION`` even though much of the physical
# spec is identical. The E1 (SINDy) description tells the LLM the basis
# will be passed into a SINDy + STLSQ fit and is allowed to depend on
# both state and controls (xu). The E3 (Koopman / EDMDc) description
# below tells the LLM the observable will be passed into an EDMDc
# linear regression where the lifted state is a function of state alone
# and the control enters separately through the B matrix. Mixing the
# two created the prompt audit issue captured on 2026-05-03:
# Q's first def under the old prompt tried to consume controls as
# x[6], x[7] before self-correcting in a later def block.
# ----------------------------------------------------------------------

KOOPMAN_SYSTEM_DESCRIPTION = """\
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
quadrotors to propose a state-only observable function suitable for an
Extended Dynamic Mode Decomposition with control (EDMDc) fit. EDMDc
fits a linear lifted-space model

    psi(x_{k+1}) = A psi(x_k) + B u_k

by ordinary least squares on the trajectory data. The A matrix captures
how the lifted state evolves under its own dynamics, and the B matrix
captures the linear contribution of control. Because B handles control
linearly and separately, your observable function must depend on state
ONLY: it is a map ``psi : R^6 -> R^N`` with N >= 6. The control inputs
u_1 and u_2 are NOT inputs to your function.

Your job is therefore to draw on your own knowledge of quadrotor physics
(gravity, rotor thrust, pitch-dependent thrust projection, differential
thrust producing torque, etc.) and decide which nonlinear functions of
the state (p_x, p_z, theta, v_x, v_z, omega) — and ONLY of the state —
are most useful as observables. Do NOT limit yourself to terms that
look like brute-force polynomials unless you believe they are physically
meaningful.
"""


KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS = """\
Your task: write a Python function ``observables(x)`` that maps a
6-dimensional planar-quadrotor state to a finite-dimensional vector
of observables suitable for an Extended Dynamic Mode Decomposition
with control (EDMDc) fit.

Function signature:

    def observables(x):
        # x is a length-6 numpy.ndarray indexable as x[0]..x[5]
        # return a 1-D numpy.ndarray (or list) of length N

Input convention:
    x[0] = p_x       (horizontal position, m)
    x[1] = p_z       (vertical position, m)
    x[2] = theta     (pitch angle, rad)
    x[3] = v_x       (horizontal velocity, m/s)
    x[4] = v_z       (vertical velocity, m/s)
    x[5] = omega     (angular velocity, rad/s)
There are no other inputs. ``x`` has length exactly 6. Your function
MUST NOT attempt to read x[6] or beyond.

Hard requirements (the sandbox will reject violations):
- The function MUST be named ``observables`` and take exactly one
  argument named ``x``.
- The output length N must satisfy 6 <= N <= 50.
- The FIRST six components of the output MUST equal the input state
  itself, in order: ``[x[0], x[1], x[2], x[3], x[4], x[5], ...]``.
  This convention lets the downstream pipeline recover the original
  state by slicing the lifted prediction with ``[:6]``.
- You may import ``numpy`` (alias ``np``) and ``math`` and use them
  inside the function body. No other imports.
- The function must be deterministic and pure: no I/O, no global
  mutable state, no random number generation.

Downstream pipeline:
- The caller provides many transitions ``(x_k, u_k, x_{k+1})`` from a
  closed-loop rollout dataset.
- ``observables`` is applied to every state. Let
  ``Psi[k] = observables(x_k)`` and ``Psi_next[k] = observables(x_{k+1})``.
- We solve for matrices A and B in
  ``Psi_next ~ A Psi + B U`` (Psi: K x N, Psi_next: K x N, U: K x 2)
  via ordinary least squares. There is no regularization and no
  sparsity penalty; every observable you propose participates in
  the regression.
- Quality is measured by one-step prediction MSE on held-out
  trajectories, projected back to the original state space via
  ``x_hat = (A Psi + B U)[:6]``, and by 50-step free-run rollout
  MSE in the lifted space (which re-applies ``observables`` after
  each [:6] projection).

Design guidance (use your own knowledge of Koopman / EDMDc):
- Choose observables so the linear regression above gives a tight
  fit of the planar-quadrotor's discrete-time dynamics under
  arbitrary control.
- Choose the smallest N you believe is sufficient. Padding with
  redundant observables wastes regression degrees of freedom and
  worsens conditioning.

Response format:
- Return ONLY the function definition inside a single ``python``
  code fence. No prose explanation outside the fence.
- Do NOT show example outputs, benchmark numbers, or a sample
  observable list — just the function definition.
- Do NOT include test code, import statements outside the function
  body, or any other top-level statements.

Function shell (the body is for you to fill in; the first six
elements of the return value are mandated, the rest are your design):

```python
import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        # ... your additional observables here ...
    ])
```
"""

KOOPMAN_OBSERVABLE_EOM_APPENDIX = """\
APPENDIX (Equations of motion, included to remove ambiguity)

The continuous-time dynamics in numpy notation are
    p_x_dot   = v_x
    p_z_dot   = v_z
    theta_dot = omega
    v_x_dot   = -(u_1 + u_2) * np.sin(theta) / m
    v_z_dot   =  (u_1 + u_2) * np.cos(theta) / m - g
    omega_dot = (u_2 - u_1) * L / I_yy

with constants ``m = 1.0`` kg, ``g = 9.81`` m/s^2, ``L = 0.25`` m,
``I_yy = 0.01`` kg*m^2. The discrete-time transitions seen by EDMDc
are the result of an RK4 integrator with ``dt = 0.02`` applied to
these dynamics.
"""


def build_koopman_observable_prompt(include_eom: bool = False) -> str:
    """Return the full Koopman observable prompt.

    Uses the dedicated ``KOOPMAN_SYSTEM_DESCRIPTION`` for the physical
    system spec (state-only observables, EDMDc lifted-space target),
    and stitches the E3-specific Koopman task instructions on top.
    When ``include_eom`` is True, the explicit equations of motion are
    appended — used for the Q condition (Qwen3.5-4B) by design.

    Note (2026-05-03 prompt audit): an earlier version of this builder reused
    ``SINDY_BASIS_SYSTEM_DESCRIPTION``, which mentioned SINDy + STLSQ
    and listed ``u_1, u_2`` in the observable input set. That created
    an internal inconsistency with the EDMDc task body and is the
    audited "old prompt" referenced by the ``_clean`` rerun.
    """
    parts = [KOOPMAN_SYSTEM_DESCRIPTION, KOOPMAN_OBSERVABLE_TASK_INSTRUCTIONS]
    if include_eom:
        parts.append(KOOPMAN_OBSERVABLE_EOM_APPENDIX)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Experiment 4 (external numbering) — Eureka-style RL reward design
#
# Externally we call this "E4 Eureka RL". The LLM is asked to write a
# per-step reward function ``reward(state, action) -> float`` that
# Stable-Baselines3 PPO will maximize. The prompt is built from two
# constants — a system description distinct from the SINDy / Koopman
# ones (no SINDy / STLSQ / EDMDc framing leaks), and a task body that
# specifies the function signature and the higher-is-better convention.
# An optional EOM appendix is included for the Q condition.
#
# Anti-leak rules:
# - No specific weight numbers in the prompt body.
# - No ``Bryson`` / ``quadratic`` / ``cost`` words that would push the
#   LLM toward a specific reward shape.
# - No worked example with concrete reward values; the function shell
#   shows placeholder slots only.
# ----------------------------------------------------------------------

EUREKA_SYSTEM_DESCRIPTION = """\
You are helping a control engineer design a reward function for a Planar
Quadrotor (a 2D simplification of a standard quadcopter, as covered in
standard robotics and data-driven control textbooks such as Brunton &
Kutz 2019). The reward will be maximized by a reinforcement-learning
agent trained with Proximal Policy Optimization (PPO) — the algorithm
introduced by the Eureka paper (Ma et al., ICLR 2024) for LLM-generated
rewards.

The system has a 6-dimensional state vector and a 2-dimensional control vector:

    state = [p_x, p_z, theta, v_x, v_z, omega] in R^6
    action = [u_1, u_2]                          in R^2

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

The control task is **hover stabilization from a disturbed initial state**:
the agent starts at a small random offset / tilt / velocity, and must
drive the state back to the hover equilibrium x = 0 with rotor thrusts
near u_hover = 4.905 N.

You are not told the equations of motion. You are asked to draw on your
own knowledge of quadrotor physics and reward shaping to design a dense
per-step reward that, when maximized over many training rollouts, will
make a PPO-trained policy stabilize the quadrotor at hover.
"""


EUREKA_REWARD_TASK_INSTRUCTIONS = """\
TASK
----
Write a Python function ``reward(state, action)`` that returns a single
scalar floating-point value the PPO agent will maximize.

Function signature:

    def reward(state, action):
        # state  is a length-6 numpy.ndarray indexable as state[0]..state[5]
        # action is a length-2 numpy.ndarray indexable as action[0], action[1]
        # return a single float (higher = better)

Higher-is-better convention: the agent maximizes. If you wish to penalize
something, the term should be negative. If you wish to encourage
something, the term should be positive.

Hard requirements (the sandbox will reject violations):

- The function MUST be named ``reward`` and take exactly two arguments
  named ``state`` and ``action`` in that order.
- It MUST return a single scalar value (not a tuple, not a list, not an
  array of length > 1).
- It MUST be deterministic and pure: no I/O, no global mutable state,
  no random number generation, no side effects.
- You may import ``numpy`` (alias ``np``) and ``math`` and use them
  inside the function body. No other imports.
- The output must be finite (no NaN, no Inf) on any reasonable input.

Episode-boundary signals (these are NOT your responsibility):
- A crash penalty is applied automatically when the quadrotor leaves the
  workspace (|p_x| > 5 or |p_z| > 5).
- A small per-step survival bonus is applied automatically while the
  episode continues.
- Episode termination at max_steps is handled automatically.

You only need to design the dense per-step reward shape — the
discontinuous boundary signals are handled by the training harness.

Design guidance (use your own knowledge of quadrotor reward shaping):

- The hover equilibrium is ``state = 0``, ``action = [u_hover, u_hover]``
  with u_hover = 4.905 N. A good reward is highest at this state and
  decreases as the state and action deviate.
- Position errors (p_x, p_z) and attitude error (theta) should typically
  matter more than velocity errors (v_x, v_z, omega).
- Control deviation from hover thrust should also be penalized so the
  policy does not learn extreme thrust patterns.
- Smoothness matters: prefer continuous, differentiable shapes that PPO
  gradients can exploit. Avoid step functions or indicator-only rewards.

Response format:

- Return ONLY the function definition inside a single ``python``
  code fence. No prose explanation outside the fence.
- Do NOT show example outputs, benchmark numbers, or specific weight
  recommendations — just the function definition.
- Do NOT include test code, import statements outside the function
  body, or any other top-level statements.

Function shell (the body is for you to fill in; the four placeholder
slots are illustrative — pick your own term shapes and weights):

```python
import numpy as np
import math

def reward(state, action):
    # Penalize position deviation from the hover origin
    pos = ...
    # Penalize attitude deviation from level
    att = ...
    # Penalize translational and angular velocity
    vel = ...
    # Penalize control effort relative to the hover-equilibrium thrust
    ctrl = ...
    return pos + att + vel + ctrl
```
"""


EUREKA_REWARD_EOM_APPENDIX = """\
APPENDIX (Equations of motion, included to remove ambiguity)

The continuous-time dynamics in numpy notation are
    p_x_dot   = v_x
    p_z_dot   = v_z
    theta_dot = omega
    v_x_dot   = -(u_1 + u_2) * np.sin(theta) / m
    v_z_dot   =  (u_1 + u_2) * np.cos(theta) / m - g
    omega_dot = (u_2 - u_1) * L / I_yy

with constants ``m = 1.0`` kg, ``g = 9.81`` m/s^2, ``L = 0.25`` m,
``I_yy = 0.01`` kg*m^2. The discrete-time transitions seen by PPO are
the result of an RK4 integrator with ``dt = 0.02`` applied to these
dynamics.
"""


def build_eureka_reward_prompt(include_eom: bool = False) -> str:
    """Return the full Eureka-style RL reward prompt.

    Uses the dedicated ``EUREKA_SYSTEM_DESCRIPTION`` (no SINDy / EDMDc
    framing) and the E4-specific task instructions. When ``include_eom``
    is True, the explicit equations of motion are appended — used for
    the Q condition (Qwen3.5-4B) by design.

    Anti-leak compliance: the assembled prompt does not
    contain SINDy / STLSQ / EDMDc framing, does not list specific
    Bryson-shape weights, and does not show worked example reward
    values.
    """
    parts = [EUREKA_SYSTEM_DESCRIPTION, EUREKA_REWARD_TASK_INSTRUCTIONS]
    if include_eom:
        parts.append(EUREKA_REWARD_EOM_APPENDIX)
    return "\n\n".join(parts)

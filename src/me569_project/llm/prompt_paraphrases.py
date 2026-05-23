"""Hand-authored semantic paraphrases of the four task prompts.

Five per task (E1-E4). Each preserves the function signature, index
convention, physical constants, and response format of the committed
build_*_prompt(); only wording / clause-order / structure vary. Used by
scripts/run_prompt_robustness.py to measure prompt-robustness.

Authoring contract (enforced by tests/test_prompt_robustness.py):
- preserve the function name (basis / stage_cost / observables / reward),
- preserve the physical constants m=1.0, I_yy=0.01, L=0.25, g=9.81,
- preserve the I/O contract and the "single python code fence" response
  format,
- hand-authored (NOT model-generated), and mutually distinct.

These store the BASE prompt only. The driver appends the equations-of-
motion appendix (E3/E4 Q condition) and prepends /no_think (E2/E3/E4 Q
condition) at call time, mirroring the existing per-experiment runners.
"""
from __future__ import annotations


# --- E1: SINDy basis (function name: basis(xu), xu length 8) ---
SINDY_PARAPHRASES: list[str] = [
    """\
A control engineer is identifying the dynamics of a Planar Quadrotor (a 2D
quadcopter from data-driven control textbooks such as Brunton & Kutz 2019).
The state is x = [p_x, p_z, theta, v_x, v_z, omega] in R^6 and the control is
u = [u_1, u_2] in R^2 (left/right rotor thrusts in Newtons, each in [0, u_max]
with u_max = 2*m*g; hover thrust per rotor is m*g/2). Physical constants:
m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2. The equations of
motion are NOT given; rely on your own knowledge of quadrotor physics
(gravity, pitch-dependent thrust projection, differential thrust torque).

Write a Python function named basis that takes ONE positional argument xu, a
length-8 vector where xu[0..5] is the state [p_x, p_z, theta, v_x, v_z, omega]
and xu[6], xu[7] are u_1, u_2. It must return a list (or numpy array) of
10-30 scalar features that a SINDy/STLSQ sparse regression can combine to
recover each of the six state-derivative equations, including the sin/cos
thrust couplings. numpy is available as np; you may also use math. Do not
import anything else and do not read past xu[7].

Respond with ONLY the function definition inside a single python code fence;
put any explanation in comments inside the function body.
""",
    """\
Task: produce a SINDy feature library for a Planar Quadrotor as a Python
function called basis.

The system has a 6D state x = [p_x, p_z, theta, v_x, v_z, omega] and a 2D
control u = [u_1, u_2] (per-rotor thrusts in Newtons, each within
[0, u_max=2*m*g]; hovering needs m*g/2 per rotor). Known physical parameters:
mass m = 1.0 kg, pitch inertia I_yy = 0.01 kg*m^2, arm length L = 0.25 m,
gravity g = 9.81 m/s^2. You are NOT told the equations of motion; use what you
know about quadrotor physics (gravity, thrust projection through pitch,
differential-thrust torque).

basis must accept a single argument xu, a length-8 array: xu[0:6] is the state
in the order above and xu[6], xu[7] are u_1 and u_2. Return between 10 and 30
scalar features (a Python list or numpy array) so that a downstream STLSQ
sparse regression can reconstruct all six derivative equations, including the
sin/cos terms coupling thrust and pitch. numpy is preinjected as np; math is
allowed; import nothing else and never index beyond xu[7].

Reply with the function definition only, placed in a single python code fence;
keep any notes as comments inside the body.
""",
    """\
I need a basis-function library for sparse identification (SINDy + STLSQ) of a
2D planar quadrotor. State vector x = [p_x, p_z, theta, v_x, v_z, omega]
(6 entries); control u = [u_1, u_2] (two rotor thrusts, Newtons). Constants
are m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81 (SI units). The dynamics are
withheld on purpose -- lean on your own understanding of how rotor thrust,
pitch angle, and differential thrust enter the equations.

Write one Python function, name it basis, taking exactly one positional
argument xu of length 8 (state in xu[0]..xu[5]; controls u_1, u_2 in xu[6],
xu[7]). It returns a list/array of roughly 10 to 30 nonlinear features rich
enough for the regression to recover each state derivative, the
thrust-projection sin/cos couplings included. You may use numpy (as np) and
math; no other imports; do not read past index 7.

Output only that function inside a single python code fence, nothing before or
after it.
""",
    """\
Context: data-driven identification of a Planar Quadrotor (Brunton & Kutz
style). The 6D state is [p_x, p_z, theta, v_x, v_z, omega]; the 2D control is
[u_1, u_2], left/right rotor thrusts in Newtons. Physical constants:
m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2. The governing
equations are deliberately not provided.

Please write a Python function named basis with these properties:
- one argument named xu, a length-8 vector; xu[0..5] = state, xu[6..7] = u_1, u_2;
- returns 10-30 scalar features as a list or numpy array;
- chosen using your quadrotor knowledge (gravity, pitch-dependent thrust,
  torque from thrust difference) so a SINDy/STLSQ fit can recover all six
  derivatives including sin/cos thrust terms;
- uses only numpy (np) and math, no other imports, and never reads beyond xu[7].

Return the function definition alone, wrapped in a single python code fence.
""",
    """\
You are assisting with system identification of a planar quadrotor (a 2D
quadcopter). Its state is x = [p_x, p_z, theta, v_x, v_z, omega] and its
control is u = [u_1, u_2], the two rotor thrusts in Newtons (each in
[0, 2*m*g]; hover thrust is m*g/2 per rotor). Take the constants as m = 1.0,
I_yy = 0.01, L = 0.25, g = 9.81. The equations of motion are not disclosed;
rely on your physical intuition about thrust, gravity, pitch projection, and
differential-thrust torque.

Deliverable: a Python function basis(xu) where xu is a length-8 array holding
the state in the first six slots and the controls u_1, u_2 in xu[6] and xu[7].
It must return a list (or numpy array) of about 10-30 candidate features for a
STLSQ sparse regression that will identify the six state derivatives,
including the trigonometric thrust/pitch couplings. numpy is available as np
and math may be used; import nothing further and do not access indices past 7.

Provide only the function, enclosed in a single python code fence.
""",
]

# --- E2: MPC stage cost (function name: stage_cost(x, u)) ---
MPC_COST_PARAPHRASES: list[str] = [
    """\
You are designing the per-step cost an MPC controller minimizes to drive a
Planar Quadrotor from a perturbed state back to hover at the origin. State
x = [p_x, p_z, theta, v_x, v_z, omega] (indices 0..5); control u = [u_1, u_2]
(indices 0,1), the left/right rotor thrusts in Newtons. Constants: m = 1.0 kg,
I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2. The hover equilibrium is
x = 0, u_1 = u_2 = m*g/2 ~= 4.905 N.

Write a Python function named stage_cost taking exactly two arguments x and u
(in that order) and returning a single scalar (smaller = better, ~0 at hover).
The function is compiled symbolically by a CasADi MPC, so you may NOT import
anything and numpy is NOT available; use only the pre-injected bare names
sin, cos, exp, sqrt, log, fabs (use fabs, never the abs builtin) plus the
operators + - * / ** and parentheses. Penalize position more than velocity,
penalize attitude but not so hard the quadrotor cannot tilt, and penalize
control effort lightly.

Return ONLY the function definition inside a single python code fence, no prose
outside the fence.
""",
    """\
Design the running cost for a model-predictive controller that returns a
Planar Quadrotor to hover. State x = [p_x, p_z, theta, v_x, v_z, omega]
(x[0]..x[5]); control u = [u_1, u_2] (u[0], u[1]), the two rotor thrusts in
Newtons. Physical constants: m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m,
g = 9.81 m/s^2. Hover sits at x = 0 with u_1 = u_2 = m*g/2 ~= 4.905 N.

Write a function named stage_cost taking exactly two positional arguments x and
u (that order) and returning a single scalar that is small (near zero) at hover
and grows as the state or control drifts. The expression is differentiated
symbolically by CasADi, so do NOT import anything and do NOT use numpy; only
the pre-bound names sin, cos, exp, sqrt, log, fabs are available (use fabs in
place of the abs builtin), together with + - * / ** and parentheses. Penalize
position errors more than velocities, penalize tilt but leave room to tilt, and
keep the control penalty light.

Return just the function definition in a single python code fence, with no
surrounding prose.
""",
    """\
Goal: an MPC stage cost for hover stabilization of a 2D quadrotor. The state is
[p_x, p_z, theta, v_x, v_z, omega], indexed x[0] through x[5]; the control is
[u_1, u_2], indexed u[0], u[1] (rotor thrusts in Newtons). Use constants
m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81. The equilibrium is the origin with
both thrusts at m*g/2 (about 4.905 N).

Provide a Python function called stage_cost(x, u) returning one scalar cost
(lower is better, ~0 at hover). Because the cost is compiled symbolically
inside a CasADi MPC, imports are forbidden and numpy is unavailable; rely
solely on the injected math names sin, cos, exp, sqrt, log, fabs (never the
builtin abs) plus ordinary arithmetic operators. Weight position and attitude
deviations above velocity terms, allow the vehicle enough tilt authority to
translate, and penalize control effort only gently.

Respond with the function definition only, inside a single python code fence.
""",
    """\
Write the per-step penalty an MPC minimizes to bring a Planar Quadrotor back to
hover. The state vector is x = [p_x, p_z, theta, v_x, v_z, omega] (six entries,
x[0..5]); the input is u = [u_1, u_2] (two rotor thrusts, u[0], u[1], in
Newtons). Constants: m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81. At hover the
state is zero and each rotor holds m*g/2 ~= 4.905 N.

Requirements for the function, which must be named stage_cost and take x then u:
- it returns a single scalar (not a tuple/list), minimal near hover;
- no imports and no numpy -- the MPC evaluates it symbolically with CasADi, so
  use only the bare helpers sin, cos, exp, sqrt, log, fabs (use fabs, never
  abs) and the operators + - * / **;
- penalize position/attitude error more than velocity, keep tilt feasible, and
  tax control lightly.

Return the definition by itself in a single python code fence.
""",
    """\
I want a hover-stabilizing stage cost for a planar-quadrotor MPC. Coordinates:
state x = [p_x, p_z, theta, v_x, v_z, omega] addressed as x[0]..x[5]; control
u = [u_1, u_2] addressed as u[0], u[1] (each a rotor thrust in Newtons).
Physical parameters are m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m,
g = 9.81 m/s^2, with the hover equilibrium at the origin and u_1 = u_2 = 4.905 N.

Produce a single Python function stage_cost(x, u) -> scalar, smallest at hover
and increasing with deviation. It will be turned into a symbolic CasADi
expression, so it must not import anything and must avoid numpy; the only
callable names available are sin, cos, exp, sqrt, log, and fabs (use fabs
rather than the abs builtin), alongside the usual arithmetic. Make position and
pitch errors dominate over velocity errors, do not over-penalize tilting, and
apply a small control-effort term.

Give only the function, wrapped in a single python code fence.
""",
]

# --- E3: Koopman/EDMDc observable (function name: observables(x), state-only) ---
KOOPMAN_PARAPHRASES: list[str] = [
    """\
Help identify a Planar Quadrotor's dynamics with EDMDc (Extended Dynamic Mode
Decomposition with control), which fits a linear lifted model
psi(x_{k+1}) = A psi(x_k) + B u_k by least squares. Because control enters
linearly through B, your observable must depend on the STATE ONLY:
x = [p_x, p_z, theta, v_x, v_z, omega] in R^6. Constants: m = 1.0 kg,
I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2. Equations of motion are not
given; use your knowledge of quadrotor physics.

Write a Python function named observables taking exactly one argument x (a
length-6 state, indexable x[0]..x[5]; never read x[6] or beyond) and returning
a length-N vector (list or numpy array) with 6 <= N <= 50, whose FIRST six
components equal x[0..5] in order (state-recovery convention). You may import
numpy (as np) and math; no other imports, no I/O, no randomness. Choose the
smallest N you believe gives a tight one-step prediction fit.

Return ONLY the function definition inside a single python code fence; show no
example outputs or test code.
""",
    """\
Build a lifting/observable function for EDMDc (Extended Dynamic Mode
Decomposition with control) identification of a Planar Quadrotor. EDMDc fits
psi(x_{k+1}) = A psi(x_k) + B u_k by least squares, and since control enters
only through B, the observable is a function of STATE ALONE:
x = [p_x, p_z, theta, v_x, v_z, omega] in R^6. Constants: m = 1.0, I_yy = 0.01,
L = 0.25, g = 9.81. The equations of motion are not given -- use your quadrotor
physics knowledge.

Name the function observables; it takes exactly one argument x (length 6,
x[0]..x[5]; never read x[6] or beyond) and returns a length-N vector (list or
numpy array) with 6 <= N <= 50 whose first six entries are exactly x[0]..x[5]
in order (so the state is recoverable by slicing [:6]). numpy (np) and math may
be imported inside the body; nothing else; the function must be pure and
deterministic. Prefer the smallest N that fits the discrete-time dynamics
tightly.

Return only the function definition inside a single python code fence; include
no example output.
""",
    """\
Task: choose Koopman observables for a 2D quadrotor so that an EDMDc regression
psi(x_{k+1}) = A psi(x_k) + B u_k fits well. Control is handled separately by
B, therefore your observable depends only on the state
x = [p_x, p_z, theta, v_x, v_z, omega] (R^6). Treat the constants as m = 1.0,
I_yy = 0.01, L = 0.25, g = 9.81; the dynamics themselves are withheld.

Write a Python function observables(x): x has length 6 (indices 0..5) and you
must not access anything past x[5]. Return a vector (list or numpy array) of
length N with 6 <= N <= 50 whose leading six components equal the input state
in order -- this lets the pipeline recover x via [:6]. You may import numpy
(as np) and math inside the function; no other imports, no I/O, no randomness.
Keep N as small as you can while still capturing the dynamics.

Reply with the function alone in a single python code fence; do not show sample
values or tests.
""",
    """\
Help me pick a state-only observable map for EDMDc system identification of a
planar quadrotor. The lifted least-squares model is
psi(x_{k+1}) = A psi(x_k) + B u_k; because B absorbs the control, your function
sees only the state x = [p_x, p_z, theta, v_x, v_z, omega] in R^6. Physical
constants: m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81. No equations of motion are
provided.

Define observables(x) with exactly one argument x of length 6 (x[0]..x[5]; do
not read x[6]+). It returns a list or numpy array of length N, 6 <= N <= 50,
and its first six elements must be x[0] through x[5] in that order
(state-recovery convention for the [:6] projection). Imports limited to numpy
(np) and math; the function is deterministic and side-effect-free. Choose the
smallest N that gives a tight one-step prediction.

Return just the function definition inside a single python code fence -- no
example outputs.
""",
    """\
For a Planar Quadrotor I am running EDMDc (Extended DMD with control), which
solves psi(x_{k+1}) = A psi(x_k) + B u_k by ordinary least squares. The control
acts through B, so the observable function must use the STATE only:
x = [p_x, p_z, theta, v_x, v_z, omega], six entries. Constants: m = 1.0 kg,
I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2; the motion equations are not
shared, so apply your own quadrotor physics intuition.

Write observables(x) taking one length-6 argument (x[0]..x[5], with no access
beyond x[5]) and returning a length-N vector (list or numpy array),
6 <= N <= 50, whose first six components are the state itself in order --
required so the downstream code recovers x by taking [:6]. You may use numpy
(np) and math; import nothing else; keep it pure. Use the fewest observables
that still model the dynamics accurately.

Output only the function, enclosed in a single python code fence, with no demo
values.
""",
]

# --- E4: Eureka PPO reward (function name: reward(state, action)) ---
EUREKA_PARAPHRASES: list[str] = [
    """\
Design a dense per-step reward for a Planar Quadrotor hover task that a PPO
agent (Proximal Policy Optimization, as in the Eureka paper, Ma et al. 2024)
will MAXIMIZE. state = [p_x, p_z, theta, v_x, v_z, omega] in R^6,
action = [u_1, u_2] in R^2 (rotor thrusts, Newtons). Constants: m = 1.0 kg,
I_yy = 0.01 kg*m^2, L = 0.25 m, g = 9.81 m/s^2; hover is state = 0 with
u_1 = u_2 ~= 4.905 N. The agent starts disturbed and must return to hover.

Write a Python function named reward taking exactly two arguments state and
action (in that order) and returning one finite float (higher = better;
penalties are negative terms). It must be pure and deterministic; you may
import numpy (as np) and math, nothing else. Crash penalties, a survival
bonus, and episode termination are handled by the harness -- design only the
dense shaping. Weight position and attitude error above velocity error, and
penalize control deviation from hover thrust; prefer smooth differentiable
shapes over step functions.

Return ONLY the function definition inside a single python code fence, no prose
outside it.
""",
    """\
Create a dense per-step reward for PPO-based hover control of a Planar
Quadrotor (the Eureka-style LLM-reward setting, PPO from Schulman et al.). The
observation is state = [p_x, p_z, theta, v_x, v_z, omega] in R^6 and the action
is action = [u_1, u_2] in R^2 (rotor thrusts, Newtons). Constants: m = 1.0,
I_yy = 0.01, L = 0.25, g = 9.81; hover is state = 0 with both thrusts near
4.905 N. The agent starts disturbed and must recover to hover.

Write a function named reward taking exactly two arguments state then action
and returning ONE finite float, where larger is better (use negative terms for
penalties). It must be pure and deterministic; you may import numpy (np) and
math and nothing else. Crash penalties, survival bonus, and episode cutoff are
added by the harness -- you only craft the dense shaping. Make position and
attitude errors outweigh velocity errors, penalize departure from hover thrust,
and favor smooth differentiable terms over step functions.

Return the function definition alone in a single python code fence.
""",
    """\
I am training a PPO policy to hover a 2D quadrotor and need a shaped reward
written by you. Inputs: state = [p_x, p_z, theta, v_x, v_z, omega] (six values)
and action = [u_1, u_2] (two rotor thrusts in Newtons). Use constants m = 1.0,
I_yy = 0.01, L = 0.25, g = 9.81; the hover target is the origin with thrusts
about 4.905 N each.

Define reward(state, action) returning a single finite float that the agent
maximizes (penalties are negative contributions). Keep it pure and
deterministic; only numpy (np) and math may be imported. Do not handle crashes,
the survival bonus, or termination -- the training harness already does.
Concentrate the reward on position and attitude accuracy (above velocity),
discourage thrusts far from hover, and use smooth, differentiable shapes rather
than discontinuous indicators.

Respond with only the function definition wrapped in a single python code fence.
""",
    """\
Design the dense reward term for a reinforcement-learning hover task on a
Planar Quadrotor, to be maximized by PPO (the Eureka LLM-reward paradigm).
state = [p_x, p_z, theta, v_x, v_z, omega] in R^6; action = [u_1, u_2] in R^2
(Newtons). Physical constants: m = 1.0 kg, I_yy = 0.01 kg*m^2, L = 0.25 m,
g = 9.81 m/s^2. The equilibrium is state = 0 with u_1 = u_2 ~= 4.905 N;
episodes begin from a disturbance.

Your function must be named reward and accept state then action, returning a
single finite scalar (higher = better; encode penalties as negative terms). It
has to be deterministic and pure, may import only numpy (np) and math, and must
stay finite on any reasonable input. The harness supplies the crash penalty,
the per-step survival bonus, and termination, so design only the continuous
shaping: weight position and tilt errors above velocity, penalize control
deviation from hover, and keep everything smooth and differentiable.

Return the function definition by itself in a single python code fence.
""",
    """\
Please write the per-step shaping reward a PPO agent will maximize to hover a
planar quadrotor. The state passed in is [p_x, p_z, theta, v_x, v_z, omega] and
the action is [u_1, u_2] (rotor thrusts in Newtons). Take the physical
constants to be m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81, with hover at the
origin and roughly 4.905 N per rotor.

The function is named reward and takes state, then action; it returns one
finite float (bigger is better, so penalties are negative). Keep it
deterministic and side-effect free; the only allowed imports are numpy (as np)
and math. Leave crash penalties, the survival bonus, and episode termination to
the harness. Prioritize position and attitude error over velocity error, add a
penalty for straying from hover thrust, and prefer smooth differentiable
expressions instead of hard thresholds.

Return only the function, placed in a single python code fence.
""",
]

PARAPHRASES: dict[str, list[str]] = {
    "E1": SINDY_PARAPHRASES,
    "E2": MPC_COST_PARAPHRASES,
    "E3": KOOPMAN_PARAPHRASES,
    "E4": EUREKA_PARAPHRASES,
}
TASK_IDS = ["E1", "E2", "E3", "E4"]

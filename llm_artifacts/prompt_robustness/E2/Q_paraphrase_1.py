"""Q E2 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Goal:** Design a running cost function for a Model-Predictive Controller (MPC) for a Planar Quadrotor.
#     *   **Objective:** Return the quadrotor to hover.
#     *   **State:** $x = [p_x, p_z, \theta, v_x, v_z, \omega]$ (6 states).
#     *   **Control:** $u = [u_1, u_2]$ (2 controls, rotor thrusts in Newtons).
#     *   **Constants:** $m = 1.0$, $I_{yy} = 0.01$, $L = 0.25$, $g = 9.81$.
#     *   **Hover Condition:** $x = [0, 0, 0, 0, 0, 0]$, $u_1 = u_2 = m*g/2 \approx 4.905$.
#     *   **Function Signature:** `stage_cost(x, u)` taking exactly two positional arguments.
#     *   **Return Type:** Single scalar.
#     *   **Behavior:** Small (near zero) at hover, grows as state/control drifts.
#     *   **Differentiation:** Must be differentiable by CasADi.
#     *   **Constraints:**
#         *   NO imports (except pre-bound names).
#         *   NO numpy.
#         *   Allowed functions: `sin, cos, exp, sqrt, log, fabs`.
#         *   Allowed operators: `+ - * / ** (parentheses)`.
#         *   Use `fabs` instead of `abs`.
#         *   Penalize position errors more than velocities.
#         *   Penalize tilt but leave room to tilt.
#         *   Keep control penalty light.
#     *   **Output Format:** Just the function definition in a single python code fence, no surrounding prose.
# 
# 2.  **Determine the Cost Structure:**
#     *   Standard quadratic cost: $J = \sum (x - x_{ref})^T Q (x - x_{ref}) + (u - u_{ref})^T R (u - u_{ref})$.
#     *   Since CasADi differentiates symbolic expressions, I need to construct a Python expression that CasADi can parse. However, the prompt asks for a *function definition* in Python. Usually, when using CasADi, one might define a function using `casadi` or just a Python function that returns a value. The prompt says "The expression is differentiated symbolically by CasADi". This implies the *body* of the function should be a mathematical expression compatible with CasADi's symbolic engine, or the function itself is used within a CasADi context where `x` and `u` are CasADi variables.
#     *   Wait, the prompt says "Write a function named stage_cost...". It doesn't explicitly say `x` and `u` are CasADi variables, but it says "The expression is differentiated symbolically by CasADi". This usually means the code provided should be compatible with CasADi's `Function` or `Symbolic` machinery, or simply a Python function that computes the cost value. Given the constraint "do NOT import anything", I cannot import `casadi`.
#     *   However, if I write a standard Python function, CasADi can't differentiate it directly unless it's a CasADi function. But the prompt asks for a *Python function definition*.
#     *   Re-reading: "The expression is differentiated symbolically by CasADi, so do NOT import anything...". This suggests the *mathematical expression* inside the function should be written in a way that CasADi understands if it were passed as a string or if the variables were CasADi variables.
#     *   Actually, usually in these types of prompts (often from a specific benchmark or dataset like "Code Generation for Control"), the expectation is a Python function that computes the cost value, but the *syntax* must be compatible with CasADi's symbolic differentiation if the inputs were CasADi variables.
#     *   Crucially, the prompt says "do NOT import anything". If I write `import casadi`, it's forbidden. If I write `x = casadi.DM(...)`, it's forbidden.
#     *   So, I must write a Python function that takes `x` and `u` (likely CasADi variables or numpy arrays, but numpy is forbidden).
#     *   Wait, if `x` and `u` are not numpy arrays, how do I compute `x[0]`?
#     *   If `x` is a CasADi variable, I can't index it like `x[0]` in standard Python unless I use CasADi's indexing. But I can't import CasADi.
#     *   Let's look at the constraint: "only the pre-bound names sin, cos, exp, sqrt, log, fabs are available".
#     *   This implies `x` and `u` are likely CasADi variables (Symbolic variables) or the function is meant to be evaluated by CasADi later.
#     *   However, standard Python indexing `x[0]` works on lists/arrays. If `x` is a CasADi variable, `x[0]` might not work directly without `casadi` functions.
#     *   BUT, the prompt asks for a "function named stage_cost taking exactly two positional arguments x and u".
#     *   If I assume `x` and `u` are CasADi variables, I cannot use `x[0]` directly in standard Python without importing casadi.
#     *   However, often in these specific "no import" constraints for CasADi, the expectation is that the code is a string or a specific format, OR the user expects me to write the math using the variables provided.
#     *   Let's reconsider the "no numpy" constraint. If I can't use numpy, I can't do `x[0]` easily if `x` is a numpy array.
#     *   If `x` is a CasADi variable, accessing components usually requires `x[0]` in CasADi context.
#     *   Wait, if I cannot import `casadi`, I cannot use `casadi` functions.
#     *   Maybe the prompt implies I should write the expression using the variables `x` and `u` assuming they are accessible as vectors, but without numpy.
#     *   Actually, looking at similar tasks, often the "function" is expected to be a Python function that *would* be used with CasADi, but the implementation details must respect the "no import" rule.
#     *   If I write `x[0]`, it assumes `x` is a list/array. If `x` is a CasADi variable, `x[0]` is valid in CasADi.
#     *   Let's assume the inputs `x` and `u` are CasADi variables (since the prompt mentions symbolic differentiation). In CasADi, variables are objects that support indexing.
#     *   However, standard Python `x[0]` works on lists.
#     *   Let's look at the constraint: "do NOT import anything".
#     *   If I write `x[0]`, it works for lists.
#     *   If I write `x[0]`, it works for CasADi variables.
#     *   So `x[0]` is safe.
#     *   What about `sin(x[0])`? `sin` is pre-bound.
#     *   So the plan is: Write a Python function that computes the cost using the allowed operators and functions.
# 
# 3.  **Constructing the Cost:**
#     *   **State:** $x = [p_x, p_z, \theta, v_x, v_z, \omega]$.
#     *   **Control:** $u = [u_1, u_2]$.
#     *   **Hover:** $p_x=0, p_z=0, \theta=0, v_x=0, v_z=0, \omega=0$. $u_1 = u_2 = m*g/2$.
#     *   **Penalties:**
#         *   Position ($p_x, p_z$): High weight.
#         *   Velocity ($v_x, v_z$): Lower weight than position.
#         *   Tilt ($\theta$): Penalize, but "leave room to tilt". This implies a small weight or a specific scaling.
#         *   Control ($u_1, u_2$): "Keep the control penalty light".
#     *   **Weights:**
#         *   $Q_p$: Weight for position.
#         *   $Q_v$: Weight for velocity.
#         *   $Q_\theta$: Weight for tilt.
#         *   $R$: Weight for control.
#     *   **Values:**
#         *   $m = 1.0, g = 9.81 \implies u_{hover} = 4.905$.
#         *   Let's define weights.
#         *   Position: $100 \times (p_x^2 + p_z^2)$.
#         *   Velocity: $1 \times (v_x^2 + v_z^2)$.
#         *   Tilt: $0.1 \times \theta^2$ (small enough to allow tilt).
#         *   Control: $0.01 \times ((u_1 - u_{hover})^2 + (u_2 - u_{hover})^2)$.
#     *   **Expression:**
#         *   $u_{hover} = m * g / 2$.
#         *   $p_x = x[0], p_z = x[1]$.
#         *   $v_x = x[3], v_z = x[4]$.
#         *   $\theta = x[2]$.
#         *   $u_1 = u[0], u_2 = u[1]$.
#         *   Cost = $Q_p (p_x^2 + p_z^2) + Q_v (v_x^2 + v_z^2) + Q_\theta \theta^2 + R ((u_1 - u_{hover})^2 + (u_2 - u_{hover})^2)$.
#     *   **Constraint Check:** "Penalize position errors more than velocities". $Q_p > Q_v$.
#     *   "Penalize tilt but leave room to tilt". $Q_\theta$ should be small compared to $Q_p$.
#     *   "Keep the control penalty light". $R$ should be small.
#     *   "Do NOT import anything".
#     *   "Only the pre-bound names sin, cos, exp, sqrt, log, fabs are available".
#     *   "Use fabs in place of the abs builtin".
#     *   "Return just the function definition".
# 
# 4.  **Refining the Expression:**
#     *   Need to ensure the math is correct.
#     *   $u_{hover} = 1.0 * 9.81 / 2 = 4.905$.
#     *   $p_x = x[0]$, $p_z = x[1]$.
#     *   $v_x = x[3]$, $v_z = x[4]$.
#     *   $\theta = x[2]$.
#     *   $u_1 = u[0]$, $u_2 = u[1]$.
#     *   Cost = $100 * (x[0]**2 + x[1]**2) + 1 * (x[3]**2 + x[4]**2) + 0.1 * x[2]**2 + 0.01 * (fabs(u[0] - 4.905)**2 + fabs(u[1] - 4.905)**2)$.
#     *   Wait, `fabs` is for absolute value. Squaring removes sign, so `**2` is fine. But `fabs` is explicitly requested to be used instead of `abs`. I should probably use `fabs` on the differences to be safe, or just square them. The prompt says "use fabs in place of the abs builtin". It doesn't mandate using it everywhere, but it's good practice if I were to use abs. Since I'm squaring, `**2` is sufficient. However, to strictly follow "use fabs in place of the abs builtin" if I were to use abs, I should check if I need abs. Squaring is fine. But maybe I should use `fabs` on the control difference just to show compliance? Or maybe the prompt implies I should use `fabs` if I calculate magnitude.
#     *   Actually, `fabs` is often used for `abs`. Since I am squaring, `**2` is mathematically equivalent to `abs(x)**2`.
#     *   Let's check the constraint: "use fabs in place of the abs builtin". This is a hint for if I need absolute values.
#     *   Let's stick to `**2` for simplicity unless I need to take a root or log.
#     *   Wait, "The expression is differentiated symbolically by CasADi".
#     *   If I use `**2`, it's fine.
#     *   Let's check the weights again.
#     *   Position > Velocity.
#     *   Tilt penalty: "leave room to tilt". This means it shouldn't be too high.
#     *   Control penalty: "light".
#     *   Let's pick weights:
#         *   $Q_p = 100$
#         *   $Q_v = 1$
#         *   $Q_\theta = 0.1$
#         *   $R = 0.01$
#     *   Wait, $u$ is in Newtons. $u_{hover} \approx 5$.
#     *   If $u$ deviates by 1N, cost is $0.01 * 1^2 = 0.01$.
#     *   If $p$ deviates by 1m, cost is $100 * 1^2 = 100$.
#     *   This seems reasonable.
#     *   Wait, I need to make sure I don't use `abs`. I will use `**2`.
#     *   Wait, the prompt says "use fabs in place of the abs builtin". It doesn't say I *must* use it, but I should if I use abs. Since I'm squaring, I don't need abs.
#     *   However, to be safe and compliant with the spirit of "use fabs", I can use `fabs` on the control difference if I want, but `**2` is cleaner.
#     *   Wait, is there any reason to use `fabs`? Maybe if I used `log` or `sqrt`? No, cost is usually quadratic.
#     *   Let's check the variable access. `x[0]` etc.
#     *   Is `x` a list or a CasADi variable?
#     *   If `x` is a CasADi variable, `x[0]` works.
#     *   If `x` is a list, `x[0]` works.
#     *   So `x[0]` is safe.
#     *   Wait, I need to define the constants inside the function or outside?
#     *   "Write a function named stage_cost...".
#     *   I can define constants inside the function.
#     *   "do NOT import anything".
#     *   So I cannot define `m, g` outside if they are not pre-bound.
#     *   The prompt says "Physical constants: m = 1.0 kg...". It lists them. It doesn't say they are pre-bound in the environment.
#     *   So I must define them inside the function or use the values directly.
#     *   Better to define them inside to be self-contained.
#     *   Wait, "only the pre-bound names sin, cos, exp, sqrt, log, fabs are available".
#     *   This implies I cannot define `m` or `g` as variables if they are not pre-bound.
#     *   So I must hardcode the values `1.0`, `9.81`, `0.01`, `0.25`.
#     *   Wait, `I_yy` and `L` are given but not used in the cost function (unless I need to calculate something else, but the cost is just state/control).
#     *   So I just need `m` and `g`.
#     *   $u_{hover} = m * g / 2$.
#     *   So inside the function: `u_hover = 1.0 * 9.81 / 2`.
#     *   Wait, is `1.0` allowed? Yes, literals are allowed.
#     *   So:
#         ```python
#         def stage_cost(x, u):
#             m = 1.0
#             g = 9.81
#             u_hover = m * g / 2
#             # ...
#         ```
#     *   Wait, the prompt says "do NOT import anything". It doesn't forbid defining variables.
#     *   However, to be most compliant with "expression is differentiated symbolically by CasADi", usually the function body should be a single expression or a return statement with an expression.
#     *   I will write the function body to return the cost expression.
# 
# 5.  **Refining the Cost Expression:**
#     *   $p_x = x[0]$
#     *   $p_z = x[1]$
#     *   $v_x = x[3]$
#     *   $v_z = x[4]$
#     *   $\theta = x[2]$
#     *   $u_1 = u[0]$
#     *   $u_2 = u[1]$
#     *   Cost = $100 * (x[0]**2 + x[1]**2) + 1 * (x[3]**2 + x[4]**2) + 0.1 * x[2]**2 + 0
# --- extracted code ---
def stage_cost(x, u):
            m = 1.0
            g = 9.81
            u_hover = m * g / 2
            # ...

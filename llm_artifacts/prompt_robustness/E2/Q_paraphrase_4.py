"""Q E2 paraphrase 4. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Goal:** Create a stage cost function for a planar quadrotor MPC.
#     *   **State:** `x = [p_x, p_z, theta, v_x, v_z, omega]` (indices 0 to 5).
#     *   **Control:** `u = [u_1, u_2]` (indices 0, 1) (rotor thrusts).
#     *   **Parameters:** `m = 1.0`, `I_yy = 0.01`, `L = 0.25`, `g = 9.81`.
#     *   **Equilibrium:** Hover at origin (`p_x=0, p_z=0, theta=0`), `u_1 = u_2 = 4.905`.
#     *   **Output:** A single Python function `stage_cost(x, u) -> scalar`.
#     *   **Constraints:**
#         *   Smallest at hover.
#         *   Increasing with deviation.
#         *   Must be compatible with symbolic CasADi (no numpy imports, no standard library functions like `abs`, only `sin, cos, exp, sqrt, log, fabs` and arithmetic).
#         *   Position and pitch errors dominate over velocity errors.
#         *   Do not over-penalize tilting (pitch).
#         *   Apply a small control-effort term.
#         *   Output format: Single Python code fence.
# 
# 2.  **Determine the Cost Structure:**
#     *   Standard MPC stage cost usually looks like: $J = \frac{1}{2} (x - x_{ref})^T Q (x - x_{ref}) + \frac{1}{2} (u - u_{ref})^T R (u - u_{ref})$.
#     *   Since it's symbolic CasADi, I need to use `fabs` for absolute values if needed, but quadratic terms are usually preferred for smoothness. However, the prompt says "smallest at hover and increasing with deviation". Quadratic is standard.
#     *   Wait, the prompt says "It will be turned into a symbolic CasADi expression". CasADi supports `abs` (via `fabs` in the prompt's allowed list) and arithmetic.
#     *   The prompt explicitly lists allowed callable names: `sin, cos, exp, sqrt, log, fabs`.
#     *   I need to define the parameters inside the function or pass them? The prompt says "Produce a single Python function". Usually, parameters are hardcoded or passed. Given the constraint "no numpy", I should hardcode the parameters to keep it self-contained and simple, or define them as local variables.
#     *   Cost components:
#         1.  Position error: $(p_x)^2 + (p_z)^2$.
#         2.  Pitch error: $\theta^2$ (or similar).
#         3.  Velocity error: $(v_x)^2 + (v_z)^2$.
#         4.  Control effort: $(u_1 - u_{eq})^2 + (u_2 - u_{eq})^2$.
#     *   Weights: Position/Pitch > Velocity. Control effort small.
#     *   Equilibrium values: $p_{ref} = [0, 0]$, $\theta_{ref} = 0$, $v_{ref} = [0, 0]$, $u_{ref} = [4.905, 4.905]$.
#     *   Note on `fabs`: The prompt says "use fabs rather than the abs builtin". This implies I should use `fabs` for any absolute value operations, though quadratic costs don't strictly need absolute values. However, to be safe and compliant with the "callable names available" constraint, I should ensure I don't use `abs`.
#     *   Note on `sqrt`: Often used for norms, but quadratic is smoother. Let's stick to quadratic for stability unless specified otherwise.
#     *   Constraint Check: "Position and pitch errors dominate over velocity errors". This means weights for position/pitch should be higher than weights for velocity.
#     *   Constraint Check: "Do not over-penalize tilting". This means the weight for $\theta$ shouldn't be excessively high compared to position, or position should be the main driver.
#     *   Constraint Check: "Apply a small control-effort term".
# 
# 3.  **Drafting the Function:**
# 
#     ```python
#     def stage_cost(x, u):
#         # Parameters
#         m = 1.0
#         I_yy = 0.01
#         L = 0.25
#         g = 9.81
#         u_eq = 4.905
# 
#         # State indices: p_x, p_z, theta, v_x, v_z, omega
#         px = x[0]
#         pz = x[1]
#         theta = x[2]
#         vx = x[3]
#         vz = x[4]
#         omega = x[5]
# 
#         # Control indices: u_1, u_2
#         u1 = u[0]
#         u2 = u[1]
# 
#         # Weights
#         # Position/Pitch dominant
#         Q_pos = 100.0
#         Q_pitch = 10.0
#         Q_vel = 1.0
#         R_ctrl = 0.01
# 
#         # Errors
#         err_pos = px**2 + pz**2
#         err_pitch = theta**2
#         err_vel = vx**2 + vz**2
#         err_ctrl = (u1 - u_eq)**2 + (u2 - u_eq)**2
# 
#         # Cost
#         cost = Q_pos * err_pos + Q_pitch * err_pitch + Q_vel * err_vel + R_ctrl * err_ctrl
#         return cost
#     ```
# 
# 4.  **Refining for CasADi Compatibility:**
#     *   CasADi expects the function to be callable.
#     *   The prompt says "It will be turned into a symbolic CasADi expression". This usually means the function body should be compatible with `casadi.DM` or `casadi.SX`.
#     *   The prompt explicitly forbids `numpy`.
#     *   The prompt explicitly lists allowed callables: `sin, cos, exp, sqrt, log, fabs`.
#     *   It says "avoid numpy".
#     *   It says "use fabs rather than the abs builtin".
#     *   I need to make sure I don't use `abs`.
#     *   I need to make sure I don't use `**` if it's not standard arithmetic? `**` is standard arithmetic in Python. CasADi usually supports `**` or `pow`. However, `x**2` is fine in Python.
#     *   Wait, the prompt says "The only callable names available are sin, cos, exp, sqrt, log, and fabs". This is a bit ambiguous. Does it mean *only* these functions are available in the environment, or just that I should use these if I need them? Usually, in these constraints, it means "Do not use anything else". Standard arithmetic operators (`+`, `-`, `*`, `/`, `**`) are usually assumed available.
#     *   However, to be extremely safe regarding the "callable names available" constraint, I should avoid `abs` (use `fabs` if needed, though quadratic doesn't need it).
#     *   I should check if `**` is allowed. It's an operator, not a callable.
#     *   I should check if `pow` is allowed.
#     *   Let's stick to standard arithmetic operators.
#     *   One detail: "smallest at hover".
#     *   One detail: "Position and pitch errors dominate over velocity errors".
#     *   One detail: "Do not over-penalize tilting".
#     *   One detail: "Apply a small control-effort term".
# 
#     Let's adjust weights to reflect the requirements.
#     $Q_{pos} = 100$
#     $Q_{pitch} = 10$ (Dominant but less than pos)
#     $Q_{vel} = 1$ (Smallest)
#     $R_{ctrl} = 0.01$ (Small)
# 
#     Wait, "Position and pitch errors dominate over velocity errors".
#     So $Q_{pos} > Q_{vel}$ and $Q_{pitch} > Q_{vel}$.
#     "Do not over-penalize tilting". This implies $Q_{pitch}$ shouldn't be huge compared to $Q_{pos}$, or at least not huge compared to $Q_{vel}$? Usually, it means don't make the cost explode just because the quadrotor tilts slightly. So $Q_{pitch}$ should be reasonable.
#     "Apply a small control-effort term". $R_{ctrl}$ should be small.
# 
#     Let's refine the weights.
#     $Q_{pos} = 1000$
#     $Q_{pitch} = 100$
#     $Q_{vel} = 10$
#     $R_{ctrl} = 0.1$
# 
#     Wait, I need to ensure the function signature matches `stage_cost(x, u)`.
#     I need to ensure no imports.
#     I need to ensure no `abs`.
#     I need to ensure no `numpy`.
# 
#     One specific constraint: "The only callable names available are sin, cos, exp, sqrt, log, and fabs".
#     This suggests I should not use `abs`. I should use `fabs` if I need absolute values. Since I am using quadratic costs ($x^2$), I don't strictly need absolute values. However, if I were to use norms like `sqrt(x*x)`, I would need `sqrt`.
#     Let's stick to quadratic forms to avoid needing `fabs` or `sqrt` unless necessary, but `sqrt` is allowed.
#     Actually, `x**2` is standard arithmetic.
#     Is `**` allowed? It's an operator.
#     Is `pow` allowed? It's a function.
#     To be safe, I will use `*` for squaring.
# 
#     Wait, I need to check if the prompt implies I *must* use `fabs` somewhere? "use fabs rather than the abs builtin". This implies I might need absolute values. But quadratic cost doesn't need them. I will just avoid `abs`.
# 
#     Wait, one more thing: "It will be turned into a symbolic CasADi expression".
#     CasADi functions like `sin`, `cos` etc. are usually imported from `casadi`. But the prompt says "The only callable names available are...". This implies I am writing the Python code that *will be* passed to CasADi or used within a context where only these are available.
#     Actually, usually, when people ask for "CasADi expression", they want a Python function that CasADi can wrap.
#     If I write `def stage_cost(x, u): ...`, CasADi can wrap this.
#     The constraint "The only callable names available are sin, cos, exp, sqrt, log, and fabs" likely refers to the math functions I can use inside the function body. I cannot use `math.sin` or `numpy.sin`. I must use `sin` directly.
#     Wait, `sin` is not a built-in in Python. It's in `math`.
#     If the prompt says "The only callable names available are sin, cos...", it implies I should treat them as if they are available in the global scope or I should just use them as if they are.
#     However, standard Python doesn't have `sin`.
#     If I write `sin(x)`, it will fail in standard Python unless `sin` is defined.
#     But the prompt says "It will be turned into a symbolic CasADi expression".
#     This usually means the function body is evaluated by CasADi's `Symbolic` engine, or the function is passed to `casadi.Function`.
#     If I am writing a Python function to be used *by* CasADi, I should use standard Python math functions if CasADi is imported.
#     BUT, the prompt says "avoid numpy; the only callable names available are sin, cos, exp, sqrt, log, and fabs".
#     This sounds like a constraint on the *content* of the function, assuming a specific environment (like a CasADi wrapper where these are exposed).
#     However, if I write `sin(x)`, and the user runs this in standard Python, it fails.
#     If the user intends to use this inside a CasADi context where `sin` is available (e.g. `casadi.sin`), then `sin` is the name.
#     Given the instruction "Make position and pitch errors dominate...", I should focus on the weights.
#     Given the instruction "It will be turned into a symbolic CasADi expression", I should write code that looks like a CasADi expression.
#     Actually, the most robust interpretation for "CasADi expression" in Python is using `casadi` functions. But I cannot import `casadi`.
#     So I must write a function that uses the names provided.
#     Wait, if I write `sin(x)`, and the environment doesn't have `sin`, it crashes.
#     But the prompt says "The only callable names available are...". This is a constraint on *my* code generation. I must assume these names exist.
#     So I will use `sin`, `cos`, etc. directly.
#     I will not use `abs`. I will use `fabs` if needed (though I won't need it for quadratic).
#     I will not use `math` or `numpy`.
# 
#     Let's check the weights again.
#     "Position and pitch errors dominate over velocity errors".
#     $Q_{pos} \gg Q_{vel}$, $Q_{pitch} \gg Q_{vel}$.
#     "Do not over-penalize tilting".
#     $Q_{pitch}$ should not be excessively large compared to $Q_{pos}$ or $Q_{vel}$? Usually means don't make it so high that the controller fights the tilt too hard.
#     "Apply a small control-effort term".
#     $R_{ctrl}$ small.
# 
#     Let's pick weights:
#     $Q_{pos} = 1000$
#     $Q_{pitch} = 100$
#     $Q_{vel} = 10$
#     $R_{ctrl} = 0.1$
# 
#     Wait, I need to check if `**` is allowed. It's an operator.
#     I'll use `*` for multiplication.
#     I'll use `+` for addition.
#     I'll use `-` for subtraction.
#     I'll use `/` for division if needed (not needed for quadratic).
# 
#     Wait, one detail: "smallest at hover".
#     At hover: $x = [0, 0, 0, 0, 0, 0]$, $u = [4.905, 4.905]$.
#     Cost = 0.
#     This satisfies "smallest at hover".
# 
#     Wait, "Coordinates: state x = [p_x, p_z, theta, v_x, v_z, omega] addressed as x[0]..x[5]".
#     "control u = [u_1, u_2] addressed as u[0], u[1]".
#     "hover equilibrium at the origin and u_1 = u_2 = 4.905 N".
#     So $p_{ref} = 0, \theta_{ref} = 0, v_{ref} = 0$.
#     $u_{ref} = 4.905$.
# 
#     Wait, I need to check if I need to use `fabs` for the control error?
#     $(u_1 - u_{eq})^2$ is always positive. No need for `fabs`.
#     However, the prompt says "use fabs rather than the abs builtin". This is a hint that I *might* need absolute values, or it's a general instruction for the environment. I will follow the instruction to avoid `abs` and use `fabs` if I encounter a situation where I need absolute value. Since I'm using squares, I don't need it. But I will ensure I don't use `abs`.
# 
#     Wait, I need to check if I need to use `sqrt`?
#     No, quadratic is fine.
#     I need to check if I need to use `exp`?
#     No, quadratic is fine.
# 
#     Wait, I need to check if I need to use `sin` or `cos`?
#     No, linear/quadratic cost.
# 
#     Wait, I need to check if I need to use `log`?
#     No.
# 
#     So the function will only use arithmetic and the allowed names (which I won't use except to avoid `abs`).
#     Wait, if I don't use `sin`, `cos`, etc., does it matter?
#     The constraint says "The only callable names available are...". It doesn't say "You must use all of them". It says "available".
#     So I can use `+`, `-`, `*`, `/`, `**` (if allowed), `fabs` (if needed).
#     I will use `**` for squaring as it's standard arithmetic.
#     Wait, `**` is an operator.
#     I'll use `*` to be safe. `x * x`.
# 
#     Wait, I need to check if I need to define the parameters inside the function.
#     Yes, to make it a single function.
#     I should define them as constants.
# 
#     Wait, one more check: "It will be turned into a symbolic CasADi expression".
#     If I write `def stage_cost(x, u): ...`, CasADi can wrap this.
#     However, if I use `**`, CasADi might not support it directly in the symbolic expression if it's not `pow`.
#     Actually, `**` is supported in CasADi `SX`/`DM`.
#     But to be safe, `x * x` is definitely supported.
# 
#     Wait, I need to check if I
# --- extracted code ---
def stage_cost(x, u):
        # Parameters
        m = 1.0
        I_yy = 0.01
        L = 0.25
        g = 9.81
        u_eq = 4.905

        # State indices: p_x, p_z, theta, v_x, v_z, omega
        px = x[0]
        pz = x[1]
        theta = x[2]
        vx = x[3]
        vz = x[4]
        omega = x[5]

        # Control indices: u_1, u_2
        u1 = u[0]
        u2 = u[1]

        # Weights
        # Position/Pitch dominant
        Q_pos = 100.0
        Q_pitch = 10.0
        Q_vel = 1.0
        R_ctrl = 0.01

        # Errors
        err_pos = px**2 + pz**2
        err_pitch = theta**2
        err_vel = vx**2 + vz**2
        err_ctrl = (u1 - u_eq)**2 + (u2 - u_eq)**2

        # Cost
        cost = Q_pos * err_pos + Q_pitch * err_pitch + Q_vel * err_vel + R_ctrl * err_ctrl
        return cost

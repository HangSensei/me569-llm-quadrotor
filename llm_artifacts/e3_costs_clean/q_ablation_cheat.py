"""Q-condition prompt ablation variant 'cheat'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Role:** Control Engineer / Assistant.
#     *   **Task:** Write a Python function `stage_cost(x, u)` for an MPC controller for a Planar Quadrotor.
#     *   **Context:** SINDy dynamics identification (though the cost function itself is for MPC, the prompt mentions the dynamics hint at the end). The prompt explicitly gives the dynamics equations at the end ("Hint: the ground-truth continuous-time dynamics are...").
#     *   **State Vector `x`:** `[p_x, p_z, theta, v_x, v_z, omega]` (6 elements).
#     *   **Control Vector `u`:** `[u_1, u_2]` (2 elements).
#     *   **Constraints:**
#         *   Function name: `stage_cost`.
#         *   Arguments: `x`, `u`.
#         *   Return: Single scalar.
#         *   **No imports** (including `numpy`).
#         *   **Allowed helpers:** `sin`, `cos`, `exp`, `sqrt`, `log`, `fabs` (bare names).
#         *   **Allowed operators:** `+`, `-`, `*`, `/`, `**`, `()`, literals.
#         *   **Forbidden:** `np.array`, `math.sqrt`, `abs`, etc.
#         *   **Goal:** Minimize deviation from hover equilibrium (`p_x=0, p_z=0, theta=0, v_x=0, v_z=0, omega=0`, `u_1=u_2=m*g/2`).
#     *   **Physics:** Planar Quadrotor.
#         *   `m = 1.0 kg`
#         *   `I_yy = 0.01 kg*m^2`
#         *   `L = 0.25 m`
#         *   `g = 9.81 m/s^2`
#         *   `u_max = 2*m*g`
#         *   `u_hover = m*g/2`
#     *   **Dynamics (provided in hint):**
#         *   `p_x_dot = v_x`
#         *   `p_z_dot = v_z`
#         *   `theta_dot = omega`
#         *   `v_x_dot = -(u_1 + u_2) * sin(theta) / m`
#         *   `v_z_dot = (u_1 + u_2) * cos(theta) / m - g`
#         *   `omega_dot = (u_2 - u_1) * L / I_yy`
#     *   **MPC Cost Design:**
#         *   Penalize position error (`p_x`, `p_z`).
#         *   Penalize velocity error (`v_x`, `v_z`).
#         *   Penalize attitude error (`theta`).
#         *   Penalize control effort (deviation from hover thrust).
#         *   Weights need to be chosen to balance these.
#         *   Must be compatible with CasADi symbolic evaluation (no numpy, no forbidden functions).
# 
# 2.  **Determine Cost Function Structure:**
#     *   Standard quadratic cost: $J = \sum (x - x_{ref})^T Q (x - x_{ref}) + (u - u_{ref})^T R (u - u_{ref})$.
#     *   Reference state: `x_ref = [0, 0, 0, 0, 0, 0]`.
#     *   Reference control: `u_ref = [u_hover, u_hover]`.
#     *   `u_hover = m * g / 2`.
#     *   `m = 1.0`, `g = 9.81` -> `u_hover = 4.905`.
#     *   Since `u` is constrained to `[0, u_max]`, and `u_max = 2*m*g = 19.62`, `u_hover` is well within bounds.
#     *   Need to handle `fabs` for absolute values if needed, but quadratic terms `x**2` are sufficient for MPC cost (smooth, differentiable).
#     *   Need to handle `sin` and `cos` if the cost depends on orientation (e.g., if we want to penalize tilt specifically relative to gravity, but usually just `theta**2` is fine).
#     *   The prompt asks to "Design a cost that penalizes deviation from hover, with awareness of these dynamics."
#     *   Since the dynamics involve `sin(theta)` and `cos(theta)` for forces, the cost should ideally be simple quadratic forms on the state and control deviations to ensure convexity and smoothness for the MPC solver.
#     *   However, the prompt mentions "Do NOT limit yourself to terms that look like brute-force polynomials unless you believe they are physically meaningful." This is advice for the *SINDy* part (basis library), but the current task is the *MPC cost function*. For the MPC cost function, standard quadratic penalties are standard practice.
#     *   Wait, the prompt says "Your task: write a Python function `stage_cost(x, u)`...". It does *not* ask me to write the SINDy basis library. It asks for the cost function.
#     *   The "Hint" at the end confirms the dynamics.
#     *   I need to calculate `u_hover` inside the function or use a literal. `m` and `g` are given in the text description. I should probably hardcode the values or use literals since I cannot import variables.
#     *   `m = 1.0`, `g = 9.81`. `u_hover = 1.0 * 9.81 / 2 = 4.905`.
#     *   I need to make sure I don't use `m` or `g` as variables unless I define them inside the function. The prompt says "You may use the standard Python operators... and integer/float literals." It doesn't explicitly forbid defining local variables, but it's safer to use literals or define them clearly.
#     *   Actually, usually in these sandbox environments, defining local variables is fine.
#     *   Let's check the constraints again: "You may NOT import anything... The math helpers... are pre-injected... use them directly".
#     *   I can define `m = 1.0`, `g = 9.81` inside the function.
#     *   Cost terms:
#         1.  Position: `x[0]**2 + x[1]**2` (scaled by weight).
#         2.  Attitude: `x[2]**2` (scaled by weight).
#         3.  Velocity: `x[3]**2 + x[4]**2` (scaled by weight).
#         4.  Angular Velocity: `x[5]**2` (scaled by weight).
#         5.  Control: `(u[0] - u_hover)**2 + (u[1] - u_hover)**2` (scaled by weight).
#     *   Weights:
#         *   Position should be penalized more than velocity.
#         *   Control effort should be penalized lightly.
#         *   Attitude needs to be penalized but allow tilt.
#     *   Let's pick weights like `Q_pos = 100`, `Q_vel = 1`, `Q_att = 10`, `Q_ctrl = 0.01`.
#     *   Wait, `u` is thrust. Deviation from hover is `u - u_hover`.
#     *   `u_hover = 4.905`.
#     *   I need to ensure `fabs` is used if I use absolute values, but `**2` is fine.
#     *   Wait, the prompt says "The function MUST be named `stage_cost` and take exactly two arguments named `x` and `u`".
#     *   I need to make sure I don't use `numpy`.
#     *   I need to make sure I use `fabs` if I need absolute value, but `x**2` is smoother and standard for MPC.
#     *   Wait, one detail: "The engineer is relying on your physical and control-theoretic knowledge... to propose a basis library." -> This was the *previous* instruction context. The *current* task is "Your task: write a Python function `stage_cost(x, u)`".
#     *   Okay, I will focus on the cost function.
# 
# 3.  **Refining the Cost Function:**
#     *   `m = 1.0`
#     *   `g = 9.81`
#     *   `u_hover = m * g / 2`
#     *   `u_max = 2 * m * g` (not strictly needed for cost, but good to know).
#     *   `x` indices: 0, 1, 2, 3, 4, 5.
#     *   `u` indices: 0, 1.
#     *   Cost = `Q_pos * (x[0]**2 + x[1]**2) + Q_att * x[2]**2 + Q_vel * (x[3]**2 + x[4]**2) + Q_omega * x[5]**2 + Q_ctrl * ((u[0] - u_hover)**2 + (u[1] - u_hover)**2)`.
#     *   Weights:
#         *   `Q_pos`: High (e.g., 1000).
#         *   `Q_vel`: Medium (e.g., 10).
#         *   `Q_att`: Medium (e.g., 100).
#         *   `Q_omega`: Low (e.g., 1).
#         *   `Q_ctrl`: Low (e.g., 0.01).
#     *   Wait, `u` is thrust. `u` values are around 5-10. `u - u_hover` is small. Squaring it makes it small.
#     *   Let's check the constraint: "u_1, u_2 : left and right rotor thrusts in Newtons. Each rotor is in [0, u_max] where u_max = 2*m*g."
#     *   `u_hover = m*g/2`.
#     *   So `u` is around 5.
#     *   If I use `Q_ctrl = 0.01`, `0.01 * (5-5)^2 = 0`.
#     *   If I use `Q_ctrl = 1.0`, `1.0 * (5-5)^2 = 0`.
#     *   If I use `Q_ctrl = 100.0`, `100.0 * (0.1)^2 = 1.0`.
#     *   Usually, control weights are tuned relative to state weights.
#     *   Let's use reasonable relative weights.
#     *   `Q_pos = 1000.0`
#     *   `Q_vel = 10.0`
#     *   `Q_att = 100.0`
#     *   `Q_omega = 1.0`
#     *   `Q_ctrl = 0.1` (to allow some thrust deviation).
#     *   Wait, `u` is thrust. `u` is in Newtons. `u` deviation is in Newtons.
#     *   If `u` is 5N, `u^2` is 25.
#     *   If `x` is 1m, `x^2` is 1.
#     *   So `Q_pos` needs to be larger than `Q_ctrl` if `u` is larger than `x`.
#     *   Let's just pick arbitrary reasonable weights that satisfy the "smaller = better" and "penalize deviation" logic.
#     *   I will define `m` and `g` inside the function to calculate `u_hover`.
#     *   I need to ensure I don't use `abs`. `fabs` is allowed. `x**2` is fine.
#     *   I need to ensure I don't use `math`.
#     *   I need to ensure I don't use `np`.
# 
# 4.  **Drafting the Code:**
#     ```python
#     def stage_cost(x, u):
#         m = 1.0
#         g = 9.81
#         u_hover = m * g / 2.0
# 
#         # Position error (x[0], x[1])
#         pos_err = x[0]**2 + x[1]**2
#         # Attitude error (x[2])
#         att_err = x[2]**2
#         # Velocity error (x[3], x[4])
#         vel_err = x[3]**2 + x[4]**2
#         # Angular velocity error (x[5])
#         om_err = x[5]**2
#         # Control effort (u[0], u[1] vs u_hover)
#         ctrl_err = (u[0] - u_hover)**2 + (u[1] - u_hover)**2
# 
#         # Weights
#         Q_pos = 1000.0
#         Q_att = 100.0
#         Q_vel = 10.0
#         Q_omega = 1.0
#         Q_ctrl = 0.1
# 
#         return Q_pos * pos_err + Q_att * att_err + Q_vel * vel_err + Q_omega * om_err + Q_ctrl * ctrl_err
#     ```
#     Wait, I need to check if `m` and `g` are allowed to be defined inside. "You may use the standard Python operators... and integer/float literals." It doesn't forbid defining local variables.
#     However, to be safe and strictly follow "integer/float literals", I can hardcode `u_hover = 4.905`.
#     `m = 1.0`, `g = 9.81` are given in the text.
#     `u_hover = 1.0 * 9.81 / 2.0 = 4.905`.
#     I will use `4.905` to avoid defining variables if that's a restriction, but defining `m` and `g` is standard Python. The restriction is "You may NOT import anything".
#     I'll stick to defining `m` and `g` for clarity, or just hardcode `4.905`. Hardcoding is safer against "variable definition" restrictions if any exist implicitly (though none are stated).
#     Actually, `m` and `g` are physical parameters provided in the prompt. Using them makes the code more readable and robust if parameters change (though they won't here).
#     Let's check the constraint: "You may use the standard Python operators... and integer/float literals."
#     It does not say "You may NOT define variables".
#     However, to be extremely safe with the "CasADi-based MPC solver" context, symbolic variables are expected. Defining `m` and `g` as floats inside the function works for both Python and CasADi (if they are symbols, `m` would need to be a symbol, but here `m` is a constant).
#     Wait, if `x` and `u` are CasADi symbolic variables, `x[0]` is a symbol. `x[0]**2` is a symbol.
#     If I define `m = 1.0`, `m` becomes a Python float.
#     If the function is called with CasADi symbols, `m` is a Python float.
#     `u_hover = m * g / 2.0` -> Python float.
#     `u[0] - u_hover` -> Symbol - Float. CasADi handles this.
#     So defining `m` and `g` is fine.
#     I will use `4.905` directly to avoid any ambiguity about variable scope or type, as it's a literal.
#     Wait, `u_hover = m * g / 2`. `m=1.0`, `g=9.81`. `1.0 * 9.81 / 2 = 4.905`.
#     I'll use `4.905`.
# 
#     Wait, one more check: "The function MUST be named `stage_cost` and take exactly two arguments named `x` and `u` in that order."
#     "It MUST return a single scalar value".
#     "You may NOT import anything".
#     "The math helpers `sin`, `cos`, `exp`, `sqrt`, `log`, `fabs` are pre-injected... use them directly".
#     "You may use the standard Python operators... and integer/float literals."
# 
#     I should check if I need `fabs`. `x**2` is always non-negative. `fabs` is for `abs`.
#     I will use `**2` for quadratic costs.
#     I will use `+`, `-`, `*`, `/`.
# 
#     One detail: "The engineer is relying on your physical and control-theoretic knowledge... to propose a basis library." -> This is context for the *previous* turn or the general task. The *current* task is specifically the `stage_cost` function.
#     The prompt says "Your task: write a Python function `stage_cost(x, u)`".
#     So I only need to output the function.
# 
#     Wait, I need to check the weights again.
#     "Position errors should typically be penalized more strongly than velocity errors."
#     "Attitude error (theta) should be penalized but not so hard that the controller refuses to tilt at all".
#     "Control effort should be penalized lightly".
#     My weights:
#     `Q_pos = 1000.0`
#     `Q_vel = 10.0`
#     `Q_att = 100.0`
#     `Q_omega = 1.0`
#     `Q_ctrl = 0.1`
#     This seems reasonable. `Q_pos` is much larger than `Q_vel`. `Q_att` is larger than `Q_omega`. `Q_ctrl` is small
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
        m = 1.0
        g = 9.81
        u_hover = m * g / 2.0

        # Position error (x[0], x[1])
        pos_err = x[0]**2 + x[1]**2
        # Attitude error (x[2])
        att_err = x[2]**2
        # Velocity error (x[3], x[4])
        vel_err = x[3]**2 + x[4]**2
        # Angular velocity error (x[5])
        om_err = x[5]**2
        # Control effort (u[0], u[1] vs u_hover)
        ctrl_err = (u[0] - u_hover)**2 + (u[1] - u_hover)**2

        # Weights
        Q_pos = 1000.0
        Q_att = 100.0
        Q_vel = 10.0
        Q_omega = 1.0
        Q_ctrl = 0.1

        return Q_pos * pos_err + Q_att * att_err + Q_vel * vel_err + Q_omega * om_err + Q_ctrl * ctrl_err

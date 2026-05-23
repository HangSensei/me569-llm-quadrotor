"""Q-condition prompt ablation variant 'cheat'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Role:** Control Engineer / Assistant.
#     *   **Task:** Write a Python function `stage_cost(x, u)` for an MPC controller for a Planar Quadrotor.
#     *   **Context:** SINDy dynamics identification (though the cost function itself is for MPC, the prompt mentions the dynamics hint at the end).
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
#         *   **Physics:** Planar Quadrotor dynamics provided in the hint.
#         *   **Goal:** Minimize deviation from hover equilibrium (`p_x=0, p_z=0, theta=0, v_x=0, v_z=0, omega=0`, `u_1=u_2=m*g/2`).
#     *   **Output Format:** ONLY the function definition inside a single `python` code fence. No prose outside.
# 
# 2.  **Determine the Cost Function Structure:**
#     *   MPC stage cost typically looks like $J = \sum (x_{k}^T Q x_{k} + u_{k}^T R u_{k})$.
#     *   Since it's a single step cost function for the solver, it needs to penalize state deviation and control deviation.
#     *   **State Weights ($Q$):**
#         *   Position ($p_x, p_z$): High weight to keep it near origin.
#         *   Velocity ($v_x, v_z$): Moderate weight.
#         *   Attitude ($\theta$): Moderate weight. Note: $\theta$ is small angle, so $\sin(\theta) \approx \theta$. Penalizing $\theta^2$ is standard.
#         *   Angular Velocity ($\omega$): Moderate weight.
#     *   **Control Weights ($R$):**
#         *   Thrusts ($u_1, u_2$): Penalize deviation from hover thrust ($u_{hover} = m*g/2$).
#         *   Since $u_{max} = 2*m*g$, $u_{hover} = 0.5 * u_{max}$.
#         *   $m = 1.0$, $g = 9.81$, so $u_{hover} = 4.905$.
#         *   Control effort penalty should encourage smoothness but allow necessary thrust.
#     *   **Constraints:**
#         *   Must use `fabs` for absolute values if needed (though squared terms avoid `fabs`).
#         *   Must use `sin`, `cos` if needed (though quadratic state costs usually don't need trig unless modeling specific nonlinearities in the cost, but standard quadratic cost is fine).
#         *   Must handle the specific equilibrium values correctly.
# 
# 3.  **Drafting the Cost Function:**
#     *   Hover equilibrium:
#         *   $x_{eq} = [0, 0, 0, 0, 0, 0]$
#         *   $u_{eq} = [4.905, 4.905]$
#     *   State Error:
#         *   $e_x = x[0]$
#         *   $e_z = x[1]$
#         *   $e_\theta = x[2]$
#         *   $e_vx = x[3]$
#         *   $e_vz = x[4]$
#         *   $e_\omega = x[5]$
#     *   Control Error:
#         *   $e_{u1} = u[0] - 4.905$
#         *   $e_{u2} = u[1] - 4.905$
#     *   Weights:
#         *   Position: 10.0 (arbitrary, needs to be high enough).
#         *   Velocity: 1.0.
#         *   Attitude: 1.0.
#         *   Angular Velocity: 0.5.
#         *   Control: 0.1.
#     *   Calculation:
#         *   `pos_cost = 10.0 * x[0]**2 + 10.0 * x[1]**2`
#         *   `att_cost = 1.0 * x[2]**2`
#         *   `vel_cost = 1.0 * x[3]**2 + 1.0 * x[4]**2 + 0.5 * x[5]**2`
#         *   `ctrl_cost = 0.1 * (u[0] - 4.905)**2 + 0.1 * (u[1] - 4.905)**2`
#         *   `total = pos_cost + att_cost + vel_cost + ctrl_cost`
# 
# 4.  **Refining for Constraints:**
#     *   Check allowed helpers: `sin`, `cos`, `exp`, `sqrt`, `log`, `fabs`.
#     *   The draft above uses `**` and `+`, `-`, `*`, `()`. This is allowed.
#     *   Does it need `fabs`? Squared terms are always non-negative, so `fabs` isn't strictly necessary for the math, but `**2` is allowed.
#     *   Does it need `sin`/`cos`? The cost function is usually quadratic in state/control space for LQR-style MPC. The dynamics hint shows trig functions in the *dynamics* ($v_x\_dot$), but the *cost* is usually simpler. However, sometimes cost functions include terms like $\sin(\theta)$ to penalize specific orientations, but standard quadratic is safer and more standard for MPC unless specified otherwise. The prompt says "Design a cost that penalizes deviation from hover". Quadratic deviation is the standard interpretation.
#     *   Wait, I need to check if I can use `**`. The prompt says: "You may use the standard Python operators `+`, `-`, `*`, `/`, `**`, parentheses, and integer/float literals." Yes, `**` is allowed.
#     *   I need to make sure I don't use `abs`. I must use `fabs` if I need absolute value. Since I'm squaring, I don't need absolute value.
#     *   I need to make sure I don't use `math`. I am using bare names `sin`, `cos` etc. if I use them. I am not using them in the draft above, but I can if I want.
#     *   Wait, the prompt says "The math helpers `sin`, `cos`, `exp`, `sqrt`, `log`, `fabs` are pre-injected as bare names — use them directly without any import."
#     *   So I can use `sin` if I want.
#     *   Is there any reason to use trig in the cost? Maybe to penalize large angles non-linearly? No, quadratic is fine.
#     *   One detail: The prompt says "The hover-equilibrium thrust per rotor is u_hover = m*g/2". $m=1.0, g=9.81 \rightarrow 4.905$.
#     *   I should hardcode this value or calculate it. Hardcoding `4.905` is safer given the constraints (no `m` or `g` variables defined in the function scope, though I could define them as literals). The prompt says "integer/float literals". So `4.905` is fine.
#     *   Wait, I should check if I need to define `m` and `g` inside the function. The prompt says "You may use the standard Python operators... and integer/float literals." It does not explicitly forbid defining local variables, but it's safer to just use literals to avoid scope issues or variable name conflicts.
#     *   Wait, the prompt says "The function MUST be named `stage_cost` and take exactly two arguments named `x` and `u`".
#     *   I need to ensure the code is valid Python.
# 
# 5.  **Final Review of Constraints:**
#     *   No imports.
#     *   No `numpy`.
#     *   No `math` module.
#     *   Use `fabs` instead of `abs`.
#     *   Use `sin`, `cos` etc. if needed (I won't need them for a quadratic cost, but I can use them if I want to make it look more "physics-aware" regarding the equilibrium, though quadratic is standard).
#     *   Actually, using `sin` or `cos` in the cost function is unnecessary for a standard quadratic MPC cost. I will stick to quadratic forms to ensure stability and simplicity, as the prompt asks to "penalize deviation from hover".
#     *   Wait, I need to check the "Hard requirements" again. "You may NOT import anything." "The math helpers ... are pre-injected".
#     *   Okay, I will write the function.
# 
# 6.  **Constructing the Code:**
#     ```python
#     def stage_cost(x, u):
#         # Hover equilibrium values
#         u_hover = 4.905
#         # State weights
#         Q_pos = 10.0
#         Q_vel = 1.0
#         Q_att = 1.0
#         Q_omega = 0.5
#         # Control weight
#         R_ctrl = 0.1
#         
#         # Position cost
#         pos_cost = Q_pos * x[0]**2 + Q_pos * x[1]**2
#         
#         # Attitude cost
#         att_cost = Q_att * x[2]**2
#         
#         # Velocity cost
#         vel_cost = Q_vel * x[3]**2 + Q_vel * x[4]**2 + Q_omega * x[5]**2
#         
#         # Control cost (deviation from hover thrust)
#         ctrl_cost = R_ctrl * (u[0] - u_hover)**2 + R_ctrl * (u[1] - u_hover)**2
#         
#         return pos_cost + att_cost + vel_cost + ctrl_cost
#     ```
#     *   Wait, I should check if defining local variables like `Q_pos` is allowed. Yes, standard Python.
#     *   Wait, I should check if I need to use `fabs`. Since I am squaring, `x**2` is always non-negative. `fabs` is not needed.
#     *   Wait, I should check if I need to use `sin`/`cos`. The prompt mentions "Design a cost that penalizes deviation from hover, with awareness of these dynamics." The dynamics involve `sin(theta)`. However, the cost function is usually on the state space. Penalizing `theta` directly is standard.
#     *   One nuance: The prompt says "The engineer is relying on your physical and control-theoretic knowledge... to propose a basis library." BUT the task is "Your task is to write a Python function `stage_cost`". The SINDy part is context for the *dynamics*, but the *task* is the cost function.
#     *   I will stick to the quadratic cost. It's robust.
#     *   Wait, I need to make sure I don't use `abs`. I am not using it.
#     *   I need to make sure I don't use `math`. I am not using it.
#     *   I need to make sure I don't use `np`. I am not using it.
#     *   I need to make sure I use `fabs` if I use absolute value. I am not using absolute value.
#     *   I need to make sure I use `sin`, `cos` if I use them. I am not using them.
#     *   Wait, I should check the `u_hover` calculation. $m=1.0, g=9.81$. $u_{hover} = 1.0 * 9.81 / 2 = 4.905$.
#     *   I will use `4.905` directly.
#     *   Wait, I should check if I need to handle the case where `x` or `u` are symbolic (CasADi). The prompt says "The function is evaluated symbolically inside a CasADi-based MPC solver". This means I should avoid operations that don't work symbolically. `**` works symbolically. `+`, `-`, `*` work symbolically. `sin`, `cos` work symbolically.
#     *   So the code is compatible.
# 
#     *   Wait, one more check: "You may use the standard Python operators `+`, `-`, `*`, `/`, `**`, parentheses, and integer/float literals."
#     *   Defining local variables `Q_pos` etc. is fine.
#     *   I will simplify the code to be as direct as possible to avoid any potential issues with variable scoping or interpretation, though local variables are standard Python.
#     *   Actually, to be extremely safe with the "no imports" and "symbolic" constraint, keeping it inline is often safer, but local variables are fine. I'll keep them for readability.
#     *   Wait, I need to ensure I don't accidentally use `abs`. I will use `**2`.
#     *   Wait, I need to ensure I don't use `math.sqrt`. I am not using it.
#     *   Wait, I need to ensure I don't use `np`. I am not using it.
# 
#     *   Wait, I should check the `u` constraint. `u` is `[u_1, u_2]`.
#     *   The cost should penalize deviation from `u_hover`.
#     *   Okay, ready to write.
# 
#     *   Wait, I need to check if I should use `fabs` for the control error. `(u[0] - u_hover)**2` is always positive. `fabs` is not needed.
#     *   Wait, I should check if I should use `sin` or `cos` in the cost. Sometimes, to penalize large angles better, one might use `sin(theta)` or `cos(theta)` in the cost, but `theta**2` is standard for small angles. Given the prompt asks to "penalize deviation from hover", `theta**2` is the most direct interpretation.
#     *   Wait, I should check if I need to use `m` and `g` as literals. `m=1.0`, `g=9.81`. `u_hover = 4.905`.
#     *   I will use `4.905`.
# 
#     *   Wait, one detail from the prompt: "The engineer does NOT tell you the equations of motion. The engineer is relying on your physical and control-theoretic knowledge... to propose a basis library."
#     *   "Your task is therefore to draw on your own knowledge... and decide which nonlinear functions... are most likely to appear in the dynamics."
#     *   "Your task: write a Python function `stage_cost`..."
#     *   Okay, the SINDy part is context for the *dynamics* identification, but the *task* is the cost function. The cost function is for the MPC.
#     *   I will proceed with the quadratic cost.
# 
#     *   Wait, I need to check if I can use `**`. Yes, "You may use the standard Python operators ... `**`".
#     *   Wait, I need to check if I can use `float` literals. Yes, "integer/float literals".
#     *   Wait, I need to check if I can use `x[0]`. Yes, "x is a length-6 vector indexable as x[0]..x[5]".
# 
#     *   Wait, I should check if I need to use `fabs` for `x[0]` etc. No, squaring handles signs.
#     *   Wait, I should check if I need to use `sin` or `cos` for the cost. No, standard quadratic cost.
#     *   Wait, I should check if I need to use `exp` or `log`. No.
#     *   Wait, I should check if I need to use `sqrt`. No.
# 
#     *   Wait, I need to make sure I don't use `abs`. I am not using it.
#     *   Wait, I need to make sure I don't use `math`. I am not using it.
#     *   Wait, I need to make sure I don't use `np`. I am not using it.
# 
#     *   Wait, I should check if I need to use `u_hover` calculation. `m*g/2`. `1.0 * 9.81 / 2 = 4.905`.
#     *   I will use `4.905`.
# 
#     *   Wait, I should check if I need to use `u_max`. `u_max = 2*m*g = 19.62`. Not needed for cost, but good to know.
#     *   Wait, I should check if I need to use `L` or `I_yy`. Not needed for cost.
# 
#     *   Wait, I should check if I need to use `sin` or `cos` in the cost.
#     *   Actually, thinking about the dynamics: `v_x_dot = -(u_1 + u_2) * sin(theta) / m`.
#     *   If I want to penalize the *force* required to move, maybe I should penalize `u` deviation.
#     *   The prompt says "Control effort should be penalized lightly so the optimizer prefers low-energy solutions but is still allowed to use enough thrust to recover from disturbances."
#     *   So
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
        # Hover equilibrium values
        u_hover = 4.905
        # State weights
        Q_pos = 10.0
        Q_vel = 1.0
        Q_att = 1.0
        Q_omega = 0.5
        # Control weight
        R_ctrl = 0.1
        
        # Position cost
        pos_cost = Q_pos * x[0]**2 + Q_pos * x[1]**2
        
        # Attitude cost
        att_cost = Q_att * x[2]**2
        
        # Velocity cost
        vel_cost = Q_vel * x[3]**2 + Q_vel * x[4]**2 + Q_omega * x[5]**2
        
        # Control cost (deviation from hover thrust)
        ctrl_cost = R_ctrl * (u[0] - u_hover)**2 + R_ctrl * (u[1] - u_hover)**2
        
        return pos_cost + att_cost + vel_cost + ctrl_cost

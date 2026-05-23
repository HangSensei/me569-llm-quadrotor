"""Q-condition prompt ablation variant 'current'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Role:** Control Engineer / Assistant.
#     *   **Task:** Write a Python function `stage_cost(x, u)` for an MPC controller for a Planar Quadrotor.
#     *   **Context:** The user is helping identify dynamics for SINDy, but the immediate task is defining the MPC cost function.
#     *   **State Vector `x`:** `[p_x, p_z, theta, v_x, v_z, omega]` (6 elements).
#     *   **Control Vector `u`:** `[u_1, u_2]` (2 elements, left and right rotor thrusts).
#     *   **Constraints:**
#         *   Function name: `stage_cost`.
#         *   Arguments: `x`, `u`.
#         *   Return: Single scalar.
#         *   **No imports** (including `numpy`).
#         *   **Allowed helpers:** `sin`, `cos`, `exp`, `sqrt`, `log`, `fabs` (bare names).
#         *   **Allowed operators:** `+`, `-`, `*`, `/`, `**`, `()`, literals.
#         *   **Forbidden:** `np.array`, `math.sqrt`, `abs`, etc.
#         *   **Physics/Control Knowledge:** Hover equilibrium is `x=0`, `u_1 = u_2 = m*g/2`. Cost should be minimized at equilibrium.
#     *   **Output Format:** ONLY the function definition inside a single `python` code fence. No prose outside.
# 
# 2.  **Determine the Cost Function Structure:**
#     *   MPC cost functions typically look like $J = \sum (x_{ref} - x)^T Q (x_{ref} - x) + (u_{ref} - u)^T R (u_{ref} - u)$.
#     *   Here, the reference state is the hover equilibrium: `p_x=0, p_z=0, theta=0, v_x=0, v_z=0, omega=0`.
#     *   The reference control is the hover thrust: `u_1 = u_2 = m*g/2`.
#     *   Given parameters: `m = 1.0`, `g = 9.81`. So `u_hover = 4.905`.
#     *   **State Weights (Q):**
#         *   Position (`p_x`, `p_z`): Should be penalized. Let's use a weight like `10.0` or `1.0`. Since `p_z` is vertical, maybe slightly different, but usually symmetric for simplicity unless specified. Let's use `10.0` for position.
#         *   Velocity (`v_x`, `v_z`): Should be penalized less than position. Let's use `1.0`.
#         *   Attitude (`theta`): Needs to be penalized to prevent drift, but not too hard. `5.0` seems reasonable.
#         *   Angular Velocity (`omega`): Often penalized to prevent spinning. `0.5` or `1.0`.
#     *   **Control Weights (R):**
#         *   Control effort should be penalized.
#         *   Deviation from hover thrust: `(u_1 - 4.905)^2 + (u_2 - 4.905)^2`.
#         *   Weight: `0.1` or `0.01`.
#     *   **Constraints:** The prompt mentions `u` is in `[0, u_max]`. The cost function itself doesn't enforce bounds (that's usually done in the MPC solver), but the cost should reflect the effort.
#     *   **CasADi Compatibility:** The prompt mentions CasADi-based MPC solver. This means the function must be compatible with symbolic differentiation. Standard arithmetic operators and the allowed math functions (`sin`, `cos`, etc.) are fine.
# 
# 3.  **Drafting the Code:**
#     *   Need to calculate `u_hover`. `m` and `g` are given in the prompt description (`m = 1.0`, `g = 9.81`). I should hardcode these values or use literals. The prompt says "Hard requirements... You may use... integer/float literals". So I can use `1.0` and `9.81`.
#     *   `u_hover = 1.0 * 9.81 / 2.0 = 4.905`.
#     *   State terms:
#         *   `pos_x = x[0]`
#         *   `pos_z = x[1]`
#         *   `theta = x[2]`
#         *   `vel_x = x[3]`
#         *   `vel_z = x[4]`
#         *   `omega = x[5]`
#     *   Control terms:
#         *   `u_1 = u[0]`
#         *   `u_2 = u[1]`
#     *   Weights:
#         *   `Q_pos = 10.0`
#         *   `Q_vel = 1.0`
#         *   `Q_att = 5.0`
#         *   `Q_omega = 0.5`
#         *   `R_ctrl = 0.1`
#     *   Calculation:
#         ```python
#         def stage_cost(x, u):
#             # State weights
#             pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
#             vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
#             att_cost = 5.0 * x[2] ** 2
#             omega_cost = 0.5 * x[5] ** 2
#             # Control weights
#             u_hover = 4.905
#             ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
#             return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost
#         ```
#     *   Wait, I need to check if `fabs` is needed. `x[0]**2` is always positive, so `abs` isn't needed for squaring. `fabs` is available but not strictly necessary for squares.
#     *   Check constraints: "You may NOT import anything." "numpy is intentionally NOT available." "math helpers ... are pre-injected".
#     *   Check return type: "return a single scalar value".
#     *   Check function signature: `def stage_cost(x, u):`.
#     *   Check arguments: `x` is length 6, `u` is length 2.
#     *   Check equilibrium: `x=0`, `u=4.905`. Cost should be 0.
#         *   `pos_cost = 0`
#         *   `vel_cost = 0`
#         *   `att_cost = 0`
#         *   `omega_cost = 0`
#         *   `ctrl_cost = 0.1 * (0 + 0) = 0`.
#         *   Total = 0. Correct.
#     *   Check differentiability: Squares and constants are differentiable. CasADi handles this.
# 
# 4.  **Refining Weights:**
#     *   The prompt says "Position errors should typically be penalized more strongly than velocity errors." -> `10.0` vs `1.0`.
#     *   "Attitude error (theta) should be penalized but not so hard that the controller refuses to tilt at all". -> `5.0` is okay.
#     *   "Control effort should be penalized lightly". -> `0.1` is okay.
#     *   I will use the hardcoded values from the prompt description (`m=1.0`, `g=9.81`) to calculate `u_hover` explicitly to be safe, or just use the literal `4.905`. The prompt says `u_hover = m*g/2`. `1.0 * 9.81 / 2.0 = 4.905`. I'll use `4.905` as a literal to avoid any potential float precision issues with `1.0 * 9.81 / 2.0` if `9.81` is treated differently, but `4.905` is exact enough. Actually, using `1.0 * 9.81 / 2.0` is more "physics-based" and robust if `g` changes, but the prompt gives specific values. I'll use `4.905` for clarity and speed, or calculate it. Let's calculate it to show understanding of the physics parameters provided. `u_hover = 1.0 * 9.81 / 2.0`.
#     *   Wait, the prompt says "You may use... integer/float literals".
#     *   I will use `4.905` directly as it's the specific hover thrust mentioned in the text (`u_hover = m*g/2` where `m=1.0`, `g=9.81`).
#     *   Wait, I should check if I need to use `fabs`. `x[0]**2` is always non-negative. `fabs` is not needed for squares.
#     *   I need to make sure I don't use `abs`. I will use `** 2`.
# 
# 5.  **Final Code Structure:**
#     ```python
#     def stage_cost(x, u):
#         # State weights
#         # Position (p_x, p_z)
#         pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
#         # Velocity (v_x, v_z)
#         vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
#         # Attitude (theta)
#         att_cost = 5.0 * x[2] ** 2
#         # Angular velocity (omega)
#         omega_cost = 0.5 * x[5] ** 2
#         # Control effort (deviation from hover thrust)
#         u_hover = 1.0 * 9.81 / 2.0
#         ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
#         return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost
#     ```
#     Wait, `u_hover` calculation: `1.0 * 9.81 / 2.0` is `4.905`.
#     Is there any reason to use `fabs`? No, squares are positive.
#     Is there any reason to use `sin`/`cos`? No, standard quadratic cost is sufficient for MPC unless there are specific nonlinearities in the cost (e.g., saturation). The prompt asks for a "stage_cost" for MPC. Quadratic is standard.
#     One detail: "The hover-equilibrium thrust per rotor is u_hover = m*g/2."
#     I will use `4.905` directly to ensure it matches the prompt's specific numbers exactly without floating point ambiguity, though `1.0 * 9.81 / 2.0` is fine too. I'll use `4.905` for clarity.
#     Actually, to be safe with the "physics" aspect, I'll calculate it. `m=1.0`, `g=9.81`.
#     Wait, I need to check if I can use `m` and `g` as variables. The prompt says "You may use... integer/float literals". It does not explicitly forbid defining variables, but it's safer to just use literals to avoid scope issues if the function is evaluated in a restricted environment. I'll use literals.
# 
#     Wait, one more check on the prompt: "The engineer does NOT tell you the equations of motion... Your job is therefore to draw on your own knowledge... and decide which nonlinear functions... are most likely to appear... Your task is therefore to write a Python function stage_cost...".
#     The first paragraph is context about SINDy. The actual task is the `stage_cost` function.
#     The prompt says "Return ONLY the function definition inside a single ``python`` code fence. No prose explanation outside the fence."
# 
#     Let's refine the weights to be reasonable.
#     Position: 10.0
#     Velocity: 1.0
#     Attitude: 5.0
#     Angular Velocity: 0.5
#     Control: 0.1
#     These seem balanced.
# 
#     One thing: `x` and `u` are lists/arrays. Indexing `x[0]` works.
#     The prompt says `x` is a length-6 vector indexable as `x[0]..x[5]`.
#     The prompt says `u` is a length-2 vector indexable as `u[0], u[1]`.
#     So `x[0]` is `p_x`, `x[1]` is `p_z`, etc.
# 
#     Final check on constraints:
#     - Function name `stage_cost`.
#     - Arguments `x`, `u`.
#     - Return scalar.
#     - No imports.
#     - Allowed helpers: `sin`, `cos`, `exp`, `sqrt`, `log`, `fabs`.
#     - Allowed operators: `+`, `-`, `*`, `/`, `**`, `()`, literals.
#     - No `abs`, `np.array`, etc.
# 
#     I will write the code now.
# 
#     Wait, I should check if `u_hover` needs to be defined inside. Yes.
#     `u_hover = 4.905` is safe.
#     Or `u_hover = 1.0 * 9.81 / 2.0`.
#     I'll use `4.905` to match the text description exactly.
# 
#     Wait, I need to make sure I don't use `abs`. I am using `** 2`.
#     I need to make sure I don't use `math`. I am not.
#     I need to make sure I don't use `np`. I am not.
# 
#     One detail: "The function MUST be named ``stage_cost`` and take exactly two arguments named ``x`` and ``u`` in that order."
#     My draft: `def stage_cost(x, u):`. Correct.
# 
#     One detail: "You may NOT import anything."
#     My draft: No imports. Correct.
# 
#     One detail: "The math helpers ``sin``, ``cos``, ``exp``, ``sqrt``, ``log``, ``fabs`` are pre-injected as bare names — use them directly without any import."
#     I am not using them, but I can if I wanted. I am using `**` and literals.
# 
#     One detail: "Return ONLY the function definition inside a single ``python`` code fence. No prose explanation outside the fence."
#     I must ensure no text outside the fence.
# 
#     Let's assemble the function.
# 
# ```python
# def stage_cost(x, u):
#     # State weights
#     # Position (p_x, p_z)
#     pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
#     # Velocity (v_x, v_z)
#     vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
#     # Attitude (theta)
#     att_cost = 5.0 * x[2] ** 2
#     # Angular velocity (omega)
#     omega_cost = 0.5 * x[5] ** 2
#     # Control effort (deviation from hover thrust)
#     u_hover = 4.905
#     ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
#     return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost
# ```
# 
#     Wait, I should check if `u_hover` calculation `1.0 * 9.81 / 2.0` is better. The prompt says `u_max = 2*m*g` and `u_hover = m*g/2`. `m=1.0`, `g=9.81`. `u_hover = 4.905`.
#     Using `4.905` is fine.
#     I will add comments to explain the weights briefly, as it's good practice, though not strictly required. The prompt example had comments.
#     I will ensure the indentation is correct (4 spaces).
# 
#     Wait, one more check: "The function MUST be named ``stage_cost``".
#     "It MUST return a single scalar value".
#     "You may NOT import anything."
#     "numpy is intentionally NOT available."
#     "math helpers ... are pre-injected".
#     "You may use the standard Python operators ... and integer/float literals."
# 
#     Looks good.
# 
#     Wait, I should check if `x` and `u` are guaranteed to be lists or if they could be symbolic variables from CasADi. The prompt says "The function is evaluated symbolically inside a CasADi-based MPC solver". CasADi variables support `**`, `+`, `-`, `*`, `/`. So the code is compatible.
# 
#     One final check on the weights.
#     Position: 10.0
#     Velocity: 1.0
#     Attitude: 5.0
#     Angular Velocity: 0.5
#     Control: 0.1
#     These are arbitrary but reasonable. I'll stick with them.
#     Actually, to make it slightly more robust against the "physics" aspect, I might want to ensure `u_hover` is calculated from `m` and `g` if I can, but I can't import `m` or `g` as variables unless I define them. Defining `m = 1.0` and `g = 9.81` inside the function
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
            # State weights
            pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
            vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
            att_cost = 5.0 * x[2] ** 2
            omega_cost = 0.5 * x[5] ** 2
            # Control weights
            u_hover = 4.905
            ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
            return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost

def stage_cost(x, u):
        # State weights
        # Position (p_x, p_z)
        pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
        # Velocity (v_x, v_z)
        vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
        # Attitude (theta)
        att_cost = 5.0 * x[2] ** 2
        # Angular velocity (omega)
        omega_cost = 0.5 * x[5] ** 2
        # Control effort (deviation from hover thrust)
        u_hover = 1.0 * 9.81 / 2.0
        ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
        return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost

def stage_cost(x, u):
    # State weights
    # Position (p_x, p_z)
    pos_cost = 10.0 * (x[0] ** 2 + x[1] ** 2)
    # Velocity (v_x, v_z)
    vel_cost = 1.0 * (x[3] ** 2 + x[4] ** 2)
    # Attitude (theta)
    att_cost = 5.0 * x[2] ** 2
    # Angular velocity (omega)
    omega_cost = 0.5 * x[5] ** 2
    # Control effort (deviation from hover thrust)
    u_hover = 4.905
    ctrl_cost = 0.1 * ((u[0] - u_hover) ** 2 + (u[1] - u_hover) ** 2)
    return pos_cost + vel_cost + att_cost + omega_cost + ctrl_cost

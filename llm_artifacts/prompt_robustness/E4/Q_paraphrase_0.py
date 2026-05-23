"""Q E4 paraphrase 0. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Task:** Design a dense per-step reward function for a Planar Quadrotor hover task.
#     *   **Agent:** PPO (Proximal Policy Optimization), specifically referencing the Eureka paper (Ma et al. 2024).
#     *   **Goal:** MAXIMIZE the reward.
#     *   **State:** `state = [p_x, p_z, theta, v_x, v_z, omega]` in $\mathbb{R}^6$.
#     *   **Action:** `action = [u_1, u_2]` in $\mathbb{R}^2$ (rotor thrusts).
#     *   **Constants:** $m = 1.0$, $I_{yy} = 0.01$, $L = 0.25$, $g = 9.81$.
#     *   **Hover Target:** $p_x=0, p_z=0, \theta=0, v_x=0, v_z=0, \omega=0$. Thrust $u_1 \approx u_2 \approx 4.905$ N.
#     *   **Constraints:**
#         *   Pure and deterministic Python function.
#         *   Return one finite float.
#         *   Higher = better.
#         *   Penalties are negative terms.
#         *   Import `numpy` (`np`) and `math` allowed.
#         *   No prose outside the code fence.
#         *   Crash/termination handled by harness (don't include them).
#         *   Weight position/attitude error > velocity error.
#         *   Penalize control deviation from hover thrust.
#         *   Prefer smooth differentiable shapes over step functions.
#     *   **Output:** ONLY the function definition inside a single python code fence.
# 
# 2.  **Deconstruct the Reward Design:**
#     *   **Objective:** Hover.
#     *   **State Space:** Position ($p_x, p_z$), Attitude ($\theta$), Velocity ($v_x, v_z$), Angular Velocity ($\omega$).
#     *   **Action Space:** Thrusts ($u_1, u_2$).
#     *   **Reward Components:**
#         1.  **Position Error:** $-(p_x^2 + p_z^2)$. Weighted higher than velocity.
#         2.  **Attitude Error:** $-(\theta^2)$. Weighted higher than velocity.
#         3.  **Velocity Error:** $-(v_x^2 + v_z^2 + \omega^2)$. Lower weight than position/attitude.
#         4.  **Control Effort/Deviation:** $-(u_1 - u_{hover})^2 - (u_2 - u_{hover})^2$. Encourages hovering thrusts.
#         5.  **Smoothness:** Squared terms are smooth and differentiable.
#     *   **Weights:** Need to tune relative weights.
#         *   Position/Attitude should be dominant.
#         *   Velocity should be secondary.
#         *   Control deviation should be a penalty to prevent large thrusts (though hovering thrust is fixed, deviation from 4.905 is the target).
#     *   **Normalization:** Since the state values can vary, but the target is 0, squared errors are standard.
#     *   **Constants:**
#         *   Hover thrust $u_{hover} = m \cdot g = 1.0 \cdot 9.81 = 9.81$ N? Wait, the prompt says "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
#         *   Let's check the physics provided in the prompt.
#         *   Dynamics:
#             *   $v_z\_dot = (u_1 + u_2) \cos(\theta) / m - g$
#             *   Hover means $v_z\_dot = 0 \implies (u_1 + u_2) \cos(\theta) = m \cdot g$.
#             *   If $\theta = 0$, $\cos(0) = 1$. So $u_1 + u_2 = m \cdot g = 1.0 \cdot 9.81 = 9.81$.
#             *   The prompt says "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
#             *   $4.905 + 4.905 = 9.81$. This matches.
#             *   So the target thrust for each rotor is $u_{target} = 4.905$.
#     *   **Reward Structure:**
#         *   $R = -w_p (p_x^2 + p_z^2) - w_\theta \theta^2 - w_v (v_x^2 + v_z^2 + \omega^2) - w_u ((u_1 - u_{target})^2 + (u_2 - u_{target})^2)$.
#         *   Wait, usually for PPO, we want to maximize reward. So negative terms are penalties.
#         *   We need to ensure the weights make sense.
#         *   Position error: If $p_x$ drifts, we want to stop it.
#         *   Velocity error: If $v_x$ is high, we want to stop it.
#         *   Control deviation: If $u$ is far from 4.905, we want to correct it.
#         *   The prompt asks to "Weight position and attitude error above velocity error".
#         *   "Penalize control deviation from hover thrust".
#         *   "Prefer smooth differentiable shapes". Squared errors are smooth.
#     *   **Weights Selection:**
#         *   Let's pick reasonable weights.
#         *   $w_p = 10.0$
#         *   $w_\theta = 10.0$ (Attitude is critical for stability)
#         *   $w_v = 1.0$ (Velocity is less critical than position for hover, but still important)
#         *   $w_u = 1.0$ (Control deviation)
#         *   Actually, usually position is weighted higher than velocity.
#         *   Let's check the magnitude of state variables.
#         *   If $p_x$ is 1m, $p_x^2 = 1$.
#         *   If $v_x$ is 1m/s, $v_x^2 = 1$.
#         *   If $\theta$ is 0.1 rad, $\theta^2 = 0.01$.
#         *   If $u$ is 5.0, deviation is 0.05, squared is 0.0025.
#         *   To make position/attitude dominant, we might need scaling.
#         *   However, the prompt asks to "Weight position and attitude error above velocity error". It doesn't explicitly ask for specific values, just the design. I should choose values that make sense physically.
#         *   Let's use a standard scaling.
#         *   $R = - (10.0 * (p_x^2 + p_z^2) + 10.0 * \theta^2 + 1.0 * (v_x^2 + v_z^2 + \omega^2) + 1.0 * ((u_1 - 4.905)^2 + (u_2 - 4.905)^2))$.
#         *   Wait, is $\theta$ in radians? Yes, standard physics.
#         *   Is $p_z$ height? Yes.
#         *   Is $u$ thrust? Yes.
#         *   One detail: "hover is state = 0". This implies $p_x=0, p_z=0, \theta=0, v_x=0, v_z=0, \omega=0$.
#         *   The prompt says "Weight position and attitude error above velocity error".
#         *   So $w_p > w_v$ and $w_\theta > w_v$.
#         *   Also "penalize control deviation".
#         *   Let's refine the weights to ensure stability.
#         *   $w_p = 100.0$
#         *   $w_\theta = 100.0$
#         *   $w_v = 10.0$
#         *   $w_u = 10.0$
#         *   Actually, let's keep it simple and robust.
#         *   $R = - (10.0 * (p_x^2 + p_z^2) + 10.0 * \theta^2 + 1.0 * (v_x^2 + v_z^2 + \omega^2) + 0.1 * ((u_1 - 4.905)^2 + (u_2 - 4.905)^2))$.
#         *   Wait, if I penalize control deviation too much, the agent might just hover with minimal thrust if possible, but thrust is constrained by physics.
#         *   Actually, the agent *must* produce thrust to hover. The reward should encourage the correct thrust.
#         *   Let's check the constraint: "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
#         *   So the target action is $[4.905, 4.905]$.
#         *   Let's use weights that reflect the importance.
#         *   Position/Attitude are the primary goal.
#         *   Velocity is secondary (damping).
#         *   Control is tertiary (efficiency/stability).
#         *   Let's set $w_p = 10.0, w_\theta = 10.0, w_v = 1.0, w_u = 1.0$.
#         *   Wait, if $p_x$ is 1, penalty is 10. If $v_x$ is 1, penalty is 1. This satisfies "Weight position... above velocity".
#         *   Let's make sure the function is deterministic and pure.
#         *   Need to import `numpy` as `np`.
#         *   Need to define constants inside the function or globally? "Write a Python function...". Constants are given. I can define them inside the function to be self-contained or use the provided values. Using provided values is safer.
#         *   Wait, the prompt says "Return ONLY the function definition inside a single python code fence".
#         *   I should define the constants inside the function to ensure it runs without external context issues, or just use the values provided in the text.
#         *   Better to define them inside to be safe.
#         *   Wait, "pure and deterministic".
#         *   "import numpy (as np) and math, nothing else."
#         *   I will put the constants inside the function.
# 
# 3.  **Refining the Reward Formula:**
#     *   $R = - (w_p (p_x^2 + p_z^2) + w_\theta \theta^2 + w_v (v_x^2 + v_z^2 + \omega^2) + w_u ((u_1 - u_{hover})^2 + (u_2 - u_{hover})^2))$
#     *   $u_{hover} = 4.905$.
#     *   Weights:
#         *   $w_p = 10.0$
#         *   $w_\theta = 10.0$
#         *   $w_v = 1.0$
#         *   $w_u = 1.0$
#     *   Wait, is $\theta$ small? Yes, usually.
#     *   Is $p_z$ height? Yes.
#     *   Is $p_x$ horizontal? Yes.
#     *   Is $v_z$ vertical velocity? Yes.
#     *   Is $v_x$ horizontal velocity? Yes.
#     *   Is $\omega$ angular velocity? Yes.
#     *   Is $u_1, u_2$ thrust? Yes.
#     *   One nuance: "Weight position and attitude error above velocity error".
#     *   So $w_p, w_\theta > w_v$.
#     *   "Penalize control deviation from hover thrust".
#     *   So $w_u > 0$.
#     *   "Prefer smooth differentiable shapes". Squared terms are smooth.
#     *   "Higher = better". So negative terms.
#     *   "Finite float".
#     *   "Crash penalties... handled by the harness". So I don't need to check bounds like $p_z < 0$.
#     *   "Episode termination... handled by the harness".
#     *   So I just return the dense reward.
# 
# 4.  **Final Check:**
#     *   Function name: `reward`.
#     *   Arguments: `state`, `action`.
#     *   Return: `float`.
#     *   Imports: `import numpy as np`.
#     *   Constants: $m, I_{yy}, L, g$ are given, but I only need $u_{hover} = m \cdot g / 2 = 4.905$.
#     *   Wait, $u_{hover} = m \cdot g / 2$ because $u_1 + u_2 = m \cdot g$ and $u_1 = u_2$.
#     *   $1.0 \cdot 9.81 / 2 = 4.905$.
#     *   I will use the constant 4.905 directly or calculate it. Calculating it is safer based on the constants provided ($m, g$).
#     *   Wait, the prompt says "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
#     *   I'll use `u_hover = m * g / 2.0` to be precise based on the constants provided in the prompt description.
#     *   Wait, the prompt says "Constants: m = 1.0 kg...".
#     *   I should define these constants inside the function to ensure it works.
#     *   Wait, "pure and deterministic".
#     *   I will define constants inside.
#     *   Weights: I'll choose values that satisfy the condition "Weight position and attitude error above velocity error".
#     *   Let's pick $w_p = 10.0, w_\theta = 10.0, w_v = 1.0, w_u = 1.0$.
#     *   Wait, if $p_x$ is 10m, $p_x^2 = 100$. If $v_x$ is 10m/s, $v_x^2 = 100$.
#     *   If I want position to be weighted *above* velocity, $w_p$ should be larger than $w_v$.
#     *   Let's use $w_p = 10.0, w_\theta = 10.0, w_v = 1.0, w_u = 1.0$.
#     *   Wait, is $\theta$ in radians? Yes.
#     *   Is $p_z$ in meters? Yes.
#     *   Is $u$ in Newtons? Yes.
#     *   Is the reward finite? Yes, assuming inputs are finite.
#     *   Is it deterministic? Yes.
#     *   Is it pure? Yes.
#     *   Does it return one float? Yes.
#     *   Does it handle the "higher = better" requirement? Yes, by using negative penalties.
#     *   Does it penalize control deviation? Yes.
#     *   Does it prefer smooth shapes? Yes, squared terms.
# 
#     *   Wait, one detail: "Weight position and attitude error above velocity error".
#     *   This implies $w_{pos} > w_{vel}$ and $w_{att} > w_{vel}$.
#     *   My choice $10 > 1$ satisfies this.
#     *   "Penalize control deviation from hover thrust".
#     *   My choice $w_u > 0$ satisfies this.
#     *   "Prefer smooth differentiable shapes".
#     *   Squared terms are smooth.
#     *   "Crash penalties... handled by the harness".
#     *   I don't need to check for $p_z < 0$.
#     *   "Return ONLY the function definition inside a single python code fence".
# 
#     *   Wait, I need to make sure I don't use `math` unnecessarily. `np` is sufficient for math.
#     *   I need to make sure I don't crash on `state` or `action` being numpy arrays.
#     *   `state` is `R^6`, `action` is `R^2`.
#     *   I should use `np.sum` or `np.square` and `np.sum`.
#     *   `state` is a numpy array.
#     *   `action` is a numpy array.
#     *   I will use `np.sum(state[:2]**2 + state[3]**2 + state[4]**2 + state[5]**2)`? No, specific indices.
#     *   `p_x = state[0]`, `p_z = state[1]`, `theta = state[2]`, `v_x = state[3]`, `v_z = state[4]`, `omega = state[5]`.
#     *   `u_1 = action[0]`, `u_2 = action[1]`.
#     *   I'll use `np.sum` for clarity.
#     *   `pos_err = state[0]**2 + state[1]**2`
#     *   `att_err = state[2]**2`
#     *   `vel_err = state[3]**2 + state[4]**2 + state[5]**2`
#     *   `ctrl_err = (action[0] - u_hover)**2 + (action[1] - u_hover)**2`
#     *   `reward = - (w_p * pos_err + w_theta * att_err + w_v * vel_err + w_u * ctrl_err)`
# 
#     *   Wait, I should check if `state` and `action` are guaranteed to be numpy arrays. The prompt says "state = [p_x, p_z
# --- extracted code ---
Thinking Process:

1.  **Analyze the Request:**
    *   **Task:** Design a dense per-step reward function for a Planar Quadrotor hover task.
    *   **Agent:** PPO (Proximal Policy Optimization), specifically referencing the Eureka paper (Ma et al. 2024).
    *   **Goal:** MAXIMIZE the reward.
    *   **State:** `state = [p_x, p_z, theta, v_x, v_z, omega]` in $\mathbb{R}^6$.
    *   **Action:** `action = [u_1, u_2]` in $\mathbb{R}^2$ (rotor thrusts).
    *   **Constants:** $m = 1.0$, $I_{yy} = 0.01$, $L = 0.25$, $g = 9.81$.
    *   **Hover Target:** $p_x=0, p_z=0, \theta=0, v_x=0, v_z=0, \omega=0$. Thrust $u_1 \approx u_2 \approx 4.905$ N.
    *   **Constraints:**
        *   Pure and deterministic Python function.
        *   Return one finite float.
        *   Higher = better.
        *   Penalties are negative terms.
        *   Import `numpy` (`np`) and `math` allowed.
        *   No prose outside the code fence.
        *   Crash/termination handled by harness (don't include them).
        *   Weight position/attitude error > velocity error.
        *   Penalize control deviation from hover thrust.
        *   Prefer smooth differentiable shapes over step functions.
    *   **Output:** ONLY the function definition inside a single python code fence.

2.  **Deconstruct the Reward Design:**
    *   **Objective:** Hover.
    *   **State Space:** Position ($p_x, p_z$), Attitude ($\theta$), Velocity ($v_x, v_z$), Angular Velocity ($\omega$).
    *   **Action Space:** Thrusts ($u_1, u_2$).
    *   **Reward Components:**
        1.  **Position Error:** $-(p_x^2 + p_z^2)$. Weighted higher than velocity.
        2.  **Attitude Error:** $-(\theta^2)$. Weighted higher than velocity.
        3.  **Velocity Error:** $-(v_x^2 + v_z^2 + \omega^2)$. Lower weight than position/attitude.
        4.  **Control Effort/Deviation:** $-(u_1 - u_{hover})^2 - (u_2 - u_{hover})^2$. Encourages hovering thrusts.
        5.  **Smoothness:** Squared terms are smooth and differentiable.
    *   **Weights:** Need to tune relative weights.
        *   Position/Attitude should be dominant.
        *   Velocity should be secondary.
        *   Control deviation should be a penalty to prevent large thrusts (though hovering thrust is fixed, deviation from 4.905 is the target).
    *   **Normalization:** Since the state values can vary, but the target is 0, squared errors are standard.
    *   **Constants:**
        *   Hover thrust $u_{hover} = m \cdot g = 1.0 \cdot 9.81 = 9.81$ N? Wait, the prompt says "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
        *   Let's check the physics provided in the prompt.
        *   Dynamics:
            *   $v_z\_dot = (u_1 + u_2) \cos(\theta) / m - g$
            *   Hover means $v_z\_dot = 0 \implies (u_1 + u_2) \cos(\theta) = m \cdot g$.
            *   If $\theta = 0$, $\cos(0) = 1$. So $u_1 + u_2 = m \cdot g = 1.0 \cdot 9.81 = 9.81$.
            *   The prompt says "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
            *   $4.905 + 4.905 = 9.81$. This matches.
            *   So the target thrust for each rotor is $u_{target} = 4.905$.
    *   **Reward Structure:**
        *   $R = -w_p (p_x^2 + p_z^2) - w_\theta \theta^2 - w_v (v_x^2 + v_z^2 + \omega^2) - w_u ((u_1 - u_{target})^2 + (u_2 - u_{target})^2)$.
        *   Wait, usually for PPO, we want to maximize reward. So negative terms are penalties.
        *   We need to ensure the weights make sense.
        *   Position error: If $p_x$ drifts, we want to stop it.
        *   Velocity error: If $v_x$ is high, we want to stop it.
        *   Control deviation: If $u$ is far from 4.905, we want to correct it.
        *   The prompt asks to "Weight position and attitude error above velocity error".
        *   "Penalize control deviation from hover thrust".
        *   "Prefer smooth differentiable shapes". Squared errors are smooth.
    *   **Weights Selection:**
        *   Let's pick reasonable weights.
        *   $w_p = 10.0$
        *   $w_\theta = 10.0$ (Attitude is critical for stability)
        *   $w_v = 1.0$ (Velocity is less critical than position for hover, but still important)
        *   $w_u = 1.0$ (Control deviation)
        *   Actually, usually position is weighted higher than velocity.
        *   Let's check the magnitude of state variables.
        *   If $p_x$ is 1m, $p_x^2 = 1$.
        *   If $v_x$ is 1m/s, $v_x^2 = 1$.
        *   If $\theta$ is 0.1 rad, $\theta^2 = 0.01$.
        *   If $u$ is 5.0, deviation is 0.05, squared is 0.0025.
        *   To make position/attitude dominant, we might need scaling.
        *   However, the prompt asks to "Weight position and attitude error above velocity error". It doesn't explicitly ask for specific values, just the design. I should choose values that make sense physically.
        *   Let's use a standard scaling.
        *   $R = - (10.0 * (p_x^2 + p_z^2) + 10.0 * \theta^2 + 1.0 * (v_x^2 + v_z^2 + \omega^2) + 1.0 * ((u_1 - 4.905)^2 + (u_2 - 4.905)^2))$.
        *   Wait, is $\theta$ in radians? Yes, standard physics.
        *   Is $p_z$ height? Yes.
        *   Is $u$ thrust? Yes.
        *   One detail: "hover is state = 0". This implies $p_x=0, p_z=0, \theta=0, v_x=0, v_z=0, \omega=0$.
        *   The prompt says "Weight position and attitude error above velocity error".
        *   So $w_p > w_v$ and $w_\theta > w_v$.
        *   Also "penalize control deviation".
        *   Let's refine the weights to ensure stability.
        *   $w_p = 100.0$
        *   $w_\theta = 100.0$
        *   $w_v = 10.0$
        *   $w_u = 10.0$
        *   Actually, let's keep it simple and robust.
        *   $R = - (10.0 * (p_x^2 + p_z^2) + 10.0 * \theta^2 + 1.0 * (v_x^2 + v_z^2 + \omega^2) + 0.1 * ((u_1 - 4.905)^2 + (u_2 - 4.905)^2))$.
        *   Wait, if I penalize control deviation too much, the agent might just hover with minimal thrust if possible, but thrust is constrained by physics.
        *   Actually, the agent *must* produce thrust to hover. The reward should encourage the correct thrust.
        *   Let's check the constraint: "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
        *   So the target action is $[4.905, 4.905]$.
        *   Let's use weights that reflect the importance.
        *   Position/Attitude are the primary goal.
        *   Velocity is secondary (damping).
        *   Control is tertiary (efficiency/stability).
        *   Let's set $w_p = 10.0, w_\theta = 10.0, w_v = 1.0, w_u = 1.0$.
        *   Wait, if $p_x$ is 1, penalty is 10. If $v_x$ is 1, penalty is 1. This satisfies "Weight position... above velocity".
        *   Let's make sure the function is deterministic and pure.
        *   Need to import `numpy` as `np`.
        *   Need to define constants inside the function or globally? "Write a Python function...". Constants are given. I can define them inside the function to be self-contained or use the provided values. Using provided values is safer.
        *   Wait, the prompt says "Return ONLY the function definition inside a single python code fence".
        *   I should define the constants inside the function to ensure it runs without external context issues, or just use the values provided in the text.
        *   Better to define them inside to be safe.
        *   Wait, "pure and deterministic".
        *   "import numpy (as np) and math, nothing else."
        *   I will put the constants inside the function.

3.  **Refining the Reward Formula:**
    *   $R = - (w_p (p_x^2 + p_z^2) + w_\theta \theta^2 + w_v (v_x^2 + v_z^2 + \omega^2) + w_u ((u_1 - u_{hover})^2 + (u_2 - u_{hover})^2))$
    *   $u_{hover} = 4.905$.
    *   Weights:
        *   $w_p = 10.0$
        *   $w_\theta = 10.0$
        *   $w_v = 1.0$
        *   $w_u = 1.0$
    *   Wait, is $\theta$ small? Yes, usually.
    *   Is $p_z$ height? Yes.
    *   Is $p_x$ horizontal? Yes.
    *   Is $v_z$ vertical velocity? Yes.
    *   Is $v_x$ horizontal velocity? Yes.
    *   Is $\omega$ angular velocity? Yes.
    *   Is $u_1, u_2$ thrust? Yes.
    *   One nuance: "Weight position and attitude error above velocity error".
    *   So $w_p, w_\theta > w_v$.
    *   "Penalize control deviation from hover thrust".
    *   So $w_u > 0$.
    *   "Prefer smooth differentiable shapes". Squared terms are smooth.
    *   "Higher = better". So negative terms.
    *   "Finite float".
    *   "Crash penalties... handled by the harness". So I don't need to check bounds like $p_z < 0$.
    *   "Episode termination... handled by the harness".
    *   So I just return the dense reward.

4.  **Final Check:**
    *   Function name: `reward`.
    *   Arguments: `state`, `action`.
    *   Return: `float`.
    *   Imports: `import numpy as np`.
    *   Constants: $m, I_{yy}, L, g$ are given, but I only need $u_{hover} = m \cdot g / 2 = 4.905$.
    *   Wait, $u_{hover} = m \cdot g / 2$ because $u_1 + u_2 = m \cdot g$ and $u_1 = u_2$.
    *   $1.0 \cdot 9.81 / 2 = 4.905$.
    *   I will use the constant 4.905 directly or calculate it. Calculating it is safer based on the constants provided ($m, g$).
    *   Wait, the prompt says "hover is state = 0 with $u_1 = u_2 \approx 4.905$ N".
    *   I'll use `u_hover = m * g / 2.0` to be precise based on the constants provided in the prompt description.
    *   Wait, the prompt says "Constants: m = 1.0 kg...".
    *   I should define these constants inside the function to ensure it works.
    *   Wait, "pure and deterministic".
    *   I will define constants inside.
    *   Weights: I'll choose values that satisfy the condition "Weight position and attitude error above velocity error".
    *   Let's pick $w_p = 10.0, w_\theta = 10.0, w_v = 1.0, w_u = 1.0$.
    *   Wait, if $p_x$ is 10m, $p_x^2 = 100$. If $v_x$ is 10m/s, $v_x^2 = 100$.
    *   If I want position to be weighted *above* velocity, $w_p$ should be larger than $w_v$.
    *   Let's use $w_p = 10.0, w_\theta = 10.0, w_v = 1.0, w_u = 1.0$.
    *   Wait, is $\theta$ in radians? Yes.
    *   Is $p_z$ in meters? Yes.
    *   Is $u$ in Newtons? Yes.
    *   Is the reward finite? Yes, assuming inputs are finite.
    *   Is it deterministic? Yes.
    *   Is it pure? Yes.
    *   Does it return one float? Yes.
    *   Does it handle the "higher = better" requirement? Yes, by using negative penalties.
    *   Does it penalize control deviation? Yes.
    *   Does it prefer smooth shapes? Yes, squared terms.

    *   Wait, one detail: "Weight position and attitude error above velocity error".
    *   This implies $w_{pos} > w_{vel}$ and $w_{att} > w_{vel}$.
    *   My choice $10 > 1$ satisfies this.
    *   "Penalize control deviation from hover thrust".
    *   My choice $w_u > 0$ satisfies this.
    *   "Prefer smooth differentiable shapes".
    *   Squared terms are smooth.
    *   "Crash penalties... handled by the harness".
    *   I don't need to check for $p_z < 0$.
    *   "Return ONLY the function definition inside a single python code fence".

    *   Wait, I need to make sure I don't use `math` unnecessarily. `np` is sufficient for math.
    *   I need to make sure I don't crash on `state` or `action` being numpy arrays.
    *   `state` is `R^6`, `action` is `R^2`.
    *   I should use `np.sum` or `np.square` and `np.sum`.
    *   `state` is a numpy array.
    *   `action` is a numpy array.
    *   I will use `np.sum(state[:2]**2 + state[3]**2 + state[4]**2 + state[5]**2)`? No, specific indices.
    *   `p_x = state[0]`, `p_z = state[1]`, `theta = state[2]`, `v_x = state[3]`, `v_z = state[4]`, `omega = state[5]`.
    *   `u_1 = action[0]`, `u_2 = action[1]`.
    *   I'll use `np.sum` for clarity.
    *   `pos_err = state[0]**2 + state[1]**2`
    *   `att_err = state[2]**2`
    *   `vel_err = state[3]**2 + state[4]**2 + state[5]**2`
    *   `ctrl_err = (action[0] - u_hover)**2 + (action[1] - u_hover)**2`
    *   `reward = - (w_p * pos_err + w_theta * att_err + w_v * vel_err + w_u * ctrl_err)`

    *   Wait, I should check if `state` and `action` are guaranteed to be numpy arrays. The prompt says "state = [p_x, p_z

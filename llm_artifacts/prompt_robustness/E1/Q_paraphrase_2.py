"""Q E1 paraphrase 2. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Goal:** Create a Python function `basis` for Sparse Identification of Nonlinear Dynamics (SINDy) / STLSQ.
#     *   **System:** 2D planar quadrotor.
#     *   **State Vector ($x$):** $[p_x, p_z, \theta, v_x, v_z, \omega]$ (6 entries).
#     *   **Control Vector ($u$):** $[u_1, u_2]$ (2 rotor thrusts).
#     *   **Input Argument:** `xu` of length 8. `xu[0:6]` are states, `xu[6:8]` are controls.
#     *   **Constants:** $m=1.0$, $I_{yy}=0.01$, $L=0.25$, $g=9.81$.
#     *   **Task:** Return a list/array of roughly 10-30 nonlinear features.
#     *   **Requirements:**
#         *   Must be rich enough to recover state derivatives.
#         *   Must include thrust-projection sin/cos couplings.
#         *   Must use only `numpy` (as `np`) and `math`.
#         *   No other imports.
#         *   No output before or after the code fence.
#         *   Function name: `basis`.
#         *   Argument: `xu` (positional, length 8).
#         *   Return: List/array of features.
# 
# 2.  **Understand Quadrotor Dynamics (2D Planar):**
#     *   State: $x = [p_x, p_z, \theta, v_x, v_z, \omega]$.
#     *   Inputs: $u_1, u_2$ (thrusts).
#     *   Physics:
#         *   Total thrust $T = u_1 + u_2$.
#         *   Moment $M = (u_1 - u_2) \cdot L$ (assuming differential thrust creates torque around the yaw axis, but in 2D planar usually we care about pitch/roll affecting $p_z$ and $p_x$). Wait, standard quadrotor dynamics:
#             *   $m \dot{v}_x = T \sin(\theta) + F_{drag} + \dots$
#             *   $m \dot{v}_z = T \cos(\theta) - mg + \dots$
#             *   $I_{yy} \dot{\omega} = T_{roll} \cdot L + \dots$
#             *   Actually, for a planar quadrotor (often modeled as a helicopter or a quadrotor moving in X-Z plane), the thrust vector is tilted by $\theta$.
#             *   $F_x = T \sin(\theta)$
#             *   $F_z = T \cos(\theta) - mg$
#             *   Torque $\tau = (u_1 - u_2) L$ (assuming $u_1, u_2$ are individual rotor thrusts).
#             *   Wait, standard quadrotor control usually sums thrusts for vertical lift and differences for torque.
#             *   Let's check the state definition: $p_x, p_z, \theta, v_x, v_z, \omega$.
#             *   Derivatives needed: $\dot{p}_x, \dot{p}_z, \dot{\theta}, \dot{v}_x, \dot{v}_z, \dot{\omega}$.
#     *   Features needed to reconstruct these derivatives:
#         *   $\dot{p}_x = v_x$
#         *   $\dot{p}_z = v_z$
#         *   $\dot{\theta} = \omega$
#         *   $\dot{v}_x = \frac{1}{m} (T \sin(\theta))$
#         *   $\dot{v}_z = \frac{1}{m} (T \cos(\theta) - mg)$
#         *   $\dot{\omega} = \frac{1}{I_{yy}} (u_1 - u_2) L$ (assuming $u_1, u_2$ are individual thrusts).
#     *   However, SINDy features are usually nonlinear combinations of states and inputs to allow the sparse regression to find the coefficients.
#     *   The prompt asks for "nonlinear features rich enough for the regression to recover each state derivative".
#     *   It explicitly mentions "thrust-projection sin/cos couplings included".
#     *   It asks for roughly 10 to 30 features.
# 
# 3.  **Constructing the Feature Set:**
#     *   **Linear terms:** $x_i$, $u_i$.
#     *   **Nonlinear terms:** $x_i^2$, $x_i x_j$, $\sin(x_i)$, $\cos(x_i)$.
#     *   **Specifics for Quadrotor:**
#         *   $\sin(\theta)$, $\cos(\theta)$ are crucial for thrust projection.
#         *   $T = u_1 + u_2$.
#         *   $T \sin(\theta)$, $T \cos(\theta)$.
#         *   $(u_1 - u_2)$ for torque.
#         *   Cross terms like $v_x v_z$, $\omega^2$, etc., might be needed if drag or coupling exists, but the prompt says "dynamics are withheld on purpose -- lean on your own understanding". This implies I should provide a standard set that *could* cover the physics if the true dynamics are simple, or a standard SINDy basis.
#         *   Standard SINDy basis often includes:
#             *   States: $x$
#             *   Squares: $x^2$
#             *   Products: $x_i x_j$
#             *   Trigonometric: $\sin(x)$, $\cos(x)$
#             *   Inputs: $u$
#             *   Input products: $u_i u_j$
#             *   Mixed: $x_i u_j$
#     *   **Constraint:** "thrust-projection sin/cos couplings included". This specifically means terms involving $\sin(\theta)$ and $\cos(\theta)$ multiplied by thrusts.
#     *   **Constraint:** "no other imports".
#     *   **Constraint:** "Output only that function inside a single python code fence".
# 
# 4.  **Drafting the Function:**
#     *   Input: `xu` (list/array of 8 floats).
#     *   Extract: `px, pz, theta, vx, vz, omega, u1, u2`.
#     *   Constants: `m, Iyy, L, g`.
#     *   Generate features.
#     *   Return list.
# 
#     *   *Feature List Idea:*
#         1.  $p_x$
#         2.  $p_z$
#         3.  $\theta$
#         4.  $v_x$
#         5.  $v_z$
#         6.  $\omega$
#         7.  $u_1$
#         8.  $u_2$
#         9.  $T = u_1 + u_2$
#         10. $\sin(\theta)$
#         11. $\cos(\theta)$
#         12. $T \sin(\theta)$
#         13. $T \cos(\theta)$
#         14. $(u_1 - u_2)$
#         15. $(u_1 - u_2) \sin(\theta)$
#         16. $(u_1 - u_2) \cos(\theta)$
#         17. $v_x^2$
#         18. $v_z^2$
#         19. $\omega^2$
#         20. $p_x^2$
#         21. $p_z^2$
#         22. $v_x v_z$
#         23. $v_x \omega$
#         24. $v_z \omega$
#         25. $u_1 u_2$
#         26. $u_1^2$
#         27. $u_2^2$
#         28. $g$ (constant)
#         29. $m$ (constant)
#         30. $I_{yy}$ (constant)
#         31. $L$ (constant)
#         32. $g/m$
#         33. $L/I_{yy}$
#         34. $g \sin(\theta)$
#         35. $g \cos(\theta)$
#         36. $v_x \sin(\theta)$
#         37. $v_z \cos(\theta)$
#         38. $v_x \cos(\theta)$
#         39. $v_z \sin(\theta)$
#         40. $\theta^2$
#         41.
# --- extracted code ---
Thinking Process:

1.  **Analyze the Request:**
    *   **Goal:** Create a Python function `basis` for Sparse Identification of Nonlinear Dynamics (SINDy) / STLSQ.
    *   **System:** 2D planar quadrotor.
    *   **State Vector ($x$):** $[p_x, p_z, \theta, v_x, v_z, \omega]$ (6 entries).
    *   **Control Vector ($u$):** $[u_1, u_2]$ (2 rotor thrusts).
    *   **Input Argument:** `xu` of length 8. `xu[0:6]` are states, `xu[6:8]` are controls.
    *   **Constants:** $m=1.0$, $I_{yy}=0.01$, $L=0.25$, $g=9.81$.
    *   **Task:** Return a list/array of roughly 10-30 nonlinear features.
    *   **Requirements:**
        *   Must be rich enough to recover state derivatives.
        *   Must include thrust-projection sin/cos couplings.
        *   Must use only `numpy` (as `np`) and `math`.
        *   No other imports.
        *   No output before or after the code fence.
        *   Function name: `basis`.
        *   Argument: `xu` (positional, length 8).
        *   Return: List/array of features.

2.  **Understand Quadrotor Dynamics (2D Planar):**
    *   State: $x = [p_x, p_z, \theta, v_x, v_z, \omega]$.
    *   Inputs: $u_1, u_2$ (thrusts).
    *   Physics:
        *   Total thrust $T = u_1 + u_2$.
        *   Moment $M = (u_1 - u_2) \cdot L$ (assuming differential thrust creates torque around the yaw axis, but in 2D planar usually we care about pitch/roll affecting $p_z$ and $p_x$). Wait, standard quadrotor dynamics:
            *   $m \dot{v}_x = T \sin(\theta) + F_{drag} + \dots$
            *   $m \dot{v}_z = T \cos(\theta) - mg + \dots$
            *   $I_{yy} \dot{\omega} = T_{roll} \cdot L + \dots$
            *   Actually, for a planar quadrotor (often modeled as a helicopter or a quadrotor moving in X-Z plane), the thrust vector is tilted by $\theta$.
            *   $F_x = T \sin(\theta)$
            *   $F_z = T \cos(\theta) - mg$
            *   Torque $\tau = (u_1 - u_2) L$ (assuming $u_1, u_2$ are individual rotor thrusts).
            *   Wait, standard quadrotor control usually sums thrusts for vertical lift and differences for torque.
            *   Let's check the state definition: $p_x, p_z, \theta, v_x, v_z, \omega$.
            *   Derivatives needed: $\dot{p}_x, \dot{p}_z, \dot{\theta}, \dot{v}_x, \dot{v}_z, \dot{\omega}$.
    *   Features needed to reconstruct these derivatives:
        *   $\dot{p}_x = v_x$
        *   $\dot{p}_z = v_z$
        *   $\dot{\theta} = \omega$
        *   $\dot{v}_x = \frac{1}{m} (T \sin(\theta))$
        *   $\dot{v}_z = \frac{1}{m} (T \cos(\theta) - mg)$
        *   $\dot{\omega} = \frac{1}{I_{yy}} (u_1 - u_2) L$ (assuming $u_1, u_2$ are individual thrusts).
    *   However, SINDy features are usually nonlinear combinations of states and inputs to allow the sparse regression to find the coefficients.
    *   The prompt asks for "nonlinear features rich enough for the regression to recover each state derivative".
    *   It explicitly mentions "thrust-projection sin/cos couplings included".
    *   It asks for roughly 10 to 30 features.

3.  **Constructing the Feature Set:**
    *   **Linear terms:** $x_i$, $u_i$.
    *   **Nonlinear terms:** $x_i^2$, $x_i x_j$, $\sin(x_i)$, $\cos(x_i)$.
    *   **Specifics for Quadrotor:**
        *   $\sin(\theta)$, $\cos(\theta)$ are crucial for thrust projection.
        *   $T = u_1 + u_2$.
        *   $T \sin(\theta)$, $T \cos(\theta)$.
        *   $(u_1 - u_2)$ for torque.
        *   Cross terms like $v_x v_z$, $\omega^2$, etc., might be needed if drag or coupling exists, but the prompt says "dynamics are withheld on purpose -- lean on your own understanding". This implies I should provide a standard set that *could* cover the physics if the true dynamics are simple, or a standard SINDy basis.
        *   Standard SINDy basis often includes:
            *   States: $x$
            *   Squares: $x^2$
            *   Products: $x_i x_j$
            *   Trigonometric: $\sin(x)$, $\cos(x)$
            *   Inputs: $u$
            *   Input products: $u_i u_j$
            *   Mixed: $x_i u_j$
    *   **Constraint:** "thrust-projection sin/cos couplings included". This specifically means terms involving $\sin(\theta)$ and $\cos(\theta)$ multiplied by thrusts.
    *   **Constraint:** "no other imports".
    *   **Constraint:** "Output only that function inside a single python code fence".

4.  **Drafting the Function:**
    *   Input: `xu` (list/array of 8 floats).
    *   Extract: `px, pz, theta, vx, vz, omega, u1, u2`.
    *   Constants: `m, Iyy, L, g`.
    *   Generate features.
    *   Return list.

    *   *Feature List Idea:*
        1.  $p_x$
        2.  $p_z$
        3.  $\theta$
        4.  $v_x$
        5.  $v_z$
        6.  $\omega$
        7.  $u_1$
        8.  $u_2$
        9.  $T = u_1 + u_2$
        10. $\sin(\theta)$
        11. $\cos(\theta)$
        12. $T \sin(\theta)$
        13. $T \cos(\theta)$
        14. $(u_1 - u_2)$
        15. $(u_1 - u_2) \sin(\theta)$
        16. $(u_1 - u_2) \cos(\theta)$
        17. $v_x^2$
        18. $v_z^2$
        19. $\omega^2$
        20. $p_x^2$
        21. $p_z^2$
        22. $v_x v_z$
        23. $v_x \omega$
        24. $v_z \omega$
        25. $u_1 u_2$
        26. $u_1^2$
        27. $u_2^2$
        28. $g$ (constant)
        29. $m$ (constant)
        30. $I_{yy}$ (constant)
        31. $L$ (constant)
        32. $g/m$
        33. $L/I_{yy}$
        34. $g \sin(\theta)$
        35. $g \cos(\theta)$
        36. $v_x \sin(\theta)$
        37. $v_z \cos(\theta)$
        38. $v_x \cos(\theta)$
        39. $v_z \sin(\theta)$
        40. $\theta^2$
        41.

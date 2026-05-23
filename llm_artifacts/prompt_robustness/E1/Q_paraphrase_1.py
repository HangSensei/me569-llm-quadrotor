"""Q E1 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Task:** Create a Python function named `basis` for a SINDy (Sparse Identification of Nonlinear Dynamics) feature library.
#     *   **System:** Planar Quadrotor.
#     *   **State ($x$):** 6D vector $[p_x, p_z, \theta, v_x, v_z, \omega]$.
#     *   **Control ($u$):** 2D vector $[u_1, u_2]$ (thrusts).
#     *   **Input:** `xu`, a length-8 array where `xu[0:6]` is state, `xu[6]` is $u_1$, `xu[7]` is $u_2$.
#     *   **Output:** A list or numpy array of scalar features (10 to 30 features).
#     *   **Constraints:**
#         *   Must reconstruct 6 derivative equations (for $p_x, p_z, \theta, v_x, v_z, \omega$).
#         *   Must include sin/cos terms coupling thrust and pitch (since quadrotors have thrust vectoring).
#         *   Known parameters: $m=1.0$, $I_{yy}=0.01$, $L=0.25$, $g=9.81$.
#         *   Physics knowledge: Gravity, thrust projection, differential thrust torque.
#         *   No imports other than `np` (which is preinjected).
#         *   No indexing beyond `xu[7]`.
#         *   Return type: Python list or numpy array.
#         *   Output format: Single python code fence with function definition only.
#         *   Comments allowed inside the body.
# 
# 2.  **Physics of Planar Quadrotor:**
#     *   **Forces:**
#         *   Total thrust $T = u_1 + u_2$.
#         *   Gravity $mg$.
#         *   Vertical force: $T - mg$.
#         *   Horizontal force: Depends on pitch angle $\theta$. Usually, thrust vectoring is modeled as $T \sin(\theta)$ for horizontal force and $T \cos(\theta)$ for vertical force (assuming small angles or specific geometry). However, the prompt mentions "sin/cos terms coupling thrust and pitch".
#         *   Standard planar quadrotor dynamics (often simplified):
#             *   $\dot{v}_x = \frac{1}{m} (T \sin(\theta) + \text{drag})$
#             *   $\dot{v}_z = \frac{1}{m} (T \cos(\theta) - mg) + \text{drag}$
#             *   $\dot{\omega} = \frac{1}{I_{yy}} (T_1 - T_2) \cdot \text{lever arm}$ (Torque).
#             *   $\dot{\theta} = \omega$.
#     *   **Torque:**
#         *   Differential thrust creates torque. $T_1$ and $T_2$ are applied at distance $L/2$ (or similar) from the center of mass.
#         *   Torque $\tau = (u_1 - u_2) \cdot L$.
#         *   $\dot{\omega} = \frac{\tau}{I_{yy}} = \frac{L(u_1 - u_2)}{I_{yy}}$.
#     *   **Coupling:**
#         *   The prompt specifically asks for "sin/cos terms coupling thrust and pitch". This implies the horizontal and vertical forces depend on $\sin(\theta)$ and $\cos(\theta)$.
#         *   So, features should include $u_1, u_2, \sin(\theta), \cos(\theta), u_1 \sin(\theta), u_2 \sin(\theta), u_1 \cos(\theta), u_2 \cos(\theta)$, etc.
#         *   Also need polynomial terms of state ($p_x, p_z, \theta, v_x, v_z, \omega$) and their products to capture nonlinearities (e.g., $v_x^2$, $\omega^2$, $p_x v_x$).
#         *   Need to ensure the count is between 10 and 30.
# 
# 3.  **Feature Selection Strategy:**
#     *   **State terms:** $p_x, p_z, \theta, v_x, v_z, \omega$.
#     *   **Control terms:** $u_1, u_2$.
#     *   **Trigonometric terms:** $\sin(\theta), \cos(\theta)$.
#     *   **Coupled terms:** $u_1 \sin(\theta), u_2 \sin(\theta), u_1 \cos(\theta), u_2 \cos(\theta)$.
#     *   **Squared terms:** $v_x^2, v_z^2, \omega^2, \theta^2$.
#     *   **Cross terms:** $v_x \omega, v_z \omega$.
#     *   **Gravity:** $g$ is constant, but $mg$ is a constant force term.
#     *   **Derivatives:**
#         *   $\dot{p}_x = v_x$
#         *   $\dot{p}_z = v_z$
#         *   $\dot{\theta} = \omega$
#         *   $\dot{v}_x = \frac{1}{m} (T \sin(\theta))$
#         *   $\dot{v}_z = \frac{1}{m} (T \cos(\theta) - mg)$
#         *   $\dot{\omega} = \frac{L}{I_{yy}} (u_1 - u_2)$
#     *   To allow SINDy to find these, the basis must contain the terms that appear on the RHS.
#     *   RHS terms needed:
#         *   $v_x$ (for $\dot{p}_x$)
#         *   $v_z$ (for $\dot{p}_z$)
#         *   $\omega$ (for $\dot{\theta}$)
#         *   $u_1 \sin(\theta) + u_2 \sin(\theta)$ (for $\dot{v}_x$) -> $u_1 \sin(\theta), u_2 \sin(\theta)$
#         *   $u_1 \cos(\theta) + u_2 \cos(\theta)$ (for $\dot{v}_z$) -> $u_1 \cos(\theta), u_2 \cos(\theta)$
#         *   $-mg$ (for $\dot{v}_z$) -> constant term
#         *   $u_1 - u_2$ (for $\dot{\omega}$) -> $u_1, u_2$
#     *   To be safe and robust for SINDy (which often needs polynomial basis for stability or higher order terms), I should include polynomial terms of state and control.
#     *   Constraint: 10 to 30 features.
#     *   Constraint: No imports other than `np`.
#     *   Constraint: `xu` is length 8.
# 
# 4.  **Drafting the Features:**
#     1.  $xu[0]$ ($p_x$)
#     2.  $xu[1]$ ($p_z$)
#     3.  $xu[2]$ ($\theta$)
#     4.  $xu[3]$ ($v_x$)
#     5.  $xu[4]$ ($v_z$)
#     6.  $xu[5]$ ($\omega$)
#     7.  $xu[6]$ ($u_1$)
#     8.  $xu[7]$ ($u_2$)
#     9.  $\sin(xu[2])$
#     10. $\cos(xu[2])$
#     11. $xu[3] * xu[5]$ ($v_x \omega$)
#     12. $xu[4] * xu[5]$ ($v_z \omega$)
#     13. $xu[3] * xu[3]$ ($v_x^2$)
#     14. $xu[4] * xu[4]$ ($v_z^2$)
#     15. $xu[5] * xu[5]$ ($\omega^2$)
#     16. $xu[6] * \sin(xu[2])$
#     17. $xu[7] * \sin(xu[2])$
#     18. $xu[6] * \cos(xu[2])$
#     19. $xu[7] * \cos(xu[2])$
#     20. $xu[6] + xu[7]$ (Total thrust)
#     21. $xu[6] - xu[7]$ (Thrust diff)
#     22. $xu[2] * xu[5]$ ($\theta \omega$)
# --- extracted code ---
Thinking Process:

1.  **Analyze the Request:**
    *   **Task:** Create a Python function named `basis` for a SINDy (Sparse Identification of Nonlinear Dynamics) feature library.
    *   **System:** Planar Quadrotor.
    *   **State ($x$):** 6D vector $[p_x, p_z, \theta, v_x, v_z, \omega]$.
    *   **Control ($u$):** 2D vector $[u_1, u_2]$ (thrusts).
    *   **Input:** `xu`, a length-8 array where `xu[0:6]` is state, `xu[6]` is $u_1$, `xu[7]` is $u_2$.
    *   **Output:** A list or numpy array of scalar features (10 to 30 features).
    *   **Constraints:**
        *   Must reconstruct 6 derivative equations (for $p_x, p_z, \theta, v_x, v_z, \omega$).
        *   Must include sin/cos terms coupling thrust and pitch (since quadrotors have thrust vectoring).
        *   Known parameters: $m=1.0$, $I_{yy}=0.01$, $L=0.25$, $g=9.81$.
        *   Physics knowledge: Gravity, thrust projection, differential thrust torque.
        *   No imports other than `np` (which is preinjected).
        *   No indexing beyond `xu[7]`.
        *   Return type: Python list or numpy array.
        *   Output format: Single python code fence with function definition only.
        *   Comments allowed inside the body.

2.  **Physics of Planar Quadrotor:**
    *   **Forces:**
        *   Total thrust $T = u_1 + u_2$.
        *   Gravity $mg$.
        *   Vertical force: $T - mg$.
        *   Horizontal force: Depends on pitch angle $\theta$. Usually, thrust vectoring is modeled as $T \sin(\theta)$ for horizontal force and $T \cos(\theta)$ for vertical force (assuming small angles or specific geometry). However, the prompt mentions "sin/cos terms coupling thrust and pitch".
        *   Standard planar quadrotor dynamics (often simplified):
            *   $\dot{v}_x = \frac{1}{m} (T \sin(\theta) + \text{drag})$
            *   $\dot{v}_z = \frac{1}{m} (T \cos(\theta) - mg) + \text{drag}$
            *   $\dot{\omega} = \frac{1}{I_{yy}} (T_1 - T_2) \cdot \text{lever arm}$ (Torque).
            *   $\dot{\theta} = \omega$.
    *   **Torque:**
        *   Differential thrust creates torque. $T_1$ and $T_2$ are applied at distance $L/2$ (or similar) from the center of mass.
        *   Torque $\tau = (u_1 - u_2) \cdot L$.
        *   $\dot{\omega} = \frac{\tau}{I_{yy}} = \frac{L(u_1 - u_2)}{I_{yy}}$.
    *   **Coupling:**
        *   The prompt specifically asks for "sin/cos terms coupling thrust and pitch". This implies the horizontal and vertical forces depend on $\sin(\theta)$ and $\cos(\theta)$.
        *   So, features should include $u_1, u_2, \sin(\theta), \cos(\theta), u_1 \sin(\theta), u_2 \sin(\theta), u_1 \cos(\theta), u_2 \cos(\theta)$, etc.
        *   Also need polynomial terms of state ($p_x, p_z, \theta, v_x, v_z, \omega$) and their products to capture nonlinearities (e.g., $v_x^2$, $\omega^2$, $p_x v_x$).
        *   Need to ensure the count is between 10 and 30.

3.  **Feature Selection Strategy:**
    *   **State terms:** $p_x, p_z, \theta, v_x, v_z, \omega$.
    *   **Control terms:** $u_1, u_2$.
    *   **Trigonometric terms:** $\sin(\theta), \cos(\theta)$.
    *   **Coupled terms:** $u_1 \sin(\theta), u_2 \sin(\theta), u_1 \cos(\theta), u_2 \cos(\theta)$.
    *   **Squared terms:** $v_x^2, v_z^2, \omega^2, \theta^2$.
    *   **Cross terms:** $v_x \omega, v_z \omega$.
    *   **Gravity:** $g$ is constant, but $mg$ is a constant force term.
    *   **Derivatives:**
        *   $\dot{p}_x = v_x$
        *   $\dot{p}_z = v_z$
        *   $\dot{\theta} = \omega$
        *   $\dot{v}_x = \frac{1}{m} (T \sin(\theta))$
        *   $\dot{v}_z = \frac{1}{m} (T \cos(\theta) - mg)$
        *   $\dot{\omega} = \frac{L}{I_{yy}} (u_1 - u_2)$
    *   To allow SINDy to find these, the basis must contain the terms that appear on the RHS.
    *   RHS terms needed:
        *   $v_x$ (for $\dot{p}_x$)
        *   $v_z$ (for $\dot{p}_z$)
        *   $\omega$ (for $\dot{\theta}$)
        *   $u_1 \sin(\theta) + u_2 \sin(\theta)$ (for $\dot{v}_x$) -> $u_1 \sin(\theta), u_2 \sin(\theta)$
        *   $u_1 \cos(\theta) + u_2 \cos(\theta)$ (for $\dot{v}_z$) -> $u_1 \cos(\theta), u_2 \cos(\theta)$
        *   $-mg$ (for $\dot{v}_z$) -> constant term
        *   $u_1 - u_2$ (for $\dot{\omega}$) -> $u_1, u_2$
    *   To be safe and robust for SINDy (which often needs polynomial basis for stability or higher order terms), I should include polynomial terms of state and control.
    *   Constraint: 10 to 30 features.
    *   Constraint: No imports other than `np`.
    *   Constraint: `xu` is length 8.

4.  **Drafting the Features:**
    1.  $xu[0]$ ($p_x$)
    2.  $xu[1]$ ($p_z$)
    3.  $xu[2]$ ($\theta$)
    4.  $xu[3]$ ($v_x$)
    5.  $xu[4]$ ($v_z$)
    6.  $xu[5]$ ($\omega$)
    7.  $xu[6]$ ($u_1$)
    8.  $xu[7]$ ($u_2$)
    9.  $\sin(xu[2])$
    10. $\cos(xu[2])$
    11. $xu[3] * xu[5]$ ($v_x \omega$)
    12. $xu[4] * xu[5]$ ($v_z \omega$)
    13. $xu[3] * xu[3]$ ($v_x^2$)
    14. $xu[4] * xu[4]$ ($v_z^2$)
    15. $xu[5] * xu[5]$ ($\omega^2$)
    16. $xu[6] * \sin(xu[2])$
    17. $xu[7] * \sin(xu[2])$
    18. $xu[6] * \cos(xu[2])$
    19. $xu[7] * \cos(xu[2])$
    20. $xu[6] + xu[7]$ (Total thrust)
    21. $xu[6] - xu[7]$ (Thrust diff)
    22. $xu[2] * xu[5]$ ($\theta \omega$)

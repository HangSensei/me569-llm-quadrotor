"""Q E4 paraphrase 4. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# Thinking Process:
# 
# 1.  **Analyze the Request:**
#     *   **Goal:** Write a per-step shaping reward function for a PPO agent to hover a planar quadrotor.
#     *   **State:** `[p_x, p_z, theta, v_x, v_z, omega]`
#     *   **Action:** `[u_1, u_2]` (rotor thrusts in Newtons)
#     *   **Constants:** `m = 1.0`, `I_yy = 0.01`, `L = 0.25`, `g = 9.81`.
#     *   **Hover Target:** Origin (`p_x=0, p_z=0`) and roughly `4.905 N` per rotor (which implies `theta=0` for hover stability in this coordinate system, or rather, the net force should balance gravity).
#     *   **Return:** One finite float (bigger is better, penalties negative).
#     *   **Constraints:** Deterministic, side-effect free, imports only `numpy` (`np`) and `math`.
#     *   **Priorities:** Position and attitude error > Velocity error.
#     *   **Specifics:** Add penalty for straying from hover thrust, prefer smooth differentiable expressions (no hard thresholds).
#     *   **Output:** Only the function in a single python code fence.
#     *   **Note:** Crash penalties, survival bonus, episode termination are handled by the harness (do not include them).
# 
# 2.  **Determine the Reward Components:**
#     *   **Position Error:** The quadrotor should stay at `p_x = 0` and `p_z = 0`.
#         *   Term: `- (p_x^2 + p_z^2)` or similar.
#     *   **Attitude Error:** The quadrotor should maintain a specific orientation. In the provided dynamics:
#         *   `v_x_dot = -(u_1 + u_2) * np.sin(theta) / m`
#         *   `v_z_dot = (u_1 + u_2) * np.cos(theta) / m - g`
#         *   For hover, `v_x = 0`, `v_z = 0`, `omega = 0`.
#         *   To hover, `u_1 + u_2` must balance `g`.
#         *   Usually, for a planar quadrotor (x-z plane), `theta` is the roll/pitch angle. If `theta = 0`, the thrust vector points straight up (positive z).
#         *   So, `theta` should be close to 0.
#         *   Term: `- theta^2`.
#     *   **Velocity Error:** The prompt says "Prioritize position and attitude error over velocity error". So velocity terms should be small or secondary.
#         *   Term: `- (v_x^2 + v_z^2)`.
#     *   **Thrust Penalty:** "Add a penalty for straying from hover thrust".
#         *   Hover thrust per rotor is roughly `4.905 N`.
#         *   Total thrust `T = u_1 + u_2`.
#         *   Hover total thrust `T_hover = m * g = 1.0 * 9.81 = 9.81 N`.
#         *   So `u_1 + u_2` should be around `9.81`.
#         *   The prompt says "roughly 4.905 N per rotor". This implies `u_1 ≈ u_2 ≈ 4.905`.
#         *   Penalty: `- ( (u_1 - 4.905)^2 + (u_2 - 4.905)^2 )` or `- ( (u_1 + u_2) - 9.81 )^2`.
#         *   Since it's a shaping reward, penalizing deviation from the expected hover thrust helps stabilize the control policy.
#     *   **Smoothness:** Use squared errors (quadratic) instead of absolute values or step functions.
# 
# 3.  **Drafting the Function:**
# 
#     ```python
#     import numpy as np
#     import math
# 
#     def reward(state, action):
#         p_x, p_z, theta, v_x, v_z, omega = state
#         u_1, u_2 = action
# 
#         # Constants
#         m = 1.0
#         g = 9.81
#         hover_thrust_per_rotor = 4.905
#         hover_total_thrust = m * g # 9.81
# 
#         # Position error (prioritized)
#         pos_error = p_x**2 + p_z**2
# 
#         # Attitude error (theta should be 0 for planar hover)
#         att_error = theta**2
# 
#         # Velocity error (lower priority)
#         vel_error = v_x**2 + v_z**2
# 
#         # Thrust penalty (prefer smooth)
#         # Target total thrust is m*g. Target per rotor is m*g/2.
#         thrust_deviation = (u_1 - hover_thrust_per_rotor)**2 + (u_2 - hover_thrust_per_rotor)**2
# 
#         # Combine
#         # Weights need to be tuned implicitly or explicitly.
#         # Since it's shaping, we want to encourage the behavior.
#         # Let's use standard weights.
#         # Position: 1.0
#         # Attitude: 1.0
#         # Velocity: 0.1
#         # Thrust: 0.1
# 
#         r = -pos_error - att_error - vel_error - thrust_deviation
#         return r
#     ```
# 
# 4.  **Refining the Logic:**
#     *   **Coordinate System:** The dynamics show `v_z_dot = (u_1 + u_2) * np.cos(theta) / m - g`.
#     *   If `theta = 0`, `cos(0) = 1`, `v_z_dot = (u_1+u_2)/m - g`. To hover, `u_1+u_2 = m*g`.
#     *   If `theta != 0`, `cos(theta) < 1`, so total thrust needed increases to maintain hover, or `v_z` changes.
#     *   The prompt says "hover at the origin". This implies `p_x=0, p_z=0`.
#     *   The prompt says "roughly 4.905 N per rotor". `m*g/2 = 9.81/2 = 4.905`.
#     *   So `u_1` and `u_2` should ideally be close to `4.905`.
#     *   The prompt asks to "Prioritize position and attitude error over velocity error".
#     *   "Add a penalty for straying from hover thrust".
#     *   "Prefer smooth differentiable expressions".
#     *   "Return only the function".
# 
#     Let's check the weights. If I just sum them, the scale matters.
#     `p_x` is in meters. `theta` is in radians. `v` is in m/s. `u` is in Newtons.
#     To make them comparable, I should normalize or use reasonable weights.
#     However, often in these tasks, the weights are tuned during training. Since I need to write the function, I should pick reasonable weights that reflect the priorities.
#     Position error (m^2) vs Attitude (rad^2) vs Velocity (m^2/s^2) vs Thrust (N^2).
#     Usually, position is the most important.
#     Let's use a standard structure:
#     `reward = - (w_pos * pos_err + w_att * att_err + w_vel * vel_err + w_thrust * thrust_err)`
#     Since I can't tune hyperparameters in the prompt response, I'll use weights that make sense for the physics.
#     `pos_err`: `p_x^2 + p_z^2`.
#     `att_err`: `theta^2`.
#     `vel_err`: `v_x^2 + v_z^2`.
#     `thrust_err`: `(u_1 - 4.905)^2 + (u_2 - 4.905)^2`.
# 
#     Wait, `theta` is in radians. `p_x` is in meters.
#     If `p_x` is 1m, `p_x^2` is 1.
#     If `theta` is 0.1 rad, `theta^2` is 0.01.
#     If `v` is 1 m/s, `v^2` is 1.
#     If `u` is 5 N, `u^2` is 25.
#     The scales are very different.
#     However, the prompt asks for "shaping reward". Shaping rewards often use specific weights to guide the agent.
#     I will use weights that reflect the priority: Position > Attitude > Velocity. Thrust is a constraint/regularization.
#     Let's try to normalize implicitly or just use standard squared errors.
#     Actually, to ensure "bigger is better", I should return negative values for penalties.
#     So `reward = - (pos_err + att_err + vel_err + thrust_err)`.
#     To make it robust, I might add a small positive constant or just rely on the negative penalties.
#     The prompt says "bigger is better, so penalties are negative".
#     So `return - ( ... )` is correct.
# 
#     Let's refine the weights.
#     Position: `p_x^2 + p_z^2`.
#     Attitude: `theta^2`.
#     Velocity: `v_x^2 + v_z^2`.
#     Thrust: `(u_1 - 4.905)^2 + (u_2 - 4.905)^2`.
# 
#     Is there a specific weight ratio needed?
#     "Prioritize position and attitude error over velocity error".
#     So `w_pos` and `w_att` should be larger than `w_vel`.
#     Let's set `w_pos = 1.0`, `w_att = 1.0`, `w_vel = 0.1`.
#     What about thrust? "Add a penalty for straying from hover thrust".
#     Let's set `w_thrust = 0.1`.
# 
#     Wait, `theta` is small. `p_x` can be small.
#     If I don't normalize, `u` terms (squared) will dominate `p` terms (squared) if `u` is around 5. `5^2 = 25`. `p` around 0.1 is `0.01`.
#     So `u` terms will dominate the reward magnitude.
#     To make it balanced, I should probably scale the thrust penalty or position/attitude.
#     However, without knowing the specific training setup, I should follow the instruction "Prioritize position and attitude error over velocity error".
#     I will use weights like `1.0` for position, `1.0` for attitude, `0.1` for velocity.
#     For thrust, since it's a constraint, maybe `0.1` is fine.
#     But to prevent the agent from just minimizing thrust error at the cost of position, I need to ensure position error is penalized heavily.
#     Actually, if I just sum them, the agent will try to minimize all.
#     Let's check the instruction again: "Prioritize position and attitude error over velocity error".
#     This implies `w_pos >= w_vel` and `w_att >= w_vel`.
#     It doesn't explicitly say `w_pos` vs `w_att`, but usually they are similar.
#     I will use `1.0` for position, `1.0` for attitude, `0.1` for velocity.
#     For thrust, I'll use `0.1` as well, but maybe slightly higher to ensure stability? No, "penalty for straying".
#     Let's stick to `1.0` for pos/att, `0.1` for vel/thrust.
#     Wait, `u` is in Newtons. `p` is in meters.
#     If `p` is 1m, penalty is 1.
#     If `u` is 5N, penalty is 25.
#     If the agent hovers, `u` is 5, penalty is 0.
#     If the agent drifts 1m, penalty is 1.
#     So `u` penalty is much larger.
#     To fix this, I should scale the thrust penalty or position/attitude.
#     Usually, in these environments, the reward is scaled.
#     But I cannot change the environment. I must write the reward function.
#     If I write `reward = - (pos + att + 0.1*vel + 0.1*thrust)`, the thrust term will dominate because `u` is around 5.
#     To make it balanced, I should divide the thrust term by `m*g` or something?
#     Or just use `0.01` for thrust?
#     Let's look at the instruction: "Add a penalty for straying from hover thrust".
#     It doesn't specify the weight.
#     However, "Prioritize position and attitude error over velocity error".
#     It implies Position/Attitude > Velocity.
#     It doesn't explicitly say Position/Attitude > Thrust.
#     But Thrust is a control input, not a state.
#     If I want the agent to hover, it needs to balance thrust.
#     If I penalize thrust deviation too much, it might fight the dynamics.
#     Let's assume the weights should be tuned to make the terms comparable in magnitude.
#     `p_x` ~ 0.1m -> `p_x^2` = 0.01.
#     `theta` ~ 0.1rad -> `theta^2` = 0.01.
#     `v` ~ 0.1m/s -> `v^2` = 0.01.
#     `u` ~ 5N -> `u^2` = 25.
#     So `u` is 1000x larger.
#     I should probably scale the thrust penalty by `1/(m*g)^2` or similar, or just use a small weight.
#     Let's use `0.01` for thrust penalty to make it comparable to position.
#     Or better, normalize the thrust error by `(m*g)^2`.
#     `thrust_err = ((u_1 - 4.905)^2 + (u_2 - 4.905)^2) / (4.905^2)`.
#     This makes the thrust error dimensionless and comparable to position/attitude.
#     Let's do that.
# 
#     Wait, `4.905` is `m*g/2`.
#     So `u_1 + u_2` should be `m*g`.
#     `u_1` and `u_2` should be `m*g/2`.
#     So `thrust_err = ((u_1 - m*g/2)^2 + (u_2 - m*g/2)^2) / (m*g/2)^2`.
#     This ensures that if `u` is correct, error is 0. If `u` is 0, error is 1.
#     This seems reasonable.
# 
#     Wait, `p_x` and `p_z` are in meters.
#     If I want `p_x` error to be comparable to `u` error (normalized), I need to decide on the scale of `p_x`.
#     Usually, `p_x` is small.
#     Let's just use `1.0` for position, `1.0` for attitude, `0.1` for velocity.
#     For thrust, I will use `0.01` * normalized error.
#     Actually, simpler:
#     `reward = - (p_x**2 + p_z**2 + theta**2 + 0.1 * (v_x**2 + v_z**2) + 0.01 * ((u_1 - 4.905)**2 + (u_2 - 4.905)**2))`
#     This might be too small for thrust.
#     Let's try to make the thrust penalty significant but not dominant.
#     Given the instruction "Prioritize position and attitude error over velocity error", I will ensure `w_pos` and `w_att` are larger than `w_vel`.
#     I will set `w_pos = 1.0`, `w_att = 1.0`, `w_vel = 0.1`.
#     For thrust, I'll set `w_thrust = 0.1` but normalize it by `(m*g/2)^2` to make it comparable to position.
#     Wait, `4.905^2` is ~24.
#     So `0.1 * 24 = 2.4`.
#     If `p_x` is 1m, `p_x^2` is 1.
#     So `u` error is still larger.
#     Let's just use `0.01` for thrust weight without normalization, or normalize.
#     Normalization is safer for "smooth differentiable expressions".
#     Let's normalize the thrust error by `(m*g/2)^2`.
#     `thrust_norm = ((u_1 - 4.905)**2 + (u_2 - 4.905)**2) / (4.905**2)`.
#     Then `reward = - (pos + att + 0.1*vel + 0.1*thrust_norm)`.
#     This makes `thrust_norm` range 0 to infinity.
#     If `u` is 0, `thrust_norm` is 1.
#     If `u` is 5, `thrust_norm` is 0.
#     So `0.1 * 1 = 0.1`.
#     `pos` is 1.
#     So Position dominates. This fits "Prioritize position...".
#     This seems like a good balance.
# 
#     Wait, `theta` is in radians. `p_x` is in meters.
#     If `p_x` is 0.1m, `pos` is 0.01.
#     If `theta` is 0.1rad, `att` is 0.01.
#     If `v` is 0.1m/s, `vel
# --- extracted code ---
import numpy as np
    import math

    def reward(state, action):
        p_x, p_z, theta, v_x, v_z, omega = state
        u_1, u_2 = action

        # Constants
        m = 1.0
        g = 9.81
        hover_thrust_per_rotor = 4.905
        hover_total_thrust = m * g # 9.81

        # Position error (prioritized)
        pos_error = p_x**2 + p_z**2

        # Attitude error (theta should be 0 for planar hover)
        att_error = theta**2

        # Velocity error (lower priority)
        vel_error = v_x**2 + v_z**2

        # Thrust penalty (prefer smooth)
        # Target total thrust is m*g. Target per rotor is m*g/2.
        thrust_deviation = (u_1 - hover_thrust_per_rotor)**2 + (u_2 - hover_thrust_per_rotor)**2

        # Combine
        # Weights need to be tuned implicitly or explicitly.
        # Since it's shaping, we want to encourage the behavior.
        # Let's use standard weights.
        # Position: 1.0
        # Attitude: 1.0
        # Velocity: 0.1
        # Thrust: 0.1

        r = -pos_error - att_error - vel_error - thrust_deviation
        return r

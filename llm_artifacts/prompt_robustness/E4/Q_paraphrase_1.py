"""Q E4 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# The user wants a dense per-step reward function for PPO-based hover control of a Planar Quadrotor.
# The observation is state = [p_x, p_z, theta, v_x, v_z, omega] in R^6.
# The action is action = [u_1, u_2] in R^2 (rotor thrusts, Newtons).
# Constants: m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81.
# Hover is state = 0 with both thrusts near 4.905 N.
# The agent starts disturbed and must recover to hover.
# The function must return ONE finite float, where larger is better.
# Use negative terms for penalties.
# It must be pure and deterministic.
# Import numpy (np) and math and nothing else.
# Crash penalties, survival bonus, and episode cutoff are added by the harness -- I only craft the dense shaping.
# Make position and attitude errors outweigh velocity errors.
# Penalize departure from hover thrust.
# Favor smooth differentiable terms over step functions.
# 
# Let's break down the reward components.
# 1.  **Hover Position Error**: The goal is to hover at position 0. So $p_x$ and $p_z$ should be close to 0.
#     Since position errors should outweigh velocity errors, I should use a squared error term for position.
#     $R_{pos} = - (p_x^2 + p_z^2)$.
#     Maybe scale it to make it significant.
# 2.  **Hover Attitude Error**: The goal is to hover with $\theta = 0$ (planar quadrotor usually implies $\theta$ is the tilt angle in the plane, but looking at the dynamics: $v_x\_dot = -(u_1 + u_2) * np.sin(theta) / m$. If $\theta=0$, $v_x\_dot = 0$. If $\theta \neq 0$, there is a horizontal force component. So $\theta$ should be 0 for pure hover).
#     $R_{att} = - \theta^2$.
#     Again, position and attitude errors should outweigh velocity errors.
# 3.  **Thrust Error**: Hover thrust is near 4.905 N.
#     $u_{target} = 4.905$.
#     $R_{thrust} = - (u_1 - 4.905)^2 - (u_2 - 4.905)^2$.
#     Or maybe just the sum of thrusts since they are coupled in vertical force.
#     $u_1 + u_2$ determines vertical force. $u_2 - u_1$ determines torque.
#     Ideally, $u_1 \approx u_2 \approx 4.905$.
#     So penalize deviation from 4.905 for both.
# 4.  **Velocity Penalty**: Velocity should be close to 0.
#     $R_{vel} = - (v_x^2 + v_z^2 + \omega^2)$.
#     But position/attitude errors should outweigh velocity errors. So maybe scale velocity penalty down or just include it with a smaller weight.
#     Actually, the prompt says "Make position and attitude errors outweigh velocity errors". This implies the weights for position/attitude should be higher than velocity.
# 5.  **Smoothness**: "Favor smooth differentiable terms over step functions."
#     Squared errors are smooth.
#     Maybe add a derivative of action or state to encourage smoothness?
#     Usually, in dense rewards, we just want the state to be close to target.
#     However, "favor smooth differentiable terms" might imply penalizing rapid changes in action or state.
#     Let's stick to state errors as they are smooth.
#     Maybe penalize action variance?
#     Let's focus on the state errors first.
# 6.  **Scaling**:
#     $p_x, p_z$ are in meters.
#     $\theta$ is in radians.
#     $v_x, v_z, \omega$ are in m/s, rad/s.
#     $u$ is in Newtons.
#     To make them comparable, I need to normalize or choose weights carefully.
#     Since position errors should outweigh velocity errors, I can use a large weight for position.
#     Let's define a base weight $W$.
#     $R = -W_1 (p_x^2 + p_z^2) - W_2 \theta^2 - W_3 (v_x^2 + v_z^2 + \omega^2) - W_4 ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
#     Wait, the prompt says "penalize departure from hover thrust".
#     So $W_4$ is important.
#     Also "position and attitude errors outweigh velocity errors".
#     So $W_1, W_2 \gg W_3$.
#     Let's check the units.
#     $p^2$ is $m^2$.
#     $\theta^2$ is $rad^2$ (dimensionless).
#     $v^2$ is $m^2/s^2$.
#     $u^2$ is $N^2$.
#     To make them comparable, I should probably normalize by some characteristic scales.
#     However, the prompt asks for a function, not necessarily normalized physics.
#     But for PPO, the magnitude of the reward matters for learning.
#     If I use raw values, $p_x$ might be small (0.1m) -> $0.01$.
#     $v_x$ might be small (0.1m/s) -> $0.01$.
#     $u$ is around 5N -> deviation 0.1N -> $0.01$.
#     So raw squared errors might be comparable.
#     But "position and attitude errors outweigh velocity errors" suggests I should explicitly weight them higher.
#     Let's use weights like 100 for position, 10 for attitude, 1 for velocity, 10 for thrust.
#     Wait, if I use 100 for position, a 0.1m error is 1.0 penalty.
#     If I use 1 for velocity, a 0.1m/s error is 0.01 penalty.
#     This satisfies "position ... outweigh velocity".
#     Let's refine the weights.
#     $R = -100 (p_x^2 + p_z^2) - 100 \theta^2 - 1 (v_x^2 + v_z^2 + \omega^2) - 10 ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
#     Wait, $\theta$ is small. $0.1$ rad is about 5 degrees.
#     $p_x$ is meters. $0.1$ m is 10 cm.
#     Usually position is more important than angle for hover stability.
#     Let's make position and angle weights similar or position slightly higher.
#     Let's use $W_{pos} = 1000$, $W_{att} = 100$, $W_{vel} = 1$, $W_{thrust} = 100$.
#     Wait, if I use 1000 for position, a 0.1m error is 10.
#     If I use 100 for angle, a 0.1 rad error is 1.
#     If I use 100 for thrust, a 0.1N error is 0.1.
#     This seems reasonable.
#     However, I need to ensure the function is deterministic and pure.
#     I need to import numpy.
#     I need to return ONE finite float.
#     I should avoid NaNs or Infs.
#     The prompt says "Crash penalties, survival bonus, and episode cutoff are added by the harness".
#     So I don't need to check for bounds or time.
#     Just the dense shaping.
#     "Favor smooth differentiable terms over step functions."
#     Squared errors are smooth.
#     Maybe I should add a term for action smoothness?
#     "Favor smooth differentiable terms" usually refers to the reward function itself being smooth (no Heaviside, no abs without square).
#     Squared errors are smooth.
#     Maybe penalize action change?
#     But the prompt says "dense per-step reward".
#     Usually, action smoothness is handled by regularization in the loss or by the reward.
#     Let's stick to state and action errors.
#     Let's check the constants again.
#     $m = 1.0$, $g = 9.81$.
#     Hover thrust $T = mg = 9.81$.
#     Wait, the prompt says "hover is state = 0 with both thrusts near 4.905 N".
#     $4.905 \times 2 = 9.81$.
#     So $u_1 + u_2 = 9.81$.
#     If $u_1 = u_2 = 4.905$, then $u_1 + u_2 = 9.81$.
#     So the target thrust for each is 4.905.
#     So penalizing deviation from 4.905 is correct.
#     Wait, if the quadrotor is planar, usually it's a 2-rotor setup or 4-rotor in a plane.
#     The dynamics given are:
#     $v_x\_dot = -(u_1 + u_2) * np.sin(theta) / m$
#     $v_z\_dot = (u_1 + u_2) * np.cos(theta) / m - g$
#     If $\theta = 0$, $v_x\_dot = 0$, $v_z\_dot = (u_1 + u_2)/m - g$.
#     For hover, $v_z\_dot = 0 \implies u_1 + u_2 = mg = 9.81$.
#     If $u_1 = u_2 = 4.905$, then $u_1 + u_2 = 9.81$.
#     So the target for each is 4.905.
#     So penalizing $(u_1 - 4.905)^2 + (u_2 - 4.905)^2$ is correct.
#     Wait, if $u_1 + u_2$ is the total thrust, maybe I should penalize $(u_1 + u_2 - 9.81)^2$?
#     But the prompt says "both thrusts near 4.905 N".
#     So I should penalize deviation of each from 4.905.
#     This ensures balanced thrusts too.
#     So $R_{thrust} = - ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
#     Or maybe just sum of squares.
#     Let's use weights to balance the terms.
#     Let's define a function.
#     I need to import numpy.
#     I need to make sure I don't use any other libraries.
#     I need to return a single float.
#     I should avoid `np.inf` or `np.nan` in the output.
#     The prompt says "Crash penalties... are added by the harness".
#     So I don't need to check for $p_z < 0$ or something.
#     Just the dense shaping.
#     Let's write the code.
# 
#     Wait, "Make position and attitude errors outweigh velocity errors".
#     This implies the magnitude of the position/attitude penalty should be larger than velocity penalty.
#     Let's use weights:
#     $W_{pos} = 1000$
#     $W_{att} = 100$
#     $W_{vel} = 1$
#     $W_{thrust} = 100$
#     Wait, if I use 1000 for position, a 0.1m error is 10.
#     If I use 1 for velocity, a 0.1m/s error is 0.01.
#     This satisfies the condition.
#     Let's check the thrust penalty.
#     If I use 100, a 0.1N error is 0.1.
#     This is comparable to velocity.
#     But thrust is critical for hover.
#     Maybe $W_{thrust} = 1000$ as well?
#     But the prompt says "penalize departure from hover thrust".
#     It doesn't explicitly say it should outweigh position.
#     But for hover control, thrust is the primary control.
#     However, "position and attitude errors outweigh velocity errors" is the specific instruction.
#     It doesn't say "thrust errors outweigh position errors".
#     But if thrust is wrong, position will drift.
#     So thrust error is correlated with position error.
#     Let's make thrust penalty significant.
#     Let's use $W_{thrust} = 1000$.
#     Wait, if I use 1000 for position and 1000 for thrust, they are comparable.
#     Let's check the units again.
#     $p^2$ is $m^2$.
#     $u^2$ is $N^2$.
#     $1 N = 1 kg \cdot m/s^2$.
#     $1 N^2 = 1 kg^2 \cdot m^2/s^4$.
#     $1 m^2 = 1 m^2$.
#     They are not dimensionally consistent.
#     So weights are necessary to make them comparable.
#     Let's assume the harness normalizes or expects relative magnitudes.
#     I will use weights to ensure position/attitude > velocity.
#     I will also make thrust significant.
#     Let's try:
#     $R = -1000 * (p_x^2 + p_z^2) - 100 * \theta^2 - 1 * (v_x^2 + v_z^2 + \omega^2) - 1000 * ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
#     Wait, if I use 1000 for thrust, it might be too strong compared to position if position is in meters.
#     But 4.905 is the target.
#     Let's check the prompt again. "penalize departure from hover thrust".
#     "Make position and attitude errors outweigh velocity errors".
#     It doesn't restrict thrust vs position.
#     So I can make thrust penalty high.
#     However, to be safe and "dense", I should ensure the reward is not dominated by one term unless intended.
#     But for hover, position and thrust are the main things.
#     Let's use a scaling factor for position to make it large.
#     Let's use `1000.0` for position and thrust, `100.0` for attitude, `1.0` for velocity.
#     Wait, if I use 1000 for position, a 0.1m error is 10.
#     If I use 1000 for thrust, a 0.1N error is 0.1.
#     Wait, $0.1^2 = 0.01$. $1000 * 0.01 = 10$.
#     So 0.1N error gives 10 penalty.
#     0.1m error gives 10 penalty.
#     This seems balanced.
#     But wait, $u$ is in Newtons. $p$ is in meters.
#     If I use 1000 for both, they are comparable in magnitude.
#     But the prompt says "position and attitude errors outweigh velocity errors".
#     So $W_{pos} \gg W_{vel}$.
#     $1000 \gg 1$. This holds.
#     $W_{att} \gg W_{vel}$.
#     $100 \gg 1$. This holds.
#     So this configuration works.
#     I should also ensure the function is deterministic.
#     I should use `np.float64` or just return the float.
#     I should avoid `np.nan`.
#     I should use `np.sqrt` if needed, but squared is better for smoothness.
#     I should use `np.sum` for sums.
#     I should use `np.square` or `**2`.
#     I should use `np.sin` or `np.cos`? No, the dynamics are given, I don't need to integrate.
#     I just need to read state and action.
#     So no `np.sin` needed in the reward function.
#     Wait, "Favor smooth differentiable terms over step functions."
#     Squared terms are smooth.
#     So `state**2` is fine.
#     I should not use `abs` without squaring.
#     I should not use `max` or `min` with step functions.
#     So `**2` is good.
#     I should not use `np.where` or `np.clip` unless necessary to avoid NaNs.
#     But the prompt says "Crash penalties... are added by the harness".
#     So I don't need to clip.
#     But I should ensure the function doesn't produce NaNs.
#     If state is valid, it won't be NaN.
#     So I can just return the sum.
#     I should import `numpy` as `np`.
#     I should not use `math` unless needed. `np` is sufficient.
#     I should not use `import math`.
#     Wait, "you may import numpy (np) and math and nothing else".
#     So I can use `math` if needed, but `np` is better for vector ops.
#     I will use `np`.
#     I will write the function.
#     I need to make sure I don't use any global variables.
#     I need to define constants inside the function or as global.
#     The prompt says "Constants: m = 1.0...".
#     I can define them inside the function or as global.
#     Since it's a function definition, I can define them inside.
#     But to be efficient, I can define them outside.
#     But the
# --- extracted code ---
The user wants a dense per-step reward function for PPO-based hover control of a Planar Quadrotor.
The observation is state = [p_x, p_z, theta, v_x, v_z, omega] in R^6.
The action is action = [u_1, u_2] in R^2 (rotor thrusts, Newtons).
Constants: m = 1.0, I_yy = 0.01, L = 0.25, g = 9.81.
Hover is state = 0 with both thrusts near 4.905 N.
The agent starts disturbed and must recover to hover.
The function must return ONE finite float, where larger is better.
Use negative terms for penalties.
It must be pure and deterministic.
Import numpy (np) and math and nothing else.
Crash penalties, survival bonus, and episode cutoff are added by the harness -- I only craft the dense shaping.
Make position and attitude errors outweigh velocity errors.
Penalize departure from hover thrust.
Favor smooth differentiable terms over step functions.

Let's break down the reward components.
1.  **Hover Position Error**: The goal is to hover at position 0. So $p_x$ and $p_z$ should be close to 0.
    Since position errors should outweigh velocity errors, I should use a squared error term for position.
    $R_{pos} = - (p_x^2 + p_z^2)$.
    Maybe scale it to make it significant.
2.  **Hover Attitude Error**: The goal is to hover with $\theta = 0$ (planar quadrotor usually implies $\theta$ is the tilt angle in the plane, but looking at the dynamics: $v_x\_dot = -(u_1 + u_2) * np.sin(theta) / m$. If $\theta=0$, $v_x\_dot = 0$. If $\theta \neq 0$, there is a horizontal force component. So $\theta$ should be 0 for pure hover).
    $R_{att} = - \theta^2$.
    Again, position and attitude errors should outweigh velocity errors.
3.  **Thrust Error**: Hover thrust is near 4.905 N.
    $u_{target} = 4.905$.
    $R_{thrust} = - (u_1 - 4.905)^2 - (u_2 - 4.905)^2$.
    Or maybe just the sum of thrusts since they are coupled in vertical force.
    $u_1 + u_2$ determines vertical force. $u_2 - u_1$ determines torque.
    Ideally, $u_1 \approx u_2 \approx 4.905$.
    So penalize deviation from 4.905 for both.
4.  **Velocity Penalty**: Velocity should be close to 0.
    $R_{vel} = - (v_x^2 + v_z^2 + \omega^2)$.
    But position/attitude errors should outweigh velocity errors. So maybe scale velocity penalty down or just include it with a smaller weight.
    Actually, the prompt says "Make position and attitude errors outweigh velocity errors". This implies the weights for position/attitude should be higher than velocity.
5.  **Smoothness**: "Favor smooth differentiable terms over step functions."
    Squared errors are smooth.
    Maybe add a derivative of action or state to encourage smoothness?
    Usually, in dense rewards, we just want the state to be close to target.
    However, "favor smooth differentiable terms" might imply penalizing rapid changes in action or state.
    Let's stick to state errors as they are smooth.
    Maybe penalize action variance?
    Let's focus on the state errors first.
6.  **Scaling**:
    $p_x, p_z$ are in meters.
    $\theta$ is in radians.
    $v_x, v_z, \omega$ are in m/s, rad/s.
    $u$ is in Newtons.
    To make them comparable, I need to normalize or choose weights carefully.
    Since position errors should outweigh velocity errors, I can use a large weight for position.
    Let's define a base weight $W$.
    $R = -W_1 (p_x^2 + p_z^2) - W_2 \theta^2 - W_3 (v_x^2 + v_z^2 + \omega^2) - W_4 ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
    Wait, the prompt says "penalize departure from hover thrust".
    So $W_4$ is important.
    Also "position and attitude errors outweigh velocity errors".
    So $W_1, W_2 \gg W_3$.
    Let's check the units.
    $p^2$ is $m^2$.
    $\theta^2$ is $rad^2$ (dimensionless).
    $v^2$ is $m^2/s^2$.
    $u^2$ is $N^2$.
    To make them comparable, I should probably normalize by some characteristic scales.
    However, the prompt asks for a function, not necessarily normalized physics.
    But for PPO, the magnitude of the reward matters for learning.
    If I use raw values, $p_x$ might be small (0.1m) -> $0.01$.
    $v_x$ might be small (0.1m/s) -> $0.01$.
    $u$ is around 5N -> deviation 0.1N -> $0.01$.
    So raw squared errors might be comparable.
    But "position and attitude errors outweigh velocity errors" suggests I should explicitly weight them higher.
    Let's use weights like 100 for position, 10 for attitude, 1 for velocity, 10 for thrust.
    Wait, if I use 100 for position, a 0.1m error is 1.0 penalty.
    If I use 1 for velocity, a 0.1m/s error is 0.01 penalty.
    This satisfies "position ... outweigh velocity".
    Let's refine the weights.
    $R = -100 (p_x^2 + p_z^2) - 100 \theta^2 - 1 (v_x^2 + v_z^2 + \omega^2) - 10 ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
    Wait, $\theta$ is small. $0.1$ rad is about 5 degrees.
    $p_x$ is meters. $0.1$ m is 10 cm.
    Usually position is more important than angle for hover stability.
    Let's make position and angle weights similar or position slightly higher.
    Let's use $W_{pos} = 1000$, $W_{att} = 100$, $W_{vel} = 1$, $W_{thrust} = 100$.
    Wait, if I use 1000 for position, a 0.1m error is 10.
    If I use 100 for angle, a 0.1 rad error is 1.
    If I use 100 for thrust, a 0.1N error is 0.1.
    This seems reasonable.
    However, I need to ensure the function is deterministic and pure.
    I need to import numpy.
    I need to return ONE finite float.
    I should avoid NaNs or Infs.
    The prompt says "Crash penalties, survival bonus, and episode cutoff are added by the harness".
    So I don't need to check for bounds or time.
    Just the dense shaping.
    "Favor smooth differentiable terms over step functions."
    Squared errors are smooth.
    Maybe I should add a term for action smoothness?
    "Favor smooth differentiable terms" usually refers to the reward function itself being smooth (no Heaviside, no abs without square).
    Squared errors are smooth.
    Maybe penalize action change?
    But the prompt says "dense per-step reward".
    Usually, action smoothness is handled by regularization in the loss or by the reward.
    Let's stick to state and action errors.
    Let's check the constants again.
    $m = 1.0$, $g = 9.81$.
    Hover thrust $T = mg = 9.81$.
    Wait, the prompt says "hover is state = 0 with both thrusts near 4.905 N".
    $4.905 \times 2 = 9.81$.
    So $u_1 + u_2 = 9.81$.
    If $u_1 = u_2 = 4.905$, then $u_1 + u_2 = 9.81$.
    So the target thrust for each is 4.905.
    So penalizing deviation from 4.905 is correct.
    Wait, if the quadrotor is planar, usually it's a 2-rotor setup or 4-rotor in a plane.
    The dynamics given are:
    $v_x\_dot = -(u_1 + u_2) * np.sin(theta) / m$
    $v_z\_dot = (u_1 + u_2) * np.cos(theta) / m - g$
    If $\theta = 0$, $v_x\_dot = 0$, $v_z\_dot = (u_1 + u_2)/m - g$.
    For hover, $v_z\_dot = 0 \implies u_1 + u_2 = mg = 9.81$.
    If $u_1 = u_2 = 4.905$, then $u_1 + u_2 = 9.81$.
    So the target for each is 4.905.
    So penalizing $(u_1 - 4.905)^2 + (u_2 - 4.905)^2$ is correct.
    Wait, if $u_1 + u_2$ is the total thrust, maybe I should penalize $(u_1 + u_2 - 9.81)^2$?
    But the prompt says "both thrusts near 4.905 N".
    So I should penalize deviation of each from 4.905.
    This ensures balanced thrusts too.
    So $R_{thrust} = - ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
    Or maybe just sum of squares.
    Let's use weights to balance the terms.
    Let's define a function.
    I need to import numpy.
    I need to make sure I don't use any other libraries.
    I need to return a single float.
    I should avoid `np.inf` or `np.nan` in the output.
    The prompt says "Crash penalties... are added by the harness".
    So I don't need to check for $p_z < 0$ or something.
    Just the dense shaping.
    Let's write the code.

    Wait, "Make position and attitude errors outweigh velocity errors".
    This implies the magnitude of the position/attitude penalty should be larger than velocity penalty.
    Let's use weights:
    $W_{pos} = 1000$
    $W_{att} = 100$
    $W_{vel} = 1$
    $W_{thrust} = 100$
    Wait, if I use 1000 for position, a 0.1m error is 10.
    If I use 1 for velocity, a 0.1m/s error is 0.01.
    This satisfies the condition.
    Let's check the thrust penalty.
    If I use 100, a 0.1N error is 0.1.
    This is comparable to velocity.
    But thrust is critical for hover.
    Maybe $W_{thrust} = 1000$ as well?
    But the prompt says "penalize departure from hover thrust".
    It doesn't explicitly say it should outweigh position.
    But for hover control, thrust is the primary control.
    However, "position and attitude errors outweigh velocity errors" is the specific instruction.
    It doesn't say "thrust errors outweigh position errors".
    But if thrust is wrong, position will drift.
    So thrust error is correlated with position error.
    Let's make thrust penalty significant.
    Let's use $W_{thrust} = 1000$.
    Wait, if I use 1000 for position and 1000 for thrust, they are comparable.
    Let's check the units again.
    $p^2$ is $m^2$.
    $u^2$ is $N^2$.
    $1 N = 1 kg \cdot m/s^2$.
    $1 N^2 = 1 kg^2 \cdot m^2/s^4$.
    $1 m^2 = 1 m^2$.
    They are not dimensionally consistent.
    So weights are necessary to make them comparable.
    Let's assume the harness normalizes or expects relative magnitudes.
    I will use weights to ensure position/attitude > velocity.
    I will also make thrust significant.
    Let's try:
    $R = -1000 * (p_x^2 + p_z^2) - 100 * \theta^2 - 1 * (v_x^2 + v_z^2 + \omega^2) - 1000 * ((u_1 - 4.905)^2 + (u_2 - 4.905)^2)$.
    Wait, if I use 1000 for thrust, it might be too strong compared to position if position is in meters.
    But 4.905 is the target.
    Let's check the prompt again. "penalize departure from hover thrust".
    "Make position and attitude errors outweigh velocity errors".
    It doesn't restrict thrust vs position.
    So I can make thrust penalty high.
    However, to be safe and "dense", I should ensure the reward is not dominated by one term unless intended.
    But for hover, position and thrust are the main things.
    Let's use a scaling factor for position to make it large.
    Let's use `1000.0` for position and thrust, `100.0` for attitude, `1.0` for velocity.
    Wait, if I use 1000 for position, a 0.1m error is 10.
    If I use 1000 for thrust, a 0.1N error is 0.1.
    Wait, $0.1^2 = 0.01$. $1000 * 0.01 = 10$.
    So 0.1N error gives 10 penalty.
    0.1m error gives 10 penalty.
    This seems balanced.
    But wait, $u$ is in Newtons. $p$ is in meters.
    If I use 1000 for both, they are comparable in magnitude.
    But the prompt says "position and attitude errors outweigh velocity errors".
    So $W_{pos} \gg W_{vel}$.
    $1000 \gg 1$. This holds.
    $W_{att} \gg W_{vel}$.
    $100 \gg 1$. This holds.
    So this configuration works.
    I should also ensure the function is deterministic.
    I should use `np.float64` or just return the float.
    I should avoid `np.nan`.
    I should use `np.sqrt` if needed, but squared is better for smoothness.
    I should use `np.sum` for sums.
    I should use `np.square` or `**2`.
    I should use `np.sin` or `np.cos`? No, the dynamics are given, I don't need to integrate.
    I just need to read state and action.
    So no `np.sin` needed in the reward function.
    Wait, "Favor smooth differentiable terms over step functions."
    Squared terms are smooth.
    So `state**2` is fine.
    I should not use `abs` without squaring.
    I should not use `max` or `min` with step functions.
    So `**2` is good.
    I should not use `np.where` or `np.clip` unless necessary to avoid NaNs.
    But the prompt says "Crash penalties... are added by the harness".
    So I don't need to clip.
    But I should ensure the function doesn't produce NaNs.
    If state is valid, it won't be NaN.
    So I can just return the sum.
    I should import `numpy` as `np`.
    I should not use `math` unless needed. `np` is sufficient.
    I should not use `import math`.
    Wait, "you may import numpy (np) and math and nothing else".
    So I can use `math` if needed, but `np` is better for vector ops.
    I will use `np`.
    I will write the function.
    I need to make sure I don't use any global variables.
    I need to define constants inside the function or as global.
    The prompt says "Constants: m = 1.0...".
    I can define them inside the function or as global.
    Since it's a function definition, I can define them inside.
    But to be efficient, I can define them outside.
    But the

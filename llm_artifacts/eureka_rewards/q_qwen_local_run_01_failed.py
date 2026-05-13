import numpy as np
        import math

        def reward(state, action):
            # Constants
            m = 1.0
            g = 9.81
            L = 0.25
            I_yy = 0.01
            u_hover = m * g / 2 # 4.905

            # Extract state
            px, pz, theta, vx, vz, omega = state
            u1, u2 = action

            # Position error (squared)
            pos_err = px**2 + pz**2
            # Attitude error (squared)
            att_err = theta**2
            # Velocity error (squared)
            vel_err = vx**2 + vz**2 + omega**2
            # Control error (squared)
            ctrl_err = (u1 - u_hover)**2 + (u2 - u_hover)**2

            # Weights
            # Position needs to be penalized heavily to keep it near 0
            # Attitude needs to be penalized heavily to keep it near 0
            # Velocity should be penalized but less than position
            # Control should be penalized to keep thrusts near hover

            # Let's choose weights that balance the scales.
            # If pos is 1m, penalty is 1.
            # If theta is 1 rad, penalty is 1.
            # If v is 1 m/s, penalty is 0.1.
            # If u is 1 N diff, penalty is 1.

            # To make it "Higher is better", we subtract penalties.
            # Or we can define a base reward and subtract.
            # Let's just return negative penalties.

            # Tuning weights:
            # Position: 1000
            # Attitude: 1000
            # Velocity: 100
            # Control: 1000

            # Wait, if I just return negative penalties, the max reward is 0 (at equilibrium).
            # PPO works better if there's a positive signal for success, but negative penalties work too.
            # The prompt says "Higher-is-better convention... If you wish to penalize something, the term should be negative."
            # So `return -pos - att - vel - ctrl` is valid.
            # However, often a small positive reward for being close helps.
            # Let's stick to the prompt's guidance: "Penalize position deviation...".
            # So `return -pos - att - vel - ctrl` is fine.
            # To make it robust, I'll add a small positive constant or just rely on the negative penalties.
            # Actually, standard practice for PPO is often `reward = -cost`.
            # Let's ensure the values are reasonable.

            # Let's refine weights based on physical scales.
            # px, pz ~ 1m. px^2 ~ 1.
            # theta ~ 0.1 rad. theta^2 ~ 0.01.
            # vx, vz ~ 0.1 m/s. vx^2 ~ 0.01.
            # u ~ 5N. (u-u_hover)^2 ~ 1.

            # If I use weights:
            # pos: 1000
            # att: 1000
            # vel: 100
            # ctrl: 1000
            # Then at 1m pos, penalty is 1000.
            # At 0.1 rad theta, penalty is 1000 * 0.01 = 10.
            # At 0.1 m/s vel, penalty is 100 * 0.01 = 1.
            # At 1N ctrl, penalty is 1000.

            # This seems balanced.
            # Let's add a small positive reward for being close to target to encourage exploration towards it?
            # No, the prompt says "Penalize position deviation...".
            # I will stick to negative penalties.
            # To ensure the agent doesn't just sit at 0 and get 0 reward, I should ensure the reward is maximized at 0.
            # So `return - (pos + att + vel + ctrl)` is correct.
            # To make it slightly more interesting, I can add a small positive term for "good behavior" but negative penalties are safer for stability.
            # Let's add a small positive constant to the return value so it's not always negative?
            # No, PPO handles negative rewards fine.
            # I will use a base reward of 0 at equilibrium.

            # Wait, one detail: "The hover equilibrium is state = 0, action = [u_hover, u_hover]".
            # So at equilibrium, reward should be 0 (or max).
            # If I return negative penalties, max is 0.
            # Let's add a small positive bonus for being close to target to encourage staying there?
            # "Higher-is-better... If you wish to penalize something, the term should be negative."
            # So `return -pos - att - vel - ctrl` works.
            # I will add a small positive term `+ 1.0` to make the reward slightly positive at equilibrium, which might help PPO convergence slightly better than 0, but 0 is fine.
            # Actually, to avoid the agent learning to just minimize penalty without caring about the sign, I'll just use the penalties.
            # Let's make sure the weights are tuned so the agent doesn't prioritize one over another incorrectly.
            # Position is usually the most critical for "hover stabilization".
            # Attitude is critical for stability.
            # Velocity is critical for dynamics.
            # Control is critical for safety (u_max).

            # Let's use weights:
            # pos: 1000
            # att: 1000
            # vel: 100
            # ctrl: 1000

            # Wait, I need to check the `u_max`. `u_max = 2*m*g = 19.62`. `u_hover = 4.905`.
            # If the agent tries to hover, `u` should be around 4.905.
            # If `u` goes to 0, penalty is `(0-4.905)^2 = 24`.
            # If `u` goes to 19.62, penalty is `(19.62-4.905)^2 = 220`.
            # So control penalty is significant.

            # Let's finalize the code.
            # I need to import numpy and math.
            # I need to handle potential NaN/Inf (though unlikely with simple math).
            # I need to ensure `state` and `action` are numpy arrays.

            # One more thing: The prompt says "The function MUST be named ``reward`` and take exactly two arguments named ``state`` and ``action`` in that order."
            # "It MUST return a single scalar value".
            # "No I/O, no global mutable state...".

            # I will write the code now.

"""P E1 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def basis(xu):
#     # State: [p_x, p_z, theta, v_x, v_z, omega], Control: [u_1, u_2]
#     px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
#     
#     # Physical parameters
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     
#     # Precompute common nonlinear terms
#     sin_theta = np.sin(theta)
#     cos_theta = np.cos(theta)
#     u_sum = u1 + u2
#     u_diff = u1 - u2
#     
#     # Linear and constant features
#     features = [
#         1.0,                    # constant
#         px, pz, theta, vx, vz, omega,  # state linear
#         u1, u2,                 # control linear
#     ]
#     
#     # Quadratic state terms (avoid redundant high-order terms; keep interpretable)
#     features += [
#         vx**2, vz**2, omega**2,
#         px*theta, pz*theta,     # position-pitch couplings
#         vx*omega, vz*omega,    # velocity-rotation couplings
#         theta**2,              # small-angle approximation support
#     ]
#     
#     # Thrust-pitch coupling (essential for dynamics)
#     features += [
#         u_sum * sin_theta,     # horizontal thrust component (x-acceleration)
#         u_sum * cos_theta,     # vertical thrust component (z-acceleration, opposes gravity)
#         u_diff * L / I_yy,     # torque (angular acceleration)
#     ]
#     
#     # Gravity term (constant vertical force)
#     features.append(-g)       # appears in pz_dot equation as -g
#     
#     # Additional physically motivated terms
#     features += [
#         u_sum,                 # total thrust (appears in z-dot with cos, but also baseline scaling)
#         u_diff,                # differential thrust (appears in theta-dot)
#         sin_theta,             # explicit sine for pitch coupling
#         cos_theta,             # explicit cosine for pitch coupling
#         vx * sin_theta,        # velocity-thrust directional coupling
#         vz * cos_theta,        # velocity-thrust directional coupling
#         theta * u_sum,         # small-angle correction to thrust projection
#         omega * theta,         # rotational coupling
#     ]
#     
#     return features
# ```
# --- extracted code ---
def basis(xu):
    # State: [p_x, p_z, theta, v_x, v_z, omega], Control: [u_1, u_2]
    px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
    
    # Physical parameters
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # Precompute common nonlinear terms
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    u_sum = u1 + u2
    u_diff = u1 - u2
    
    # Linear and constant features
    features = [
        1.0,                    # constant
        px, pz, theta, vx, vz, omega,  # state linear
        u1, u2,                 # control linear
    ]
    
    # Quadratic state terms (avoid redundant high-order terms; keep interpretable)
    features += [
        vx**2, vz**2, omega**2,
        px*theta, pz*theta,     # position-pitch couplings
        vx*omega, vz*omega,    # velocity-rotation couplings
        theta**2,              # small-angle approximation support
    ]
    
    # Thrust-pitch coupling (essential for dynamics)
    features += [
        u_sum * sin_theta,     # horizontal thrust component (x-acceleration)
        u_sum * cos_theta,     # vertical thrust component (z-acceleration, opposes gravity)
        u_diff * L / I_yy,     # torque (angular acceleration)
    ]
    
    # Gravity term (constant vertical force)
    features.append(-g)       # appears in pz_dot equation as -g
    
    # Additional physically motivated terms
    features += [
        u_sum,                 # total thrust (appears in z-dot with cos, but also baseline scaling)
        u_diff,                # differential thrust (appears in theta-dot)
        sin_theta,             # explicit sine for pitch coupling
        cos_theta,             # explicit cosine for pitch coupling
        vx * sin_theta,        # velocity-thrust directional coupling
        vz * cos_theta,        # velocity-thrust directional coupling
        theta * u_sum,         # small-angle correction to thrust projection
        omega * theta,         # rotational coupling
    ]
    
    return features

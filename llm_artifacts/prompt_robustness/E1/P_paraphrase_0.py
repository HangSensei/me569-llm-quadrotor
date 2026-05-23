"""P E1 paraphrase 0. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def basis(xu):
#     # Extract state and control from xu
#     px, pz, theta, vx, vz, omega = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5]
#     u1, u2 = xu[6], xu[7]
#     
#     # Precompute common nonlinear terms for efficiency and clarity
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     u_sum = u1 + u2          # total thrust
#     u_diff = u1 - u2          # differential thrust (torque-related)
#     
#     # Linear terms: all state variables and controls
#     # [1, px, pz, theta, vx, vz, omega, u1, u2]
#     features = [1.0, px, pz, theta, vx, vz, omega, u1, u2]
#     
#     # Quadratic and interaction terms that capture key physics:
#     # - Gravity appears as constant acceleration in z (but coupled with cos/sin for thrust projection)
#     # - Thrust projection: u_sum * cos(theta) -> x-acceleration, u_sum * sin(theta) -> z-acceleration
#     # - Torque: L * u_diff / I_yy -> angular acceleration
#     # - Drag is often neglected in basic models, but we include vx, vz, omega linearly as potential damping
#     # - Nonlinear couplings: theta with u_sum (thrust projection), u_diff with theta (torque has no direct theta coupling in simple model, but higher-order effects may exist)
#     # - Also include sin(theta), cos(theta), u_sum*cos(theta), u_sum*sin(theta), u_diff
#     
#     features.extend([
#         cos_theta,
#         sin_theta,
#         u_sum * cos_theta,      # x-direction thrust component
#         u_sum * sin_theta,      # z-direction thrust component (opposes gravity)
#         u_diff,                 # raw torque input (proportional to angular acceleration)
#         vx**2,                  # optional quadratic drag proxy
#         vz**2,                  # optional quadratic drag proxy
#         omega**2,               # optional rotational drag proxy
#         theta**2,               # small-angle approximations suggest theta^2 may appear in higher-order expansions
#         vx * cos_theta,         # velocity-thrust alignment coupling
#         vz * sin_theta,         # velocity-thrust alignment coupling
#         u_sum * theta,          # first-order approximation of thrust projection nonlinearity
#         u_diff * theta,         # coupling between torque and attitude
#         u1 * cos_theta,         # individual rotor x-component
#         u2 * cos_theta,         # individual rotor x-component
#         u1 * sin_theta,         # individual rotor z-component
#         u2 * sin_theta,         # individual rotor z-component
#         vx * u_sum,             # velocity-thrust interaction
#         vz * u_sum,             # velocity-thrust interaction
#         omega * u_diff,         # angular rate-torque interaction
#     ])
#     
#     # Return as list (SINDy/STLSQ expects iterable of scalars)
#     return features
# ```
# --- extracted code ---
def basis(xu):
    # Extract state and control from xu
    px, pz, theta, vx, vz, omega = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5]
    u1, u2 = xu[6], xu[7]
    
    # Precompute common nonlinear terms for efficiency and clarity
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    u_sum = u1 + u2          # total thrust
    u_diff = u1 - u2          # differential thrust (torque-related)
    
    # Linear terms: all state variables and controls
    # [1, px, pz, theta, vx, vz, omega, u1, u2]
    features = [1.0, px, pz, theta, vx, vz, omega, u1, u2]
    
    # Quadratic and interaction terms that capture key physics:
    # - Gravity appears as constant acceleration in z (but coupled with cos/sin for thrust projection)
    # - Thrust projection: u_sum * cos(theta) -> x-acceleration, u_sum * sin(theta) -> z-acceleration
    # - Torque: L * u_diff / I_yy -> angular acceleration
    # - Drag is often neglected in basic models, but we include vx, vz, omega linearly as potential damping
    # - Nonlinear couplings: theta with u_sum (thrust projection), u_diff with theta (torque has no direct theta coupling in simple model, but higher-order effects may exist)
    # - Also include sin(theta), cos(theta), u_sum*cos(theta), u_sum*sin(theta), u_diff
    
    features.extend([
        cos_theta,
        sin_theta,
        u_sum * cos_theta,      # x-direction thrust component
        u_sum * sin_theta,      # z-direction thrust component (opposes gravity)
        u_diff,                 # raw torque input (proportional to angular acceleration)
        vx**2,                  # optional quadratic drag proxy
        vz**2,                  # optional quadratic drag proxy
        omega**2,               # optional rotational drag proxy
        theta**2,               # small-angle approximations suggest theta^2 may appear in higher-order expansions
        vx * cos_theta,         # velocity-thrust alignment coupling
        vz * sin_theta,         # velocity-thrust alignment coupling
        u_sum * theta,          # first-order approximation of thrust projection nonlinearity
        u_diff * theta,         # coupling between torque and attitude
        u1 * cos_theta,         # individual rotor x-component
        u2 * cos_theta,         # individual rotor x-component
        u1 * sin_theta,         # individual rotor z-component
        u2 * sin_theta,         # individual rotor z-component
        vx * u_sum,             # velocity-thrust interaction
        vz * u_sum,             # velocity-thrust interaction
        omega * u_diff,         # angular rate-torque interaction
    ])
    
    # Return as list (SINDy/STLSQ expects iterable of scalars)
    return features

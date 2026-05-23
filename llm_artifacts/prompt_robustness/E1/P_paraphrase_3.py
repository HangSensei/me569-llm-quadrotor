"""P E1 paraphrase 3. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# import math
# 
# def basis(xu):
#     px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
#     c = np.cos(theta)
#     s = np.sin(theta)
#     u_sum = u1 + u2
#     u_diff = u1 - u2
#     
#     # Constant and linear terms
#     features = [1.0, px, pz, theta, vx, vz, omega, u1, u2]
#     
#     # Quadratic and interaction terms relevant to dynamics
#     features.extend([
#         u_sum,           # total thrust (affects vertical acceleration)
#         u_diff,          # differential thrust (affects angular acceleration)
#         c,               # cos(theta) for gravity projection and thrust components
#         s,               # sin(theta) for gravity projection and thrust components
#         vx * c,          # x-velocity projected along body frame (drag-like, but also appears in Coriolis)
#         vz * s,          # z-velocity projected along body frame
#         vx * s,
#         vz * c,
#         u_sum * c,       # horizontal component of total thrust
#         u_sum * s,       # vertical component of total thrust (counteracts gravity)
#         u_diff * theta,  # torque modulation by attitude (linearized coupling)
#         vx * omega,      # Coriolis-like term (from rotating frame)
#         vz * omega,
#         omega**2,        # centrifugal-like term
#         theta**2,        # nonlinear stiffness-like term
#         u1 * u2,         # thrust interaction (e.g., cross-coupling)
#         u1**2,
#         u2**2,
#         c * u_diff,      # torque component modulated by attitude
#         s * u_diff,
#         vx**2,
#         vz**2,
#         omega * theta,
#         u_sum * theta,
#         u_diff * c,
#         u_diff * s
#     ])
#     
#     return np.array(features)
# ```
# --- extracted code ---
import numpy as np
import math

def basis(xu):
    px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
    c = np.cos(theta)
    s = np.sin(theta)
    u_sum = u1 + u2
    u_diff = u1 - u2
    
    # Constant and linear terms
    features = [1.0, px, pz, theta, vx, vz, omega, u1, u2]
    
    # Quadratic and interaction terms relevant to dynamics
    features.extend([
        u_sum,           # total thrust (affects vertical acceleration)
        u_diff,          # differential thrust (affects angular acceleration)
        c,               # cos(theta) for gravity projection and thrust components
        s,               # sin(theta) for gravity projection and thrust components
        vx * c,          # x-velocity projected along body frame (drag-like, but also appears in Coriolis)
        vz * s,          # z-velocity projected along body frame
        vx * s,
        vz * c,
        u_sum * c,       # horizontal component of total thrust
        u_sum * s,       # vertical component of total thrust (counteracts gravity)
        u_diff * theta,  # torque modulation by attitude (linearized coupling)
        vx * omega,      # Coriolis-like term (from rotating frame)
        vz * omega,
        omega**2,        # centrifugal-like term
        theta**2,        # nonlinear stiffness-like term
        u1 * u2,         # thrust interaction (e.g., cross-coupling)
        u1**2,
        u2**2,
        c * u_diff,      # torque component modulated by attitude
        s * u_diff,
        vx**2,
        vz**2,
        omega * theta,
        u_sum * theta,
        u_diff * c,
        u_diff * s
    ])
    
    return np.array(features)

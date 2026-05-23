"""P E1 paraphrase 2. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# import math
# 
# def basis(xu):
#     px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
#     cth = math.cos(theta)
#     sth = math.sin(theta)
#     # Constants
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     
#     # Linear terms
#     features = [1.0, px, pz, theta, vx, vz, omega, u1, u2]
#     
#     # Quadratic and interaction terms
#     features.extend([
#         px*px, pz*pz, theta*theta, vx*vx, vz*vz, omega*omega,
#         u1*u1, u2*u2, u1*u2,
#         px*vx, pz*vz, theta*omega,
#         vx*cth, vx*sth, vz*cth, vz*sth,
#         u1*cth, u1*sth, u2*cth, u2*sth,
#         (u1 - u2)*theta,  # differential thrust coupling with angle
#         (u1 + u2)*cth,   # total thrust projection in x
#         (u1 + u2)*sth,   # total thrust projection in z (opposes gravity)
#         (u1 + u2),       # total thrust magnitude
#         (u1 - u2),       # differential thrust (torque-related)
#         theta*cth, theta*sth,
#         vx*vx*cth, vz*vz*sth,
#         omega*omega*theta
#     ])
#     
#     return features
# ```
# --- extracted code ---
import numpy as np
import math

def basis(xu):
    px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
    cth = math.cos(theta)
    sth = math.sin(theta)
    # Constants
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # Linear terms
    features = [1.0, px, pz, theta, vx, vz, omega, u1, u2]
    
    # Quadratic and interaction terms
    features.extend([
        px*px, pz*pz, theta*theta, vx*vx, vz*vz, omega*omega,
        u1*u1, u2*u2, u1*u2,
        px*vx, pz*vz, theta*omega,
        vx*cth, vx*sth, vz*cth, vz*sth,
        u1*cth, u1*sth, u2*cth, u2*sth,
        (u1 - u2)*theta,  # differential thrust coupling with angle
        (u1 + u2)*cth,   # total thrust projection in x
        (u1 + u2)*sth,   # total thrust projection in z (opposes gravity)
        (u1 + u2),       # total thrust magnitude
        (u1 - u2),       # differential thrust (torque-related)
        theta*cth, theta*sth,
        vx*vx*cth, vz*vz*sth,
        omega*omega*theta
    ])
    
    return features

"""P E3 paraphrase 3. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# 
# def observables(x):
#     px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
#     
#     # Base state (required first 6)
#     psi = [px, pz, theta, vx, vz, omega]
#     
#     # Add nonlinear terms motivated by quadrotor dynamics:
#     # - sin/cos(theta) for attitude-coupled forces/torques
#     # - quadratic velocity terms for drag-like effects (though minimal in standard model, they help capture residual dynamics and improve EDMDc fit)
#     # - products like theta*vx, theta*vz for coupling
#     # - theta^2, omega^2 for inertial/centrifugal corrections
#     
#     psi.append(np.sin(theta))
#     psi.append(np.cos(theta))
#     psi.append(vx**2)
#     psi.append(vz**2)
#     psi.append(omega**2)
#     psi.append(theta * vx)
#     psi.append(theta * vz)
#     psi.append(theta * omega)
#     psi.append(vx * vz)
#     
#     # Total: 6 + 9 = 15 observables
#     return psi
# ```
# --- extracted code ---
import numpy as np

def observables(x):
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Base state (required first 6)
    psi = [px, pz, theta, vx, vz, omega]
    
    # Add nonlinear terms motivated by quadrotor dynamics:
    # - sin/cos(theta) for attitude-coupled forces/torques
    # - quadratic velocity terms for drag-like effects (though minimal in standard model, they help capture residual dynamics and improve EDMDc fit)
    # - products like theta*vx, theta*vz for coupling
    # - theta^2, omega^2 for inertial/centrifugal corrections
    
    psi.append(np.sin(theta))
    psi.append(np.cos(theta))
    psi.append(vx**2)
    psi.append(vz**2)
    psi.append(omega**2)
    psi.append(theta * vx)
    psi.append(theta * vz)
    psi.append(theta * omega)
    psi.append(vx * vz)
    
    # Total: 6 + 9 = 15 observables
    return psi

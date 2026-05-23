"""P E3 paraphrase 2. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# 
# def observables(x):
#     p_x, p_z, theta, v_x, v_z, omega = x[0], x[1], x[2], x[3], x[4], x[5]
#     
#     # Start with identity observables (state itself)
#     psi = [p_x, p_z, theta, v_x, v_z, omega]
#     
#     # Add trigonometric functions of angle for rotational coupling
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     psi.extend([cos_theta, sin_theta])
#     
#     # Add products of velocity and trig functions (appear in dynamics: e.g., v_x*cos(theta), v_z*sin(theta))
#     psi.extend([v_x * cos_theta, v_x * sin_theta, v_z * cos_theta, v_z * sin_theta])
#     
#     # Add angular velocity squared (appears in centrifugal/Coriolis terms)
#     psi.append(omega**2)
#     
#     # Add cross terms involving omega and velocities (Coriolis-like)
#     psi.extend([omega * v_x, omega * v_z])
#     
#     # Add second-order trig terms that arise from acceleration expressions
#     psi.extend([cos_theta**2, sin_theta**2, cos_theta * sin_theta])
#     
#     # Add position-velocity couplings (less critical but help with nonlinearities)
#     psi.extend([p_x * v_x, p_z * v_z])
#     
#     return psi
# ```
# --- extracted code ---
import numpy as np

def observables(x):
    p_x, p_z, theta, v_x, v_z, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Start with identity observables (state itself)
    psi = [p_x, p_z, theta, v_x, v_z, omega]
    
    # Add trigonometric functions of angle for rotational coupling
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    psi.extend([cos_theta, sin_theta])
    
    # Add products of velocity and trig functions (appear in dynamics: e.g., v_x*cos(theta), v_z*sin(theta))
    psi.extend([v_x * cos_theta, v_x * sin_theta, v_z * cos_theta, v_z * sin_theta])
    
    # Add angular velocity squared (appears in centrifugal/Coriolis terms)
    psi.append(omega**2)
    
    # Add cross terms involving omega and velocities (Coriolis-like)
    psi.extend([omega * v_x, omega * v_z])
    
    # Add second-order trig terms that arise from acceleration expressions
    psi.extend([cos_theta**2, sin_theta**2, cos_theta * sin_theta])
    
    # Add position-velocity couplings (less critical but help with nonlinearities)
    psi.extend([p_x * v_x, p_z * v_z])
    
    return psi

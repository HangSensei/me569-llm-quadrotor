"""P E3 paraphrase 0. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# 
# def observables(x):
#     # State: x = [p_x, p_z, theta, v_x, v_z, omega]
#     p_x, p_z, theta, v_x, v_z, omega = x[0], x[1], x[2], x[3], x[4], x[5]
#     
#     # Start with identity observables (state-recovery convention)
#     psi = [p_x, p_z, theta, v_x, v_z, omega]
#     
#     # Add nonlinear terms that appear in quadrotor dynamics:
#     # Forces: thrust components depend on cos(theta), sin(theta)
#     # Moments: torque depends on theta (for coupling) and omega (damping)
#     # Also include quadratic velocity terms for drag-like effects (common in modeling)
#     # and cross terms that capture coupling between translation and rotation
#     
#     # Trigonometric terms (essential for attitude-coupled forces)
#     psi.append(np.cos(theta))
#     psi.append(np.sin(theta))
#     
#     # Quadratic state terms: velocities squared (appear in drag, kinetic energy)
#     psi.append(v_x**2)
#     psi.append(v_z**2)
#     psi.append(omega**2)
#     
#     # Cross terms: v_x * v_z, v_x * omega, v_z * omega (coupling terms)
#     psi.append(v_x * v_z)
#     psi.append(v_x * omega)
#     psi.append(v_z * omega)
#     
#     # Theta-coupled velocity terms: cos(theta)*v_x, cos(theta)*v_z, sin(theta)*v_x, sin(theta)*v_z
#     # These arise from rotation of thrust vector
#     psi.append(np.cos(theta) * v_x)
#     psi.append(np.cos(theta) * v_z)
#     psi.append(np.sin(theta) * v_x)
#     psi.append(np.sin(theta) * v_z)
#     
#     # Theta-squared and higher trig: theta^2 (small-angle approximation term), cos(2*theta), sin(2*theta)
#     psi.append(theta**2)
#     psi.append(np.cos(2*theta))
#     psi.append(np.sin(2*theta))
#     
#     # Additional coupling: theta * v_x, theta * v_z, theta * omega (for small-angle expansions)
#     psi.append(theta * v_x)
#     psi.append(theta * v_z)
#     psi.append(theta * omega)
#     
#     # Lifted state dimension: 6 (original) + 22 added = 28
#     # This set captures essential nonlinearities in planar quadrotor dynamics:
#     # - Rigid-body rotation of thrust (cos/sin theta)
#     # - Coriolis/centrifugal-like terms (v_i*v_j, v_i*omega)
#     # - Small-angle expansions (theta^2, theta*v_i)
#     # - Harmonic terms (cos2t, sin2t) to better approximate nonlinear force mapping
#     
#     return np.array(psi)
# ```
# --- extracted code ---
import numpy as np

def observables(x):
    # State: x = [p_x, p_z, theta, v_x, v_z, omega]
    p_x, p_z, theta, v_x, v_z, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Start with identity observables (state-recovery convention)
    psi = [p_x, p_z, theta, v_x, v_z, omega]
    
    # Add nonlinear terms that appear in quadrotor dynamics:
    # Forces: thrust components depend on cos(theta), sin(theta)
    # Moments: torque depends on theta (for coupling) and omega (damping)
    # Also include quadratic velocity terms for drag-like effects (common in modeling)
    # and cross terms that capture coupling between translation and rotation
    
    # Trigonometric terms (essential for attitude-coupled forces)
    psi.append(np.cos(theta))
    psi.append(np.sin(theta))
    
    # Quadratic state terms: velocities squared (appear in drag, kinetic energy)
    psi.append(v_x**2)
    psi.append(v_z**2)
    psi.append(omega**2)
    
    # Cross terms: v_x * v_z, v_x * omega, v_z * omega (coupling terms)
    psi.append(v_x * v_z)
    psi.append(v_x * omega)
    psi.append(v_z * omega)
    
    # Theta-coupled velocity terms: cos(theta)*v_x, cos(theta)*v_z, sin(theta)*v_x, sin(theta)*v_z
    # These arise from rotation of thrust vector
    psi.append(np.cos(theta) * v_x)
    psi.append(np.cos(theta) * v_z)
    psi.append(np.sin(theta) * v_x)
    psi.append(np.sin(theta) * v_z)
    
    # Theta-squared and higher trig: theta^2 (small-angle approximation term), cos(2*theta), sin(2*theta)
    psi.append(theta**2)
    psi.append(np.cos(2*theta))
    psi.append(np.sin(2*theta))
    
    # Additional coupling: theta * v_x, theta * v_z, theta * omega (for small-angle expansions)
    psi.append(theta * v_x)
    psi.append(theta * v_z)
    psi.append(theta * omega)
    
    # Lifted state dimension: 6 (original) + 22 added = 28
    # This set captures essential nonlinearities in planar quadrotor dynamics:
    # - Rigid-body rotation of thrust (cos/sin theta)
    # - Coriolis/centrifugal-like terms (v_i*v_j, v_i*omega)
    # - Small-angle expansions (theta^2, theta*v_i)
    # - Harmonic terms (cos2t, sin2t) to better approximate nonlinear force mapping
    
    return np.array(psi)

"""P E3 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# import numpy as np
# 
# def observables(x):
#     px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
#     
#     # Base state (first 6 elements)
#     psi = [px, pz, theta, vx, vz, omega]
#     
#     # Add nonlinear features that appear in quadrotor dynamics:
#     # - sin(theta), cos(theta) for gravity projection and thrust components
#     # - vx^2, vz^2, vx*vz for drag-like quadratic terms (common in modeling)
#     # - theta^2 for small-angle approximations and nonlinearities
#     # - vx*theta, vz*theta for coupling terms
#     # - omega^2 for rotational inertia effects
#     # - sin(2*theta), cos(2*theta) for harmonic content
#     
#     psi.append(np.sin(theta))
#     psi.append(np.cos(theta))
#     psi.append(vx**2)
#     psi.append(vz**2)
#     psi.append(vx * vz)
#     psi.append(theta**2)
#     psi.append(vx * theta)
#     psi.append(vz * theta)
#     psi.append(omega**2)
#     psi.append(np.sin(2*theta))
#     psi.append(np.cos(2*theta))
#     
#     # Add cross terms involving control-relevant nonlinearities: 
#     # vx*sin(theta), vx*cos(theta), vz*sin(theta), vz*cos(theta) — appear in thrust projection
#     psi.append(vx * np.sin(theta))
#     psi.append(vx * np.cos(theta))
#     psi.append(vz * np.sin(theta))
#     psi.append(vz * np.cos(theta))
#     
#     # Add higher-order but physically motivated: theta*omega, vx*omega, vz*omega
#     psi.append(theta * omega)
#     psi.append(vx * omega)
#     psi.append(vz * omega)
#     
#     # Total length = 6 + 17 = 23 ≤ 50, minimal sufficient set capturing dominant nonlinearities
#     # in planar quadrotor continuous-time dynamics (before discretization):
#     #   d/dt [px; pz; theta; vx; vz; omega] = 
#     #     [vx; vz; omega; 
#     #      -(u1 + u2)*sin(theta)/m; 
#     #      (u1 + u2)*cos(theta)/m - g; 
#     #      L*(u1 - u2)/I_yy]
#     # So right-hand side depends on sin(theta), cos(theta), and products like vx*sin(theta), etc.
#     # EDMDc requires observables rich enough to represent the true Koopman eigenfunctions
#     # for the unknown discrete-time map — these terms span the dominant nonlinear dependencies.
#     
#     return np.array(psi)
# ```
# --- extracted code ---
import numpy as np

def observables(x):
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Base state (first 6 elements)
    psi = [px, pz, theta, vx, vz, omega]
    
    # Add nonlinear features that appear in quadrotor dynamics:
    # - sin(theta), cos(theta) for gravity projection and thrust components
    # - vx^2, vz^2, vx*vz for drag-like quadratic terms (common in modeling)
    # - theta^2 for small-angle approximations and nonlinearities
    # - vx*theta, vz*theta for coupling terms
    # - omega^2 for rotational inertia effects
    # - sin(2*theta), cos(2*theta) for harmonic content
    
    psi.append(np.sin(theta))
    psi.append(np.cos(theta))
    psi.append(vx**2)
    psi.append(vz**2)
    psi.append(vx * vz)
    psi.append(theta**2)
    psi.append(vx * theta)
    psi.append(vz * theta)
    psi.append(omega**2)
    psi.append(np.sin(2*theta))
    psi.append(np.cos(2*theta))
    
    # Add cross terms involving control-relevant nonlinearities: 
    # vx*sin(theta), vx*cos(theta), vz*sin(theta), vz*cos(theta) — appear in thrust projection
    psi.append(vx * np.sin(theta))
    psi.append(vx * np.cos(theta))
    psi.append(vz * np.sin(theta))
    psi.append(vz * np.cos(theta))
    
    # Add higher-order but physically motivated: theta*omega, vx*omega, vz*omega
    psi.append(theta * omega)
    psi.append(vx * omega)
    psi.append(vz * omega)
    
    # Total length = 6 + 17 = 23 ≤ 50, minimal sufficient set capturing dominant nonlinearities
    # in planar quadrotor continuous-time dynamics (before discretization):
    #   d/dt [px; pz; theta; vx; vz; omega] = 
    #     [vx; vz; omega; 
    #      -(u1 + u2)*sin(theta)/m; 
    #      (u1 + u2)*cos(theta)/m - g; 
    #      L*(u1 - u2)/I_yy]
    # So right-hand side depends on sin(theta), cos(theta), and products like vx*sin(theta), etc.
    # EDMDc requires observables rich enough to represent the true Koopman eigenfunctions
    # for the unknown discrete-time map — these terms span the dominant nonlinear dependencies.
    
    return np.array(psi)

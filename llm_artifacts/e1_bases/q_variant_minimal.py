"""Q-condition basis from prompt variant 'minimal'. Saved by scripts/e1_prompt_ablation.py.
"""
import numpy as np

def basis(xu):
    # Extract state and control
    p_x, p_z, theta, v_x, v_z, omega = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5]
    u_1, u_2 = xu[6], xu[7]
    
    # Flatten to numpy array for easier computation
    x = np.array([p_x, p_z, theta, v_x, v_z, omega, u_1, u_2])
    
    # Generate basis functions
    basis_list = []
    
    # Constant
    basis_list.append(1.0)
    
    # Linear terms (state and control)
    basis_list.extend([x[i] for i in range(8)])
    
    # Quadratic terms
    for i in range(8):
        basis_list.append(x[i] ** 2)
    
    # Cross terms
    for i in range(8):
        for j in range(i+1, 8):
            basis_list.append(x[i] * x[j])
    
    # Trigonometric terms (for periodic behavior)
    basis_list.append(np.sin(x[2]))  # theta
    basis_list.append(np.cos(x[2]))  # theta
    
    return basis_list

import numpy as np

def basis(xu):
    # Extract state and control
    p_x, p_z, theta, v_x, v_z, omega = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5]
    u_1, u_2 = xu[6], xu[7]
    
    # Create state vector for easier indexing
    x = np.array([p_x, p_z, theta, v_x, v_z, omega, u_1, u_2])
    
    basis_list = []
    
    # Constant term
    basis_list.append(1.0)
    
    # Linear terms (state and control)
    for i in range(8):
        basis_list.append(x[i])
    
    # Quadratic terms
    for i in range(8):
        basis_list.append(x[i] ** 2)
    
    # Cross terms (all pairs)
    for i in range(8):
        for j in range(i + 1, 8):
            basis_list.append(x[i] * x[j])
    
    # Trigonometric terms for orientation (theta)
    basis_list.append(np.sin(theta))
    basis_list.append(np.cos(theta))
    
    # Cross terms with trigonometric terms
    basis_list.append(np.sin(theta) * x[0])  # sin(theta) * p_x
    basis_list.append(np.cos(theta) * x[0])  # cos(theta) * p_x
    basis_list.append(np.sin(theta) * x[1])  # sin(theta) * p_z
    basis_list.append(np.cos(theta) * x[1])  # cos(theta) * p_z
    basis_list.append(np.sin(theta) * x[2])  # sin(theta) * theta
    basis_list.append(np.cos(theta) * x[2])  # cos(theta) * theta
    
    return basis_list

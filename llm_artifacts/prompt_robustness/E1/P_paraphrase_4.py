"""P E1 paraphrase 4. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def basis(xu):
#     px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
#     
#     # Constants
#     m = 1.0
#     I_yy = 0.01
#     L = 0.25
#     g = 9.81
#     
#     # Trigonometric functions
#     cth = np.cos(theta)
#     sth = np.sin(theta)
#     
#     # Basic monomials and interactions
#     features = [
#         1.0,  # bias
#         px, pz, theta, vx, vz, omega, u1, u2,
#         u1 + u2,  # total thrust
#         u1 - u2,  # differential thrust
#         cth, sth,
#         u1 * cth, u1 * sth, u2 * cth, u2 * sth,
#         (u1 + u2) * cth, (u1 + u2) * sth,
#         (u1 - u2) * cth, (u1 - u2) * sth,
#         theta**2, vx**2, vz**2, omega**2,
#         u1**2, u2**2,
#         u1 * u2,
#         vx * u1, vx * u2, vz * u1, vz * u2,
#         theta * u1, theta * u2,
#         vx * cth, vx * sth, vz * cth, vz * sth,
#         omega * u1, omega * u2,
#         theta * vx, theta * vz, theta * omega,
#         u1 * vx * cth, u1 * vz * sth, u2 * vx * cth, u2 * vz * sth,
#         (u1 + u2) * g,
#         (u1 - u2) * L / I_yy,
#         (u1 + u2) * cth / m,
#         (u1 + u2) * sth / m,
#         -g * cth, -g * sth,
#         omega * theta,
#         vx * vz,
#         u1 * u2 * cth, u1 * u2 * sth,
#     ]
#     
#     return features
# ```
# --- extracted code ---
def basis(xu):
    px, pz, theta, vx, vz, omega, u1, u2 = xu[0], xu[1], xu[2], xu[3], xu[4], xu[5], xu[6], xu[7]
    
    # Constants
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # Trigonometric functions
    cth = np.cos(theta)
    sth = np.sin(theta)
    
    # Basic monomials and interactions
    features = [
        1.0,  # bias
        px, pz, theta, vx, vz, omega, u1, u2,
        u1 + u2,  # total thrust
        u1 - u2,  # differential thrust
        cth, sth,
        u1 * cth, u1 * sth, u2 * cth, u2 * sth,
        (u1 + u2) * cth, (u1 + u2) * sth,
        (u1 - u2) * cth, (u1 - u2) * sth,
        theta**2, vx**2, vz**2, omega**2,
        u1**2, u2**2,
        u1 * u2,
        vx * u1, vx * u2, vz * u1, vz * u2,
        theta * u1, theta * u2,
        vx * cth, vx * sth, vz * cth, vz * sth,
        omega * u1, omega * u2,
        theta * vx, theta * vz, theta * omega,
        u1 * vx * cth, u1 * vz * sth, u2 * vx * cth, u2 * vz * sth,
        (u1 + u2) * g,
        (u1 - u2) * L / I_yy,
        (u1 + u2) * cth / m,
        (u1 + u2) * sth / m,
        -g * cth, -g * sth,
        omega * theta,
        vx * vz,
        u1 * u2 * cth, u1 * u2 * sth,
    ]
    
    return features

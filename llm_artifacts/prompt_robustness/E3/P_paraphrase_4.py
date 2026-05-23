"""P E3 paraphrase 4. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def observables(x):
#     px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
#     cth = np.cos(theta)
#     sth = np.sin(theta)
#     # First 6: identity (required)
#     psi = [px, pz, theta, vx, vz, omega]
#     # Quadratic terms capturing key couplings: thrust-induced accelerations depend on cos/sin(theta), 
#     # angular acceleration depends on torque (u1-u2) and thus on control inputs (handled by B, not psi),
#     # but state nonlinearities needed for accurate EDMDc: vx*cth, vz*cth, vx*sth, vz*sth (horizontal/vertical projections)
#     psi.append(vx * cth)
#     psi.append(vz * cth)
#     psi.append(vx * sth)
#     psi.append(vz * sth)
#     # Angular acceleration coupling: omega^2 appears in centrifugal terms, but more critical is theta dependence of gravity projection
#     # Gravity affects z-acceleration as -g*cos(theta), so include cos(theta) and sin(theta) explicitly
#     psi.append(cth)
#     psi.append(sth)
#     # Cross terms for rotational dynamics: theta*omega, omega*cth, omega*sth
#     psi.append(theta * omega)
#     psi.append(omega * cth)
#     psi.append(omega * sth)
#     # Velocity squared terms appear in drag (if modeled) but minimal model omits drag; instead, include v_x^2, v_z^2 for possible quadratic drag or Coriolis
#     psi.append(vx**2)
#     psi.append(vz**2)
#     # To capture coupling between translation and rotation in moments: vx*omega, vz*omega
#     psi.append(vx * omega)
#     psi.append(vz * omega)
#     return psi
# ```
# --- extracted code ---
def observables(x):
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    cth = np.cos(theta)
    sth = np.sin(theta)
    # First 6: identity (required)
    psi = [px, pz, theta, vx, vz, omega]
    # Quadratic terms capturing key couplings: thrust-induced accelerations depend on cos/sin(theta), 
    # angular acceleration depends on torque (u1-u2) and thus on control inputs (handled by B, not psi),
    # but state nonlinearities needed for accurate EDMDc: vx*cth, vz*cth, vx*sth, vz*sth (horizontal/vertical projections)
    psi.append(vx * cth)
    psi.append(vz * cth)
    psi.append(vx * sth)
    psi.append(vz * sth)
    # Angular acceleration coupling: omega^2 appears in centrifugal terms, but more critical is theta dependence of gravity projection
    # Gravity affects z-acceleration as -g*cos(theta), so include cos(theta) and sin(theta) explicitly
    psi.append(cth)
    psi.append(sth)
    # Cross terms for rotational dynamics: theta*omega, omega*cth, omega*sth
    psi.append(theta * omega)
    psi.append(omega * cth)
    psi.append(omega * sth)
    # Velocity squared terms appear in drag (if modeled) but minimal model omits drag; instead, include v_x^2, v_z^2 for possible quadratic drag or Coriolis
    psi.append(vx**2)
    psi.append(vz**2)
    # To capture coupling between translation and rotation in moments: vx*omega, vz*omega
    psi.append(vx * omega)
    psi.append(vz * omega)
    return psi

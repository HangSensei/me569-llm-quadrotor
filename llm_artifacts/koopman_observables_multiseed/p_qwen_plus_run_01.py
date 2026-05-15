import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Physical nonlinearities essential for planar quadrotor dynamics:
    # - Gravity projection depends on cos(theta) and sin(theta)
    # - Thrust projection into inertial frame: cos(theta) for horizontal, sin(theta) for vertical
    # - Angular acceleration depends on differential thrust (u1-u2) * L / I_yy, but since u is separate in EDMDc,
    #   we need state terms that multiply with control in the lifted space — however, observables are state-only,
    #   so we include terms that naturally appear in the dynamics: sin(theta), cos(theta), theta^2, etc.
    # - Velocity products appear in Coriolis-like terms (e.g., omega * vx, omega * vz) due to rotating frame
    # - Quadratic velocity terms model drag-like effects (even if not explicitly modeled, they improve fit)
    # - Cross-coupling terms between position/angle and velocity reflect geometric nonlinearities
    
    # Core physically motivated observables beyond identity:
    # 1. sin(theta) — appears in vertical thrust component
    # 2. cos(theta) — appears in horizontal thrust component and gravity projection
    # 3. theta^2 — captures higher-order angular potential or approximation of cos/sin
    # 4. vx^2, vz^2 — kinetic energy components, common in quadratic drag or inertia coupling
    # 5. vx * vz — cross-velocity coupling (e.g., in rotating frame accelerations)
    # 6. omega^2 — rotational kinetic energy
    # 7. theta * vx, theta * vz — linearized coupling of attitude and translation
    # 8. theta * omega — coupling of attitude and angular rate (e.g., in gyroscopic terms)
    # 9. vx * omega, vz * omega — Coriolis-type terms in body-to-inertial transformation
    # 10. sin(2*theta), cos(2*theta) — capture second-harmonic effects in thrust projection
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin2_theta = np.sin(2.0 * theta)
    cos2_theta = np.cos(2.0 * theta)
    
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        sin_theta,
        cos_theta,
        sin2_theta,
        cos2_theta,
        px * theta,
        pz * theta,
        vx * vx,
        vz * vz,
        omega * omega,
        vx * vz,
        vx * omega,
        vz * omega,
        theta * omega,
        theta * vx,
        theta * vz,
        px * vx,
        pz * vz,
        px * px,
        pz * pz,
        theta * theta,
        vx * sin_theta,
        vz * cos_theta,
        vx * cos_theta,
        vz * sin_theta
    ])
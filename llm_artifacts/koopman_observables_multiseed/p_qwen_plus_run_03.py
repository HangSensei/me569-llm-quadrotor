import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Physical nonlinearities in planar quadrotor dynamics:
    # - Gravity acts vertically: appears as constant offset in z-acceleration
    # - Thrust components depend on cos(theta) and sin(theta) for vertical/horizontal projection
    # - Angular acceleration depends on differential thrust (u1 - u2) * L / I_yy
    # - Coriolis/centrifugal terms in body-frame to inertial-frame velocity mapping: 
    #   v_x_dot = ... + omega * vz (rotational coupling), v_z_dot = ... - omega * vx
    # - Kinetic energy terms: vx^2, vz^2, omega^2
    # - Potential energy term: pz (for gravity coupling)
    # - Trigonometric terms essential for attitude-dependent forces: cos(theta), sin(theta)
    # - Coupling terms: theta * vx, theta * vz, theta * omega, vx * omega, vz * omega
    # - Quadratic velocity terms appear in drag-like corrections (even if not modeled explicitly, they aid Koopman lift)
    # - Also include theta^2 for symmetry and curvature near equilibrium
    
    # We include minimal physically motivated nonlinearities:
    # 1. Trigonometric basis for attitude dependence
    # 2. Velocity couplings from rotation
    # 3. Quadratic terms for kinetic energy and coupling
    # 4. Position-attitude coupling (e.g., for potential energy gradients in feedback)
    
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        # Trigonometric terms — essential for thrust projection
        cos_th,
        sin_th,
        # Quadratic velocity terms — appear in centripetal/Coriolis accelerations and kinetic energy
        vx**2,
        vz**2,
        omega**2,
        # Cross-coupling terms from rotation: Coriolis-like effects in translation
        vx * omega,
        vz * omega,
        # Attitude-velocity couplings — important for pitch-dependent force generation
        theta * vx,
        theta * vz,
        theta * omega,
        # Position-attitude coupling — relevant for potential energy gradients and visual servoing analogs
        px * theta,
        pz * theta,
        # Higher-order but low-rank: theta^2 (curvature of gravity projection), vx*vz (off-diagonal kinetic coupling)
        theta**2,
        vx * vz,
        # Angular acceleration related: sin(theta)*theta (small-angle approximation derivative), but include sin(theta)*vx etc.
        sin_th * vx,
        sin_th * vz,
        cos_th * vx,
        cos_th * vz,
        # Vertical position appears linearly in gravity, but quadratic in potential — include pz^2 for completeness
        pz**2,
        # Composite: cos(theta)*omega, sin(theta)*omega — angular rate modulation by attitude
        cos_th * omega,
        sin_th * omega,
        # Additional coupling: theta * cos_th, theta * sin_th — captures nonlinear stiffness near equilibrium
        theta * cos_th,
        theta * sin_th,
        # Velocity magnitude squared (inertial frame) — total kinetic energy proxy
        vx**2 + vz**2,
        # Angular momentum proxy
        omega * theta,
        # Small set of cubic terms with strong physical grounding: cos(theta)*vx^2 (thrust-modulated inertia), sin(theta)*vz^2
        cos_th * vx**2,
        sin_th * vz**2,
        # And one critical term: the horizontal acceleration coupling due to pitch — sin(theta) is primary, but sin(theta)*theta adds curvature
        sin_th * theta,
        # Finally, a stabilizing observable: the total energy-like term (not exact, but Koopman-informed): 0.5*(vx^2+vz^2) + 0.5*omega^2 + pz (gravity potential)
        # We omit factor 0.5 for scaling neutrality in regression; include linear pz already present, so add nonlinear part:
        vx**2 + vz**2 + omega**2 + pz
    ])
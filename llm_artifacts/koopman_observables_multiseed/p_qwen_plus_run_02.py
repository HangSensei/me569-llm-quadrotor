import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Physical nonlinearities essential for planar quadrotor dynamics:
    # - Gravity projection: cos(theta), sin(theta) for thrust decomposition
    # - Quadratic velocity terms: drag-like couplings (even if idealized, they capture inertial coupling)
    # - Angular-velocity coupling: omega*vx, omega*vz (Coriolis-like in rotating frame)
    # - Position-velocity products: px*vx, pz*vz (for potential energy gradients in lifted space)
    # - Trigonometric products: theta*cos(theta), theta*sin(theta) (to capture small-angle nonlinearities)
    # - Higher-order trig: cos(2*theta), sin(2*theta) (for torque coupling harmonics)
    # - Velocity magnitude squared: vx**2 + vz**2 (kinetic energy term)
    # - Angular kinetic energy: omega**2
    # - Cross terms: vx*vz, theta*omega (inertial coupling)
    
    # We avoid arbitrary high-degree polynomials; instead select physically motivated terms
    # that appear in the continuous-time equations of motion (after discretization):
    #   p_x' = v_x
    #   p_z' = v_z
    #   theta' = omega
    #   v_x' = (u1 + u2)/m * sin(theta)  [horizontal thrust component]
    #   v_z' = (u1 + u2)/m * cos(theta) - g  [vertical thrust minus gravity]
    #   omega' = (u2 - u1)*L / I_yy  [torque from differential thrust]
    
    # Thus, key nonlinear state dependencies are: sin(theta), cos(theta), 
    # and their products with velocities (e.g., sin(theta)*vx appears in Coriolis when transforming accelerations),
    # plus quadratic velocity terms from centripetal/Coriolis effects in rotating frame.
    
    # Also include: theta^2 (for potential energy curvature near equilibrium), 
    # and tan(theta) is unstable near ±π/2 — avoid; stick to sin/cos.
    
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        np.sin(theta),
        np.cos(theta),
        theta**2,
        vx**2,
        vz**2,
        omega**2,
        vx * vz,
        vx * omega,
        vz * omega,
        theta * vx,
        theta * vz,
        theta * omega,
        vx * np.sin(theta),
        vx * np.cos(theta),
        vz * np.sin(theta),
        vz * np.cos(theta),
        omega * np.sin(theta),
        omega * np.cos(theta),
        np.sin(2*theta),
        np.cos(2*theta),
        px * vx,
        pz * vz,
        px * theta,
        pz * theta,
        px * np.sin(theta),
        pz * np.cos(theta),
        # Add a constant term for affine dynamics (offsets like -g in v_z dynamics)
        1.0
    ])
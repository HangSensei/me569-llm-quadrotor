import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    px, pz, theta, vx, vz, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Physical constants
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # Core nonlinearities from planar quadrotor dynamics:
    # 1. Gravity projection: cos(theta) and sin(theta) for thrust decomposition
    # 2. Quadratic velocity terms from drag-like coupling (common in Koopman models for mechanical systems)
    # 3. Angular-velocity coupling: theta*omega, vx*omega, vz*omega (Coriolis-type interactions)
    # 4. Position-velocity couplings that appear in lifted representations of second-order systems
    # 5. Trigonometric terms essential for capturing pitch-dependent force projection
    # 6. Squared angular terms for rotational kinetic energy coupling
    
    # Trigonometric terms (essential for thrust projection: u1*cos(theta), u2*cos(theta), (u2-u1)*sin(theta))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Quadratic state terms that appear in bilinear/second-order dynamics
    vx_sq = vx * vx
    vz_sq = vz * vz
    omega_sq = omega * omega
    vx_vz = vx * vz
    px_theta = px * theta
    pz_theta = pz * theta
    vx_theta = vx * theta
    vz_theta = vz * theta
    theta_omega = theta * omega
    
    # Velocity-angle products critical for Coriolis and gyroscopic effects
    vx_cos = vx * cos_theta
    vz_cos = vz * cos_theta
    vx_sin = vx * sin_theta
    vz_sin = vz * sin_theta
    
    # Additional physically motivated terms: position-velocity cross terms and trig-velocity couplings
    px_vx = px * vx
    pz_vz = pz * vz
    theta_sq = theta * theta
    
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        cos_theta,
        sin_theta,
        vx_sq,
        vz_sq,
        omega_sq,
        vx_vz,
        theta_omega,
        vx_cos,
        vz_cos,
        vx_sin,
        vz_sin,
        theta_sq,
        px_theta,
        pz_theta,
        vx_theta,
        vz_theta,
        px_vx,
        pz_vz
    ])
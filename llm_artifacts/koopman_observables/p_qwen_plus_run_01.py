import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    p_x, p_z, theta, v_x, v_z, omega = x[0], x[1], x[2], x[3], x[4], x[5]
    
    # Physical constants
    m = 1.0
    I_yy = 0.01
    L = 0.25
    g = 9.81
    
    # Key nonlinear terms from planar quadrotor physics:
    # - Trigonometric functions of theta for thrust projection: cos(theta), sin(theta)
    # - Quadratic velocity terms for drag-like effects (even if not explicitly modeled, often needed for identification)
    # - Products of state and control: u1 and u2 appear linearly in dynamics but multiplied by trig functions
    # - Angular acceleration coupling: theta * omega, omega^2 (centrifugal/Coriolis-like)
    # - Cross-terms between linear and angular motion: v_x * sin(theta), v_z * cos(theta), etc.
    # - Since u is not part of x, we cannot include u-dependent terms here — observables are only function of x
    #   (EDMDc handles control separately via B matrix; so only state-based nonlinearities go here)
    
    # Essential trigonometric terms for rotation-coupled dynamics
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Velocity magnitude squared (common in aerodynamic drag approximations)
    v_sq = v_x**2 + v_z**2
    
    # Coupling terms between translation and rotation
    v_x_cos = v_x * cos_theta
    v_x_sin = v_x * sin_theta
    v_z_cos = v_z * cos_theta
    v_z_sin = v_z * sin_theta
    
    # Angular terms
    omega_sq = omega**2
    theta_omega = theta * omega
    
    # Higher-order but physically motivated: sin(2*theta), cos(2*theta) for torque modulation
    cos_2theta = np.cos(2.0 * theta)
    sin_2theta = np.sin(2.0 * theta)
    
    # Position-velocity cross terms (less critical but sometimes helpful for Koopman lift)
    p_x_v_x = p_x * v_x
    p_z_v_z = p_z * v_z
    
    # Return observables: state + selected nonlinear features
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        cos_theta,
        sin_theta,
        v_sq,
        v_x_cos,
        v_x_sin,
        v_z_cos,
        v_z_sin,
        omega_sq,
        theta_omega,
        cos_2theta,
        sin_2theta,
        p_x_v_x,
        p_z_v_z
    ])
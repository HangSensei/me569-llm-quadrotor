"""Controllers for the Planar Quadrotor.

Currently provides:
- LQRHoverController: classical linear-quadratic regulator linearized around
  the hover equilibrium. Used as the sanity-check baseline for Week 1 and as
  the initial guess / terminal controller for later MPC experiments.
"""

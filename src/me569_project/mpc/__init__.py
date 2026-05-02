"""Model Predictive Control (MPC) infrastructure for Experiment 3.

E3 evaluates whether LLM-generated stage cost functions can replace
hand-written quadratic costs in an MPC controller for the Planar
Quadrotor hover task. The pipeline mirrors E1's structure:

    LLM response -> extract_python_code -> sandbox.load_callable
                 -> mpc_wrapper.build_mpc(stage_cost=fn)
                 -> closed-loop rollout
                 -> hover success rate / settling time / control energy

Submodules:

- ``mpc_math``: re-exports of ``casadi`` math functions (sin, cos, exp,
  sqrt, log, fabs) plus the ``MPC_MATH_NAMESPACE`` dict for sandbox
  injection. Using these instead of ``numpy.sin`` etc. ensures the
  LLM-generated stage cost works on both numerical and CasADi symbolic
  inputs.
- ``mpc_wrapper`` (planned): do-mpc wrapper that takes a Python
  callable ``stage_cost(x, u) -> scalar``, builds the corresponding
  symbolic ``lterm`` for ``do_mpc.controller.MPC.set_objective``, and
  returns a ready-to-call MPC controller.
- ``baseline_cost`` (planned): hand-written quadratic stage cost that
  serves as the B condition for E3.
- ``evaluation`` (planned): closed-loop evaluation harness that runs
  N initial states through MPC and computes the three E3 metrics.
"""

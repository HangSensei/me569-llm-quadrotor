"""do-mpc wrapper for the Planar Quadrotor with a pluggable stage cost.

Why this wrapper exists
-----------------------
do-mpc's native API takes a CasADi symbolic expression for the stage
cost (``set_objective(lterm=...)``). E3's whole point is to plug in
**LLM-generated Python callables** as the stage cost and compare them
against a hand-written baseline through the same MPC machinery.

This wrapper bridges those two worlds: it accepts a Python callable
``stage_cost(x, u) -> scalar`` and constructs the do-mpc model and
controller around it. The trick is that CasADi's symbolic variables
support the same arithmetic operators as numpy scalars (``+``, ``-``,
``*``, ``**``, indexing), so passing the symbolic state and control
through the user's Python callable produces a symbolic expression
that do-mpc can compile into the optimization problem.

The math functions the LLM is allowed to use (``sin``, ``cos``, etc.)
are CasADi's own (see ``mpc_math.py``), so they propagate through
symbolic variables natively.

Architecture
------------
1. Build the do-mpc model with the same continuous-time dynamics as
   ``quadrotor_dynamics.quadrotor_dynamics``, but written in CasADi
   symbolic form.
2. Call the user's ``stage_cost(x_sym, u_sym)`` once during setup to
   build the symbolic ``lterm``.
3. Wrap that in a ``do_mpc.controller.MPC`` with horizon, control
   bounds, and an Ipopt-based NLP solver.
4. Expose ``make_step(x_current) -> u`` for use inside the closed-loop
   evaluation harness in ``mpc.evaluation``.

Configuration is held in an immutable ``MPCConfig`` dataclass so that
the closed-loop harness can swap horizons or terminal weights without
re-defining the dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import casadi
import do_mpc
import numpy as np

from me569_project.quadrotor_dynamics import QuadrotorParams


@dataclass(frozen=True)
class MPCConfig:
    """Configuration for the do-mpc PlanarQuadrotor wrapper."""

    horizon: int = 20             # prediction horizon (steps)
    dt: float = 0.02              # discretization step (s); matches the env
    terminal_weight: float = 10.0  # multiplier on stage cost at the final state
    suppress_ipopt_output: bool = True


class PlanarQuadrotorMPC:
    """do-mpc controller for the Planar Quadrotor with a pluggable stage cost.

    Parameters
    ----------
    params : QuadrotorParams
        Physical parameters used to build the symbolic model.
    stage_cost : Callable[[SX, SX], SX]
        A Python callable taking the 6-dim state and the 2-dim control
        and returning a scalar cost. Must work on CasADi symbolic
        variables (use ``mpc_math`` for trig / exp / sqrt rather than
        numpy).
    config : MPCConfig, optional
        Horizon, dt, terminal weight, and verbosity settings.
    """

    def __init__(
        self,
        params: QuadrotorParams,
        stage_cost: Callable,
        config: MPCConfig | None = None,
    ) -> None:
        self.params = params
        self.stage_cost = stage_cost
        self.config = config if config is not None else MPCConfig()

        self._model = self._build_model()
        self._mpc = self._build_controller(self._model)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    def _build_model(self) -> do_mpc.model.Model:
        """Build the symbolic continuous-time model that mirrors
        ``quadrotor_dynamics.quadrotor_dynamics``."""
        model = do_mpc.model.Model("continuous")

        # State variables
        p_x = model.set_variable(var_type="_x", var_name="p_x")
        p_z = model.set_variable(var_type="_x", var_name="p_z")
        theta = model.set_variable(var_type="_x", var_name="theta")
        v_x = model.set_variable(var_type="_x", var_name="v_x")
        v_z = model.set_variable(var_type="_x", var_name="v_z")
        omega = model.set_variable(var_type="_x", var_name="omega")

        # Control variables
        u_1 = model.set_variable(var_type="_u", var_name="u_1")
        u_2 = model.set_variable(var_type="_u", var_name="u_2")

        m = self.params.m
        I_yy = self.params.I_yy
        L = self.params.L
        g = self.params.g

        u_sum = u_1 + u_2
        sin_theta = casadi.sin(theta)
        cos_theta = casadi.cos(theta)

        # Continuous-time dynamics, identical to quadrotor_dynamics.py
        model.set_rhs("p_x", v_x)
        model.set_rhs("p_z", v_z)
        model.set_rhs("theta", omega)
        model.set_rhs("v_x", -u_sum * sin_theta / m)
        model.set_rhs("v_z", u_sum * cos_theta / m - g)
        model.set_rhs("omega", (u_2 - u_1) * L / I_yy)

        model.setup()
        return model

    # ------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------
    def _build_controller(
        self, model: do_mpc.model.Model
    ) -> do_mpc.controller.MPC:
        mpc = do_mpc.controller.MPC(model)

        suppress = {} if not self.config.suppress_ipopt_output else {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

        mpc.set_param(
            n_horizon=self.config.horizon,
            t_step=self.config.dt,
            store_full_solution=False,
            nlpsol_opts=suppress,
        )

        # Build the symbolic state and control vectors that the user's
        # stage cost will receive. We expose them as 6-/2- element
        # CasADi vertcat objects so the user can index x[0], x[1], ...
        x_vec = casadi.vertcat(
            model.x["p_x"],
            model.x["p_z"],
            model.x["theta"],
            model.x["v_x"],
            model.x["v_z"],
            model.x["omega"],
        )
        u_vec = casadi.vertcat(
            model.u["u_1"],
            model.u["u_2"],
        )

        lterm_expr = self.stage_cost(x_vec, u_vec)
        # Make sure the result is a scalar CasADi expression.
        lterm_expr = casadi.SX(lterm_expr)
        if lterm_expr.numel() != 1:
            raise ValueError(
                f"stage_cost must return a scalar, got shape {lterm_expr.shape}"
            )

        # do-mpc forbids the terminal cost ``mterm`` from depending on
        # control variables (there is no control to apply at the
        # terminal state). To build a state-only terminal cost from the
        # same user-supplied stage_cost, we re-evaluate it with ``u``
        # replaced by the per-rotor hover thrust as a numerical constant.
        # The resulting expression contains only symbolic state plus
        # numerical constants, which CasADi treats as a valid mterm.
        u_hover_const = casadi.DM(
            [self.params.u_hover, self.params.u_hover]
        )
        mterm_raw = self.stage_cost(x_vec, u_hover_const)
        mterm_expr = self.config.terminal_weight * casadi.SX(mterm_raw)

        mpc.set_objective(mterm=mterm_expr, lterm=lterm_expr)
        # Smoothing on control rate (small but non-zero so the QP is well-posed)
        mpc.set_rterm(u_1=1e-4, u_2=1e-4)

        # Action bounds: each rotor in [0, u_max]
        mpc.bounds["lower", "_u", "u_1"] = 0.0
        mpc.bounds["lower", "_u", "u_2"] = 0.0
        mpc.bounds["upper", "_u", "u_1"] = self.params.u_max
        mpc.bounds["upper", "_u", "u_2"] = self.params.u_max

        mpc.setup()
        return mpc

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def make_step(self, state: np.ndarray) -> np.ndarray:
        """Compute the optimal first control action for the given state.

        Parameters
        ----------
        state : np.ndarray of shape (6,)
            Current quadrotor state in the same ordering as
            ``quadrotor_dynamics.quadrotor_dynamics``.

        Returns
        -------
        np.ndarray of shape (2,)
            Per-rotor thrusts ``[u_1, u_2]``, clipped to ``[0, u_max]``.
        """
        x0 = np.asarray(state, dtype=np.float64).reshape(6, 1)
        self._mpc.x0 = x0
        self._mpc.set_initial_guess()

        u_opt = self._mpc.make_step(x0)
        u_arr = np.asarray(u_opt, dtype=np.float64).reshape(-1)
        # Defensive clip in case Ipopt returns something a hair outside
        # the bounds due to slack tolerances.
        u_arr = np.clip(u_arr, 0.0, self.params.u_max)
        return u_arr

    def reset(self) -> None:
        """Reset the controller's internal state guess so the next
        ``make_step`` starts fresh.
        """
        self._mpc.reset_history()

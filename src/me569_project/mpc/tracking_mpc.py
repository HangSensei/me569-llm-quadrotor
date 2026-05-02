"""Tracking MPC wrapper using do-mpc time-varying parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import casadi
import do_mpc
import numpy as np

from me569_project.quadrotor_dynamics import QuadrotorParams


@dataclass(frozen=True)
class TrackingMPCConfig:
    """Configuration for the tracking MPC wrapper."""

    horizon: int = 20
    dt: float = 0.02
    terminal_weight: float = 10.0
    suppress_ipopt_output: bool = True


class PlanarQuadrotorTrackingMPC:
    """do-mpc controller for reference tracking with time-varying references.

    ``stage_cost`` operates on the tracking error ``e = x - x_ref`` and the
    absolute control ``u``. The future reference trajectory enters via do-mpc
    TVPs so the optimizer sees the entire reference horizon at each step.
    """

    def __init__(
        self,
        params: QuadrotorParams,
        reference_fn: Callable[[float], np.ndarray],
        stage_cost: Callable,
        config: TrackingMPCConfig | None = None,
    ) -> None:
        self.params = params
        self.reference_fn = reference_fn
        self.stage_cost = stage_cost
        self.config = config if config is not None else TrackingMPCConfig()

        self._model = self._build_model()
        self._mpc = self._build_controller(self._model)

    def _build_model(self) -> do_mpc.model.Model:
        model = do_mpc.model.Model("continuous")

        p_x = model.set_variable(var_type="_x", var_name="p_x")
        p_z = model.set_variable(var_type="_x", var_name="p_z")
        theta = model.set_variable(var_type="_x", var_name="theta")
        v_x = model.set_variable(var_type="_x", var_name="v_x")
        v_z = model.set_variable(var_type="_x", var_name="v_z")
        omega = model.set_variable(var_type="_x", var_name="omega")

        u_1 = model.set_variable(var_type="_u", var_name="u_1")
        u_2 = model.set_variable(var_type="_u", var_name="u_2")

        for idx in range(6):
            model.set_variable(var_type="_tvp", var_name=f"x_ref_{idx}")

        m = self.params.m
        I_yy = self.params.I_yy
        L = self.params.L
        g = self.params.g

        u_sum = u_1 + u_2
        sin_theta = casadi.sin(theta)
        cos_theta = casadi.cos(theta)

        model.set_rhs("p_x", v_x)
        model.set_rhs("p_z", v_z)
        model.set_rhs("theta", omega)
        model.set_rhs("v_x", -u_sum * sin_theta / m)
        model.set_rhs("v_z", u_sum * cos_theta / m - g)
        model.set_rhs("omega", (u_2 - u_1) * L / I_yy)

        model.setup()
        return model

    def _reference_tvp_vector(self, model: do_mpc.model.Model):
        return casadi.vertcat(
            model.tvp["x_ref_0"],
            model.tvp["x_ref_1"],
            model.tvp["x_ref_2"],
            model.tvp["x_ref_3"],
            model.tvp["x_ref_4"],
            model.tvp["x_ref_5"],
        )

    def _build_controller(self, model: do_mpc.model.Model) -> do_mpc.controller.MPC:
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

        x_vec = casadi.vertcat(
            model.x["p_x"],
            model.x["p_z"],
            model.x["theta"],
            model.x["v_x"],
            model.x["v_z"],
            model.x["omega"],
        )
        u_vec = casadi.vertcat(model.u["u_1"], model.u["u_2"])
        x_ref_vec = self._reference_tvp_vector(model)
        error_vec = x_vec - x_ref_vec

        lterm_expr = casadi.SX(self.stage_cost(error_vec, u_vec))
        if lterm_expr.numel() != 1:
            raise ValueError(
                f"stage_cost must return a scalar, got shape {lterm_expr.shape}"
            )

        u_hover_const = casadi.DM([self.params.u_hover, self.params.u_hover])
        mterm_expr = self.config.terminal_weight * casadi.SX(
            self.stage_cost(error_vec, u_hover_const)
        )

        mpc.set_objective(mterm=mterm_expr, lterm=lterm_expr)
        mpc.set_rterm(u_1=1e-4, u_2=1e-4)

        mpc.bounds["lower", "_u", "u_1"] = 0.0
        mpc.bounds["lower", "_u", "u_2"] = 0.0
        mpc.bounds["upper", "_u", "u_1"] = self.params.u_max
        mpc.bounds["upper", "_u", "u_2"] = self.params.u_max

        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now: float):
            # do-mpc may pass t0 as a scalar-like array; normalize it once here.
            t_scalar = float(np.asarray(t_now, dtype=np.float64).reshape(-1)[0])
            for k in range(self.config.horizon + 1):
                ref = np.asarray(
                    self.reference_fn(t_scalar + k * self.config.dt),
                    dtype=np.float64,
                ).reshape(6)
                for idx, value in enumerate(ref):
                    tvp_template["_tvp", k, f"x_ref_{idx}"] = value
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        return mpc

    def make_step(self, state: np.ndarray) -> np.ndarray:
        x0 = np.asarray(state, dtype=np.float64).reshape(6, 1)
        self._mpc.x0 = x0
        self._mpc.set_initial_guess()
        u_opt = self._mpc.make_step(x0)
        u_arr = np.asarray(u_opt, dtype=np.float64).reshape(-1)
        return np.clip(u_arr, 0.0, self.params.u_max)

    def reset(self) -> None:
        self._mpc.reset_history()
        self._mpc.t0 = 0.0

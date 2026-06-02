"""Experiment Cascade — does E1's structural win propagate to closed-loop control?

Replaces do-mpc's symbolic ground-truth dynamics with the E1-fitted SINDy
model ``\\dot x = \\Xi \\Phi(x, u)`` for each condition (B / P / Q).
Cost is held constant at Bryson's-rule baseline; only the prediction
model varies. The hover-task evaluation harness is identical to the one
used in E2, so settling time, control energy, and hover-success rate
are directly comparable to the E2 single-seed numbers.

Conditions:
- B: hand-rolled polynomial-degree-3 basis (165 features, identical to E1
  baseline), STLSQ threshold 0.1.
- P: Qwen 3.6-Plus generated basis from `llm_artifacts/e1_bases/p_qwen_plus_run_01.py`.
- Q: Qwen3.5-4B generated basis from `llm_artifacts/e1_bases/q_qwen_local_run_01.py`.
- R: tuned RBF basis (cascade-capped M, from `llm_artifacts/rbf/e1_rbf_cascade.npz`).

Output:
- ``results/cascade_results.csv`` — one row per condition with closed-loop metrics.

Env vars:
- ``CASCADE_SKIP_B=1`` — skip the polynomial baseline (no LLM call needed; only
  used during smoke testing of the LLM conditions).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable

import casadi
import do_mpc
import numpy as np
from pysindy.optimizers import STLSQ

from me569_project.data.trajectory_generator import (
    TrajectoryData,
    generate_trajectories,
)
from me569_project.mpc.baseline_cost import make_baseline_stage_cost
from me569_project.mpc.evaluation import evaluate_controller
from me569_project.quadrotor_dynamics import QuadrotorParams
from me569_project.sysid.rbf_basis import build_rbf_e1


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "cascade_results.csv"
E1_ARTIFACTS = REPO_ROOT / "llm_artifacts" / "e1_bases"


# ---------------------------------------------------------------------------
# CasADi shim for LLM-generated bases that import numpy
# ---------------------------------------------------------------------------
# E1 LLM bases use ``np.sin(theta)`` and ``np.cos(theta)`` plus arithmetic
# on the elements of ``xu``. ``casadi.sin/cos`` work on both numpy floats
# and CasADi symbols, so we expose them under the ``np`` name. This lets
# the *same* basis function evaluate on numpy vectors during the STLSQ
# fit and on CasADi SX symbols inside the MPC RHS construction.
class _CasadiNumPyShim:
    """Drop-in replacement for ``numpy`` with the ops LLM bases actually use."""
    sin = staticmethod(casadi.sin)
    cos = staticmethod(casadi.cos)
    exp = staticmethod(casadi.exp)
    sqrt = staticmethod(casadi.sqrt)
    log = staticmethod(casadi.log)
    fabs = staticmethod(casadi.fabs)
    abs = staticmethod(casadi.fabs)
    tanh = staticmethod(casadi.tanh)

    @staticmethod
    def array(x):
        # If the LLM ever wraps the output in np.array, return a CasADi vertcat.
        if isinstance(x, (list, tuple)):
            return casadi.vertcat(*x)
        return x

    @staticmethod
    def power(a, b):
        # CasADi handles ** directly; np.power calls map cleanly.
        return a ** b


def load_basis_with_shim(code_path: Path) -> Callable:
    """Exec the saved E1 basis file with a CasADi-shim ``np`` and return ``basis``.

    Returned callable works on both numpy arrays (during fit) and CasADi
    symbols (during MPC RHS construction).
    """
    code = code_path.read_text()
    shim = _CasadiNumPyShim()
    namespace: dict = {"np": shim, "numpy": shim}
    exec(compile(code, str(code_path), "exec"), namespace)
    if "basis" not in namespace:
        raise RuntimeError(f"{code_path} does not define `basis`.")
    return namespace["basis"]


# ---------------------------------------------------------------------------
# Polynomial-degree-3 baseline (B) basis
# ---------------------------------------------------------------------------
_POLY_DEG3_TUPLES = []
for d in range(1, 4):
    for combo in combinations_with_replacement(range(8), d):
        _POLY_DEG3_TUPLES.append(combo)


def polynomial_basis_deg3(xu):
    """1 + 8 + 36 + 120 = 165 features (matches E1 baseline)."""
    feats = [1.0]
    for combo in _POLY_DEG3_TUPLES:
        term = xu[combo[0]]
        for i in combo[1:]:
            term = term * xu[i]
        feats.append(term)
    return feats


# ---------------------------------------------------------------------------
# STLSQ fit
# ---------------------------------------------------------------------------
def fit_xi_from_basis(
    basis_fn: Callable, train: TrajectoryData, threshold: float = 0.1,
) -> np.ndarray:
    """Build ``Phi`` and finite-difference ``Y``, fit ``Xi`` via STLSQ.

    Returns ``Xi`` of shape ``(n_basis, 6)``.
    """
    N = train.num_trajectories
    T = train.steps_per_trajectory
    dt = train.dt

    probe = np.zeros(8, dtype=np.float64)
    n_basis = len(np.asarray(basis_fn(probe), dtype=np.float64).reshape(-1))

    Phi = np.zeros((N * T, n_basis), dtype=np.float64)
    Y = np.zeros((N * T, 6), dtype=np.float64)

    row = 0
    for n in range(N):
        states = train.states[n]
        actions = train.actions[n]
        x_dot_true = (states[1:] - states[:-1]) / dt  # (T, 6)
        for t in range(T):
            xu = np.concatenate([states[t], actions[t]])
            phi = np.asarray(basis_fn(xu), dtype=np.float64).reshape(-1)
            Phi[row] = phi
            Y[row] = x_dot_true[t]
            row += 1

    optimizer = STLSQ(threshold=threshold)
    optimizer.fit(Phi, Y)
    Xi = np.asarray(optimizer.coef_, dtype=np.float64).T  # (n_basis, 6)
    return Xi


# ---------------------------------------------------------------------------
# Cascade MPC: do-mpc model whose RHS = Xi.T @ phi(x, u)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CascadeMPCConfig:
    horizon: int = 20
    dt: float = 0.02
    terminal_weight: float = 10.0
    suppress_ipopt_output: bool = True


class CascadePlanarQuadrotorMPC:
    """do-mpc controller whose continuous-time RHS comes from a SINDy fit.

    The dynamics are ``\\dot x = \\Xi^T \\Phi([x; u])`` — a linear map from
    the LLM/poly basis features to state derivatives, in continuous time
    (the same regime as the STLSQ fit).
    """

    def __init__(
        self,
        params: QuadrotorParams,
        basis_fn: Callable,
        Xi: np.ndarray,
        stage_cost: Callable,
        config: CascadeMPCConfig | None = None,
    ) -> None:
        if Xi.shape[1] != 6:
            raise ValueError(f"Xi must have 6 output rows; got {Xi.shape}")
        self.params = params
        self.basis_fn = basis_fn
        self.Xi = Xi
        self.stage_cost = stage_cost
        self.config = config or CascadeMPCConfig()
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

        # Build symbolic basis vector by passing CasADi SX symbols through
        # the basis function (the CasADi shim makes sin/cos work).
        xu_sym = [p_x, p_z, theta, v_x, v_z, omega, u_1, u_2]
        phi_sym = self.basis_fn(xu_sym)
        # phi_sym is a Python list of CasADi SX scalars. Stack to a column vec.
        phi_vec = casadi.vertcat(*phi_sym)
        if phi_vec.numel() != self.Xi.shape[0]:
            raise ValueError(
                f"basis returned {phi_vec.numel()} features symbolically, "
                f"Xi expects {self.Xi.shape[0]}"
            )

        # dx/dt = Xi.T @ phi  (since Xi has shape (n_basis, 6), Xi.T is (6, n_basis))
        Xi_dm = casadi.DM(self.Xi.T)
        dxdt = Xi_dm @ phi_vec  # (6, 1)

        names = ["p_x", "p_z", "theta", "v_x", "v_z", "omega"]
        for i, name in enumerate(names):
            model.set_rhs(name, dxdt[i])

        model.setup()
        return model

    def _build_controller(self, model) -> do_mpc.controller.MPC:
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
            model.x["p_x"], model.x["p_z"], model.x["theta"],
            model.x["v_x"], model.x["v_z"], model.x["omega"],
        )
        u_vec = casadi.vertcat(model.u["u_1"], model.u["u_2"])
        lterm_expr = casadi.SX(self.stage_cost(x_vec, u_vec))
        if lterm_expr.numel() != 1:
            raise ValueError(
                f"stage_cost must return scalar; got shape {lterm_expr.shape}"
            )

        u_hover_const = casadi.DM([self.params.u_hover, self.params.u_hover])
        mterm_expr = self.config.terminal_weight * casadi.SX(
            self.stage_cost(x_vec, u_hover_const)
        )

        mpc.set_objective(mterm=mterm_expr, lterm=lterm_expr)
        mpc.set_rterm(u_1=1e-4, u_2=1e-4)
        mpc.bounds["lower", "_u", "u_1"] = 0.0
        mpc.bounds["lower", "_u", "u_2"] = 0.0
        mpc.bounds["upper", "_u", "u_1"] = self.params.u_max
        mpc.bounds["upper", "_u", "u_2"] = self.params.u_max
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


# ---------------------------------------------------------------------------
# One-condition runner
# ---------------------------------------------------------------------------
def run_condition(
    label: str,
    basis_fn: Callable,
    train: TrajectoryData,
    description: str,
) -> dict | None:
    print(f"\n[{label}] Fitting STLSQ on E1-style basis ...")
    t0 = time.time()
    Xi = fit_xi_from_basis(basis_fn, train, threshold=0.1)
    fit_time = time.time() - t0
    n_basis = Xi.shape[0]
    n_active = int(np.count_nonzero(Xi))
    print(
        f"[{label}]   fit done in {fit_time:.2f}s, n_basis={n_basis}, "
        f"active_terms={n_active}"
    )

    print(f"[{label}] Building cascade MPC and running hover-task eval ...")
    params = QuadrotorParams()
    stage_cost = make_baseline_stage_cost(params)
    try:
        mpc = CascadePlanarQuadrotorMPC(
            params=params, basis_fn=basis_fn, Xi=Xi, stage_cost=stage_cost,
        )
    except Exception as e:
        print(f"[{label}]   MPC build FAILED: {e}")
        return {
            "condition": label,
            "n_basis": n_basis,
            "active_terms": n_active,
            "fit_time_s": fit_time,
            "build_failed": 1,
            "build_error": str(e)[:200],
            "description": description,
        }

    t1 = time.time()
    metrics = evaluate_controller(
        mpc, params=QuadrotorParams(), num_initial_states=20,
    )
    eval_time = time.time() - t1
    print(
        f"[{label}]   eval done in {eval_time:.1f}s: "
        f"success={metrics.hover_success_rate:.2f}, "
        f"settle={metrics.mean_settling_time_s:.2f}, "
        f"energy={metrics.mean_control_energy:.4f}"
    )

    return {
        "condition": label,
        "description": description,
        "n_basis": n_basis,
        "active_terms": n_active,
        "fit_time_s": fit_time,
        "eval_time_s": eval_time,
        "build_failed": 0,
        "hover_success_rate": metrics.hover_success_rate,
        "mean_settling_time_s": metrics.mean_settling_time_s,
        "mean_control_energy": metrics.mean_control_energy,
        "mean_final_position_error": metrics.mean_final_position_error,
        "mean_final_attitude_error": metrics.mean_final_attitude_error,
        "failed_due_to_crash": metrics.failed_due_to_crash,
        "failed_due_to_timeout": metrics.failed_due_to_timeout,
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"\nWrote {len(rows)} row(s) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("Generating 500 train + 100 val trajectories (same as E1) ...")
    t0 = time.time()
    train = generate_trajectories(
        num_trajectories=500, steps_per_trajectory=400, seed=0,
    )
    print(
        f"Data done in {time.time() - t0:.1f}s "
        f"({train.num_transitions} train transitions)"
    )

    rows: list[dict] = []

    # B — polynomial degree 3 (165 features), no LLM call.
    if not os.environ.get("CASCADE_SKIP_B"):
        b_row = run_condition(
            "B", polynomial_basis_deg3, train,
            description="polynomial degree 3 (165 features)",
        )
        if b_row is not None:
            rows.append(b_row)

    # P — Qwen 3.6-Plus from committed E1 artifact.
    p_path = E1_ARTIFACTS / "p_qwen_plus_run_01.py"
    if p_path.exists():
        try:
            p_basis = load_basis_with_shim(p_path)
            p_row = run_condition(
                "P", p_basis, train,
                description=f"Qwen 3.6-Plus E1 basis ({p_path.name})",
            )
            if p_row is not None:
                rows.append(p_row)
        except Exception as e:
            print(f"[P] FAILED to load basis: {e}")
    else:
        print(f"[P] Skipped: {p_path} not found.")

    # Q — Qwen3.5-4B from committed E1 artifact.
    q_path = E1_ARTIFACTS / "q_qwen_local_run_01.py"
    if q_path.exists():
        try:
            q_basis = load_basis_with_shim(q_path)
            q_row = run_condition(
                "Q", q_basis, train,
                description=f"Qwen3.5-4B E1 basis ({q_path.name})",
            )
            if q_row is not None:
                rows.append(q_row)
        except Exception as e:
            print(f"[Q] FAILED to load basis: {e}")
    else:
        print(f"[Q] Skipped: {q_path} not found.")

    # R — tuned RBF (cascade-capped M), module-built basis like B.
    try:
        r_basis = build_rbf_e1(REPO_ROOT, for_cascade=True)
        r_row = run_condition(
            "R", r_basis, train,
            description="tuned RBF basis (cascade-capped M)",
        )
        if r_row is not None:
            rows.append(r_row)
    except Exception as e:
        print(f"[R] FAILED to build/run: {e}")

    print()
    print("=" * 76)
    print("Cascade — closed-loop hover task with E1 SINDy plant model")
    print("=" * 76)
    print(
        f"{'cond':4s} {'n_basis':>8s} {'active':>8s} {'success':>8s} "
        f"{'settle':>8s} {'energy':>10s}"
    )
    for r in rows:
        if r.get("build_failed"):
            print(
                f"{r['condition']:4s} {r['n_basis']:8d} {r['active_terms']:8d} "
                f"BUILD FAILED: {r['build_error']}"
            )
            continue
        print(
            f"{r['condition']:4s} {r['n_basis']:8d} {r['active_terms']:8d} "
            f"{r['hover_success_rate']:8.2f} {r['mean_settling_time_s']:8.2f} "
            f"{r['mean_control_energy']:10.4f}"
        )
    print("=" * 76)

    write_csv(rows, RESULTS_CSV)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

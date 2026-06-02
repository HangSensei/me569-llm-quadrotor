# scripts/tune_rbf.py
"""Tune the RBF baseline (R) for E1 + E3 on a separate tune split (seed 7).

Grid: M in {50,100,200,400} x sigma in {0.5,1,2} * median heuristic. Selects
the lowest tune-set one-step MSE per stage; for the cascade (E1 closed loop)
selects the best config with M <= 100 (Ipopt tractability). Persists the chosen
centers/sigma/mean/std to llm_artifacts/rbf/{e1_rbf,e1_rbf_cascade,e3_rbf}.npz
and a human-readable manifest.json. No LLM calls.

Run:  uv run python scripts/tune_rbf.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.sysid.edmdc import edmdc_one_step_mse, fit_edmdc
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn
from me569_project.sysid.rbf_basis import (
    make_rbf_basis_fn,
    make_rbf_observable_fn,
    median_sigma,
    save_rbf_artifact,
    select_rbf_centers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RBF_DIR = REPO_ROOT / "llm_artifacts" / "rbf"

M_GRID = [50, 100, 200, 400]
# EDMDc validates the lifted-observable dim into [6, 50], and the E3 observable
# is [state(6), M kernels], so 6 + M <= 50 => M <= 44. Give E3 its own grid;
# this still gives the RBF a larger lift budget than the degree-2 polynomial
# baseline (27 dims), so it remains a fair steelman within the EDMDc contract.
E3_M_GRID = [10, 20, 30, 44]
SIGMA_SCALES = [0.5, 1.0, 2.0]
CASCADE_M_CAP = 100
KMEANS_SEED = 0
THRESHOLD = 0.1


def select_best(records: list[dict], m_cap: int | None = None) -> dict | None:
    """Return the record with the smallest ``mse`` (optionally M <= m_cap)."""
    pool = [r for r in records if (m_cap is None or r["M"] <= m_cap)]
    if not pool:
        return None
    return min(pool, key=lambda r: r["mse"])


def _flatten(data):
    X = data.states[:, :-1, :].reshape(-1, 6)
    U = data.actions.reshape(-1, 2)
    X_next = data.next_states.reshape(-1, 6)
    return X, U, X_next


def _xu_matrix(data) -> np.ndarray:
    X, U, _ = _flatten(data)
    return np.concatenate([X, U], axis=1)


def tune_e1(train, tune) -> list[dict]:
    XU = _xu_matrix(train)
    records: list[dict] = []
    for M in M_GRID:
        t0 = time.time()
        centers, mean, std = select_rbf_centers(XU, M, seed=KMEANS_SEED)
        base_sigma = median_sigma(centers)
        for scale in SIGMA_SCALES:
            sigma = scale * base_sigma
            basis, names = make_rbf_basis_fn(centers, sigma, mean, std)
            model = fit_with_basis_fn(train, basis, names, threshold=THRESHOLD)
            mse = model.one_step_mse(tune)
            print(f"  [E1] M={M:>3d} scale={scale} sigma={sigma:.3f} tune_mse={mse:.4e}")
            records.append({
                "M": M, "sigma": float(sigma), "scale": scale, "mse": float(mse),
                "centers": centers.copy(), "mean": mean, "std": std,
            })
        print(f"  [E1] M={M} done in {time.time()-t0:.1f}s")
    return records


def tune_e3(train, tune) -> list[dict]:
    Xtr, Utr, Xtr_next = _flatten(train)
    Xtu, Utu, Xtu_next = _flatten(tune)
    Xstate = Xtr
    records: list[dict] = []
    for M in E3_M_GRID:
        t0 = time.time()
        centers, mean, std = select_rbf_centers(Xstate, M, seed=KMEANS_SEED)
        base_sigma = median_sigma(centers)
        for scale in SIGMA_SCALES:
            sigma = scale * base_sigma
            obs = make_rbf_observable_fn(centers, sigma, mean, std)
            res = fit_edmdc(Xtr, Utr, Xtr_next, obs, observable_name=f"R-M{M}-s{scale}")
            mse = edmdc_one_step_mse(res, Xtu, Utu, Xtu_next)
            print(f"  [E3] M={M:>3d} scale={scale} sigma={sigma:.3f} tune_mse={mse:.4e}")
            records.append({
                "M": M, "sigma": float(sigma), "scale": scale, "mse": float(mse),
                "centers": centers.copy(), "mean": mean, "std": std,
            })
        print(f"  [E3] M={M} done in {time.time()-t0:.1f}s")
    return records


def _save(rec: dict, path: Path) -> None:
    save_rbf_artifact(path, rec["centers"], rec["sigma"], rec["mean"], rec["std"])


def main() -> int:
    RBF_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating train(seed0,500) + tune(seed7,100) ...")
    train = generate_trajectories(500, steps_per_trajectory=400, seed=0)
    tune = generate_trajectories(100, steps_per_trajectory=400, seed=7)

    print("E1 tuning ...")
    e1 = tune_e1(train, tune)
    e1_open = select_best(e1)
    e1_casc = select_best(e1, m_cap=CASCADE_M_CAP)

    print("E3 tuning ...")
    e3 = tune_e3(train, tune)
    e3_best = select_best(e3)

    assert e1_open is not None and e1_casc is not None and e3_best is not None, \
        "select_best returned None — check M_GRID vs CASCADE_M_CAP"

    _save(e1_open, RBF_DIR / "e1_rbf.npz")
    _save(e1_casc, RBF_DIR / "e1_rbf_cascade.npz")
    _save(e3_best, RBF_DIR / "e3_rbf.npz")

    manifest = {
        "tune_seed": 7,
        "kmeans_seed": KMEANS_SEED,
        "threshold": THRESHOLD,
        "m_grid": M_GRID,
        "e3_m_grid": E3_M_GRID,
        "sigma_scales": SIGMA_SCALES,
        "cascade_m_cap": CASCADE_M_CAP,
        "e1_open": {k: e1_open[k] for k in ("M", "sigma", "scale", "mse")},
        "e1_cascade": {k: e1_casc[k] for k in ("M", "sigma", "scale", "mse")},
        "e3": {k: e3_best[k] for k in ("M", "sigma", "scale", "mse")},
    }
    (RBF_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote artifacts + manifest to {RBF_DIR}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

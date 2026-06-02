import numpy as np
import pytest

from me569_project.data.trajectory_generator import generate_trajectories
from me569_project.sysid.edmdc import edmdc_one_step_mse, fit_edmdc
from me569_project.sysid.llm_sysid_model import fit_with_basis_fn
from me569_project.sysid.rbf_basis import (
    standardize_params,
    select_rbf_centers,
    median_sigma,
    make_rbf_basis_fn,
    make_rbf_observable_fn,
    save_rbf_artifact,
    load_rbf_artifact,
    build_rbf_e1,
    build_rbf_e3,
)


def test_standardize_params_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=3.0, size=(1000, 4))
    mean, std = standardize_params(data)
    assert mean.shape == (4,) and std.shape == (4,)
    z = (data - mean) / std
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-9)


def test_standardize_params_constant_column_no_divzero():
    data = np.ones((10, 3))            # zero variance everywhere
    mean, std = standardize_params(data)
    assert np.all(std > 0)             # floored, never zero


def test_select_rbf_centers_shape_and_determinism():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(5000, 8))
    c1, mean, std = select_rbf_centers(data, n_centers=50, seed=3)
    c2, _, _ = select_rbf_centers(data, n_centers=50, seed=3)
    assert c1.shape == (50, 8)
    assert mean.shape == (8,) and std.shape == (8,)
    assert np.allclose(c1, c2)         # deterministic for fixed seed


def test_median_sigma_positive():
    # centers [[0,0],[3,4],[6,8]]: pairwise distances 5, 10, 5 -> median 5.0
    centers = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    s = median_sigma(centers)
    assert s > 0
    assert np.isclose(s, 5.0)


def test_median_sigma_requires_two_centers():
    with pytest.raises(ValueError):
        median_sigma(np.zeros((1, 3)))


def _identity_scaling(d):
    return np.zeros(d), np.ones(d)


def test_rbf_basis_kernel_one_at_center_zero_far():
    centers = np.array([[0.0] * 8, [10.0] * 8])
    mean, std = _identity_scaling(8)
    basis, names = make_rbf_basis_fn(centers, sigma=1.0, mean=mean, std=std,
                                     include_affine=True)
    # at center 0: rbf0 == 1, rbf1 ~ 0
    feats = np.asarray(basis(np.zeros(8)), dtype=float)
    # layout: [1, z0..z7 (8), rbf0, rbf1]
    assert feats.shape[0] == 1 + 8 + 2
    assert np.isclose(feats[9], 1.0)        # rbf0 at its own center
    assert feats[10] < 1e-6                  # rbf1 (center at 10s) ~ 0
    assert names[0] == "1" and names[9] == "rbf0"


def test_rbf_basis_numpy_and_casadi_agree():
    casadi = pytest.importorskip("casadi")
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(5, 8))
    mean, std = rng.normal(size=8), np.abs(rng.normal(size=8)) + 0.5
    basis, _ = make_rbf_basis_fn(centers, sigma=1.3, mean=mean, std=std)
    xu_val = rng.normal(size=8)

    np_feats = np.asarray(basis(xu_val), dtype=float)

    syms = [casadi.SX.sym(f"x{i}") for i in range(8)]
    sym_feats = basis(syms)                      # list of SX (symbolic path)
    f = casadi.Function("f", syms, [casadi.vertcat(*sym_feats)])
    cas_feats = np.asarray(f(*list(xu_val))).reshape(-1)

    assert cas_feats.shape == np_feats.shape
    assert np.allclose(cas_feats, np_feats, atol=1e-9)


def test_rbf_basis_rejects_wrong_input_dim():
    centers = np.zeros((3, 8))
    mean, std = _identity_scaling(8)
    basis, _ = make_rbf_basis_fn(centers, sigma=1.0, mean=mean, std=std)
    with pytest.raises(ValueError):
        basis(np.zeros(5))


def test_rbf_observable_state_recovery_and_shape():
    rng = np.random.default_rng(4)
    centers = rng.normal(size=(7, 6))
    mean, std = np.zeros(6), np.ones(6)
    obs = make_rbf_observable_fn(centers, sigma=1.0, mean=mean, std=std)
    x = rng.normal(size=6)
    psi = np.asarray(obs(x), dtype=float)
    assert psi.shape[0] == 6 + 7
    assert np.allclose(psi[:6], x)        # first six entries == raw state


def test_rbf_observable_rejects_non_6dim_centers():
    with pytest.raises(ValueError):
        make_rbf_observable_fn(np.zeros((3, 8)), 1.0, np.zeros(8), np.ones(8))


def test_artifact_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    centers = rng.normal(size=(4, 8))
    mean, std = rng.normal(size=8), np.abs(rng.normal(size=8)) + 0.1
    p = tmp_path / "x.npz"
    save_rbf_artifact(p, centers, 1.7, mean, std)
    c2, s2, m2, st2 = load_rbf_artifact(p)
    assert np.allclose(c2, centers) and np.isclose(s2, 1.7)
    assert np.allclose(m2, mean) and np.allclose(st2, std)


def test_builders_load_from_repo_layout(tmp_path):
    rng = np.random.default_rng(6)
    rbf_dir = tmp_path / "llm_artifacts" / "rbf"
    rbf_dir.mkdir(parents=True)
    # E1 open-loop + cascade artifacts (8-dim), E3 (6-dim).
    save_rbf_artifact(rbf_dir / "e1_rbf.npz", rng.normal(size=(5, 8)), 1.0,
                      np.zeros(8), np.ones(8))
    save_rbf_artifact(rbf_dir / "e1_rbf_cascade.npz", rng.normal(size=(3, 8)),
                      1.0, np.zeros(8), np.ones(8))
    save_rbf_artifact(rbf_dir / "e3_rbf.npz", rng.normal(size=(4, 6)), 1.0,
                      np.zeros(6), np.ones(6))

    e1 = build_rbf_e1(tmp_path, for_cascade=False)
    e1c = build_rbf_e1(tmp_path, for_cascade=True)
    e3 = build_rbf_e3(tmp_path)
    assert len(e1(np.zeros(8))) == 1 + 8 + 5
    assert len(e1c(np.zeros(8))) == 1 + 8 + 3
    assert len(e3(np.zeros(6))) == 6 + 4


# --- New guard tests (fix #9) ---

def test_make_rbf_basis_fn_rejects_zero_sigma():
    centers = np.zeros((3, 8))
    mean, std = _identity_scaling(8)
    with pytest.raises(ValueError, match="sigma must be positive"):
        make_rbf_basis_fn(centers, sigma=0, mean=mean, std=std)


def test_make_rbf_basis_fn_rejects_negative_sigma():
    centers = np.zeros((3, 8))
    mean, std = _identity_scaling(8)
    with pytest.raises(ValueError, match="sigma must be positive"):
        make_rbf_basis_fn(centers, sigma=-1.0, mean=mean, std=std)


def test_rbf_basis_casadi_path_rejects_wrong_length():
    casadi = pytest.importorskip("casadi")
    centers = np.zeros((3, 8))
    mean, std = _identity_scaling(8)
    basis, _ = make_rbf_basis_fn(centers, sigma=1.0, mean=mean, std=std)
    # pass a list of length 3 instead of 8 — hits the CasADi branch guard
    with pytest.raises(ValueError, match="basis expects 8-dim input"):
        basis([casadi.SX.sym("a"), casadi.SX.sym("b"), casadi.SX.sym("c")])


def test_make_rbf_observable_fn_rejects_mismatched_mean_std_dim():
    # centers (4, 6), but mean/std are 5-dim — should raise
    centers = np.zeros((4, 6))
    with pytest.raises(ValueError, match="mean/std dim must match centers dim"):
        make_rbf_observable_fn(centers, sigma=1.0, mean=np.zeros(5), std=np.ones(5))


def test_make_rbf_observable_fn_rejects_zero_sigma():
    centers = np.zeros((4, 6))
    with pytest.raises(ValueError, match="sigma must be positive"):
        make_rbf_observable_fn(centers, sigma=0, mean=np.zeros(6), std=np.ones(6))


def test_rbf_smoke_fit_e1_and_e3():
    """End-to-end: an RBF basis/observable fits through STLSQ (E1) and EDMDc
    (E3) on a tiny dataset and yields a finite, non-negative MSE. Guards the
    integration path between rbf_basis and the fitting harnesses."""
    data = generate_trajectories(5, steps_per_trajectory=20, seed=0)

    # E1: RBF SINDy basis through fit_with_basis_fn
    xu = np.concatenate(
        [data.states[:, :-1, :].reshape(-1, 6), data.actions.reshape(-1, 2)],
        axis=1,
    )
    centers, mean, std = select_rbf_centers(xu, n_centers=5, seed=0)
    basis, names = make_rbf_basis_fn(centers, median_sigma(centers), mean, std)
    model = fit_with_basis_fn(data, basis, names, threshold=0.1)
    mse_e1 = model.one_step_mse(data)
    assert np.isfinite(mse_e1) and mse_e1 >= 0

    # E3: RBF Koopman observable through fit_edmdc (lift dim 6+5=11 <= 50)
    x_state = data.states[:, :-1, :].reshape(-1, 6)
    u = data.actions.reshape(-1, 2)
    x_next = data.next_states.reshape(-1, 6)
    c6, m6, s6 = select_rbf_centers(x_state, n_centers=5, seed=0)
    obs = make_rbf_observable_fn(c6, median_sigma(c6), m6, s6)
    res = fit_edmdc(x_state, u, x_next, obs, observable_name="R-smoke")
    mse_e3 = edmdc_one_step_mse(res, x_state, u, x_next)
    assert np.isfinite(mse_e3) and mse_e3 >= 0

"""src.yukawa (exp 030): convention self-checks and reproduction of exps 014, 022, 023 from their stored inputs."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from src import yukawa as Y
from src.mixing import extract_angles, standard_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def _load(path, name):
    """Import an experiment module under a unique name (exps 022 and 023 both have model.py)."""
    d = str(path.parent)
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if added:
            sys.path.remove(d)
    return mod


def _wrapdiff(a, b, period):
    return (a - b + period / 2) % period - period / 2


# ---------------------------------------------------------------- convention self-checks

def _random_unitary(rng):
    z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_takagi_reconstructs():
    rng = np.random.default_rng(1)
    for _ in range(20):
        U = _random_unitary(rng)
        m = np.sort(rng.uniform(0.01, 1, 3))
        M = U.conj() @ np.diag(m) @ U.conj().T
        V, mi = Y.takagi(M)
        assert np.allclose(mi, m, rtol=1e-10)
        assert np.allclose(V.T @ M @ V, np.diag(mi), atol=1e-12)
        assert np.allclose(V.conj() @ np.diag(mi) @ V.conj().T, M, atol=1e-12)


def test_planted_pmns_and_majorana_phases():
    """Build Y_e diagonal and m_nu = U* D U^dag from a PDG matrix with Majorana phases; recover all."""
    t12, t13, t23, d, a21, a31 = 0.59, 0.15, 0.84, 4.1, 1.3, 5.0
    U = standard_matrix(t12, t13, t23, d) @ np.diag([1, np.exp(0.5j * a21), np.exp(0.5j * a31)])
    m = np.array([0.003, 0.0095, 0.05])
    mnu = U.conj() @ np.diag(m) @ U.conj().T
    I = np.diag([1e-6, 1e-3, 1e-1])
    o = Y.observables(I, I * 2, np.diag([3e-6, 6e-4, 1e-2]), mnu)
    assert o["s12sq"] == pytest.approx(np.sin(t12) ** 2, abs=1e-12)
    assert o["s13sq"] == pytest.approx(np.sin(t13) ** 2, abs=1e-12)
    assert o["s23sq"] == pytest.approx(np.sin(t23) ** 2, abs=1e-12)
    assert np.radians(o["dcp_deg"]) == pytest.approx(d, abs=1e-10)
    assert abs(_wrapdiff(np.radians(o["alpha21_deg"]), a21, 2 * np.pi)) < 1e-9
    assert abs(_wrapdiff(np.radians(o["alpha31_deg"]), a31, 2 * np.pi)) < 1e-9
    assert o["dm_ratio"] == pytest.approx((m[1] ** 2 - m[0] ** 2) / (m[2] ** 2 - m[0] ** 2), rel=1e-12)
    assert o["mr_d_s"] == pytest.approx(1e-3, rel=1e-12) and o["mr_b_tau"] == pytest.approx(20.0, rel=1e-12)
    # charged-lepton rotation enters as U_e^dag: Y_e = W diag W'^dag gives U_PMNS = W^dag U
    W = standard_matrix(0.2, 0.01, 0.05, 1.0)
    o2, det = Y.observables(I, I, W @ np.diag([3e-6, 6e-4, 1e-2]), mnu, details=True)
    U2 = W.conj().T @ U
    assert np.allclose(np.abs(det["U_pmns"]), np.abs(U2), atol=1e-12)
    assert np.radians(o2["dcp_deg"]) == pytest.approx(float(extract_angles(U2)[3]), abs=1e-10)


def test_ckm_planted():
    V = standard_matrix(0.227, 0.0037, 0.042, 1.14)
    Yu = np.diag([1e-5, 3e-3, 0.9])
    Yd = V @ np.diag([2e-5, 4e-4, 2e-2]) @ standard_matrix(0.3, 0.2, 0.1, 0.5)  # right rotation irrelevant
    o = Y.observables(Yu, Yd, np.eye(3), np.diag([0.001, 0.009, 0.05]))
    t12, t13, t23, d = extract_angles(V)
    assert o["ckm_s12"] == pytest.approx(np.sin(t12), abs=1e-12)
    assert o["ckm_s13"] == pytest.approx(np.sin(t13), abs=1e-12)
    assert o["ckm_s23"] == pytest.approx(np.sin(t23), abs=1e-12)
    assert o["ckm_delta_rad"] == pytest.approx(float(d), abs=1e-9)


def test_io_labelling_and_seesaw():
    U = standard_matrix(0.59, 0.15, 0.84, 1.0)
    m = np.array([0.0495, 0.05, 0.001])  # m1, m2, m3: IO
    mnu = U.conj() @ np.diag(m) @ U.conj().T
    I = np.diag([1e-3, 1e-2, 1.0])
    o, det = Y.observables(I, I, I, mnu, details=True, nu_mass_scale_eV=1.0)
    assert det["ordering"] == "IO"
    assert o["s12sq"] == pytest.approx(np.sin(0.59) ** 2, abs=1e-10)
    assert o["s13sq"] == pytest.approx(np.sin(0.15) ** 2, abs=1e-10)
    assert o["dm3l_sq_eV2"] == pytest.approx(m[2] ** 2 - m[1] ** 2, rel=1e-10) and o["dm3l_sq_eV2"] < 0
    assert o["dm_ratio"] == pytest.approx((m[1] ** 2 - m[0] ** 2) / abs(m[2] ** 2 - m[1] ** 2), rel=1e-8)
    # seesaw: -Y^* M^-1 Y^dag equals the direct matrix
    rng = np.random.default_rng(3)
    Yn = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    MR = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    MR = MR + MR.T
    a = Y.observables(I, I, I, Y_nu=Yn, M_R=MR)
    b = Y.observables(I, I, I, -Yn.conj() @ np.linalg.inv(MR) @ Yn.conj().T)
    assert all(a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9) for k in a)
    with pytest.raises(ValueError):
        Y.observables(I, I, I)
    with pytest.raises(ValueError):
        Y.observables(I, I, I, mnu, Y_nu=Yn, M_R=MR)


def test_evaluate_consumes_output():
    from src import evaluate as ev
    U = standard_matrix(0.59, 0.15, 0.84, 3.6)
    mnu = U.conj() @ np.diag([0.001, 0.0087, 0.05]) @ U.conj().T
    o = Y.observables(np.diag([1e-5, 3e-3, 1]), np.diag([1e-5, 2e-4, 1e-2]), np.diag([3e-6, 6e-4, 1e-2]), mnu)
    r = ev.evaluate(o, "NO", use=("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio", "mr_d_s"), k=0)
    assert np.isfinite(r.chi2)


# ---------------------------------------------------------------- exp 014 (SO(10) FN, Weyl RL)

def test_reproduces_exp014_fn():
    so = _load(EXP / "20261118-014-so10-fits" / "so10fit.py", "_exp014_so10fit")
    for tag in ("fit_FN_ours.json", "fit_FN_DR.json"):
        d = json.loads((EXP / "20261118-014-so10-fits" / tag).read_text())
        th = np.array([[d["best_theta"][k] for k in so.FN_NAMES]])
        Mu, Md, Me, Mnu = (M[0] for M in so.fn_mats(th))
        ref = d["best_pred"]
        dag = lambda M: M.conj().T
        # mapping: Y_f = M_f^dag; m_nu = M_nu as given; seesaw path Y_nu = M_D^dag, M_R = F
        v = so.fn_values(th)[0]
        F = np.zeros((3, 3))
        k = 2
        for i in range(3):
            for j in range(i, 3):
                F[i, j] = F[j, i] = v[k]
                k += 1
        H = np.diag([v[0], v[1], 1.0])
        G = np.zeros((3, 3))
        for i, j in ((0, 1), (0, 2), (1, 2)):
            G[i, j], G[j, i] = v[k], -v[k]
            k += 1
        MD = H - 3 * v[12] * F + 1j * v[14] * G
        for o in (Y.observables(dag(Mu), dag(Md), dag(Me), Mnu, ordering="IO" if d["best_IO"] else "NO"),
                  Y.observables(dag(Mu), dag(Md), dag(Me), Y_nu=dag(MD), M_R=F + 0j,
                                ordering="IO" if d["best_IO"] else "NO")):
            if "lg_mu_mc" in ref:  # exp 011/014 z vector (fit to our GUT data)
                got = {"lg_mu_mc": np.log10(o["mr_u_c"]), "lg_mc_mt": np.log10(o["mr_c_t"]),
                       "lg_md_ms": np.log10(o["mr_d_s"]), "lg_ms_mb": np.log10(o["mr_s_b"]),
                       "lg_me_mmu": np.log10(o["mr_e_mu"]), "lg_mmu_mtau": np.log10(o["mr_mu_tau"]),
                       "lg_mtau_mb": -np.log10(o["mr_b_tau"]), "s12q": o["ckm_s12"],
                       "lg_s13q": np.log10(o["ckm_s13"]), "lg_s23q": np.log10(o["ckm_s23"]),
                       "s12sq_l": o["s12sq"], "s13sq_l": o["s13sq"], "s23sq_l": o["s23sq"],
                       "lg_dm21_dm3l": np.log10(o["dm_ratio"])}
                want, dq, dl = ref, ref["dq"], ref["dl"]
            else:  # 1306.4468 DR observables (GeV, eV^2): compare the dimensionless combinations
                got = {"d_s": o["mr_d_s"], "s_b": o["mr_s_b"], "u_c": o["mr_u_c"], "c_t": o["mr_c_t"],
                       "e_mu": o["mr_e_mu"], "mu_tau": o["mr_mu_tau"], "b_tau": o["mr_b_tau"],
                       "s12q": o["ckm_s12"], "s13q": o["ckm_s13"], "s23q": o["ckm_s23"], "s12sq": o["s12sq"],
                       "s13sq": o["s13sq"], "s23sq": o["s23sq"], "dm": o["dm_ratio"]}
                want = {"d_s": ref["md"] / ref["ms"], "s_b": ref["ms"] / ref["mb"], "u_c": ref["mu"] / ref["mc"],
                        "c_t": ref["mc"] / ref["mt"], "e_mu": ref["me"] / ref["mmu"],
                        "mu_tau": ref["mmu"] / ref["mtau"], "b_tau": ref["mb"] / ref["mtau"],
                        "s12q": ref["s12q"], "s13q": ref["s13q"], "s23q": ref["s23q"], "s12sq": ref["s12sq"],
                        "s13sq": ref["s13sq"], "s23sq": ref["s23sq"], "dm": ref["dm21"] / ref["dm31"]}
                dq, dl = ref["dCKM"], None
            for key, val in got.items():
                assert val == pytest.approx(want[key], rel=1e-7, abs=1e-9), (tag, key)
            assert abs(_wrapdiff(o["ckm_delta_rad"], dq, 2 * np.pi)) < 1e-8
            if dl is None:
                continue
            assert abs(_wrapdiff(np.radians(o["dcp_deg"]), dl, 2 * np.pi)) < 1e-7


# ---------------------------------------------------------------- exp 022 (SU(5)+CSD3, LR, m m^dag)

def test_reproduces_exp022():
    M = _load(EXP / "20261119-022-su5-csd3" / "model.py", "_exp022_model")
    d = json.loads((EXP / "20261119-022-su5-csd3" / "fit_primary.json").read_text())
    th, ref = d["params"], d["pred_gut"]
    u = (th["u11"], th["u12"], th["u13"], th["u22"], th["u23"], 1.0)
    dd = (th["d11"], th["d12"], th["d22"], 1.0)
    Yu, Yd, Ye = M.yukawas(u, dd)
    mnu = M.mnu(th["r"], d["eta"])
    o = Y.observables(Yu, Yd, Ye, mnu.conj(), ordering="NO")  # mapping: m_nu = conj(m_022)
    for key in ("ckm_s12", "ckm_s13", "ckm_s23", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau",
                "mr_b_tau", "s12sq", "s13sq", "s23sq", "dm_ratio", "R_clebsch"):
        assert o[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key
    assert abs(_wrapdiff(o["ckm_delta_rad"], ref["ckm_delta_rad"], 2 * np.pi)) < 1e-9
    assert abs(_wrapdiff(o["dcp_deg"], ref["dcp_deg"], 360)) < 1e-7
    # the paper benchmark (exp 022 reproduce.json, tan beta 5): published CP signs need m_nu = conj(m_022)
    rep = json.loads((EXP / "20261119-022-su5-csd3" / "reproduce.json").read_text())["tb5_selected"]
    b = dict(u=(0.9566, 0.7346, 0.7198, 0.5961, 0.3224, 0.5435), d=(2.133, 0.8363, 1.108, 1.021))
    Yu, Yd, Ye = M.yukawas(b["u"], b["d"])
    o = Y.observables(Yu, Yd, Ye, M.mnu(2.684e-3 / 26.57e-3, ma=26.57e-3).conj(), nu_mass_scale_eV=1.0)
    assert np.degrees(np.arcsin(np.sqrt(o["s13sq"]))) == pytest.approx(rep["th13l"], abs=1e-8)
    assert _wrapdiff(o["dcp_deg"], 0, 360) == pytest.approx(rep["dl"], abs=1e-7)
    assert o["dm21sq_eV2"] == pytest.approx(rep["dm21"], rel=1e-9)
    assert o["dm3l_sq_eV2"] == pytest.approx(rep["dm31"], rel=1e-9)
    # without the conjugation delta_l flips sign (exp 022's rejected convention)
    o_bad = Y.observables(Yu, Yd, Ye, M.mnu(2.684e-3 / 26.57e-3, ma=26.57e-3))
    assert np.sign(_wrapdiff(o_bad["dcp_deg"], 0, 360)) == -np.sign(rep["dl"])


# ---------------------------------------------------------------- exp 023 (S4xSO(10), LR, Ue^T m* Ue)

def test_reproduces_exp023():
    M = _load(EXP / "20261119-023-s4so10" / "model.py", "_exp023_model")
    rep = json.loads((EXP / "20261119-023-s4so10" / "reproduce.json").read_text())["conj=False"]["at_published_point"]["pred"]
    Yu, Yd, Ye, mnu = M.matrices(M.PUBLISHED)
    o, det = Y.observables(Yu, Yd, Ye, mnu.conj(), details=True, nu_mass_scale_eV=1e-3)  # mu_i in meV
    l12, l13, l23 = (np.degrees(np.arcsin(np.sqrt(o[k]))) for k in ("s12sq", "s13sq", "s23sq"))
    assert l12 == pytest.approx(rep["th12l_deg"], abs=1e-8)
    assert l13 == pytest.approx(rep["th13l_deg"], abs=1e-8)
    assert l23 == pytest.approx(rep["th23l_deg"], abs=1e-8)
    assert o["dcp_deg"] == pytest.approx(rep["dl_deg"], abs=1e-7)
    assert np.degrees(o["ckm_delta_rad"]) % 360 == pytest.approx(rep["dq_deg"], abs=1e-7)
    assert np.degrees(np.arcsin(o["ckm_s13"])) == pytest.approx(rep["th13q_deg"], abs=1e-9)
    assert abs(_wrapdiff(o["alpha21_deg"], rep["alpha21_deg"], 360)) < 1e-6
    assert abs(_wrapdiff(o["alpha31_deg"], rep["alpha31_deg"], 360)) < 1e-6
    assert o["dm21sq_eV2"] / 1e-5 == pytest.approx(rep["dm21_1e-5eV2"], rel=1e-9)
    assert o["dm3l_sq_eV2"] / 1e-3 == pytest.approx(rep["dm31_1e-3eV2"], rel=1e-9)
    assert det["y_d"][2] == pytest.approx(rep["yb"], rel=1e-10)
    # the eb08 refit point (fit_eb08.json): keys not touched by exp 023's linear eta_b treatment
    d = json.loads((EXP / "20261119-023-s4so10" / "fit_eb08.json").read_text())
    Yu, Yd, Ye, mnu = M.matrices(d["params"])
    o = Y.observables(Yu, Yd, Ye, mnu.conj())
    ref = d["pred_gut"]
    for key in ("s12sq", "s13sq", "s23sq", "dm_ratio", "ckm_s12", "mr_u_c", "mr_c_t", "mr_d_s", "mr_e_mu",
                "mr_mu_tau"):
        assert o[key] == pytest.approx(ref[key], rel=1e-9), key
    eb = d["eta_b"]  # exp 023 fitlib.apply_eta_b: s13, s23, m_s/m_b divided by (1 + eta_b), m_b/m_tau times it
    for key in ("ckm_s13", "ckm_s23", "mr_s_b"):
        assert o[key] / (1 + eb) == pytest.approx(ref[key], rel=1e-9), key
    assert o["mr_b_tau"] * (1 + eb) == pytest.approx(ref["mr_b_tau"], rel=1e-9)
    assert abs(_wrapdiff(o["dcp_deg"], ref["dcp_deg"], 360)) < 1e-7
    assert abs(_wrapdiff(o["ckm_delta_rad"], ref["ckm_delta_rad"], 2 * np.pi)) < 1e-9

import numpy as np
import pytest

from src import rg
from src.mixing import unitarity_defect

D = np.pi / 180


@pytest.mark.parametrize("model,tb", [("SM", None), ("MSSM", 10), ("MSSM", 50)])
def test_up_then_down_returns_inputs(model, tb):
    s = rg.build_mz_state("IO", 0.05, (0.3, 1.2))
    mz = s["mu"]
    up = rg.run(mz, 2e16, s, model, tb)
    down = rg.run(2e16, mz, up, model, tb)
    assert down["normalisation"] == "SM"
    for k in ("g", "Yu", "Yd", "Ye", "kappa"):
        assert np.abs(down[k] - s[k]).max() / np.abs(s[k]).max() < 1e-8, k
    assert abs(down["lam"] - s["lam"]) < 1e-9


def test_mz_state_reproduces_inputs():
    pm = (33.0 * D, 8.5 * D, 45.5 * D, 200 * D)
    ck = rg.AHS_EQ6_CKM
    s = rg.build_mz_state("NO", 0.02, (0.4, 1.0), ckm=ck, pmns=pm)
    o = rg.observables(s)
    assert np.allclose([o["ckm"][k] for k in ("theta12", "theta13", "theta23", "delta")], [ck[0], ck[1], ck[2], ck[3]],
                       rtol=1e-9)
    assert np.allclose([o["pmns"][k] for k in ("theta12", "theta13", "theta23", "delta")],
                       [pm[0], pm[1], pm[2], pm[3]], rtol=1e-9)
    assert np.allclose(o["pmns"]["m"], rg.neutrino_masses("NO", 0.02), rtol=1e-9)


def test_unitarity_preserved_at_gut():
    s = rg.run_mssm(91.1879, 2e16, rg.build_mz_state("NO", 0.05), tan_beta=50)
    _, Uu = rg._left_basis(s["Yu"])
    _, Ud = rg._left_basis(s["Yd"])
    assert unitarity_defect(Uu.conj().T @ Ud) < 1e-12
    k = s["kappa"]
    assert np.abs(k - k.T).max() / np.abs(k).max() < 1e-12  # kappa stays symmetric


def test_derived_mz_yukawas_match_published():
    """Our M_Z Yukawas from data/ vs arXiv:2510.01312 Eq. (4) (PDG 2024 inputs): within 1.5%."""
    ours = rg.derived_yukawas_mz()
    for k, v in rg.AHS_EQ4_YUKAWAS.items():
        assert abs(ours[k] / v - 1) < 0.015, k


# arXiv:2510.01312v2 Table 2 (SM, 1e16 GeV) and Table 6 (MSSM, M_SUSY = 3 TeV, 2e16 GeV)
SM_1E16 = {"theta12": 0.22708, "theta23": 4.743e-2, "theta13": 4.19e-3, "delta": 1.139}
MSSM_GUT = {10: {"theta23": 3.874e-2, "theta13": 3.41e-3}, 50: {"theta23": 3.679e-2, "theta13": 3.24e-3}}


def test_sm_ckm_running_matches_published():
    c = rg.ckm_at(1e16, yukawas=rg.AHS_EQ4_YUKAWAS, ckm=rg.AHS_EQ6_CKM)
    assert abs(c["theta23"] / SM_1E16["theta23"] - 1) < 0.003
    assert abs(c["theta13"] / SM_1E16["theta13"] - 1) < 0.003
    assert abs(c["theta12"] / SM_1E16["theta12"] - 1) < 5e-4
    assert abs(c["delta"] - SM_1E16["delta"]) < 1e-3
    assert abs(c["yukawas"]["y_t"] / 0.4454 - 1) < 0.02  # one-loop Yukawa running: ~1% high
    assert np.allclose(c["g"], [0.575240, 0.52296, 0.5310], rtol=0.01)


@pytest.mark.parametrize("tb", [10, 50])
def test_mssm_ckm_running_matches_published(tb):
    """One-loop Yukawa RGEs reproduce 73-88% of the published two-loop MSSM shift: tolerance 3%."""
    c = rg.ckm_at(2e16, "MSSM", tb, 3000.0, yukawas=rg.AHS_EQ4_YUKAWAS, ckm=rg.AHS_EQ6_CKM)
    for k, v in MSSM_GUT[tb].items():
        assert abs(c[k] / v - 1) < 0.03, k
        assert c[k] < {"theta23": 4.194e-2, "theta13": 3.70e-3}[k]  # runs down in the MSSM
    assert abs(c["theta12"] / 0.22702 - 1) < 5e-4


def _pmns_rates():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "20261118-008-rg-module" / "run_tables.py"
    spec = importlib.util.spec_from_file_location("rg_tables", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("case", [("NO", 0.05, (0.0, 0.0), "SM", None), ("IO", 0.05, (0.0, 0.0), "SM", None),
                                  ("NO", 0.0, (0.0, 0.0), "MSSM", 50), ("NO", 0.1, (0.5, 1.0), "MSSM", 30)])
def test_pmns_rates_match_analytic(case):
    """dtheta/dt at M_Z vs hep-ph/0305273 eqs. (8)-(10) at small theta13: within 1.5%."""
    num, ana = _pmns_rates().numeric_rates(*case)
    assert np.all(np.abs(num / ana - 1) < 0.015), (num, ana)


def test_sm_pmns_running_small_and_mssm_theta12_decreases():
    p0 = rg.pmns_at(91.1879, "NO", 0.0)
    p1 = rg.pmns_at(2e16, "NO", 0.0)
    assert abs(p1["theta12"] - p0["theta12"]) < 0.01 * D
    q = rg.pmns_at(2e16, "NO", 0.05, model="MSSM", tan_beta=10)
    assert q["theta12"] < p0["theta12"] - 1 * D


def test_qcd_alpha_s_monotone():
    a = [rg.qcd_alpha_s(mu) for mu in (2.0, 5.0, 91.1879, 500.0)]
    assert np.all(np.diff(a) < 0)
    assert abs(a[2] - 0.1180) < 1e-6

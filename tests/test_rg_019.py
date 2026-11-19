"""Exp 019: src.rg quark-only paths (no PMNS crosstalk), m_lightest != 0, uncertainties."""
import numpy as np
import pytest

from src import data, rg
from src.leakprobe import leak_probe

MZ = rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"]
SPLIT_KEYS = {"dm21sq", "dm31sq", "dm32sq"}


def _quark_model():
    c = rg.ckm_at(2e16, cache=False)
    y = rg.derived_yukawas_mz()
    return {"t12": c["theta12"], "t13": c["theta13"], "t23": c["theta23"], "delta": c["delta"],
            **{"r_" + k: v for k, v in c["ratios"].items()}, "y_t": y["y_t"], "y_c": y["y_c"]}


def test_quark_only_model_has_no_pmns_flags():
    flags = leak_probe(_quark_model, files=("pmns.json", "masses.json"))
    pmns_like = [f for f in flags if f.record.startswith("pmns.json") or f.key in SPLIT_KEYS]
    assert pmns_like == []
    # the probe still sees the quark inputs the model really uses
    assert {"m_u", "m_c", "m_t_direct"} <= {f.key for f in flags}


def test_ckm_at_equals_old_path_with_neutrinos():
    old = rg.observables(rg.run_sm(MZ, 2e16, rg.build_mz_state()))
    new = rg.ckm_at(2e16, cache=False)
    for k in ("theta12", "theta13", "theta23", "delta", "J"):
        assert new[k] == pytest.approx(old["ckm"][k], rel=1e-9)
    for k, v in old["y"].items():
        assert new["yukawas"][k] == pytest.approx(v, rel=1e-9)
    s = rg.build_mz_state(neutrinos=False)
    assert not np.any(s["kappa"]) and rg.observables(s)["pmns"] is None


@pytest.mark.parametrize("ordering", ("NO", "IO"))
@pytest.mark.parametrize("ml", (0.0, 0.015, 0.08))
def test_pmns_at_m_lightest(ordering, ml):
    p = rg.pmns_at(MZ, ordering, m_lightest=ml)
    sp = data.load_masses()["neutrino_splittings"][data.DEFAULT_ANALYSIS][ordering]
    d3l = sp["dm31sq" if ordering == "NO" else "dm32sq"]["value"]
    assert p["m_lightest"] == pytest.approx(ml, abs=1e-9)
    assert p["dm21sq_eV2"] == pytest.approx(sp["dm21sq"]["value"], rel=1e-6)
    assert p["dm3l_sq_eV2"] == pytest.approx(d3l, rel=1e-6)
    assert p["dm21_over_dm3l"] == pytest.approx(sp["dm21sq"]["value"] / abs(d3l), rel=1e-6)
    rec = data.load_pmns(ordering)
    assert np.degrees(p["theta12"]) == pytest.approx(rec["theta12_deg"]["value"], abs=1e-6)
    assert p["sum_m"] == pytest.approx(sum(p["m"]))


def test_pmns_at_m_lightest_changes_mssm_running_and_validates():
    a = rg.pmns_at(2e16, "NO", 0.0, model="MSSM", tan_beta=30.0)
    b = rg.pmns_at(2e16, "NO", 0.08, model="MSSM", tan_beta=30.0)
    assert abs(b["theta12"] - a["theta12"]) > 1e-3  # quasi-degenerate spectra run more
    assert b["dm21_over_dm3l"] != pytest.approx(a["dm21_over_dm3l"], rel=1e-4)
    with pytest.raises(ValueError):
        rg.pmns_at(MZ, "NO", -0.01)
    with pytest.raises(ValueError):
        rg.neutrino_masses("IO", 0.0, splittings=(7.5e-5, 2.5e-3))  # wrong sign for IO
    q = rg.pmns_at(MZ, "NO", 0.01, splittings=(8e-5, 2.6e-3))
    assert q["dm21sq_eV2"] == pytest.approx(8e-5, rel=1e-6) and q["dm3l_sq_eV2"] == pytest.approx(2.6e-3, rel=1e-6)
    c = rg.pmns_at(MZ, "NO", 0.01)
    c["m"][0] = -1.0  # returned copies, cache intact
    assert rg.pmns_at(MZ, "NO", 0.01)["m"][0] > 0


def test_propagate_ratio_is_exact_for_a_simple_function():
    ms = data.load_masses()

    def fn(m, s):
        mm = {k: ms[k]["value"] for k in rg.UNC_MASS_INPUTS}
        mm.update(m or {})
        return {"r": mm["m_u"] / mm["m_c"]}

    out = rg.propagate(fn)
    su, sc = (ms[k]["err_plus"] / ms[k]["value"] for k in ("m_u", "m_c"))
    assert out["err"]["r"] / out["value"]["r"] == pytest.approx(np.hypot(su, sc), rel=1e-4)
    assert out["inputs"][-1] == "pdg2026/alpha_s_MZ" and out["jac"].shape == (1, 10)


def test_yukawa_and_ratio_uncertainties():
    ms = data.load_masses()
    u = rg.derived_yukawas_mz_unc()
    assert u["value"] == rg.derived_yukawas_mz()
    rel = {k: u["err"][k] / u["value"][k] for k in u["names"]}
    assert ms["m_u"]["err_plus"] / ms["m_u"]["value"] <= rel["y_u"] < 0.05
    assert rel["y_e"] == pytest.approx(ms["m_e"]["err_plus"] / ms["m_e"]["value"], rel=0.02)
    assert rel["y_c"] > ms["m_c"]["err_plus"] / ms["m_c"]["value"]  # alpha_s running adds to m_c
    r = rg.mass_ratios()
    c = rg.ckm_at(MZ)["ratios"]
    for k, name in rg.MASS_RATIO_KEYS.items():
        assert r["value"][k] == pytest.approx(c[name], rel=1e-8)
    assert np.all(np.linalg.eigvalsh(r["cov"]) > 0)
    assert np.allclose(r["cov_log"], r["cov"] / np.outer(list(r["value"].values()), list(r["value"].values())))
    assert np.sqrt(r["cov_log"][0, 0]) >= ms["m_u"]["err_plus"] / ms["m_u"]["value"]
    g = rg.mass_ratios(2e16, errors=False)["value"]
    cg = rg.ckm_at(2e16)["ratios"]
    assert all(g[k] == pytest.approx(cg[n], rel=1e-12) for k, n in rg.MASS_RATIO_KEYS.items())

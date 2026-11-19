"""Exp 013: src.rg input overrides and ckm_at memoisation (backward compatible)."""
import numpy as np
import pytest

from src import data, rg


def test_no_override_equals_default_and_old_hooks_work():
    q = rg.quark_masses_mz()
    assert rg.quark_masses_mz(masses=None, sm_inputs=None) == q
    assert rg.quark_masses_mz.__wrapped__() == q          # exp 010 monkeypatch path
    assert dict(rg._derived_yukawas_tuple.__wrapped__()) == rg.derived_yukawas_mz()
    q["m_s"] = -1.0                                        # callers cannot corrupt the cache
    assert rg.quark_masses_mz()["m_s"] > 0


def test_mass_override_scales_light_quark():
    ms = data.load_masses()["m_s"]["value"]
    base, shifted = rg.quark_masses_mz(), rg.quark_masses_mz(masses={"m_s": 1.1 * ms})
    assert shifted["m_s"] / base["m_s"] == pytest.approx(1.1, rel=1e-9)  # QCD running is linear in m
    assert shifted["m_c"] == base["m_c"]
    y = rg.derived_yukawas_mz(masses={"m_s": {"value": 1.1 * ms}})
    assert y["y_s"] / rg.derived_yukawas_mz()["y_s"] == pytest.approx(1.1, rel=1e-9)
    with pytest.raises(KeyError):
        rg.quark_masses_mz(masses={"m_strange": 0.1})


def test_sm_inputs_override_moves_alpha_s_dependence():
    base = rg.quark_masses_mz()
    hi = rg.quark_masses_mz(sm_inputs={"pdg2026": {"alpha_s_MZ": 0.1200}})
    assert hi["m_s"] < base["m_s"]  # larger alpha_s -> more running from 2 GeV down to M_Z
    with pytest.raises(KeyError):
        rg.quark_masses_mz(sm_inputs={"pdg2026": {"alpha_s": 0.12}})
    assert rg.quark_masses_mz() == base  # the override did not leak into the default


def test_ckm_at_cached_and_identical():
    rg.ckm_at.cache_clear()
    a = rg.ckm_at(2e16)
    b = rg.ckm_at(2e16)
    c = rg.ckm_at(2e16, cache=False)
    for k in ("theta12", "theta13", "theta23", "delta", "J"):
        assert a[k] == b[k] == c[k]
    assert rg.ckm_at.cache_info().hits >= 1
    a["theta12"] = 0.0
    a["ratios"]["md_over_ms"] = 0.0
    d = rg.ckm_at(2e16)
    assert d["theta12"] > 0 and d["ratios"]["md_over_ms"] > 0  # returned copies, cache intact
    m = rg.ckm_at(2e16, masses={"m_s": 1.1 * data.load_masses()["m_s"]["value"]})
    assert m["ratios"]["md_over_ms"] == pytest.approx(d["ratios"]["md_over_ms"] / 1.1, rel=1e-3)
    assert np.isclose(m["theta12"], d["theta12"], rtol=1e-6)

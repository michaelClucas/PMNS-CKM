import numpy as np
import pytest

from src.data import (ANALYSES, ORDERINGS, load_ckm, load_masses, load_pmns,
                      ordering_offset, pmns_delta_chi2)

RECORD_KEYS = {"value", "err_plus", "err_minus", "scale", "scheme", "source"}


def _records(d):
    return {k: v for k, v in d.items() if isinstance(v, dict) and "value" in v}


def test_records_complete():
    for d in (load_ckm(), load_pmns("NO"), load_pmns("IO"), load_masses()):
        recs = _records(d)
        assert recs
        for name, r in recs.items():
            assert RECORD_KEYS <= set(r), name
            assert r["err_plus"] >= 0 and r["err_minus"] >= 0, name


def test_ckm_unitarity_rows_and_columns():
    c = load_ckm()
    V = np.array([[c[f"V{u}{d}"]["value"] for d in "dsb"] for u in "uct"])
    E = np.array([[max(c[f"V{u}{d}"]["err_plus"], c[f"V{u}{d}"]["err_minus"]) for d in "dsb"] for u in "uct"])
    for axis in (0, 1):
        sums = (V ** 2).sum(axis=axis)
        errs = np.sqrt(((2 * V * E) ** 2).sum(axis=axis))
        assert np.all(np.abs(sums - 1) < 3 * errs + 1e-5), (sums, errs)


def test_ckm_consistency():
    c = load_ckm()
    assert c["s12"]["value"] == c["lambda"]["value"]
    assert c["s23"]["value"] == c["Vcb"]["value"]
    assert c["s13"]["value"] == c["Vub"]["value"]
    # Jarlskog from the standard parametrization, J = c12 c23 c13^2 s12 s23 s13 sin(delta)
    s12, s13, s23, d = (c[k]["value"] for k in ("s12", "s13", "s23", "delta_rad"))
    c12, c13, c23 = (np.sqrt(1 - s ** 2) for s in (s12, s13, s23))
    J = c12 * c23 * c13 ** 2 * s12 * s23 * s13 * np.sin(d)
    assert abs(J - c["J"]["value"]) < 2 * c["J"]["err_plus"]


def test_masses():
    m = load_masses()
    order = ["m_u", "m_d", "m_s", "m_c", "m_b", "m_t_pole_xsec"]
    assert all(m[a]["value"] < m[b]["value"] for a, b in zip(order, order[1:]))
    assert m["m_e"]["value"] < m["m_mu"]["value"] < m["m_tau"]["value"]
    ns = m["neutrino_splittings"]["IC23_woSKatm"]
    assert ns["NO"]["dm31sq"]["value"] > 0 > ns["IO"]["dm32sq"]["value"]


@pytest.mark.parametrize("analysis", list(ANALYSES))
@pytest.mark.parametrize("ordering", ORDERINGS)
def test_chi2_zero_at_best_fit_and_grows(ordering, analysis):
    p = load_pmns(ordering, analysis)
    bf = dict(s12sq=p["s12sq"]["value"], s13sq=p["s13sq"]["value"],
              s23sq=p["s23sq"]["value"], dcp_deg=p["dcp_deg"]["value"])
    for method in ("auto", "1d"):
        assert pmns_delta_chi2(ordering, **bf, analysis=analysis, method=method) < 0.1
    # each parameter moved to its 3 sigma edge gives a clearly larger chi^2
    for k in ("s12sq", "s13sq"):
        lo, hi = p[k]["range_3sigma"]
        for edge in (lo, hi):
            v = pmns_delta_chi2(ordering, **{**bf, k: edge}, analysis=analysis)
            assert 6 < v < 13, (k, edge, v)
    far = dict(bf, s12sq=bf["s12sq"] + 0.05, s13sq=bf["s13sq"] + 0.003)
    assert pmns_delta_chi2(ordering, **far, analysis=analysis) > 25
    assert pmns_delta_chi2(ordering, s13sq=0.5, analysis=analysis) == np.inf


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_1d_profiles_match_published_table(ordering):
    """1 sigma intervals read from the chi^2 tables agree with the NuFIT parameter table."""
    p = load_pmns(ordering)
    for k in ("s12sq", "s13sq"):
        r = p[k]
        for v in (r["value"] - r["err_minus"], r["value"] + r["err_plus"]):
            assert 0.6 < pmns_delta_chi2(ordering, **{k: v}) < 1.4, (k, v)
    r = p["dm21sq_1e-5eV2"]
    assert pmns_delta_chi2(ordering, dm21sq_eV2=r["value"] * 1e-5) < 0.1
    r = p["dm3l_sq_1e-3eV2"]
    assert pmns_delta_chi2(ordering, dm3l_sq_eV2=r["value"] * 1e-3) < 0.1


def test_orderings_differ():
    no, io = load_pmns("NO"), load_pmns("IO")
    assert no["s23sq"]["value"] != io["s23sq"]["value"]
    assert no["dcp_deg"]["value"] != io["dcp_deg"]["value"]
    assert ordering_offset("NO") == pytest.approx(0, abs=1e-6)
    assert ordering_offset("IO") == pytest.approx(no["delta_chi2_IO_minus_NO"], abs=0.05)
    assert ordering_offset("IO", "IC24_wSKatm") == pytest.approx(5.9, abs=0.05)
    pt = dict(s12sq=0.31, s13sq=0.0225, s23sq=0.47, dcp_deg=200)
    assert pmns_delta_chi2("NO", **pt) != pytest.approx(pmns_delta_chi2("IO", **pt), abs=0.5)
    assert pmns_delta_chi2("IO", **pt, relative_to="global") > pmns_delta_chi2("IO", **pt)


def test_dcp_periodic():
    assert pmns_delta_chi2("NO", dcp_deg=207) == pytest.approx(pmns_delta_chi2("NO", dcp_deg=-153))
    assert pmns_delta_chi2("NO", s23sq=0.47, dcp_deg=567) == pytest.approx(
        pmns_delta_chi2("NO", s23sq=0.47, dcp_deg=207))


def test_bad_args():
    with pytest.raises(ValueError):
        load_pmns("XO")
    with pytest.raises(ValueError):
        pmns_delta_chi2("NO", method="6d")

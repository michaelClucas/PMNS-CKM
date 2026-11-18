import numpy as np
import pytest

from src import baselines as B
from src.data import load_ckm, load_masses, load_pmns, ordering_offset, pmns_delta_chi2
from src.evaluate import (PMNS_KEYS, evaluate, evaluate_baseline, fit_and_evaluate,
                          observable_keys, pmns_best)
from src.stats import p_value

ANALYSES = ("IC23_woSKatm", "IC24_wSKatm")


@pytest.mark.parametrize("analysis", ANALYSES)
@pytest.mark.parametrize("ordering", ("NO", "IO"))
def test_chi2_near_zero_at_nufit_best_fit(ordering, analysis):
    r = evaluate(pmns_best(ordering, analysis), ordering, analysis)
    # published best-fit values are rounded and the tables are gridded: ~0.01, not 0
    assert r.chi2 < 0.1
    assert r.N == 4 and r.k == 0 and r.dof == 4
    assert r.max_abs_pull < 0.3


@pytest.mark.parametrize("analysis", ANALYSES)
def test_io_offset_counted_once(analysis):
    off = ordering_offset("IO", analysis)
    assert off > 1.0
    for use in (("s12sq",), ("s23sq", "dcp_deg"), PMNS_KEYS):
        r = evaluate(pmns_best("IO", analysis), "IO", analysis, use=use)
        assert r.ordering_offset == pytest.approx(off)
        assert r.chi2_global - r.chi2 == pytest.approx(off)
    rno = evaluate(pmns_best("NO", analysis), "NO", analysis)
    assert rno.chi2_global == pytest.approx(rno.chi2)
    # no PMNS observable -> ordering irrelevant, no offset
    rm = evaluate({"m_tau": load_masses()["m_tau"]["value"]}, "IO", analysis)
    assert rm.ordering_offset == 0.0 and rm.chi2_global == rm.chi2


def test_joint_pmns_matches_single_loader_call():
    pred = {"s12sq": 0.32, "s13sq": 0.0225, "s23sq": 0.55, "dcp_deg": 100.0}
    for o in ("NO", "IO"):
        r = evaluate(pred, o)
        assert r.chi2 == pytest.approx(pmns_delta_chi2(o, **pred))
        assert r.chi2_global == pytest.approx(pmns_delta_chi2(o, **pred, relative_to="global"))


def test_2d_table_used_for_s23_and_delta():
    pred = {"s23sq": 0.56, "dcp_deg": 90.0}
    joint = evaluate(pred, "NO").chi2
    sum1d = pmns_delta_chi2("NO", s23sq=0.56) + pmns_delta_chi2("NO", dcp_deg=90.0)
    assert joint == pytest.approx(pmns_delta_chi2("NO", **pred))
    assert joint != pytest.approx(sum1d, abs=1e-6)


def test_gaussian_terms_and_pvalue():
    c = load_ckm()
    v, ep = c["Vus"]["value"], c["Vus"]["err_plus"]
    r = evaluate({"ckm_Vus": v + 2 * ep}, "NO")
    assert r.chi2 == pytest.approx(4.0)
    assert r.pulls["ckm_Vus"] == pytest.approx(2.0)
    assert r.p == pytest.approx(p_value(4.0, 1, 0))
    m = load_masses()["m_tau"]
    r = evaluate({"m_tau": m["value"] - m["err_minus"]}, "NO")
    assert r.chi2 == pytest.approx(1.0) and r.pulls["m_tau"] == pytest.approx(-1.0)


def test_pmns_1d_pull_sign_and_size():
    best = pmns_best("NO")
    rec = load_pmns("NO")["s12sq"]
    r = evaluate({"s12sq": best["s12sq"] + 2 * rec["err_plus"]}, "NO")
    assert 1.6 < r.pulls["s12sq"] < 2.4
    r = evaluate({"s12sq": best["s12sq"] - 2 * rec["err_minus"]}, "NO")
    assert -2.4 < r.pulls["s12sq"] < -1.6


def test_delta_periodic_and_pull_sign():
    best = pmns_best("NO")["dcp_deg"]
    a = evaluate({"dcp_deg": best - 60}, "NO")
    b = evaluate({"dcp_deg": best - 60 + 720}, "NO")
    assert a.chi2 == pytest.approx(b.chi2) and a.pulls["dcp_deg"] < 0 and b.pulls["dcp_deg"] < 0
    c = evaluate({"dcp_deg": best + 30 - 360}, "NO")
    assert c.pulls["dcp_deg"] > 0


def test_dof_k_and_p_fit():
    best = pmns_best("NO")
    r = evaluate(best, "NO", k=3, k_discrete=1)
    assert r.dof == 1 and np.isfinite(r.p) and r.p_fit == pytest.approx(p_value(r.chi2, 4, 2))
    r = evaluate(best, "NO", k=4, k_discrete=2)
    assert np.isnan(r.p) and np.isfinite(r.p_fit)
    with pytest.raises(ValueError):
        evaluate(best, "NO", k=1, k_discrete=2)


def test_inputs_cannot_be_predictions():
    with pytest.raises(ValueError):
        evaluate({"ckm_s12": 0.225, "s12sq": 0.3}, "NO", use=["ckm_s12", "s12sq"], inputs=["ckm_s12"])
    r = evaluate({"s12sq": 0.3}, "NO", inputs=["ckm_s12"])
    assert r.N == 1 and r.inputs == ("ckm_s12",)
    with pytest.raises(KeyError):
        evaluate({"s12sq": 0.3}, "NO", inputs=["not_an_observable"])


def test_missing_and_nan_predictions_raise():
    with pytest.raises(KeyError):
        evaluate({"s12sq": 0.3}, "NO", use=["s12sq", "s13sq"])
    with pytest.raises(ValueError):
        evaluate({"s12sq": np.nan}, "NO")
    with pytest.raises(KeyError):
        evaluate({"s12sq": 0.3}, "NO", use=["bogus"])


def test_outside_table_gives_inf_and_zero_p():
    r = evaluate({"s13sq": 0.2}, "NO")
    assert np.isinf(r.chi2) and r.p == 0.0


def test_observable_keys_cover_sectors():
    keys = set(observable_keys())
    assert set(PMNS_KEYS) <= keys
    assert {"ckm_Vus", "ckm_s12", "ckm_J", "m_tau", "m_d"} <= keys


def test_fit_recovers_known_minimum():
    best = pmns_best("NO")

    def model(th):
        return {"s12sq": th["x"], "s13sq": best["s13sq"]}

    fr = fit_and_evaluate(model, {"x": (0.2, 0.4)}, "NO", use=("s12sq", "s13sq"), n_starts=4)
    assert fr.params["x"] == pytest.approx(best["s12sq"], abs=2e-3)
    assert fr.result.chi2 < 0.05 and fr.result.k == 1 and fr.result.dof == 1


def test_fit_periodic_phase_wraps():
    best = pmns_best("NO")

    def model(th):
        return {"dcp_deg": np.degrees(th["d"])}

    fr = fit_and_evaluate(model, {"d": (0.0, 2 * np.pi)}, "NO", use=("dcp_deg",),
                          periodic=("d",), n_starts=4)
    assert 0.0 <= fr.params["d"] < 2 * np.pi
    assert abs(((np.degrees(fr.params["d"]) - best["dcp_deg"] + 180) % 360) - 180) < 15
    assert fr.result.chi2 < 0.05


# ---------------------------------------------------------------- baselines registry


def test_registry_counts_consistent():
    for b in B.BASELINES.values():
        assert b.N == len(b.predicts) > 0
        assert set(b.nuisance) <= set(b.params) and set(b.periodic) <= set(b.params)
        assert set(b.nuisance) <= set(b.predicts), b.id
        assert not set(b.inputs) & set(b.predicts), b.id
        assert set(b.inputs) <= set(B.INPUT_RECORDS), b.id
        assert set(b.predicts) <= set(observable_keys()), b.id
    assert B.L1_ID in B.BASELINES


def test_known_simple_cases():
    ctx = B.data_context()
    # QLC solar: chi^2 equals the 1D s12sq profile at sin^2(pi/4 - theta_C)
    fr = evaluate_baseline(B.BASELINES["A1s"], "NO", ctx=ctx)
    s = np.sin(np.pi / 4 - np.arcsin(load_ckm()["s12"]["value"])) ** 2
    assert fr.result.chi2 == pytest.approx(pmns_delta_chi2("NO", s12sq=s))
    assert fr.result.chi2 > 9  # excluded at > 3 sigma (exp 003: -4.3 sigma)
    # Koide m_tau: known +0.4 sigma
    fr = evaluate_baseline(B.BASELINES["C3"], "NO", ctx=ctx)
    assert 0.2 < fr.result.pulls["m_tau"] < 0.7
    # LO TBM: s12sq = 1/3, s13sq = 0, s23sq = 1/2
    o = B.lo_pattern("TBM")
    assert o["s12sq"] == pytest.approx(1 / 3) and "dcp_deg" not in o


def test_data_context_shift():
    c0, c1 = B.data_context(), B.data_context({"ckm_s12": 1.0})
    rec = load_ckm()["s12"]
    assert c1["ckm_s12"] - c0["ckm_s12"] == pytest.approx(rec["err_plus"])
    with pytest.raises(KeyError):
        B.data_context({"nope": 1.0})


def test_ported_relations_match_constructions():
    rng = np.random.default_rng(1)
    for _ in range(20):
        th, ph = rng.uniform(0.05, 0.3), rng.uniform(0, 2 * np.pi)
        o1 = B.obs_from_matrix(B.lo_matrix("TBM") @ B.r23(th, ph))
        t = B.tm1(o1["th13"], o1["th23"])
        assert np.isclose(t["th12"], o1["th12"]) and np.isclose(t["cos_delta"], np.cos(o1["delta"]))
        o2 = B.obs_from_matrix(B.lo_matrix("TBM") @ B.r13(th, ph))
        t = B.tm2(o2["th13"], o2["th23"])
        assert np.isclose(t["th12"], o2["th12"]) and np.isclose(t["cos_delta"], np.cos(o2["delta"]))
    o = B.charged_lepton_corrected(B.lo_matrix("TBM"), 0.2, np.pi)
    assert np.isclose(np.sin(o["th13"]), np.sin(0.2) / np.sqrt(2))
    for _ in range(20):
        t1, t2, t3 = rng.uniform(0.05, 1.4, 3)
        d = rng.uniform(0, 2 * np.pi)
        U = B.phase_diag(*rng.uniform(0, 2 * np.pi, 3)) @ B.km_matrix(t1, t2, t3, d) \
            @ B.phase_diag(*rng.uniform(0, 2 * np.pi, 3))
        assert abs(np.angle(np.exp(1j * (B.km_phase(U) - d)))) < 1e-7
    o = B.ckm_dagger_x(np.eye(3), "TBM")
    assert np.isclose(o["s2_12"], 1 / 3) and np.isclose(o["th13"], 0, atol=1e-12)

"""src.nulls on toy pipelines whose answer is known.

PLANTED family: two CKM-dependent relations for all four PMNS parameters (1 discrete bit):
    th12 = pi/4 - theta_C,  s13^2 = lambda^2 / 2,  th23 = pi/4 -/+ 3 theta_cb,  delta = 5 delta_q.
On a universe where the first variant is true (make_injector), the N1 null (fake CKM) should
pass and the power check should be adequate.
FLAT family: the same structure with no CKM dependence (fixed numbers, 1 bit). Made true by
injection it fits as well as the planted one, but every N1 fake fits equally well, so the
null fails and its power is 0: N1 cannot confirm a family with no CKM link.
"""
import numpy as np
import pytest

from src import data, evaluate as ev
from src.nulls import (N1_PRIOR, Universe, decide, is_hit, make_injector, max_hits_to_pass, n1_universe,
                       n2_universe, pmns_noise, power_check, run_null)

PMNS4 = ev.PMNS_KEYS


def planted_pred(ctx, v=0):
    sgn = (-1.0, 1.0)[v]
    th12 = np.pi / 4 - ctx["theta_C"]
    th23 = np.pi / 4 + sgn * 3 * ctx["theta_cb"]
    return {"s12sq": np.sin(th12) ** 2, "s13sq": ctx["ckm_s12"] ** 2 / 2, "s23sq": np.sin(th23) ** 2,
            "dcp_deg": np.degrees(5 * ctx["delta_q"]) % 360}


FLAT = ({"s12sq": 0.30, "s13sq": 0.022, "s23sq": 0.45, "dcp_deg": 240.0},
        {"s12sq": 0.34, "s13sq": 0.024, "s23sq": 0.58, "dcp_deg": 120.0})


def _best(u, preds):
    rs = [u.evaluate(p, PMNS4, k=1, k_discrete=1) for p in preds]
    return max(rs, key=lambda r: (r.p_global, -r.chi2_global))


def planted_pipeline(u):
    ctx = u.context()
    return _best(u, [planted_pred(ctx, v) for v in (0, 1)])


def flat_pipeline(u):
    return _best(u, FLAT)


def _planted_universe(seed=5):
    return make_injector(planted_pred(Universe.real().context()))(np.random.default_rng(seed), Universe.real())


# ---------------------------------------------------------------- universes and translation


@pytest.mark.parametrize("ordering", ("NO", "IO"))
def test_real_universe_scores_like_evaluate(ordering):
    u = Universe.real(ordering)
    pred = {"s12sq": 0.31, "s13sq": 0.0222, "s23sq": 0.53, "dcp_deg": 250.0, "ckm_Vub": 0.0038, "m_tau": 1.777}
    a = u.evaluate(pred, tuple(pred))
    b = ev.evaluate(pred, ordering, use=tuple(pred))
    assert a.chi2 == b.chi2 and a.chi2_global == b.chi2_global and a.pulls == b.pulls
    ctx, ref = u.context(), __import__("src.baselines", fromlist=["x"]).data_context()
    assert all(ctx[k] == ref[k] for k in ref if k != "V_ckm") and np.array_equal(ctx["V_ckm"], ref["V_ckm"])


@pytest.mark.parametrize("ordering", ("NO", "IO"))
def test_translated_tables(ordering):
    t = ev.pmns_best(ordering)
    same = 0.46 if t["s23sq"] < 0.5 else 0.56
    other = 1.0 - same
    c = {"s12sq": t["s12sq"] + 0.03, "s13sq": t["s13sq"] - 0.002, "s23sq": same, "dcp_deg": t["dcp_deg"] + 100.0}
    u = Universe(kind="N2", ordering=ordering, pmns_centre={ordering: c})
    for d in (-0.02, 0.0, 0.015):
        assert u.pmns_delta_chi2(ordering, s12sq=c["s12sq"] + d) == data.pmns_delta_chi2(ordering, s12sq=t["s12sq"] + d)
        assert u.pmns_delta_chi2(ordering, s13sq=c["s13sq"] + d / 10) == pytest.approx(
            data.pmns_delta_chi2(ordering, s13sq=t["s13sq"] + d / 10), abs=1e-12)
    # delta is periodic; same-octant s23 is a pure shift; the minimum moves to the centre
    assert u.pmns_delta_chi2(ordering, dcp_deg=c["dcp_deg"] + 400) == pytest.approx(
        data.pmns_delta_chi2(ordering, dcp_deg=t["dcp_deg"] + 40), abs=1e-12)
    assert u.pmns_delta_chi2(ordering, s23sq=same + 0.01, dcp_deg=c["dcp_deg"]) == pytest.approx(
        data.pmns_delta_chi2(ordering, s23sq=t["s23sq"] + 0.01, dcp_deg=t["dcp_deg"]), abs=1e-12)
    assert u.evaluate(c, PMNS4).chi2 == pytest.approx(ev.evaluate(t, ordering, use=PMNS4).chi2, abs=1e-12)
    # other octant: the real shape reflected about the fake centre
    c2 = dict(c, s23sq=other)
    u2 = Universe(kind="N2", ordering=ordering, pmns_centre={ordering: c2})
    for d in (-0.05, 0.03):
        assert u2.pmns_delta_chi2(ordering, s23sq=other + d) == pytest.approx(
            data.pmns_delta_chi2(ordering, s23sq=t["s23sq"] - d), abs=1e-12)
    # pulls are signed about the fake centre
    r = u.evaluate({"s12sq": c["s12sq"] + 0.02}, ("s12sq",))
    assert r.pulls["s12sq"] > 1


def test_n1_and_n2_generators():
    rng = np.random.default_rng(0)
    base = Universe.real("NO")
    f = n1_universe(base, rng)
    w = f.truth["wolfenstein"]
    assert all(N1_PRIOR[k][0] <= w[k] <= N1_PRIOR[k][1] for k in w)
    ck = f.load_ckm()
    assert ck["lambda"]["value"] == w["lambda"] and ck["direct_Vus"]["value"] == ck["Vus"]["value"]
    assert ck["Vus"]["err_plus"] == data.load_ckm()["Vus"]["err_plus"]
    assert f.pmns_centre == {} and f.context()["ckm_s12"] == pytest.approx(w["lambda"])
    g = n2_universe(base, rng)
    for o in ("NO", "IO"):
        c = g.pmns_centre[o]
        lo, hi = data.load_pmns(o)["s23sq"]["range_3sigma"]
        assert (lo + hi) / 2 - 1.5 * (hi - lo) <= c["s23sq"] <= (lo + hi) / 2 + 1.5 * (hi - lo)
        assert 0 <= c["dcp_deg"] < 360
    assert g.ckm is None


def test_pmns_noise_has_nufit_width():
    rng = np.random.default_rng(2)
    d = np.array([pmns_noise("NO", rng)["s12sq"] for _ in range(1500)])
    rec = data.load_pmns("NO")["s12sq"]
    assert abs(np.mean(d)) < 0.003
    assert np.std(d) == pytest.approx((rec["err_plus"] + rec["err_minus"]) / 2, rel=0.15)


# ---------------------------------------------------------------- hit and decision rule


def test_decision_rule():
    assert max_hits_to_pass(1000) == 4 and max_hits_to_pass(300) == 0 and max_hits_to_pass(100) == -1
    assert decide(4, 1000)["passed"] and not decide(5, 1000)["passed"]
    ref = {"N": 4, "k": 2, "max_abs_pull": 0.5, "p_global": 0.3}
    assert is_hit({"N": 4, "k": 2, "max_abs_pull": 1.9, "p_global": 0.3}, ref)
    assert not is_hit({"N": 4, "k": 3, "max_abs_pull": 0.1, "p_global": 0.9}, ref)   # k > N - 2
    assert not is_hit({"N": 4, "k": 2, "max_abs_pull": 2.1, "p_global": 0.9}, ref)   # a 2 sigma pull
    assert not is_hit({"N": 4, "k": 2, "max_abs_pull": 0.1, "p_global": 0.29}, ref)  # worse p
    assert not is_hit(None, ref)


# ---------------------------------------------------------------- toy pipelines


def test_planted_relation_passes_n1():
    u = _planted_universe()
    ref = planted_pipeline(u)
    assert ref.fit_ok and ref.compression_ok
    r = run_null(planted_pipeline, "N1", "NO", n_fakes=1000, base=u, seed=11)
    assert r.passed and r.n_run == 1000 and r.n_hits <= 4


def test_flat_family_fails_n1_with_early_stop():
    u = make_injector(FLAT[0])(np.random.default_rng(5), Universe.real())
    r = run_null(flat_pipeline, "N1", "NO", n_fakes=1000, base=u, seed=11)
    assert not r.passed and r.stopped_early and r.n_run == 5 and r.n_hits == 5


def test_power_check_planted_vs_flat():
    inj = make_injector(planted_pred(Universe.real().context()))
    p = power_check(planted_pipeline, inj, n_inject=20, n_fakes=300, null="N1", seed=3)
    assert p.adequate and p.fraction >= 0.5
    f = power_check(flat_pipeline, make_injector(FLAT[0]), n_inject=20, n_fakes=300, null="N1", seed=3)
    assert f.fraction == 0.0 and not f.adequate and f.n_candidate_ok > 0
    assert all(n.n_run == 1 for n in f.nulls if n is not None)  # fails on its first fake


def test_n2_runs_and_workers_do_not_change_results():
    u = _planted_universe()
    a = run_null(planted_pipeline, "N2", "NO", n_fakes=24, base=u, seed=4, early_stop=False)
    b = run_null(planted_pipeline, "N2", "NO", n_fakes=24, base=u, seed=4, early_stop=False, workers=2)
    assert [x["hit"] for x in a.fakes] == [x["hit"] for x in b.fakes]
    assert [x["result"]["chi2"] for x in a.fakes] == [x["result"]["chi2"] for x in b.fakes]
    assert a.n_run == b.n_run == 24

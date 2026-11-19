"""Exp 026: SUSY threshold corrections (src.rg, src.evaluate scale=), fit_and_evaluate methods,
Objective default, null_info in src.nulls.run_null."""
import json

import numpy as np
import pytest

from src import evaluate as ev
from src import infogain as ig
from src import rg
from src.nulls import Universe, run_null
from src.stats import Objective

GUT, MS = 2e16, 1000.0
SC = {"mu": GUT, "model": "MSSM", "tan_beta": 5.0, "m_susy": MS}
USE = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "mr_b_tau",
       "s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")


# ---------------------------------------------------------------- src.rg thresholds

def test_rg_eta_zero_is_bit_identical():
    a = rg.ckm_at(GUT, "MSSM", 10, MS)
    b = rg.ckm_at(GUT, "MSSM", 10, MS, thresholds={"eta_b": 0.0, "eta_q": 0.0}, cache=False)
    for k in ("theta12", "theta13", "theta23", "delta", "J"):
        assert a[k] == b[k]
    assert a["ratios"] == b["ratios"]
    s = rg.build_mz_state("NO", 0.01)
    u1 = rg.run(s["mu"], GUT, s, "MSSM", 10, MS)
    u2 = rg.run(s["mu"], GUT, s, "MSSM", 10, MS, thresholds={"eta_b": 0.0})
    for k in ("Yu", "Yd", "Ye", "kappa"):
        assert np.array_equal(u1[k], u2[k])


def test_rg_leading_order_matching():
    """Antusch-Maurer 1306.6879 Eqs. (17)-(24) just above M_SUSY."""
    mu1 = MS * (1 + 1e-9)
    r0 = rg.ckm_at(mu1, "MSSM", 20, MS)
    eb, eq, el = -0.3, 0.2, -0.1
    r = rg.ckm_at(mu1, "MSSM", 20, MS, thresholds={"eta_b": eb, "eta_q": eq, "eta_l": el})
    y0, y1 = r0["yukawas"], r["yukawas"]
    T = (1 + eq) / (1 + eb)
    assert y1["y_b"] * (1 + eb) / y0["y_b"] == pytest.approx(1, abs=2e-3)
    assert y1["y_s"] * (1 + eq) / y0["y_s"] == pytest.approx(1, abs=2e-3)
    assert y1["y_d"] * (1 + eq) / y0["y_d"] == pytest.approx(1, abs=2e-3)
    assert y1["y_mu"] * (1 + el) / y0["y_mu"] == pytest.approx(1, abs=1e-9)
    assert y1["y_e"] * (1 + el) / y0["y_e"] == pytest.approx(1, abs=1e-9)
    assert y1["y_tau"] / y0["y_tau"] == pytest.approx(1, abs=1e-9)
    assert r["theta13"] * T / r0["theta13"] == pytest.approx(1, abs=2e-3)
    assert r["theta23"] * T / r0["theta23"] == pytest.approx(1, abs=2e-3)
    assert r["theta12"] / r0["theta12"] == pytest.approx(1, abs=2e-3)


def test_rg_threshold_up_down_identity():
    s = rg.build_mz_state("NO", 0.01)
    eta = {"eta_b": -0.4, "eta_q": 0.1, "eta_l": 0.05}
    up = rg.run(s["mu"], GUT, s, "MSSM", 20, MS, thresholds=eta)
    dn = rg.run(GUT, s["mu"], up, "MSSM", 20, MS, thresholds=eta)
    for k in ("Yu", "Yd", "Ye", "kappa"):
        assert np.max(np.abs(dn[k] - s[k])) <= 1e-9 * np.max(np.abs(s[k]))


def test_rg_threshold_errors():
    with pytest.raises(ValueError):
        rg.ckm_at(GUT, "SM", thresholds={"eta_b": -0.2})
    with pytest.raises(ValueError):
        rg.ckm_at(GUT, "MSSM", 10, MS, thresholds={"eta_x": 0.1})
    with pytest.raises(ValueError):
        rg.ckm_at(GUT, "MSSM", 10, MS, thresholds={"eta_b": -1.0})


def test_rg_eq25_invariants_independent_of_eta():
    """(1+eta_l) y_mu / ((1+eta_q) y_s) at M_GUT barely depends on the eta (paper Eq. (25))."""
    def ratio(e):
        y = rg.ckm_at(GUT, "MSSM", 10, MS, thresholds=e)["yukawas"]
        return (1 + e.get("eta_l", 0)) * y["y_mu"] / ((1 + e.get("eta_q", 0)) * y["y_s"])
    a, b = ratio({}), ratio({"eta_b": -0.3, "eta_q": 0.2, "eta_l": 0.1})
    assert b / a == pytest.approx(1, abs=2e-3)
    assert a == pytest.approx(4.36, abs=0.23)  # published 4.36 +- 0.23 (PDG 2012 inputs)


# ---------------------------------------------------------------- src.evaluate scale thresholds

def test_scale_eta_zero_unchanged():
    a = ev.Scorer(USE, "NO", mass_floor=0.02, scale=SC)
    b = ev.Scorer(USE, "NO", mass_floor=0.02, scale=dict(SC, eta_b=0.0, eta_q=0.0, eta_l=0.0))
    assert a.scale_shift == b.scale_shift and a.scale_systematic == b.scale_systematic
    assert a.scale == b.scale and "eta_b" not in a.scale and b.scale_slope == {}
    pred = {k: v * 1.01 for k, v in a._prep({k: 0.5 for k in USE}).items()}
    pred.update({"s12sq": 0.3, "s13sq": 0.022, "s23sq": 0.5, "dcp_deg": 230.0, "dm_ratio": 0.03})
    assert a.chi2(pred) == b.chi2(pred)


def test_scale_eta_data_point_maps_to_data():
    """A prediction equal to the data's own GUT values (with eta) translates to the M_Z data."""
    eb = -0.5
    sc = ev.Scorer(USE, "NO", mass_floor=0.02, scale=dict(SC, eta_b=eb))
    assert sc.scale["eta_b"] == eb and sc.scale_slope["ckm_s13"] == pytest.approx(1 / (1 + eb))
    g = rg.ckm_at(GUT, "MSSM", 5.0, MS, thresholds={"eta_b": eb})
    r = rg.mass_ratios(GUT, "MSSM", 5.0, MS, errors=False, thresholds={"eta_b": eb})["value"]
    lo = rg.ckm_at(rg._mz())
    rlo = rg.mass_ratios(None, errors=False)["value"]
    pred = {k: 0.1 for k in USE}
    pred.update(ckm_s13=np.sin(g["theta13"]), ckm_s23=np.sin(g["theta23"]), mr_s_b=r["mr_s_b"], mr_b_tau=r["mr_b_tau"])
    p = sc._prep(pred)
    assert p["ckm_s13"] == pytest.approx(np.sin(lo["theta13"]), rel=1e-12)
    assert p["ckm_s23"] == pytest.approx(np.sin(lo["theta23"]), rel=1e-12)
    assert p["mr_s_b"] == pytest.approx(rlo["mr_s_b"], rel=1e-12)
    assert p["mr_b_tau"] == pytest.approx(rlo["mr_b_tau"], rel=1e-12)


def test_scale_eta_small_matches_exp023_linear():
    """At small eta_b the full treatment equals exp 023's linear rescaling (to O(eta) running)."""
    eb = -0.05
    base = ev.Scorer(USE, "NO", scale=SC)
    full = ev.Scorer(USE, "NO", scale=dict(SC, eta_b=eb))
    g = rg.ckm_at(GUT, "MSSM", 5.0, MS, thresholds={"eta_b": eb})
    r = rg.mass_ratios(GUT, "MSSM", 5.0, MS, errors=False, thresholds={"eta_b": eb})["value"]
    pred = {k: 0.1 for k in USE}
    pred.update(ckm_s13=1.02 * np.sin(g["theta13"]), ckm_s23=0.99 * np.sin(g["theta23"]), mr_s_b=r["mr_s_b"],
                mr_b_tau=r["mr_b_tau"])
    lin = dict(pred, ckm_s13=pred["ckm_s13"] / (1 + eb), ckm_s23=pred["ckm_s23"] / (1 + eb),
               mr_s_b=pred["mr_s_b"] / (1 + eb), mr_b_tau=pred["mr_b_tau"] * (1 + eb))
    pf, pl = full._prep(pred), base._prep(lin)
    for k in ("ckm_s13", "ckm_s23", "mr_s_b", "mr_b_tau"):
        assert pf[k] == pytest.approx(pl[k], rel=2e-4)


def test_scale_eta_errors():
    with pytest.raises(ValueError):
        ev.Scorer(("mr_s_b",), "NO", scale={"mu": GUT, "model": "SM", "eta_b": -0.2})
    with pytest.raises(ValueError):
        ev.Scorer(("mr_s_b",), "NO", scale=dict(SC, eta_b=-1.2))


# ---------------------------------------------------------------- fit_and_evaluate methods

def _toy():
    """Rastrigin-like multi-modal landscape on (ckm_s12, ckm_s23); global minimum at the data."""
    c = ev._data.load_ckm()
    s12, s23 = c["s12"]["value"], c["s23"]["value"]

    def model(t):
        return {"ckm_s12": s12 + 0.01 * t["a"] + 0.004 * np.sin(9 * t["a"]) ** 2,
                "ckm_s23": s23 + 0.01 * t["b"] + 0.004 * np.sin(9 * t["b"]) ** 2}
    return model, {"a": (-3.0, 3.0), "b": (-3.0, 3.0)}


@pytest.mark.parametrize("method", ["lbfgsb", "de"])
def test_fit_methods_find_global_minimum(method):
    model, bounds = _toy()
    fr = ev.fit_and_evaluate(model, bounds, "NO", ("ckm_s12", "ckm_s23"), n_starts=8, seed=1, method=method,
                             optimizer_options={"maxiter": 300} if method == "de" else None)
    assert fr.result.chi2 < 1e-6
    assert fr.n_starts == (1 if method == "de" else 8)


def test_fit_default_method_unchanged():
    model, bounds = _toy()
    a = ev.fit_and_evaluate(model, bounds, "NO", ("ckm_s12", "ckm_s23"), n_starts=4, seed=3)
    b = ev.fit_and_evaluate(model, bounds, "NO", ("ckm_s12", "ckm_s23"), n_starts=4, seed=3, method="nelder-mead")
    assert a.params == b.params and a.start_chi2 == b.start_chi2 and a.nfev == b.nfev


def test_fit_method_errors():
    model, bounds = _toy()
    with pytest.raises(ValueError):
        ev.fit_and_evaluate(model, bounds, "NO", ("ckm_s12",), method="bfgs")
    with pytest.raises(ValueError):
        ev.fit_and_evaluate(model, bounds, "NO", ("ckm_s12",), method="de", optimizer_options={"bogus": 1})


def test_fit_de_maxfev_budget():
    model, bounds = _toy()
    fr = ev.fit_and_evaluate(model, bounds, "NO", ("ckm_s12", "ckm_s23"), method="de",
                             optimizer_options={"maxfev": 300, "polish": False})
    assert fr.nfev < 300 + 2 * 15 * 2 + 4000 * 2 * 2  # DE budget + one generation + NM polish cap


# ---------------------------------------------------------------- Objective default, null_info

def test_objective_default_joint():
    calls = []

    def f(ordering, **kw):
        calls.append(kw)
        return 0.0
    o = Objective(pmns_delta_chi2=f)
    assert o.pmns_mode == "joint"
    o({"s12sq": 0.3, "s13sq": 0.02, "s23sq": 0.5, "dcp_deg": 200.0})
    assert len(calls[-1]) == 4  # one joint call with all four parameters


def test_run_null_reports_null_info():
    u = Universe.real("NO")

    def pipe(univ):
        ctx = univ.context()
        th12 = np.pi / 4 - ctx["theta_C"]
        pred = {"s12sq": np.sin(th12) ** 2, "s13sq": ctx["ckm_s12"] ** 2 / 2, "s23sq": 0.5, "dcp_deg": 230.0}
        return univ.evaluate(pred, ev.PMNS_KEYS, k=0)
    r = run_null(pipe, "N1", "NO", n_fakes=12, base=u, seed=3, early_stop=False)
    fk = [f["result"]["chi2_global"] for f in r.fakes]
    assert r.info is not None and r.info["n"] == sum(np.isfinite(fk)) >= 6  # null_info drops non-finite fakes
    want = ig.null_info(r.ref["chi2_global"], fk)
    assert r.info == want
    json.dumps(r.info)
    r2 = run_null(pipe, "N1", "NO", n_fakes=12, base=u, seed=3, early_stop=False, keep_fakes=False)
    assert r2.info == want


def test_chi2_soft_include_finite():
    use = ("ckm_s12", "s12sq", "s13sq", "s23sq", "dcp_deg")
    sc = ev.Scorer(use, "NO")
    c = ev._data.load_ckm()["s12"]
    out = {"ckm_s12": c["value"] + 10 * c["err_plus"], "s12sq": 0.9, "s13sq": 0.022, "s23sq": 0.5, "dcp_deg": 230.0}
    a, b = sc.chi2_soft(out), sc.chi2_soft(out, include_finite=True)
    assert a >= ev.CHI2_CAP and b == pytest.approx(a + 100.0, rel=1e-9)
    inside = dict(out, s12sq=0.31)
    assert sc.chi2_soft(inside) == sc.chi2_soft(inside, include_finite=True) == sc.chi2(inside)

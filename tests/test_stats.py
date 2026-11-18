import numpy as np
import pytest
from scipy import stats

from src.stats import (
    Objective,
    asym_pull,
    chi2_asym,
    chi2_cov,
    clopper_pearson,
    fake_universe_rate,
    p_value,
    pvalue_to_sigma,
    sigma_to_pvalue,
    trials_factor_pvalue,
)


def test_asym_pull_sides():
    assert asym_pull(1.2, 1.0, 0.1, 0.4) == pytest.approx(2.0)
    assert asym_pull(0.6, 1.0, 0.1, 0.4) == pytest.approx(-1.0)
    assert asym_pull(0.6, 1.0, 0.1, -0.4) == pytest.approx(-1.0)  # sign of err_minus ignored
    assert asym_pull(1.0, 1.0, 0.1, 0.4) == 0.0


def test_chi2_asym_vectorized():
    c2, pulls = chi2_asym([1.2, 0.6], [1.0, 1.0], [0.1, 0.1], [0.4, 0.4])
    assert c2 == pytest.approx(5.0)
    c2b, _ = chi2_asym(np.array([[1.2, 0.6], [1.0, 1.0]]), [1.0, 1.0], 0.1, 0.4)
    assert np.allclose(c2b, [5.0, 0.0])


def test_chi2_cov_matches_diagonal_and_correlated():
    c2, _ = chi2_cov([1.0, 2.0], [0.0, 0.0], np.diag([1.0, 4.0]))
    assert c2 == pytest.approx(2.0)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    r = np.array([1.0, -1.0])
    c2, _ = chi2_cov(r, [0, 0], cov)
    assert c2 == pytest.approx(r @ np.linalg.solve(cov, r))


def test_chi2_cov_distribution():
    rng = np.random.default_rng(0)
    cov = np.array([[2.0, 0.6, 0.1], [0.6, 1.0, -0.3], [0.1, -0.3, 0.5]])
    x = rng.multivariate_normal(np.zeros(3), cov, size=50000)
    c2, _ = chi2_cov(x, np.zeros(3), cov)
    assert stats.kstest(c2, stats.chi2(3).cdf).pvalue > 1e-3


def fake_pmns(ordering, s12sq=None, s13sq=None, s23sq=None, dcp_deg=None):
    best = {"s12sq": 0.30, "s13sq": 0.022, "s23sq": 0.57, "dcp_deg": 200.0}
    sig = {"s12sq": 0.012, "s13sq": 0.0006, "s23sq": 0.02, "dcp_deg": 30.0}
    vals = {"s12sq": s12sq, "s13sq": s13sq, "s23sq": s23sq, "dcp_deg": dcp_deg}
    off = 0.0 if ordering == "NO" else 2.0
    return off + sum(((v - best[k]) / sig[k]) ** 2 for k, v in vals.items() if v is not None)


CKM = {"Vus": {"value": 0.2250, "err_plus": 0.0007, "err_minus": 0.0007},
       "Vcb": {"value": 0.0418, "err_plus": 0.0008, "err_minus": 0.0007},
       "Vub": {"value": 0.00369, "err_plus": 0.00011, "err_minus": 0.00011}}
MASSES = {"m_e/m_mu": {"value": 0.004836, "err_plus": 1e-6, "err_minus": 1e-6}}


def test_objective_total_and_pulls():
    obj = Objective(ckm=CKM, pmns_delta_chi2=fake_pmns, pmns_ordering="NO",
                    pmns_best={"s12sq": 0.30, "dcp_deg": 200.0}, masses=MASSES)
    assert obj.n_obs == 3 + 4 + 1
    pred = {"Vus": 0.2257, "Vcb": 0.0418 - 0.0014, "Vub": 0.00369,
            "s12sq": 0.30 - 0.024, "s13sq": 0.022, "s23sq": 0.57, "dcp_deg": 260.0,
            "m_e/m_mu": 0.004836}
    r = obj(pred)
    assert r.pulls["Vus"] == pytest.approx(1.0)
    assert r.pulls["Vcb"] == pytest.approx(-2.0)
    assert r.pulls["s12sq"] == pytest.approx(-2.0)
    assert r.pulls["dcp_deg"] == pytest.approx(2.0)
    assert r.chi2 == pytest.approx(1 + 4 + 4 + 4)
    assert r.max_abs_pull() == pytest.approx(2.0)
    assert r.p_value(k=2) == pytest.approx(stats.chi2.sf(13, 6))


def test_objective_joint_mode_and_ordering():
    obj = Objective(pmns_delta_chi2=fake_pmns, pmns_ordering="IO", pmns_mode="joint")
    r = obj({"s12sq": 0.30, "s13sq": 0.022, "s23sq": 0.57, "dcp_deg": 200.0})
    assert r.chi2 == pytest.approx(2.0)  # ordering offset counted once in joint mode
    assert obj.n_obs == 4


def test_objective_subset_and_missing_key():
    obj = Objective(ckm=CKM, pmns_delta_chi2=fake_pmns, pmns_params=("s13sq",))
    assert obj.n_obs == 4
    with pytest.raises(KeyError):
        obj({"Vus": 0.225, "Vcb": 0.0418, "Vub": 0.00369})
    with pytest.raises(ValueError):
        Objective(pmns_delta_chi2=fake_pmns, pmns_params=("theta12",))
    with pytest.raises(ValueError):
        Objective(pmns_ordering="normal")


def test_objective_cov():
    cov = np.array([[0.0007**2, 0.5 * 0.0007 * 0.0008], [0.5 * 0.0007 * 0.0008, 0.0008**2]])
    obj = Objective(ckm=CKM, ckm_cov=(["Vus", "Vcb"], cov))
    pred = {"Vus": 0.2257, "Vcb": 0.0426, "Vub": 0.00369}
    r = obj(pred)
    rr = np.array([0.0007, 0.0008])
    assert r.terms["ckm_cov"] == pytest.approx(rr @ np.linalg.solve(cov, rr))
    assert "Vus" not in r.terms and r.terms["Vub"] == 0.0
    assert r.chi2 == pytest.approx(2 / 1.5)


def test_objective_no_pmns():
    obj = Objective(ckm=CKM)
    assert obj.n_obs == 3
    assert obj({"Vus": 0.225, "Vcb": 0.0418, "Vub": 0.00369}).chi2 == 0.0


def test_p_value():
    assert p_value(0.0, 5, 2) == pytest.approx(1.0)
    assert p_value(3.0, 4, 1) == pytest.approx(stats.chi2.sf(3.0, 3))
    # chi2 = dof gives p ~ 0.4-0.5 for moderate dof
    assert 0.3 < p_value(10, 12, 2) < 0.5
    with pytest.raises(ValueError):
        p_value(1.0, 3, 3)
    with pytest.raises(ValueError):
        p_value(1.0, 3, 4)


def test_sigma_conversions():
    assert sigma_to_pvalue(1.0) == pytest.approx(0.3173, abs=1e-4)
    assert pvalue_to_sigma(0.05) == pytest.approx(1.95996, abs=1e-4)
    assert pvalue_to_sigma(sigma_to_pvalue(5.0)) == pytest.approx(5.0)


def test_trials_factor():
    assert trials_factor_pvalue(0.01, 1) == pytest.approx(0.01)
    assert trials_factor_pvalue(0.01, 100) == pytest.approx(1 - 0.99**100)
    assert trials_factor_pvalue(1e-12, 1000) == pytest.approx(1e-9, rel=1e-6)  # no cancellation
    assert trials_factor_pvalue(0.0, 10) == 0.0
    assert trials_factor_pvalue(1.0, 10) == 1.0
    with pytest.raises(ValueError):
        trials_factor_pvalue(1.5, 10)


def test_clopper_pearson_and_rate():
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 100))
    lo, hi = clopper_pearson(100, 100)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 100))
    lo, hi = clopper_pearson(5, 100)
    assert lo < 0.05 < hi
    r = fake_universe_rate(0, 100)
    assert r["rate"] == 0.0 and r["upper"] == pytest.approx(1 - 0.05 ** (1 / 100))  # ~3%
    assert fake_universe_rate(0, 300)["upper"] < 0.01
    with pytest.raises(ValueError):
        clopper_pearson(5, 3)


def test_clopper_pearson_coverage():
    rng = np.random.default_rng(1)
    p, n = 0.03, 200
    ks = rng.binomial(n, p, size=2000)
    cover = np.mean([lo <= p <= hi for lo, hi in (clopper_pearson(k, n) for k in ks)])
    assert cover >= 0.95 - 0.01  # Clopper-Pearson is conservative

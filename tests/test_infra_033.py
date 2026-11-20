"""Exp 033: src.nulls N1lam (lambda-matched N1) and src.data IO-aware Delta m^2 ratio."""
import json
import pathlib

import numpy as np
import pytest
from scipy import stats

from src import data, evaluate as ev
from src import nulls
from src.nulls import N1LAM_PRIOR, Universe, make_fake, n1lam_universe, run_null, split_normal

ROOT = pathlib.Path(__file__).resolve().parents[1]
E028 = ROOT / "experiments" / "20261119-028-nn-info-decisive"
WKEYS = ("lambda", "A", "rhobar", "etabar")


def _w(u):
    return [u.truth["wolfenstein"][k] for k in WKEYS]


# ---------------------------------------------------------------- N1lam


def test_n1lam_registered_without_changing_existing_nulls():
    assert nulls.NULLS[:3] == ("N1", "N2", "N3") and "N1lam" in nulls.NULLS
    assert nulls._NULL_CODE["N1"] == 1 and nulls._NULL_CODE["N1lam"] not in (1, 2, 3, 4)
    assert nulls.N1_PRIOR == {"lambda": (0.05, 0.5), "A": (0.3, 2.0), "rhobar": (-1.0, 1.0), "etabar": (-1.0, 1.0)}


@pytest.mark.parametrize("ordering", ("NO", "IO"))
def test_n1lam_reproduces_exp028_fakes_exactly(ordering):
    """exp 028 analyze.fakes("N1lam"): rng default_rng([28, 2, i]), split normal lambda then A,
    then n1_universe with lambda, A pinned. Stored values are rounded to 5 decimals."""
    ref = np.array(json.loads((E028 / f"results_{ordering}.json").read_text())["fake_wolfenstein"]["N1lam"])
    base = Universe.real(ordering)
    got = np.array([_w(n1lam_universe(base, np.random.default_rng([28, 2, i]))) for i in range(len(ref))])
    assert len(ref) == 2000
    assert np.max(np.abs(got - ref)) <= 5.01e-6


def test_n1lam_fake_universe_is_n1_with_lambda_A_pinned():
    base = Universe.real("NO")
    u = n1lam_universe(base, np.random.default_rng(7))
    rng = np.random.default_rng(7)
    lam, A = split_normal(rng, data.load_ckm()["lambda"]), split_normal(rng, data.load_ckm()["A"])
    v = nulls.n1_universe(base, rng, prior={"lambda": (lam, lam), "A": (A, A)})
    assert u.kind == "N1lam" and u.ckm == v.ckm and _w(u) == _w(v)
    assert u.pmns_centre == base.pmns_centre and u.masses is None
    assert u.truth["lambda_A_centre"] == (data.load_ckm()["lambda"]["value"], data.load_ckm()["A"]["value"])
    c = u.load_ckm()
    assert c["lambda"]["value"] == lam and c["lambda"]["err_plus"] == data.load_ckm()["lambda"]["err_plus"]


def test_n1lam_distribution():
    """Fresh-seed fakes: lambda, A split normal about the PDG records; rhobar, etabar U[-1, 1];
    and the joint distribution agrees with exp 028's stored fakes (KS per coordinate)."""
    base = Universe.real("NO")
    W = np.array([_w(n1lam_universe(base, np.random.default_rng([33, 5, i]))) for i in range(2000)])
    ck = data.load_ckm()
    for j, k in enumerate(("lambda", "A")):
        r = ck[k]
        x = W[:, j] - r["value"]
        up, dn = x[x > 0], x[x <= 0]
        assert abs(len(up) / len(x) - 0.5) < 4 * 0.5 / np.sqrt(len(x))
        # half-normal: E|x| = err sqrt(2 / pi)
        assert abs(up.mean() / (r["err_plus"] * np.sqrt(2 / np.pi)) - 1) < 0.12
        assert abs(-dn.mean() / (r["err_minus"] * np.sqrt(2 / np.pi)) - 1) < 0.12
    for j in (2, 3):
        assert stats.kstest((W[:, j] + 1) / 2, "uniform").pvalue > 1e-3
    ref = np.array(json.loads((E028 / "results_NO.json").read_text())["fake_wolfenstein"]["N1lam"])
    for j in range(4):
        assert stats.ks_2samp(W[:, j], ref[:, j]).pvalue > 1e-3, WKEYS[j]


def test_n1lam_prior_overrides():
    base = Universe.real("NO")
    u = n1lam_universe(base, np.random.default_rng(1), prior={"width": 0.0, "rhobar": (0.1, 0.1)})
    ck = data.load_ckm()
    assert u.truth["wolfenstein"]["lambda"] == ck["lambda"]["value"]
    assert u.truth["wolfenstein"]["A"] == ck["A"]["value"]
    assert u.truth["wolfenstein"]["rhobar"] == 0.1
    with pytest.raises(ValueError):
        n1lam_universe(base, np.random.default_rng(1), prior={"lambda": (0.1, 0.2)})
    assert N1LAM_PRIOR["width"] == 1.0


def test_n1lam_follows_base_universe_ckm():
    """On an injected / fake base the lambda, A centres are the base universe's records."""
    base = Universe.real("NO")
    moved = nulls.n1_universe(base, np.random.default_rng(3))
    u = n1lam_universe(moved, np.random.default_rng(4), prior={"width": 0.0})
    assert u.truth["wolfenstein"]["lambda"] == moved.load_ckm()["lambda"]["value"]
    assert u.truth["lambda_A_centre"][0] == moved.truth["wolfenstein"]["lambda"]


def flat_pipeline(u):
    return u.evaluate({"s12sq": 0.30, "s13sq": 0.022, "s23sq": 0.45, "dcp_deg": 240.0}, ev.PMNS_KEYS, k=1,
                      k_discrete=1)


def lam_pipeline(u):
    """Depends on the CKM only through lambda: s13^2 = lambda^2 / 2.25."""
    lam = u.load_ckm()["lambda"]["value"]
    return u.evaluate({"s13sq": lam ** 2 / 2.25}, ("s13sq",), k=0)


def test_run_null_n1lam():
    u = nulls.make_injector({"s12sq": 0.30, "s13sq": 0.022, "s23sq": 0.45, "dcp_deg": 240.0})(
        np.random.default_rng(5), Universe.real())
    r = run_null(flat_pipeline, "N1lam", "NO", n_fakes=1000, base=u, seed=11)
    assert not r.passed and r.stopped_early and r.n_run == 5 and r.n_hits == 5
    # seed path: fake i uses default_rng([*seed, 5, ordering code, i])
    f = make_fake("N1lam", u, np.random.default_rng([11, 5, 0, 0]))
    assert f.kind == "N1lam"
    assert nulls.summarize(lam_pipeline(f)) is not None


def test_n1lam_vs_n1_on_lambda_only_relation():
    """A relation that uses only lambda is untouched by N1lam (every fake fits about as well as the
    real data) but separated by N1 (broad lambda)."""
    real = Universe.real("NO")
    ref = nulls.summarize(lam_pipeline(real))
    chi_lam = [nulls.summarize(lam_pipeline(make_fake("N1lam", real, np.random.default_rng([1, i]))))["chi2"]
               for i in range(100)]
    chi_n1 = [nulls.summarize(lam_pipeline(make_fake("N1", real, np.random.default_rng([1, i]))))["chi2"]
              for i in range(100)]
    assert np.median(chi_lam) < ref["chi2"] + 1.0
    assert np.median(chi_n1) > 100 * max(ref["chi2"], 1.0)


# ---------------------------------------------------------------- Delta m^2 ratio


@pytest.mark.parametrize("analysis", tuple(data.ANALYSES))
@pytest.mark.parametrize("ordering", ("NO", "IO"))
def test_load_dm_ratio(ordering, analysis):
    r = data.load_dm_ratio(ordering, analysis)
    p = data.load_pmns(ordering, analysis)
    d21 = p["dm21sq_1e-5eV2"]["value"] * 1e-5
    d3l = p["dm3l_sq_1e-3eV2"]["value"] * 1e-3
    assert (d3l > 0) == (ordering == "NO")
    assert r["value"] > 0 and r["value"] == pytest.approx(d21 / abs(d3l), rel=1e-15)
    assert r["value"] == ev._dm_best(ordering, analysis)["dm_ratio"]
    assert r["dm3l_sq_eV2"] == d3l and r["dm3l_abs_eV2"] == abs(d3l)
    assert 0 < r["err_plus"] < 0.05 * r["value"] and 0 < r["err_minus"] < 0.05 * r["value"]
    # symmetrised error close to the minimizer fallback of src.evaluate
    assert 0.5 * (r["err_plus"] + r["err_minus"]) == pytest.approx(ev._dm_sigma(ordering, analysis)["dm_ratio"],
                                                                    rel=0.1)
    # the scorer's profile chi^2 is ~1 at value +- err (independent-splitting approximation)
    for s in (r["err_plus"], -r["err_minus"]):
        assert 0.5 < ev.dm_ratio_delta_chi2(ordering, r["value"] + s, analysis) < 1.6
    assert ev.dm_ratio_delta_chi2(ordering, r["value"], analysis) < 0.05


def test_load_dm_ratio_error_sides_io():
    r = data.load_dm_ratio("IO")
    p = data.load_pmns("IO")
    e21, e3l = p["dm21sq_1e-5eV2"], p["dm3l_sq_1e-3eV2"]
    # IO: dm32 < 0, so the ratio goes up when |dm32| goes down, i.e. dm32 moves by +err_plus
    exp_plus = r["value"] * np.hypot(e21["err_plus"] / e21["value"], e3l["err_plus"] / abs(e3l["value"]))
    assert r["err_plus"] == pytest.approx(exp_plus, rel=1e-12)
    with pytest.raises(ValueError):
        data.load_dm_ratio("XO")


def test_dm_ratio_function():
    assert data.dm_ratio(7.5e-5, -2.5e-3) == pytest.approx(0.03)
    assert data.dm_ratio(7.5e-5, 2.5e-3) == pytest.approx(0.03)
    a = data.dm_ratio([1.0, 2.0], [-4.0, 4.0])
    assert isinstance(a, np.ndarray) and np.allclose(a, [0.25, 0.5])

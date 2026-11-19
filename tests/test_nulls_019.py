"""Exp 019: src.nulls tolerance, fit-only rule, ok_fn, N3 mass-side null, shared pool."""
import multiprocessing as mp

import numpy as np
import pytest

from src import data, evaluate as ev, rg
from src.nulls import (FIT_ONLY, N3_PRIOR, P_RTOL, Universe, candidate_ok, candidate_ok_fit_only, is_hit,
                       is_hit_fit_only, make_injector, n3_universe, power_check, run_null)
from test_nulls import FLAT, PMNS4, planted_pred


def _best(u, preds, k):
    rs = [u.evaluate(p, PMNS4, k=k, k_discrete=1) for p in preds]
    return max(rs, key=lambda r: (r.p_global, -r.chi2_global))


def planted_k3_pipeline(u):
    """The planted CKM-PMNS family of test_nulls, counted with k = 3 > N - 2 = 2."""
    ctx = u.context()
    return _best(u, [planted_pred(ctx, v) for v in (0, 1)], k=3)


def flat_k3_pipeline(u):
    return _best(u, FLAT, k=3)


REAL_R = rg.mass_ratios(errors=False)["value"]
T_NO = ev.pmns_best("NO")
A12 = T_NO["s12sq"] / np.sqrt(REAL_R["mr_u_c"])
A13 = T_NO["s13sq"] / REAL_R["mr_d_s"]
USE2 = ("s12sq", "s13sq")


def mass_pred(r):
    return {"s12sq": A12 * np.sqrt(r["mr_u_c"]), "s13sq": A13 * r["mr_d_s"]}


def mass_pipeline(u):
    """Toy mass-side relation: PMNS angles from same-scale quark mass ratios (k = 0)."""
    r = rg.mass_ratios(masses=u.rg_masses(), errors=False)["value"]
    return u.evaluate(mass_pred(r), USE2)


def flat_mass_pipeline(u):
    return u.evaluate({k: T_NO[k] for k in USE2}, USE2)


# ---------------------------------------------------------------- hit rules


def test_p_tolerance_and_fit_only_rule():
    ref = {"N": 4, "k": 2, "max_abs_pull": 0.5, "p_global": 0.3}
    assert P_RTOL == 1e-5
    assert is_hit({"N": 4, "k": 2, "max_abs_pull": 0.5, "p_global": 0.3 * (1 - 1e-7)}, ref)  # noise-level tie
    assert not is_hit({"N": 4, "k": 2, "max_abs_pull": 0.5, "p_global": 0.3 * (1 - 1e-3)}, ref)
    assert not is_hit({"N": 4, "k": 2, "max_abs_pull": 0.5, "p_global": 0.3 * (1 - 1e-7)}, ref, rtol=0.0)
    big = {"N": 4, "k": 3, "max_abs_pull": 0.5, "p_global": 0.5}
    assert not is_hit(big, ref) and is_hit_fit_only(big, ref)
    assert not is_hit_fit_only(dict(big, max_abs_pull=2.1), ref) and not is_hit_fit_only(None, ref)
    assert not candidate_ok(big) and candidate_ok_fit_only(big)
    assert not candidate_ok_fit_only(dict(big, p_global=0.04))


def test_fit_only_power_detects_planted_relation_with_k_above_n_minus_2():
    inj = make_injector(planted_pred(Universe.real().context()))
    kw = dict(n_inject=6, n_fakes=100, null="N1", seed=7, limit=0.05)
    strict = power_check(planted_k3_pipeline, inj, **kw)
    assert strict.fraction == 0.0 and strict.n_candidate_ok == 0 and strict.hit_rule == "strict"
    fit = power_check(planted_k3_pipeline, inj, **kw, **FIT_ONLY)
    assert fit.adequate and fit.n_pass == fit.n_candidate_ok >= 3 and fit.hit_rule.startswith("fit_only")
    assert all(n.hit_rule.startswith("fit_only") for n in fit.nulls if n is not None)
    flat = power_check(flat_k3_pipeline, make_injector(FLAT[0]), **kw, **FIT_ONLY)
    assert flat.fraction == 0.0 and flat.n_candidate_ok > 0


def test_power_check_ok_fn_is_used():
    inj = make_injector(planted_pred(Universe.real().context()))
    p = power_check(planted_k3_pipeline, inj, n_inject=2, n_fakes=5, seed=1, ok_fn=lambda r: False)
    assert p.n_candidate_ok == 0 and p.nulls == [None, None]


def test_shared_pool_matches_serial():
    inj = make_injector(planted_pred(Universe.real().context()))
    kw = dict(n_inject=3, n_fakes=12, null="N1", seed=5, limit=0.3)
    a = power_check(planted_k3_pipeline, inj, workers=1, **kw, **FIT_ONLY)
    b = power_check(planted_k3_pipeline, inj, workers=2, **kw, **FIT_ONLY)
    sig = lambda p: [None if n is None else (n.n_run, n.n_hits, [f["hit"] for f in n.fakes]) for n in p.nulls]
    assert sig(a) == sig(b) and a.fraction == b.fraction
    u = make_injector(FLAT[0])(np.random.default_rng(5), Universe.real())
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        r = run_null(flat_k3_pipeline, "N1", "NO", 1000, base=u, seed=11, workers=2, pool=pool, hit_fn=is_hit_fit_only)
        pool.terminate()
    assert r.stopped_early and r.n_run == 5 and r.n_hits == 5


# ---------------------------------------------------------------- N3


def test_n3_changes_mass_inputs():
    base = Universe.real("NO")
    real = data.load_masses()
    for i in range(3):
        f = n3_universe(base, np.random.default_rng([19, i]))
        fac = f.truth["factors"]
        assert f.ckm is None and f.pmns_centre == {} and f.kind == "N3"
        assert all(10 ** -N3_PRIOR["decades"] <= fac[k] <= 10 ** N3_PRIOR["decades"] for k in N3_PRIOR["ratios"])
        assert fac["c_t"] == fac["b_tau"] == 1.0  # heavy ratios not in the default prior
        m = f.load_masses()
        assert m["m_u"]["value"] == pytest.approx(real["m_u"]["value"] * fac["u_c"])
        assert m["m_d"]["value"] == pytest.approx(real["m_d"]["value"] * fac["d_s"] * fac["s_b"])
        assert m["m_e"]["value"] == pytest.approx(real["m_e"]["value"] * fac["e_mu"] * fac["mu_tau"])
        assert m["m_u"]["err_plus"] / m["m_u"]["value"] == pytest.approx(real["m_u"]["err_plus"] / real["m_u"]["value"])
        for k in ("m_c", "m_b", "m_t_direct", "m_tau"):
            assert m[k] == real[k]
        assert m["m_c_over_m_s"]["value"] == pytest.approx(real["m_c_over_m_s"]["value"] / (fac["s_b"]))
        # the fake masses reach src.rg through masses= and move the M_Z ratios by the drawn factors
        r = rg.mass_ratios(masses=f.rg_masses(), errors=False)["value"]
        for key, x in (("mr_u_c", fac["u_c"]), ("mr_d_s", fac["d_s"]), ("mr_s_b", fac["s_b"])):
            assert r[key] / REAL_R[key] == pytest.approx(x, rel=1e-7)
        for key, x in (("mr_e_mu", fac["e_mu"]), ("mr_mu_tau", fac["mu_tau"])):
            assert r[key] / REAL_R[key] == pytest.approx(x, rel=0.02)  # QED log term is not linear in M
        for key in ("mr_c_t", "mr_b_tau"):
            assert r[key] == pytest.approx(REAL_R[key], rel=1e-6)  # light Yukawas enter the y_t solve only via traces
        # and reach the scorer: the real ratios are now far off
        assert f.evaluate(REAL_R, ("mr_u_c",)).chi2 > 0 or fac["u_c"] == 1.0
    assert Universe.real().rg_masses() is None


def test_n3_heavy_ratios_optional_and_clipped():
    prior = {"ratios": ("c_t", "b_tau"), "heavy_decades": 1.0}
    for i in range(20):
        m = n3_universe(Universe.real(), np.random.default_rng([3, i]), prior).load_masses()
        assert 1.0 <= m["m_c"]["value"] <= 0.8 * m["m_b"]["value"] + 1e-12 and 2.5 <= m["m_b"]["value"] <= 50.0
    with pytest.raises(ValueError):
        n3_universe(Universe.real(), np.random.default_rng(0), {"ratios": ("u_t",)})


def test_n3_null_passes_mass_relation_and_fails_flat_family():
    u = make_injector(mass_pred(REAL_R))(np.random.default_rng(4), Universe.real())
    ref = mass_pipeline(u)
    assert ref.fit_ok
    r = run_null(mass_pipeline, "N3", "NO", n_fakes=40, base=u, seed=19, early_stop=False)
    assert r.n_hits == 0 and r.n_run == 40
    f = run_null(flat_mass_pipeline, "N3", "NO", n_fakes=1000, base=u, seed=19)
    assert not f.passed and f.stopped_early and f.n_run == 5

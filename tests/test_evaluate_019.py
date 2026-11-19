"""Exp 019: src.evaluate same-scale mass ratios, splittings, theory floor, high-scale scoring."""
import numpy as np
import pytest

from src import data, evaluate as ev, rg
from src.nulls import Universe, n2_universe

MR = ev.MASS_RATIO_KEYS
MZ = rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"]


def _targets():
    v, names, cl = ev.mass_ratio_targets()
    return v, names, cl


# ---------------------------------------------------------------- mass ratios


def test_mass_ratio_block_chi2_and_pulls():
    v, names, cl = _targets()
    r = ev.evaluate(v, "NO", use=MR)
    assert r.chi2 == 0.0 and r.N == 7 and r.max_abs_pull == 0.0 and r.chi2_global == r.chi2
    full = rg.mass_ratios()
    p = dict(v)
    p["mr_u_c"] += full["err"]["mr_u_c"]
    r = ev.evaluate(p, "NO", use=MR)
    assert r.pulls["mr_u_c"] == pytest.approx(1.0, rel=1e-3)
    res = np.array([p[n] - v[n] for n in names])
    assert r.chi2 == pytest.approx(res @ np.linalg.solve(full["cov"], res), rel=1e-3)
    assert r.chi2 > 1.0  # correlated with mr_c_t through m_c and alpha_s
    one = ev.evaluate(p, "NO", use=("mr_u_c",))
    assert one.chi2 == pytest.approx(1.0, rel=2e-3)


def test_mass_floor_on_ratios_and_raw_masses():
    v, _, _ = _targets()
    p = dict(v, mr_e_mu=v["mr_e_mu"] * 1.02)
    assert ev.evaluate(p, "NO", use=("mr_e_mu",)).chi2 > 1e10  # ppb errors without a floor
    r = ev.evaluate(p, "NO", use=("mr_e_mu",), mass_floor=0.02)
    assert r.chi2 == pytest.approx(1.0, rel=1e-3) and r.mass_floor == 0.02
    r = ev.evaluate(p, "NO", use=("mr_e_mu", "mr_u_c"), mass_floor={"mr_e_mu": 0.02})
    assert r.mass_floor == {"mr_e_mu": 0.02} and r.chi2 == pytest.approx(1.0, rel=1e-3)
    me = data.load_masses()["m_e"]["value"]
    r = ev.evaluate({"m_e": me * 1.01}, "NO", use=("m_e",), mass_floor=0.01)
    assert r.pulls["m_e"] == pytest.approx(1.0, rel=1e-6)
    assert ev.evaluate({"m_e": me}, "NO", use=("m_e",)).mass_floor == 0.0
    assert "mass_floor" in r.row()
    with pytest.raises(ValueError):
        ev.evaluate({"m_e": me}, "NO", use=("m_e",), mass_floor=-0.1)


def test_mass_ratio_alias_rules():
    ev.check_observables(MR)
    ev.check_observables(("mr_u_c", "mr_c_t", "ckm_s12", "s12sq", "s13sq"))
    for bad in (("mr_u_c", "m_u"), ("mr_c_t", "m_c_over_m_s"), ("mr_b_tau", "m_tau"), ("mr_c_t", "m_t_direct")):
        with pytest.raises(ev.AliasError):
            ev.check_observables(bad)
    with pytest.raises(ev.AliasError):
        ev.check_observables(("mr_u_c",), inputs=("m_c",))
    with pytest.raises(ev.AliasError):
        ev.check_observables(("mr_c_t",), inputs=("mr_u_c",))
    ev.check_observables(("mr_d_s",), inputs=("mr_u_c",))  # no shared flavour


def test_mass_ratio_targets_follow_mass_overrides():
    u = Universe(kind="x", masses={k: dict(r) for k, r in data.load_masses().items()
                                   if isinstance(r, dict) and "value" in r})
    u.masses["m_u"] = dict(u.masses["m_u"], value=2 * u.masses["m_u"]["value"])
    v, _, _ = _targets()
    sc = u.scorer(("mr_u_c",))
    assert sc._mr_value[0] == pytest.approx(2 * v["mr_u_c"], rel=1e-9)
    assert u.evaluate(dict(v, mr_u_c=2 * v["mr_u_c"]), ("mr_u_c",)).chi2 == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- splittings


@pytest.mark.parametrize("ordering", ("NO", "IO"))
def test_splitting_observables(ordering):
    b = ev._dm_best(ordering, data.DEFAULT_ANALYSIS)
    for key in ("dm21sq_eV2", "dm3l_sq_eV2"):
        for f in (0.98, 1.0, 1.02):
            x = b[key] * f
            r = ev.evaluate({key: x}, ordering, use=(key,))
            d1 = data.pmns_delta_chi2(ordering, **{key: x})
            assert r.chi2 == pytest.approx(d1, abs=1e-12)
            assert r.pulls[key] == pytest.approx(np.sign(x - b[key]) * np.sqrt(d1), abs=1e-12)
            assert r.chi2_global == pytest.approx(r.chi2 + data.ordering_offset(ordering))
    # ratio: brute-force profile over the DMA axis agrees
    x, y = data.pmns_profile_1d(ordering, "dm3l_sq_eV2")
    grid = np.linspace(x[0], x[-1], 400001)
    for f in (0.97, 1.0, 1.04):
        rr = b["dm_ratio"] * f
        brute = np.min(data.pmns_profile_1d(ordering, "dm21sq_eV2", rr * np.abs(grid) * 1e-3)
                       + np.interp(grid, x, y))
        got = ev.dm_ratio_delta_chi2(ordering, rr)
        assert got == pytest.approx(brute, abs=2e-4)
    assert ev.dm_ratio_delta_chi2(ordering, b["dm_ratio"]) < 0.05
    assert ev.dm_ratio_delta_chi2(ordering, b["dm_ratio"] * 1.1) > 9


def test_splitting_alias_ordering_bit_and_universes():
    with pytest.raises(ev.AliasError):
        ev.check_observables(("dm_ratio", "dm21sq_eV2"))
    ev.check_observables(("dm21sq_eV2", "dm3l_sq_eV2", "s12sq"))
    b = ev._dm_best("NO", data.DEFAULT_ANALYSIS)
    r = ev.evaluate({"dm_ratio": b["dm_ratio"]}, "best", use=("dm_ratio",))
    assert r.ordering == "NO" and r.k == 1 and r.k_discrete == 1
    # N2 fakes translate the angles only; splittings score as on real data
    u = n2_universe(Universe.real("NO"), np.random.default_rng(1))
    pred = {"dm21sq_eV2": b["dm21sq_eV2"] * 1.01, "dm_ratio": b["dm_ratio"] * 0.99}
    assert u.evaluate({"dm21sq_eV2": pred["dm21sq_eV2"]}, ("dm21sq_eV2",)).chi2 == pytest.approx(
        ev.evaluate(pred, "NO", use=("dm21sq_eV2",)).chi2, abs=1e-12)
    assert u.evaluate(pred, ("dm_ratio",)).chi2 == ev.evaluate(pred, "NO", use=("dm_ratio",)).chi2


# ---------------------------------------------------------------- high-scale scoring


def _data_at(mu_, ordering="NO", **kw):
    mu = mu_
    c = rg.ckm_at(mu, kw.get("model", "SM"), kw.get("tan_beta"), kw.get("m_susy", 3000.0))
    t = ev.pmns_best(ordering)
    pm = tuple(float(np.arcsin(np.sqrt(t[k]))) for k in ("s12sq", "s13sq", "s23sq")) + (np.radians(t["dcp_deg"]),)
    p = rg.pmns_at(mu, ordering, kw.get("m_lightest", 0.0), model=kw.get("model", "SM"), tan_beta=kw.get("tan_beta"),
                   m_susy=kw.get("m_susy", 3000.0), pmns=pm)
    out = {"ckm_s12": np.sin(c["theta12"]), "ckm_s13": np.sin(c["theta13"]), "ckm_s23": np.sin(c["theta23"]),
           "ckm_delta_rad": c["delta"], "s12sq": np.sin(p["theta12"]) ** 2, "s13sq": np.sin(p["theta13"]) ** 2,
           "s23sq": np.sin(p["theta23"]) ** 2, "dcp_deg": np.degrees(p["delta"]), "dm_ratio": p["dm21_over_dm3l"]}
    out.update(rg.mass_ratios(mu, kw.get("model", "SM"), kw.get("tan_beta"), kw.get("m_susy", 3000.0),
                              errors=False)["value"])
    return out


USE_GUT = ("ckm_s12", "ckm_s13", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio") + MR


@pytest.mark.parametrize("spec", ({"mu": 2e16}, {"mu": 2e16, "model": "MSSM", "tan_beta": 10.0}))
def test_data_run_up_scores_like_data_at_low_scale(spec):
    hi = _data_at(spec["mu"], **spec)
    lo = _data_at(MZ)
    a = ev.evaluate(hi, "NO", use=USE_GUT, scale=spec)
    b = ev.evaluate(lo, "NO", use=USE_GUT)
    assert a.chi2 == pytest.approx(b.chi2, abs=1e-6)
    assert a.scale["mu"] == 2e16 and set(a.scale_shift) == set(USE_GUT)
    assert all(s >= 0 for s in a.scale_systematic.values())
    assert a.scale_shift["mr_b_tau"] > 1.5  # m_b/m_tau falls by ~40% or more from M_Z to the GUT scale
    assert b.scale is None and b.scale_shift == {}


def test_high_scale_prediction_off_by_one_sigma_scores_one_sigma():
    ck = data.load_ckm()
    s12 = ck["s12"]["value"] + ck["s12"]["err_plus"]
    inp = (np.arcsin(s12), np.arcsin(ck["s13"]["value"]), np.arcsin(ck["s23"]["value"]), ck["delta_rad"]["value"])
    hi = rg.ckm_at(2e16, ckm=inp)
    r = ev.evaluate({"ckm_s12": np.sin(hi["theta12"])}, "NO", use=("ckm_s12",), scale={"mu": 2e16})
    assert r.pulls["ckm_s12"] == pytest.approx(1.0, abs=1e-3)
    m = data.load_masses()["m_u"]
    shifted = {"m_u": m["value"] + m["err_plus"]}
    ghi = rg.mass_ratios(2e16, masses=shifted, errors=False)["value"]["mr_u_c"]
    glo = rg.mass_ratios(masses=shifted, errors=False)["value"]["mr_u_c"]
    a = ev.evaluate({"mr_u_c": ghi}, "NO", use=("mr_u_c",), scale={"mu": 2e16})
    b = ev.evaluate({"mr_u_c": glo}, "NO", use=("mr_u_c",))
    assert a.chi2 == pytest.approx(b.chi2, rel=1e-4) and 0.8 < b.chi2 < 1.2


def test_high_scale_rejects_unsupported_keys_and_bad_specs():
    with pytest.raises(ValueError):
        ev.evaluate({"m_tau": 1.7}, "NO", use=("m_tau",), scale={"mu": 2e16})
    with pytest.raises(ValueError):
        ev.evaluate({"s12sq": 0.3}, "NO", use=("s12sq",), scale={"mu": 2e16, "model": "MSSM"})
    with pytest.raises(ValueError):
        ev.evaluate({"s12sq": 0.3}, "NO", use=("s12sq",), scale={"mu": 2e16, "tanb": 3})


def test_mssm_systematic_includes_threshold_variation_and_fit_passes_options():
    spec = {"mu": 2e16, "model": "MSSM", "tan_beta": 10.0}
    sm = ev.evaluate(_data_at(2e16), "NO", use=("mr_b_tau",), scale={"mu": 2e16})
    ms = ev.evaluate(_data_at(2e16, **spec), "NO", use=("mr_b_tau",), scale=spec)
    assert sm.scale_systematic["mr_b_tau"] == pytest.approx(0.05 * np.log(sm.scale_shift["mr_b_tau"]))
    assert ms.scale_systematic["mr_b_tau"] > 0.05 * abs(np.log(ms.scale_shift["mr_b_tau"]))
    target = _data_at(2e16)["mr_b_tau"]
    fr = ev.fit_and_evaluate(lambda th: {"mr_b_tau": th["x"]}, {"x": (0.3, 1.0)}, "NO", ("mr_b_tau",),
                             n_starts=4, scale={"mu": 2e16}, mass_floor=0.01)
    assert fr.params["x"] == pytest.approx(target, rel=1e-4)
    assert fr.result.mass_floor == 0.01 and fr.result.scale["mu"] == 2e16 and fr.result.k == 1

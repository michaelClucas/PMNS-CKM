"""Clebsch double ratio R_clebsch in src.evaluate, and the Objective sum1d warning (exp 024)."""
import json
import math
import pathlib
import warnings

import numpy as np
import pytest

from src import evaluate as ev
from src import rg
from src.stats import Objective

ROOT = pathlib.Path(__file__).resolve().parents[1]
C1 = json.loads((ROOT / "experiments/20261119-021-theory-scout/checks.json").read_text())["C1_double_ratio"]


def _scorer(use, **kw):
    return ev.Scorer(tuple(use), "NO", **kw)


def test_r_value_matches_exp021_and_ratio_of_mr_keys():
    sc = _scorer(["R_clebsch"])
    vals, _, _ = ev.mass_ratio_targets()
    assert sc._mr_value[0] == pytest.approx(C1["R_double_ratio"], rel=1e-12)
    assert sc._mr_value[0] == pytest.approx(vals["mr_e_mu"] / vals["mr_d_s"], rel=1e-14)


def test_r_error_is_linear_propagation_of_cov_log():
    r = rg.mass_ratios()
    n = list(r["names"])
    c = np.zeros(len(n))
    c[n.index("mr_e_mu")], c[n.index("mr_d_s")] = 1, -1
    var = c @ r["cov_log"] @ c
    sc = _scorer(["R_clebsch"])
    assert (sc._mr_sig[0] / sc._mr_value[0]) ** 2 == pytest.approx(var, rel=1e-12)
    # same as exp 021's PDG-only relative error (m_e/m_mu error is ppm, m_d and m_s errors uncorrelated)
    assert sc._mr_sig[0] / sc._mr_value[0] == pytest.approx(C1["rho_rel_err_pdg"], rel=1e-3)


def test_r_mass_floor_in_quadrature():
    s0 = _scorer(["R_clebsch"])
    s2 = _scorer(["R_clebsch"], mass_floor=0.02)
    rel0, rel2 = s0._mr_sig[0] / s0._mr_value[0], s2._mr_sig[0] / s2._mr_value[0]
    assert rel2 == pytest.approx(math.hypot(rel0, 0.02), rel=1e-12)
    sk = _scorer(["R_clebsch"], mass_floor={"R_clebsch": 0.03, "default": 0.0})
    assert sk._mr_sig[0] / sk._mr_value[0] == pytest.approx(math.hypot(rel0, 0.03), rel=1e-12)


def test_r_chi2_and_pull():
    sc = _scorer(["R_clebsch"], mass_floor=0.02)
    v, s = sc._mr_value[0], sc._mr_sig[0]
    res = sc({"R_clebsch": v + 1.5 * s})
    assert res.chi2 == pytest.approx(2.25, rel=1e-10)
    assert res.pulls["R_clebsch"] == pytest.approx(1.5, rel=1e-10)
    assert res.N == 1 and res.chi2_global == res.chi2  # no PMNS: no ordering offset


def test_r_joint_block_covariance():
    """R with mr_e_mu: covariance from cov_log; chi2 equals the explicit quadratic form."""
    sc = _scorer(("R_clebsch", "mr_u_c", "mr_e_mu"), mass_floor=0.02)
    use = sc.mr_use   # block order (MR_BLOCK_KEYS), not the order given
    assert use == ("mr_u_c", "mr_e_mu", "R_clebsch")
    r = rg.mass_ratios()
    n = list(r["names"])
    C = np.zeros((3, len(n)))
    C[0, n.index("mr_u_c")] = 1
    C[1, n.index("mr_e_mu")] = 1
    C[2, n.index("mr_e_mu")], C[2, n.index("mr_d_s")] = 1, -1
    cl = C @ r["cov_log"] @ C.T + np.eye(3) * 0.02 ** 2
    v = sc._mr_value
    cov = cl * np.outer(v, v)
    pred = dict(zip(use, v * np.array([1.01, 0.97, 1.05])))
    d = np.array([pred[k] for k in use]) - v
    assert sc.chi2(pred) == pytest.approx(d @ np.linalg.solve(cov, d), rel=1e-9)
    assert cov[1, 2] > 0   # R and mr_e_mu share m_e / m_mu


def test_r_alias_rules():
    ev.check_observables(("R_clebsch", "mr_e_mu"))
    ev.check_observables(("R_clebsch", "mr_d_s", "mr_s_b", "ckm_s12"))
    with pytest.raises(ev.AliasError):
        ev.check_observables(("R_clebsch", "mr_e_mu", "mr_d_s"))
    for other in ("m_d", "m_s", "m_e", "m_mu"):
        if other in ev.observable_keys():
            with pytest.raises(ev.AliasError):
                ev.check_observables(("R_clebsch", other))
    with pytest.raises(ev.AliasError):
        ev.check_observables(("R_clebsch",), inputs=("mr_d_s",))
    assert ev.alias_atoms("R_clebsch") == frozenset({"mass:e", "mass:mu", "mass:d", "mass:s"})
    assert "R_clebsch" in ev.observable_keys()


def test_r_scale_translation_is_ratio_of_factors():
    spec = {"mu": 2e16}
    sc = _scorer(["R_clebsch", "mr_e_mu"], scale=spec)
    sc2 = _scorer(["mr_d_s"], scale=spec)
    f = sc.scale_shift
    assert f["R_clebsch"] == pytest.approx(f["mr_e_mu"] / sc2.scale_shift["mr_d_s"], rel=1e-12)
    assert abs(f["R_clebsch"] - 1) < 1e-2   # near RG-invariant in the SM (exp 021)


def test_r_follows_mass_overrides():
    from src import data
    recs = {k: dict(v) for k, v in data.load_masses().items() if isinstance(v, dict)}
    recs["m_d"]["value"] *= 1.1
    sc = ev.Scorer(("R_clebsch",), "NO", mass_records=recs)
    base = _scorer(["R_clebsch"])
    assert sc._mr_value[0] == pytest.approx(base._mr_value[0] / 1.1, rel=1e-3)


def test_mr_only_scores_unchanged_path():
    """Without R the block uses the pre-exp-024 code path (bit-identical)."""
    use = ev.MASS_RATIO_KEYS
    sc = _scorer(use, mass_floor=0.02)
    vals, names, cov_log = ev.mass_ratio_targets()
    idx = [names.index(n) for n in use]
    cl = cov_log[np.ix_(idx, idx)].copy()
    cl[np.diag_indices_from(cl)] += 0.02 ** 2
    v = np.array([vals[n] for n in use])
    assert np.array_equal(sc._mr_cov, cl * np.outer(v, v))


def _fake(ordering, **kw):
    return sum((v or 0.0) * 0 for v in kw.values())


def test_objective_default_is_joint_without_warning():
    # exp 026 flipped the default to joint (exp 024 had a warning on the implicit sum1d default)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Objective(pmns_delta_chi2=_fake).pmns_mode == "joint"
        Objective(pmns_delta_chi2=_fake, pmns_mode="sum1d")
        Objective(pmns_delta_chi2=_fake, pmns_mode="joint")
        Objective(pmns_delta_chi2=_fake, pmns_params=("s13sq",))
        o = Objective(pmns_delta_chi2=_fake, pmns_mode="sum1d")
    assert o.pmns_mode == "sum1d"

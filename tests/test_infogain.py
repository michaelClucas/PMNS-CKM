"""src.infogain reproduces exp 021 bound.json (exp 024)."""
import csv
import json
import math
import pathlib
import sys

import numpy as np
import pytest

from src import infogain as ig

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
BOUND = json.loads((EXP / "20261119-021-theory-scout/bound.json").read_text())
REL = 1e-12


def _close(got: dict, ref: dict, keys):
    for k in keys:
        if isinstance(ref[k], list):
            assert got[k] == pytest.approx(ref[k], rel=REL, abs=1e-15), k
        else:
            assert got[k] == pytest.approx(ref[k], rel=REL, abs=1e-15), k


NULL_KEYS = ("real_chi2", "n", "hits", "rate", "rate_ci95", "S_rank", "S_rank_max_ci", "B_ckm", "fake_median")


def test_data_bits_match_exp021():
    for o in ("NO", "IO"):
        _close(ig.pmns_data_bits(o), BOUND["data_budget_bits_vs_haar"][o],
               ("s12sq", "s13sq", "s23sq", "dcp_deg", "total_4"))


def test_null_info_f1_f2_match_exp021():
    ref = BOUND["F1_F2_hierarchical_null"]
    rp = json.loads((EXP / "20261118-007-f1-shared-rotation/real_pipeline.json").read_text())["pipeline"]
    rows = list(csv.DictReader(open(EXP / "20261118-007-f1-shared-rotation/null_b.csv")))
    for o in ("NO", "IO"):
        for v in ("cont", "struct"):
            got = ig.null_info(rp[f"table_{o}"][f"best_{v}"], [float(r[f"best_{v}_{o}"]) for r in rows])
            _close(got, ref[f"F1_{o}_{v}"], NULL_KEYS)
    pr = json.loads((EXP / "20261118-009-f2-clebsch/pipeline_real.json").read_text())
    rows = list(csv.DictReader(open(EXP / "20261118-009-f2-clebsch/null_b.csv")))
    n = 0
    for sc in ("MZ", "GUT"):
        for o in ("NO", "IO"):
            for v in ("fit", "struct"):
                for t in ("A", "ABC", "ALL"):
                    col = f"{sc}_tab_{o}_{v}_{t}"
                    got = ig.null_info(pr[sc][f"tab_{o}"][v][t], [float(r[col]) for r in rows])
                    _close(got, ref[f"F2_{col}"], NULL_KEYS)
                    n += 1
    assert n == 24


def test_null_info_protocol_match_exp021():
    for f in sorted((EXP / "20261118-018-protocol-reruns").glob("rate_*.json")):
        d = json.loads(f.read_text())
        got = ig.null_info(d["ref"]["chi2_global"], [x["result"]["chi2_global"] for x in d["fakes"] if x.get("result")])
        _close(got, BOUND["protocol_rate_runs"][f.stem], NULL_KEYS)


def test_i_gauss_matches_exp021():
    for case, hj in (("MN_ours", "heldout_MN"), ("MN_drrel", "heldout_MN_drrel"), ("FN_ours", "heldout_FN")):
        h = json.loads((EXP / f"20261118-014-so10-fits/{hj}.json").read_text())["regression_R2_post"]
        for t, ref in BOUND["so10_heldout"][case]["I_gauss_post"].items():
            best = max(x for x in (h[m][t] for m in ("linear", "quadratic", "gbm")) if x is not None)
            assert ig.i_gauss(best) == pytest.approx(ref["I_gauss_bits"], rel=REL, abs=1e-15)


def test_sharing_matches_exp021():
    occam = math.log2(3.5 / 0.04 ** 2)
    for lab, j, l, q in (("2O_ours_NO", 68.8, 14.4, 0.0), ("A4_cat_IO", 446.5, 1.6, 254.8)):
        got = ig.sharing_net_bits(j, l, q, occam)
        ref = BOUND["modular_sharing"][lab]
        assert got["net_bits"] == pytest.approx(ref["net_bits"], rel=REL)
        assert got["delta_chi2_sharing"] == pytest.approx(ref["delta_chi2_sharing"], rel=REL)


def test_null_info_edge_cases():
    r = ig.null_info(1.0, [2.0, 3.0, np.inf, np.nan, 4.0])
    assert r["n"] == 3 and r["hits"] == 0 and r["S_rank"] == pytest.approx(math.log2(6))
    # identical real and fakes: B_ckm = 0
    assert ig.null_info(5.0, [5.0] * 10)["B_ckm"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        ig.null_info(1.0, [np.inf])


def test_predictive_gain_gaussian_analytic():
    rng = np.random.default_rng(0)
    x, s = 0.3, 0.01
    post = rng.normal(x, 0.01, 400000)       # E exp(-chi2/2) = s / sqrt(s^2 + 0.01^2)
    prior = rng.uniform(0, 1, 400000)        # E exp(-chi2/2) = s sqrt(2 pi)
    g = ig.predictive_gain(post, prior, [x], [s])
    expect = math.log2((s / math.sqrt(s ** 2 + 0.01 ** 2)) / (s * math.sqrt(2 * math.pi)))
    assert g["G_pred"] == pytest.approx(expect, abs=0.02)
    # periodic column: wrapping makes 359 deg and 1 deg equivalent
    a = ig.predictive_gain(np.array([[359.0]]), np.array([[1.0]]), [0.0], [5.0], periodic=[360.0])
    assert a["G_pred"] == pytest.approx(0.0, abs=1e-12)


def test_kl_shift_basic():
    rng = np.random.default_rng(1)
    b = rng.uniform(0, 1, 100000)
    assert ig.kl_shift(rng.uniform(0, 1, 100000), b) < 0.01
    # all posterior mass in one of 40 equal-probability bins (+0.5 pseudo-count per bin)
    P = np.array([1000.5] + [0.5] * 39) / 1019.5
    assert ig.kl_shift(np.full(1000, 0.51), b) == pytest.approx(float(np.sum(P * np.log2(P * 40))), abs=0.01)


@pytest.mark.skipif(not (ROOT / "artifacts/so10/pt_MN_ql_1.npz").exists(), reason="exp 014 PT samples (gitignored)")
def test_so10_mn_ours_matches_exp021():
    """MN_ours used the first draw of bound.py's default_rng(5), so it is reproducible alone."""
    sys.path.insert(0, str(EXP / "20261118-014-so10-fits"))
    import so10fit as S
    z = np.concatenate([np.load(S.ART / f"pt_MN_ql_{i}.npz")["z"] for i in range(1, 6)])
    io = np.concatenate([np.load(S.ART / f"pt_MN_ql_{i}.npz")["io"] for i in range(1, 6)])
    prob = S.Problem("MN", "ours")
    lo, hi = S.mn_box("ours", prob.val)
    zp, op = prob.predict(lo + (hi - lo) * np.random.default_rng(5).random((200000, len(lo))))
    okp = np.all(np.isfinite(zp), 1)
    zp, iop = zp[okp], op["io"][okp]
    ok = np.all(np.isfinite(z), 1)
    z, io = z[ok], io[ok]
    z, zp = z[~io.astype(bool)], zp[~iop.astype(bool)]
    ref = BOUND["so10_heldout"]["MN_ours"]
    per = [None, None, None, 2 * math.pi]
    js = [12, 13, 14, 15]
    g = ig.predictive_gain(z[:, js], zp[:, js], prob.val[js], prob.err[js], periodic=per)
    assert g["G_pred"] == pytest.approx(ref["joint4_G_pred_sigma_x1"], rel=1e-9)
    assert [g["min_chi2_post"], g["min_chi2_prior"]] == pytest.approx(ref["joint4_min_chi2_post_prior_x1"], rel=1e-9)
    g3 = ig.predictive_gain(z[:, js], zp[:, js], prob.val[js], prob.err[js], periodic=per, sigma_scale=3.0)
    assert g3["G_pred"] == pytest.approx(ref["joint4_G_pred_sigma_x3"], rel=1e-9)
    assert ig.kl_shift(z[:, 13], zp[:, 13]) == pytest.approx(ref["s13sq"]["KL_shift_bits"], rel=1e-9)

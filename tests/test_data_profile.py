"""Exp 013: vectorized 1D profile accessor and direct uniform-grid indexing in src.data."""
import numpy as np
import pytest

from src import data
from src.data import ANALYSES, ORDERINGS, pmns_delta_chi2, pmns_profile_1d


def _ref_pmns(ordering, s12sq=None, s13sq=None, s23sq=None, dcp_deg=None, analysis=data.DEFAULT_ANALYSIS):
    """The pre-exp-013 implementation (searchsorted, shift recomputed per call)."""
    def interp_1d(x, y, v):
        if v < x[0] or v > x[-1]:
            return np.inf
        return float(np.interp(v, x, y))

    def interp_2d(x, y, z, u, v):
        if u < x[0] or u > x[-1] or v < y[0] or v > y[-1]:
            return np.inf
        i = min(max(np.searchsorted(x, u) - 1, 0), len(x) - 2)
        j = min(max(np.searchsorted(y, v) - 1, 0), len(y) - 2)
        tx = (u - x[i]) / (x[i + 1] - x[i])
        ty = (v - y[j]) / (y[j + 1] - y[j])
        return float((1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i + 1, j]
                     + (1 - tx) * ty * z[i, j + 1] + tx * ty * z[i + 1, j + 1])

    one = {"T12": s12sq, "T13": s13sq, "T23": s23sq,
           "DCP": None if dcp_deg is None else (dcp_deg + 180.0) % 360.0 - 180.0}
    tot = 0.0
    if s23sq is not None and dcp_deg is not None:
        x, y, z = data._grid_2d(analysis, ordering, "T23-DCP")
        tot += interp_2d(x, y, z - z.min(), s23sq, one["DCP"])
        one["T23"] = one["DCP"] = None
    for tag, v in one.items():
        if v is None:
            continue
        x, y = data._profile_1d(analysis, ordering, tag)
        tot += interp_1d(x, y - y.min(), v)
    return tot


@pytest.mark.parametrize("analysis", tuple(ANALYSES))
@pytest.mark.parametrize("ordering", ORDERINGS)
def test_pmns_delta_chi2_bitwise_unchanged(ordering, analysis):
    rng = np.random.default_rng(13)
    x, y, _ = data._grid_2d(analysis, ordering, "T23-DCP")
    pts = [dict(s12sq=a, s13sq=b, s23sq=c, dcp_deg=d) for a, b, c, d in zip(
        rng.uniform(0.15, 0.85, 400), rng.uniform(-0.005, 0.075, 400),
        rng.uniform(0.24, 0.76, 400), rng.uniform(-400, 400, 400))]
    pts += [dict(s23sq=float(u), dcp_deg=float(v)) for u in x[::7] for v in y[::9]]  # exact grid nodes
    pts += [dict(s23sq=float(x[0]), dcp_deg=180.0), dict(s23sq=float(x[-1]), dcp_deg=-180.0)]
    worst = 0.0
    for p in pts:
        a, b = pmns_delta_chi2(ordering, analysis=analysis, **p), _ref_pmns(ordering, analysis=analysis, **p)
        if np.isinf(b):
            assert np.isinf(a)
            continue
        worst = max(worst, abs(a - b))
    assert worst <= 1e-12


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("param,lo,hi", [("s12sq", 0.1, 0.9), ("s13sq", -0.01, 0.08), ("s23sq", 0.2, 0.8),
                                         ("dcp_deg", -720, 720), ("dm21sq_eV2", -1e-5, 4e-4),
                                         ("dm3l_sq_eV2", -3e-3, 8e-3)])
def test_profile_1d_matches_scalar(ordering, param, lo, hi):
    v = np.random.default_rng(1).uniform(lo, hi, 300)
    vec = pmns_profile_1d(ordering, param, v)
    sca = np.array([pmns_delta_chi2(ordering, **{param: float(t)}) for t in v])
    assert vec.shape == v.shape
    assert np.array_equal(np.isinf(vec), np.isinf(sca))
    fin = np.isfinite(sca)
    assert fin.sum() > 50
    assert np.array_equal(vec[fin], sca[fin])  # same np.interp on the same table: bitwise


def test_profile_1d_table_and_global():
    x, d = pmns_profile_1d("IO", "T12")
    assert d.min() == 0.0 and len(x) == len(d)
    g = pmns_profile_1d("IO", "s12sq", [0.31], relative_to="global")
    assert g[0] == pytest.approx(pmns_delta_chi2("IO", s12sq=0.31, relative_to="global"))
    with pytest.raises(ValueError):
        pmns_profile_1d("NO", "theta12")

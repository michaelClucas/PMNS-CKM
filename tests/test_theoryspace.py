"""src.theoryspace and src.yukawa.observables_batch (exp 033).

GEN and CTRL are ported from exp 025 gen.py: same rng calls, so the same Generator state must
give the same draws (to rounding of the engine's Takagi phase convention). The acceptance rates
must match exp 025's committed gen_stats.json within sampling error (cluster-robust: draws share
a discrete structure in batches of 64), and the accepted marginals must match exp 025's data
(artifacts/025/data.npz, local only: those tests skip when it is absent).
"""
import json
import pathlib

import numpy as np
import pytest

from src import theoryspace as ts
from src import yukawa as yk

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN_STATS = ROOT / "experiments" / "20261119-025-nn-ckm-info" / "gen_stats.json"
DATA_025 = ROOT / "artifacts" / "025" / "data.npz"
ANGLE_KEYS = ("dcp_deg", "alpha21_deg", "alpha31_deg")


def _cplx(rng, B):
    return rng.normal(size=(B, 3, 3)) + 1j * rng.normal(size=(B, 3, 3))


def _angle_diff(a, b, period):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


# ---------------------------------------------------------------- batched engine


@pytest.mark.parametrize("ordering", ("auto", "NO", "IO"))
def test_observables_batch_matches_scalar(ordering):
    rng = np.random.default_rng(3301)
    B = 150
    Yu, Yd, Ye = _cplx(rng, B), _cplx(rng, B), _cplx(rng, B)
    m = _cplx(rng, B)
    m = m + np.swapaxes(m, 1, 2)
    o = yk.observables_batch(Yu, Yd, Ye, m, ordering=ordering, nu_mass_scale_eV=0.05)
    seen = set()
    for i in range(B):
        s, det = yk.observables(Yu[i], Yd[i], Ye[i], m[i], ordering=ordering, nu_mass_scale_eV=0.05, details=True)
        seen.add(det["ordering"])
        assert o["io"][i] == float(det["ordering"] == "IO")
        for k, v in s.items():
            if k in ANGLE_KEYS:
                assert _angle_diff(o[k][i], v, 360.0) < 1e-9, k
            elif k == "ckm_delta_rad":
                assert _angle_diff(o[k][i], v, 2 * np.pi) < 1e-12, k
            else:
                assert abs(o[k][i] - v) <= 1e-12 * max(1.0, abs(v)), k
    if ordering == "auto":
        assert seen == {"NO", "IO"}


def test_observables_batch_bad_rows_are_nan():
    rng = np.random.default_rng(3302)
    Yu, Yd, Ye, m = _cplx(rng, 4), _cplx(rng, 4), _cplx(rng, 4), _cplx(rng, 4)
    m = m + np.swapaxes(m, 1, 2)
    Yu[1, 0, 0] = np.nan
    m[2, 0, 1] += 1.0  # not symmetric
    o = yk.observables_batch(Yu, Yd, Ye, m)
    for k in yk.OUTPUT_KEYS + ("io",):
        assert np.isfinite(o[k][[0, 3]]).all() and np.isnan(o[k][[1, 2]]).all()
    with pytest.raises(ValueError):
        yk.observables_batch(Yu[0], Yd, Ye, m)


# ---------------------------------------------------------------- registry


def test_registry():
    names = ts.families()
    for f in ("GEN", "CTRL", "F1", "F2", "MN", "FN", "MOD", "SU5", "S4SO"):
        assert f in names
    assert ts.get("GEN").native and ts.get("CTRL").native and not ts.get("MN").native
    assert ts.get("GEN").shared is True and ts.get("CTRL").shared is False
    with pytest.raises(ValueError):
        ts.register(ts.Family("GEN", ts.gen_textures))
    with pytest.raises(KeyError):
        ts.get("nope")


@pytest.mark.parametrize("fam", ("GEN", "CTRL"))
def test_native_keys_and_shapes(fam):
    o = ts.sample(fam, 300, np.random.default_rng(1))
    assert set(o) == set(ts.OBS_KEYS)
    assert all(v.shape == (300,) for v in o.values())
    ok = np.isfinite(o["s12sq"])
    assert ok.mean() > 0.9
    f = ok & (o["mr_d_s"] > 0)
    assert np.allclose(o["R_clebsch"][f], o["mr_e_mu"][f] / o["mr_d_s"][f])
    assert np.all(o["dm_ratio"][ok] > 0)
    assert set(np.unique(o["io"][ok])) <= {0.0, 1.0}


# ---------------------------------------------------------------- port == exp 025


@pytest.mark.parametrize("fam", ("GEN", "CTRL"))
def test_port_reproduces_exp025_same_seed(fam):
    g = ts._exp025_module()
    a = ts.from_exp025(g.GENS[fam](np.random.default_rng([25, 7]), 8000))
    b = ts.sample(fam, 8000, np.random.default_rng([25, 7]))
    oka, okb = ts.accept(a, ordering=None), ts.accept(b, ordering=None)
    assert oka.sum() > 30 and np.array_equal(oka, okb)
    for k in ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dm_ratio", "io",
              *ts.MR_KEYS, "R_clebsch"):
        assert np.allclose(a[k][oka], b[k][oka], rtol=1e-9, atol=1e-12), k
    assert _angle_diff(a["dcp_deg"][oka], b["dcp_deg"][oka], 360.0).max() < 1e-7


def test_features_match_exp025():
    g = ts._exp025_module()
    raw = g.GENS["GEN"](np.random.default_rng([25, 7]), 4000)
    c0, m0, y0, _, ok0 = g.features(raw)
    o = ts.from_exp025(raw)
    ok = ts.accept(o)
    assert np.array_equal(ok, ok0)
    c, m, y = ts.features_025(o)
    assert np.allclose(c[ok], c0[ok], atol=1e-12) and np.allclose(m[ok], m0[ok], atol=1e-12)
    assert np.allclose(y[ok][:, [0, 1, 2, 4]], y0[ok][:, [0, 1, 2, 4]], atol=1e-12)
    assert _angle_diff(y[ok][:, 3], y0[ok][:, 3], 2 * np.pi).max() < 1e-12


@pytest.mark.skipif(not DATA_025.exists(), reason="artifacts/025/data.npz is local only")
@pytest.mark.parametrize("fam,idx", (("GEN", 7), ("CTRL", 8)))
def test_port_reproduces_exp025_training_rows(fam, idx):
    """exp 025 gen._task: rng = default_rng([25, i]), first chunk 7500 draws; its accepted rows
    are the first rows of the family in data.npz."""
    D = np.load(DATA_025)
    sel = D["fam"] == list(D["families"]).index(fam)
    o = ts.sample(fam, 7500, np.random.default_rng([25, idx]))
    ok = ts.accept(o)
    c, m, y = ts.features_025({k: v[ok] for k, v in o.items()})
    n = int(ok.sum())
    assert n > 50
    assert np.allclose(c, D["ctx_ckm"][sel][:n], atol=1e-9)
    assert np.allclose(m, D["ctx_m"][sel][:n], atol=1e-9)
    assert np.allclose(y[:, [0, 1, 2, 4]], D["y"][sel][:n][:, [0, 1, 2, 4]], atol=1e-9)
    assert _angle_diff(y[:, 3], D["y"][sel][:n][:, 3], 2 * np.pi).max() < 1e-9


# ---------------------------------------------------------------- statistics vs exp 025


N_RAW = 64 * 1000
BLOCK = 64 * 20


@pytest.fixture(scope="module")
def fresh():
    """64k raw draws per family from seeds exp 025 never used."""
    out = {}
    for i, fam in enumerate(("GEN", "CTRL")):
        o = ts.sample(fam, N_RAW, np.random.default_rng([33, 1, i]))
        out[fam] = (o, ts.accept(o))
    return out


def _cluster_mean(x, ok, block=BLOCK):
    """Mean of x[ok] and its cluster-robust SE with clusters = contiguous blocks of raw draws."""
    b = np.arange(len(ok)) // block
    s = np.bincount(b, weights=np.where(ok, x, 0.0))
    w = np.bincount(b, weights=ok.astype(float))
    mu = s.sum() / w.sum()
    r = s - mu * w
    return mu, np.sqrt(len(w) / (len(w) - 1) * np.sum(r ** 2)) / w.sum()


@pytest.mark.parametrize("fam", ("GEN", "CTRL"))
def test_acceptance_matches_gen_stats(fresh, fam):
    ref = json.loads(GEN_STATS.read_text())["stats"][fam]["acceptance"]
    o, ok = fresh[fam]
    rate = ok.mean()
    blocks = ok[: len(ok) // BLOCK * BLOCK].reshape(-1, BLOCK).mean(1)
    se = blocks.std(ddof=1) / np.sqrt(len(blocks))
    assert abs(rate - ref) < 4 * se, (rate, ref, se)
    assert abs(rate - ref) / ref < 0.25


def test_sample_accepted():
    o, st = ts.sample_accepted("GEN", 40, np.random.default_rng(5))
    assert st["kept"] == 40 and st["n_pass"] >= 40 and len(o["s12sq"]) == 40
    assert ts.accept(o).all()
    assert abs(st["acceptance"] - st["n_pass"] / st["n_raw"]) < 1e-15
    o2, _ = ts.sample_accepted("GEN", 10, np.random.default_rng(5), ordering=None)
    assert len(o2["s12sq"]) == 10


@pytest.mark.skipif(not DATA_025.exists(), reason="artifacts/025/data.npz is local only")
@pytest.mark.parametrize("fam", ("GEN", "CTRL"))
def test_marginals_match_exp025_data(fresh, fam):
    """Each of the 16 flow features: fresh-seed mean within 4.5 cluster-robust SE of exp 025's
    30k accepted rows (whose own SE is negligible next to it), and the PMNS targets pass a KS test."""
    from scipy import stats

    D = np.load(DATA_025)
    sel = D["fam"] == list(D["families"]).index(fam)
    R = np.concatenate([D["ctx_ckm"][sel], D["ctx_m"][sel], D["y"][sel]], 1)
    o, ok = fresh[fam]
    X = np.concatenate(ts.features_025(o), 1)
    for j in range(16):
        mu, se = _cluster_mean(np.where(ok, X[:, j], 0.0), ok)
        assert abs(mu - R[:, j].mean()) < 4.5 * se, (j, mu, R[:, j].mean(), se)
    for j in range(11, 16):
        assert stats.ks_2samp(X[ok, j], R[:, j]).pvalue > 1e-3, j


# ---------------------------------------------------------------- adapters


@pytest.mark.parametrize("fam", ("F1", "F2"))
def test_exp025_adapter_light_families(fam):
    g = ts._exp025_module()
    a = ts.sample(fam, 30, np.random.default_rng([33, 2]))
    raw = g.GENS[fam](np.random.default_rng([33, 2]), 30)
    assert set(a) == set(ts.OBS_KEYS)
    assert np.array_equal(a["s12sq"], raw["s12sq"]) and np.allclose(a["dcp_deg"], np.degrees(raw["dcp"]))
    assert np.all(np.isnan(a["mr_b_tau"]))
    assert ts.accept(a).sum() > 0

import numpy as np
import pytest
from scipy import stats

from src.haar import (
    fake_universe_pair,
    haar_angle_uniforms,
    haar_special_unitary,
    haar_unitary,
    sample_haar_angles,
    shifted_data_universe,
    split_normal,
)
from src.mixing import standard_matrix, unitarity_defect

N = 100_000
KS_ALPHA = 1e-3  # fixed seeds, so this is a deterministic check, not a flaky one


@pytest.fixture(scope="module")
def haar_batch():
    return haar_unitary(np.random.default_rng(12345), N)


def test_shapes_and_unitarity(haar_batch):
    assert haar_unitary(0).shape == (3, 3)
    assert haar_unitary(0, 5).shape == (5, 3, 3)
    assert haar_batch.shape == (N, 3, 3)
    assert unitarity_defect(haar_batch).max() < 1e-13


def test_special_unitary():
    U = haar_special_unitary(1, 10000)
    assert unitarity_defect(U).max() < 1e-13
    assert np.max(np.abs(np.linalg.det(U) - 1)) < 1e-12


@pytest.mark.parametrize("idx,name", [(0, "sin2_t12"), (1, "cos4_t13"), (2, "sin2_t23"), (3, "delta")])
def test_haar_angle_marginals_uniform(haar_batch, idx, name):
    u = haar_angle_uniforms(haar_batch)[idx]
    p = stats.kstest(u, "uniform").pvalue
    assert p > KS_ALPHA, (name, p)


def test_naive_qr_is_not_haar():
    # sanity check that the KS test has power: QR without the phase fix is not Haar
    rng = np.random.default_rng(99)
    Z = (rng.standard_normal((N, 3, 3)) + 1j * rng.standard_normal((N, 3, 3))) / np.sqrt(2)
    Q, _ = np.linalg.qr(Z)
    # wrong-measure alternative: sin^2 t13 uniform instead of cos^4 t13
    fake = standard_matrix(np.arcsin(np.sqrt(rng.random(N))), np.arcsin(np.sqrt(rng.random(N))),
                           np.arcsin(np.sqrt(rng.random(N))), 2 * np.pi * rng.random(N))
    assert stats.kstest(haar_angle_uniforms(fake)[1], "uniform").pvalue < 1e-10
    # QR without phase fix: entries of Q's e-row still fine in modulus (column phases only),
    # so angle marginals are unaffected; check instead the phase of a diagonal entry,
    # which is Haar-uniform only with the fix.
    ph_naive = np.angle(Q[:, 0, 0])
    ph_haar = np.angle(haar_unitary(rng, N)[:, 0, 0])
    assert stats.kstest(ph_haar / (2 * np.pi) + 0.5, "uniform").pvalue > KS_ALPHA
    assert stats.kstest(ph_naive / (2 * np.pi) + 0.5, "uniform").pvalue < 1e-10


def test_angle_correlations_small(haar_batch):
    u = np.array(haar_angle_uniforms(haar_batch))
    c = np.corrcoef(u)
    assert np.max(np.abs(c - np.eye(4))) < 5 * 1 / np.sqrt(N) * 2


def test_direct_angle_sampler_matches():
    t = sample_haar_angles(3, N)
    U = standard_matrix(*t)
    for i, u in enumerate(haar_angle_uniforms(U)):
        assert stats.kstest(u, "uniform").pvalue > KS_ALPHA, i


def test_fake_universe_pair_independent():
    rng = np.random.default_rng(5)
    a, b = fake_universe_pair(rng)
    assert a.shape == b.shape == (3, 3)
    assert not np.allclose(a, b)
    A, B = fake_universe_pair(np.random.default_rng(6), 20000)
    ua, ub = haar_angle_uniforms(A)[0], haar_angle_uniforms(B)[0]
    assert abs(np.corrcoef(ua, ub)[0, 1]) < 0.03
    # same seed reproduces
    a2, b2 = fake_universe_pair(np.random.default_rng(5))
    assert np.allclose(a, a2) and np.allclose(b, b2)


def test_split_normal():
    rng = np.random.default_rng(7)
    x = split_normal(rng, 1.0, 0.2, 0.5, size=200_000)
    assert abs(np.median(x) - 1.0) < 0.005
    assert abs(np.mean(x > 1.0) - 0.5) < 0.005
    hi, lo = x[x > 1] - 1, 1 - x[x < 1]
    assert abs(np.sqrt(np.mean(hi**2)) - 0.2) < 0.005
    assert abs(np.sqrt(np.mean(lo**2)) - 0.5) < 0.01


def test_shifted_data_universe():
    ckm = {"Vus": {"value": 0.225, "err_plus": 0.001, "err_minus": 0.001, "scale": "low"},
           "J": {"value": 3.0e-5, "err_plus": 0.1e-5, "err_minus": -0.2e-5}}
    pmns = {"s12sq": {"value": 0.30, "err_plus": 0.012, "err_minus": 0.011},
            "limit": {"value": None, "err_plus": None, "err_minus": None}}
    c2, p2 = shifted_data_universe(np.random.default_rng(1), ckm, pmns)
    assert c2["Vus"]["value"] != ckm["Vus"]["value"]
    assert c2["Vus"]["value_measured"] == 0.225
    assert c2["Vus"]["scale"] == "low"
    assert p2["limit"]["value"] is None
    assert ckm["Vus"]["value"] == 0.225  # inputs untouched
    only = shifted_data_universe(np.random.default_rng(1), ckm, keys=["J"])
    assert only["Vus"]["value"] == 0.225 and only["J"]["value"] != 3.0e-5
    vals = [shifted_data_universe(np.random.default_rng(s), ckm)["Vus"]["value"] for s in range(2000)]
    assert abs(np.std(vals) - 0.001) < 1e-4

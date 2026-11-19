"""Tests for modforms.py: run with  .venv/Scripts/python -m pytest -q experiments/20261118-017-modular-search"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import modforms as mf  # noqa: E402

CASES = [(N, w) for N in mf.LEVELS for w in mf.WEIGHTS]


def _pts(seed, n=12):
    return mf._rand_tau(np.random.default_rng(seed), n)


@pytest.mark.parametrize("N,w", CASES)
def test_dimension(N, w):
    assert mf.space(N, w).D == mf.dim_formula(N, w)


@pytest.mark.parametrize("N,w", CASES)
def test_S_and_T_transformation(N, w):
    V = mf.space(N, w)
    t = _pts(100 + 10 * N + w)
    f = V(t)
    fS = V(-1 / t) / (t ** w)[:, None]
    assert np.abs(fS - f @ V.rhoS.T).max() / np.abs(f).max() < 1e-9
    fT = V(t + 1)
    assert np.abs(fT - f @ V.rhoT.T).max() / np.abs(f).max() < 1e-11


@pytest.mark.parametrize("N,w", CASES)
def test_rho_presentation(N, w):
    V = mf.space(N, w)
    I = np.eye(V.D)
    S, T = V.rhoS, V.rhoT
    assert np.abs(S @ S - I).max() < 1e-8
    assert np.abs(np.linalg.matrix_power(S @ T, 3) - I).max() < 1e-8
    assert np.abs(np.linalg.matrix_power(T, N) - I).max() < 1e-8


@pytest.mark.parametrize("N", mf.LEVELS)
def test_irreps(N):
    words = mf.group_words(N)
    assert len(words) == mf.GROUP_ORDER[N]
    for name, (S, T) in mf.IRREPS[N].items():
        d = S.shape[0]
        I = np.eye(d)
        assert np.abs(S @ S - I).max() < 1e-12
        assert np.abs(np.linalg.matrix_power(S @ T, 3) - I).max() < 1e-12
        assert np.abs(np.linalg.matrix_power(T, N) - I).max() < 1e-12
        assert np.abs(S - S.T).max() < 1e-12 and np.abs(T - T.T).max() < 1e-12  # symmetric basis (gCP)
    ch = mf.characters(N)
    assert sum(abs(c[0]) ** 2 for c in ch.values()) == pytest.approx(mf.GROUP_ORDER[N])
    for a, b in itertools.combinations(ch, 2):  # orthogonality
        assert abs((ch[a] * np.conj(ch[b])).sum()) < 1e-8


# irrep content per weight: Feruglio 1706.08749 (A4), Penedo-Petcov 1806.11040 and
# Novichkov et al. 1811.04933 (S4), Novichkov et al. 1812.02158 / Ding et al. 1903.12588 (A5)
LITERATURE = {
    (3, 2): {"3": 1}, (3, 4): {"1": 1, "1p": 1, "3": 1}, (3, 6): {"1": 1, "3": 2},
    (3, 8): {"1": 1, "1p": 1, "1pp": 1, "3": 2},
    (4, 2): {"2": 1, "3p": 1}, (4, 4): {"1": 1, "2": 1, "3": 1, "3p": 1},
    (4, 6): {"1": 1, "1p": 1, "2": 1, "3": 1, "3p": 2}, (4, 8): {"1": 1, "2": 2, "3": 2, "3p": 2},
    (5, 2): {"3": 1, "3p": 1, "5": 1}, (5, 4): {"1": 1, "3": 1, "3p": 1, "4": 1, "5": 2},
    (5, 6): {"1": 1, "3": 2, "3p": 2, "4": 2, "5": 2},
    (5, 8): {"1": 1, "3": 2, "3p": 2, "4": 2, "5": 4},
}


@pytest.mark.parametrize("N,w", CASES)
def test_irrep_content(N, w):
    V = mf.space(N, w)
    dec = {k: int(round(v.real)) for k, v in mf.decompose(N, V.rhoS, V.rhoT).items() if abs(v) > 1e-6}
    assert dec == LITERATURE[(N, w)]


def test_A4_weight2_matches_feruglio():
    """Feruglio 1706.08749 Eq. (3.?) eta-quotient form, written with E2 (eta'/eta = i pi E2 / 12):
    Y1 = -(1/24)[E2(t/3) + E2((t+1)/3) + E2((t+2)/3) - 27 E2(3t)],
    Y2 = (1/12)[E2(t/3) + w^2 E2((t+1)/3) + w E2((t+2)/3)], Y3 = same with w <-> w^2,
    and the q-expansions Y1 = 1 + 12q + 36q^2 + 12q^3, Y2 = -6 q^{1/3}(1 + 7q + 8q^2),
    Y3 = -18 q^{2/3}(1 + 2q + 5q^2)."""
    Y = mf.multiplet(3, 2, "3")
    assert Y.shape[0] == 1
    c = Y[0]
    assert np.allclose(c[0, [0, 3, 6, 9]], [1, 12, 36, 12])
    assert np.allclose(c[1, [1, 4, 7]], [-6, -42, -48])
    assert np.allclose(c[2, [2, 5, 8]], [-18, -36, -90])

    def E2(t, nmax=400):
        n = np.arange(1, nmax)
        sig = np.array([sum(d for d in range(1, k + 1) if k % d == 0) for k in n])
        return 1 - 24 * (sig[None, :] * np.exp(2j * np.pi * np.outer(t, n))).sum(1)

    w = np.exp(2j * np.pi / 3)
    t = np.array([0.1 + 1.0j, -0.3 + 0.9j, 0.45 + 1.3j])
    e = [E2((t + j) / 3) for j in range(3)]
    Y1 = -(e[0] + e[1] + e[2] - 27 * E2(3 * t)) / 24
    Y2 = (e[0] + w ** 2 * e[1] + w * e[2]) / 12
    Y3 = (e[0] + w * e[1] + w ** 2 * e[2]) / 12
    ours = mf.eval_series(c, t, 3)
    assert np.allclose(ours, np.stack([Y1, Y2, Y3], 1), atol=1e-10)


@pytest.mark.parametrize("N,w", CASES)
def test_multiplets_transform(N, w):
    for name, (S, T) in mf.IRREPS[N].items():
        Y = mf.multiplet(N, w, name)
        t = _pts(7 * w + N, 6)
        for c in Y:
            y = mf.eval_series(c, t, N)
            yS = mf.eval_series(c, -1 / t, N) / (t ** w)[:, None]
            assert np.abs(yS - y @ S.T).max() < 1e-8 * np.abs(y).max()
            assert np.abs(mf.eval_series(c, t + 1, N) - y @ T.T).max() < 1e-10 * np.abs(y).max()


def _row_cases():
    for N in mf.LEVELS:
        for w in mf.WEIGHTS:
            for r in mf.SINGLETS[N]:
                for q in mf.TRIPLETS[N]:
                    yield N, w, r, q


@pytest.mark.parametrize("N,w,r,q", list(_row_cases()))
def test_yukawa_rows_invariant(N, w, r, q):
    """Row R_j(tau) of f^c(r) Q(q) Y^(w): invariance <=> R(-1/tau) = tau^w a_S^-1 (rho_Q(S)^T)^-1 R(tau)."""
    R = mf.yukawa_row_structures(N, w, r, q)
    (aS, aT), (QS, QT) = [mf.IRREPS[N][r][0][0, 0], mf.IRREPS[N][r][1][0, 0]], mf.IRREPS[N][q]
    t = _pts(3 * w + N, 6)
    for c in R:
        v = mf.eval_series(c, t, N)
        vS = mf.eval_series(c, -1 / t, N) / (t ** w)[:, None]
        assert np.abs(vS - v @ (np.linalg.inv(QS.T) / aS).T).max() < 1e-8 * np.abs(v).max()
        vT = mf.eval_series(c, t + 1, N)
        assert np.abs(vT - v @ (np.linalg.inv(QT.T) / aT).T).max() < 1e-10 * np.abs(v).max()
        assert np.isrealobj(c)


@pytest.mark.parametrize("N,w,q", [(N, w, q) for N in mf.LEVELS for w in mf.WEIGHTS for q in mf.TRIPLETS[N]])
def test_weinberg_invariant(N, w, q):
    Ms = mf.weinberg_structures(N, w, q)
    assert Ms.shape[0] >= 1
    QS, QT = mf.IRREPS[N][q]
    t = _pts(5 * w + N, 5)
    for c in Ms:
        m = mf.eval_series(c, t, N)  # (B, 3, 3)
        assert np.abs(m - np.swapaxes(m, 1, 2)).max() < 1e-12 * np.abs(m).max()
        mS = mf.eval_series(c, -1 / t, N) / (t ** w)[:, None, None]
        iS = np.linalg.inv(QS)
        pred = np.einsum("ij,bjk,kl->bil", iS.T, m, iS)
        assert np.abs(mS - pred).max() < 1e-8 * np.abs(m).max()


def test_coupling_counts_A4():
    """Known A4 counts: weight-2 Weinberg with L ~ 3 has one coupling (Feruglio 1706.08749 Eq. (4.?)
    model: m_nu fixed up to scale); weight-4 has 3 (1, 1', 3_S)."""
    assert mf.weinberg_structures(3, 2, "3").shape[0] == 1
    assert mf.weinberg_structures(3, 4, "3").shape[0] == 3
    for r in ("1", "1p", "1pp"):
        assert mf.yukawa_row_structures(3, 2, r, "3").shape[0] == 1
        assert mf.yukawa_row_structures(3, 6, r, "3").shape[0] == 2

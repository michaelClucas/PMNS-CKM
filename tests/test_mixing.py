import numpy as np
import pytest

from src.mixing import (
    extract_angles,
    jarlskog,
    rephase_to_standard,
    standard_matrix,
    standard_to_wolfenstein,
    unitarity_defect,
    wolfenstein_matrix,
    wolfenstein_to_standard,
)

HALF_PI = np.pi / 2
EDGE = [0.0, 1e-9, 1e-5, 0.3, HALF_PI - 1e-9, HALF_PI]


def random_angles(rng, n):
    return (
        rng.uniform(0, HALF_PI, n),
        rng.uniform(0, HALF_PI, n),
        rng.uniform(0, HALF_PI, n),
        rng.uniform(0, 2 * np.pi, n),
    )


def random_phases(rng, n):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (n, 3)))[..., None] * np.eye(3)


def dphase(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def test_standard_matrix_unitary_and_shape():
    rng = np.random.default_rng(1)
    U = standard_matrix(*random_angles(rng, 1000))
    assert U.shape == (1000, 3, 3)
    assert unitarity_defect(U).max() < 1e-14
    assert standard_matrix(0.1, 0.2, 0.3, 0.4).shape == (3, 3)


def test_jarlskog_formula():
    rng = np.random.default_rng(2)
    t12, t13, t23, d = random_angles(rng, 1000)
    J = jarlskog(standard_matrix(t12, t13, t23, d))
    Jf = (np.sin(t12) * np.cos(t12) * np.sin(t23) * np.cos(t23)
          * np.cos(t13) ** 2 * np.sin(t13) * np.sin(d))
    assert np.max(np.abs(J - Jf)) < 1e-15
    # PDG's CKM form Im(V_us V_cb V_ub* V_cs*) agrees
    V = standard_matrix(t12, t13, t23, d)
    J2 = np.imag(V[:, 0, 1] * V[:, 1, 2] * np.conj(V[:, 0, 2]) * np.conj(V[:, 1, 1]))
    assert np.max(np.abs(J - J2)) < 1e-15


def test_round_trip_random():
    rng = np.random.default_rng(3)
    t = random_angles(rng, 20000)
    e = extract_angles(standard_matrix(*t))
    for a, b in zip(e[:3], t[:3]):
        assert np.max(np.abs(a - b)) < 1e-12
    # delta precision degrades as 1e-16 / (s12 c12 s23 c23 s13); keep generic samples
    D = np.sin(t[0]) * np.cos(t[0]) * np.sin(t[2]) * np.cos(t[2]) * np.sin(t[1])
    m = D > 1e-4
    assert np.max(dphase(e[3][m], t[3][m])) < 1e-10
    assert np.all((e[3] >= 0) & (e[3] < 2 * np.pi))


@pytest.mark.parametrize("t12", EDGE)
@pytest.mark.parametrize("t13", EDGE)
@pytest.mark.parametrize("t23", EDGE)
def test_round_trip_edges(t12, t13, t23):
    d = 1.234
    U = standard_matrix(t12, t13, t23, d)
    e12, e13, e23, ed = extract_angles(U)
    assert abs(e13 - t13) < 1e-12
    if np.cos(t13) > 1e-12:
        assert abs(e12 - t12) < 1e-12
        assert abs(e23 - t23) < 1e-12
    # whatever the convention in degenerate cases, the extracted angles must reproduce U
    # up to rephasing: compare all |U_ij| and J
    U2 = standard_matrix(e12, e13, e23, ed)
    assert np.max(np.abs(np.abs(U2) - np.abs(U))) < 1e-8
    assert abs(jarlskog(U2) - jarlskog(U)) < 1e-12
    # delta is physical only if all of s12 c12 s23 c23 s13 c13 are non-zero
    D = np.sin(t12) * np.cos(t12) * np.sin(t23) * np.cos(t23) * np.sin(t13) * np.cos(t13)
    if D > 1e-6:
        assert dphase(ed, d) < 1e-9


def test_rephasing_invariance():
    rng = np.random.default_rng(4)
    n = 20000
    t = random_angles(rng, n)
    U = standard_matrix(*t)
    V = random_phases(rng, n) @ U @ random_phases(rng, n)
    e0, e1 = extract_angles(U), extract_angles(V)
    for a, b in zip(e0[:3], e1[:3]):
        assert np.max(np.abs(a - b)) < 1e-12
    D = np.sin(t[0]) * np.cos(t[0]) * np.sin(t[2]) * np.cos(t[2]) * np.sin(t[1])
    m = D > 1e-4
    assert np.max(dphase(e0[3][m], e1[3][m])) < 1e-10
    assert np.max(np.abs(jarlskog(U) - jarlskog(V))) < 1e-15


def test_overall_phase_and_complex_conjugate():
    U = standard_matrix(0.59, 0.15, 0.85, 3.4)
    t = extract_angles(np.exp(0.7j) * U)
    assert dphase(t[3], 3.4) < 1e-12
    tc = extract_angles(np.conj(U))  # CP conjugate flips delta
    assert dphase(tc[3], 2 * np.pi - 3.4) < 1e-12


def test_rephase_to_standard_generic():
    rng = np.random.default_rng(5)
    n = 5000
    U = standard_matrix(*random_angles(rng, n))
    V = np.exp(1j * rng.uniform(0, 6, n))[:, None, None] * random_phases(rng, n) @ U @ random_phases(rng, n)
    S, P, Q = rephase_to_standard(V)
    assert np.max(np.abs(P @ S @ Q - V)) < 1e-12
    assert np.max(np.abs(P[:, 0, 0] - 1)) < 1e-14
    off = ~np.eye(3, dtype=bool)
    assert np.max(np.abs(P[:, off])) == 0 and np.max(np.abs(Q[:, off])) == 0
    assert np.max(np.abs(np.abs(np.diagonal(Q, axis1=1, axis2=2)) - 1)) < 1e-14
    # S really is standard form: e1, e2, mu3, tau3 real non-negative
    for i, j in [(0, 0), (0, 1), (1, 2), (2, 2)]:
        assert np.max(np.abs(S[:, i, j].imag)) < 1e-15 and np.min(S[:, i, j].real) >= 0


@pytest.mark.parametrize("t13", EDGE)
@pytest.mark.parametrize("t12", [0.0, 1e-9, 0.5, HALF_PI])
@pytest.mark.parametrize("t23", [0.0, 1e-7, 0.7, HALF_PI])
def test_rephase_to_standard_edges(t12, t13, t23):
    rng = np.random.default_rng(6)
    U = standard_matrix(t12, t13, t23, 1.0)
    V = random_phases(rng, 1)[0] @ U @ random_phases(rng, 1)[0]
    S, P, Q = rephase_to_standard(V)
    # near-degenerate corners (D < 1e-15) lose delta; error then bounded by ~1e-9
    assert np.max(np.abs(P @ S @ Q - V)) < 1e-8


def test_haar_like_batch_round_trip():
    # generic U(3) (not built from standard_matrix) reconstructs from its rephasing
    rng = np.random.default_rng(7)
    Z = rng.standard_normal((3000, 3, 3)) + 1j * rng.standard_normal((3000, 3, 3))
    V, _ = np.linalg.qr(Z)
    S, P, Q = rephase_to_standard(V)
    assert np.max(np.abs(P @ S @ Q - V)) < 1e-12


# ------------------------------------------------------------- Wolfenstein


@pytest.mark.parametrize("pars", [
    (0.22500, 0.826, 0.159, 0.348),
    (0.3, 1.2, -0.5, 0.9),
    (0.1, 0.5, 0.0, 0.0),
    (0.22, 0.8, 0.2, -0.3),
])
def test_wolfenstein_exact(pars):
    lam, A, rb, eb = pars
    V = wolfenstein_matrix(*pars)
    assert unitarity_defect(V) < 1e-15
    assert abs(abs(V[0, 1]) / np.hypot(abs(V[0, 0]), abs(V[0, 1])) - lam) < 1e-15
    t12, t13, t23, d = wolfenstein_to_standard(*pars)
    assert abs(np.sin(t12) - lam) < 1e-15
    assert abs(np.sin(t23) - A * lam**2) < 1e-15
    # rhobar + i etabar = -V_ud V_ub* / (V_cd V_cb*), exactly
    z = -V[0, 0] * np.conj(V[0, 2]) / (V[1, 0] * np.conj(V[1, 2]))
    assert abs(z - (rb + 1j * eb)) < 1e-13
    back = standard_to_wolfenstein(V)
    assert np.allclose(back, pars, atol=1e-13, rtol=0)


def test_wolfenstein_leading_order():
    lam, A, rb, eb = 0.225, 0.83, 0.16, 0.35
    V = wolfenstein_matrix(lam, A, rb, eb)
    # |V_ub| ~ A lam^3 |rho - i eta| at leading order, with rho ~ rhobar/(1 - lam^2/2)
    rho, eta = rb / (1 - lam**2 / 2), eb / (1 - lam**2 / 2)
    assert abs(abs(V[0, 2]) / (A * lam**3 * np.hypot(rho, eta)) - 1) < 1e-3
    assert abs(jarlskog(V) / (A**2 * lam**6 * eta) - 1) < 0.05


def test_wolfenstein_vectorized():
    rng = np.random.default_rng(8)
    n = 1000
    pars = (rng.uniform(0.1, 0.3, n), rng.uniform(0.5, 1.2, n),
            rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n))
    V = wolfenstein_matrix(*pars)
    assert V.shape == (n, 3, 3)
    assert unitarity_defect(V).max() < 1e-14
    back = standard_to_wolfenstein(V)
    for a, b in zip(back, pars):
        assert np.max(np.abs(a - b)) < 1e-12

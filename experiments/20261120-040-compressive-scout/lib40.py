"""Exp 040 model builders (matrices as written in the papers; equation numbers in models.md).

All matrices are LR (rows = left-handed doublets), the src.yukawa convention, unless noted.
"""
import numpy as np

from src import yukawa as yk


def sv(M):
    """Singular values ascending."""
    return np.linalg.svd(np.asarray(M, complex), compute_uv=False)[::-1]


def ratios(M):
    s = sv(M)
    return s[0] / s[1], s[1] / s[2]


# ---------------------------------------------------------------- Albright-Barr lopsided SO(10)
# hep-ph/0502161 Eq. (2) (charged sector, from Albright-Barr hep-ph/0002155, hep-ph/0003251),
# Eq. (4) M_R, Eq. (3) benchmark. Rows = left-handed fields, columns = left-handed conjugates.

AB_BENCH = dict(sigma=1.83, eps=0.147, delta=0.00946, deltap=0.00827, phi=np.radians(119.4), eta=6e-6)


def ab_D(sigma, eps, delta, deltap, phi, **_):
    dp = deltap * np.exp(1j * phi)
    return np.array([[0, delta, dp], [delta, 0, -eps / 3], [dp, sigma + eps / 3, 1]], complex)


def ab_L(sigma, eps, delta, deltap, phi, **_):
    dp = deltap * np.exp(1j * phi)
    return np.array([[0, delta, dp], [delta, 0, sigma + eps], [dp, -eps, 1]], complex)


def ab_U(eps, eta, dN=0.0, dNp=0.0, **_):
    return np.array([[eta, dN, dNp], [dN, 0, -eps / 3], [dNp, eps / 3, 1]], complex)


def ab_N(eps, eta, dN=0.0, dNp=0.0, **_):
    return np.array([[eta, dN, dNp], [dN, 0, eps], [dNp, -eps, 1]], complex)


def ab_MR(eps, eta, a, b, c, **_):
    return np.array([[c * c * eta * eta, -b * eps * eta, a * eta], [-b * eps * eta, eps * eps, -eps],
                     [a * eta, -eps, 1]], complex)


def ab_obs(p, details=False):
    """Paper: M_nu = -M_N M_R^-1 M_N^T (Eq. 5). Passed to src as its complex conjugate, the map that
    reproduces the UTZ paper's PMNS from the same kind of formula (reproduce_utz.py)."""
    N, M = ab_N(**p), ab_MR(**p)
    m = (-N @ np.linalg.solve(M, N.T)).conj()
    return yk.observables(ab_U(**p), ab_D(**p), ab_L(**p), m_nu=0.5 * (m + m.T), details=details)


# ---------------------------------------------------------------- UTZ (1710.01741)
# Eq. (V.1) with alpha = beta = 0 (vacuum-alignment phases; the paper keeps two physical phases
# gamma, delta at LO). M = a e^{i g} A + b e^{i d} B + E33, r-factor (II.1, II.3) on the
# Sigma-dependent b term: b_d = b_e * (r_d / r_e) = -b_e / 3 (kappa = 0 case of Sec. II.A).
# Higher order (II.8), the Sigma S operator: d' e^{i psi} (theta23 theta3 + theta3 theta23)
# = d' e^{i psi} [[0,0,0],[0,0,1],[0,1,2]], also carrying the r factor (it contains Sigma).

UTZ_A = np.array([[0, 1, 1], [1, 2, 0], [1, 0, -2]], float)
UTZ_B = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], float)
UTZ_D = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 2]], float)
UTZ_E = np.diag([0.0, 0.0, 1.0])


def utz_M(a, b, g, d, dp=0.0, psi=0.0):
    return a * np.exp(1j * g) * UTZ_A + b * np.exp(1j * d) * UTZ_B + dp * np.exp(1j * psi) * UTZ_D + UTZ_E


def utz_de(a_e, b_e, g, d, dp_d=0.0, psi=0.0):
    """(M_d, M_e) from the shared charged-lepton parameters (Table III convention)."""
    Me = utz_M(a_e, b_e, g, d, -3.0 * dp_d, psi)
    Md = utz_M(a_e, -b_e / 3.0, g, d, dp_d, psi)
    return Md, Me


# ---------------------------------------------------------------- Babu-Pati-Rastogi (BPW extended)
# hep-ph/0502152 Eq. (1) (rows = left-handed fields, columns = right-handed: LR), Eq. (4) M_R,
# Eq. (6) fit A. (eps, eps') -> -3 (eps, eps') from quarks to leptons; zeta22 has no Clebsch.

BPR_FITA = dict(sigma=0.109 - 0.012j, eta=0.122 - 0.0464j, eps=-0.103, etap=2.4e-3,
                epsp=2.35e-4 * np.exp(1j * np.radians(69)), z22d=9.8e-3 * np.exp(-1j * np.radians(149)),
                z22u=0.0)


def bpr_U(sigma, eps, epsp, z22u, **_):
    return np.array([[0, epsp, 0], [-epsp, z22u, sigma + eps], [0, sigma - eps, 1]], complex)


def bpr_D(eta, eps, etap, epsp, z22d, **_):
    return np.array([[0, etap + epsp, 0], [etap - epsp, z22d, eta + eps], [0, eta - eps, 1]], complex)


def bpr_N(sigma, eps, epsp, z22u, **_):
    return np.array([[0, -3 * epsp, 0], [3 * epsp, z22u, sigma - 3 * eps], [0, sigma + 3 * eps, 1]], complex)


def bpr_L(eta, eps, etap, epsp, z22d, **_):
    return np.array([[0, etap - 3 * epsp, 0], [etap + 3 * epsp, z22d, eta - 3 * eps], [0, eta + 3 * eps, 1]],
                    complex)


def bpr_MR(x, y, z):
    return np.array([[x, 0, z], [0, 0, y], [z, y, 1]], complex)

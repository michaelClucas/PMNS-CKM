"""Exp 042 model builders, matrices as printed in the papers (equation numbers in models.md).

All Yukawas are returned LR (rows = left-handed doublets), the src.yukawa convention.

AKS  Antusch-King-Spinrath, arXiv:1301.6764, A4 x SU(5) with CSD2.
     Eq. (3.8) Y_d, Y_e (LR in the paper, App. A Eq. A.1), Eq. (3.10) Y_u, Eq. (2.1) M_nu.
CM   Chen-Huang-Mahanthappa-Wijangco, arXiv:1307.7711, SUSY SU(5) x T' with eta''.
     Eq. (9): -L = U_R M_u Q_L + D_R M_d Q_L + E_R M_e l_L, so M is RL and Y_LR = M^T.
     Eq. (13), (14) normalised M_u, M_d, M_e^T; Eq. (23) M_nu = -M_D M_RR^-1 M_D^T with M_D, M_RR
     from p. 7-8.
"""
import numpy as np

from src import yukawa as yk

OMB = np.exp(4j * np.pi / 3)  # omega-bar of AKS Eq. (3.8)

# ---------------------------------------------------------------- AKS (1301.6764)
# Table 4 (fit at M_SUSY = 750 GeV, tan beta 1.49; GUT-scale MSSM Yukawas).
AKS_T4 = dict(au=-3.01e-5, bu=-2.66e-4, cu=-2.57e-3, du=3.09e-2, eu=2.05,
              e2=-3.57e-5, e102=3.17e-5, e23=1.62e-4, e3=1.24e-2, tan_beta=1.49, m_susy=750.0)


def aks_Yu(au, bu, cu, du, eu, **_):
    return np.array([[au, bu, 0], [bu, cu, du], [0, du, eu]], complex)


def aks_Yd(e2, e102, e23, e3, **_):
    return np.array([[0, 1j * e2, 0], [OMB * e102, e23, 2 * OMB * e102 - e23], [0, 0, e3]], complex)


def aks_Ye(e2, e102, e23, e3, **_):
    return np.array([[0, -1.5 * OMB * e102, 0], [-1.5j * e2, 4.5 * e23, 0],
                     [0, -3 * OMB * e102 - 4.5 * e23, e3]], complex)


def aks_Mnu(eps, alpha=np.pi / 3, **_):
    """Eq. (2.1) with m_a = 1: M_nu = A0 + eps e^{i alpha} B0."""
    A0 = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]], float)
    B0 = np.array([[1, 0, 2], [0, 0, 0], [2, 0, 4]], float)
    return A0 + eps * np.exp(1j * alpha) * B0


def aks_obs(p, conj=True, details=False):
    """App. A: L_nu = -1/2 Lbar M_nu L^c, i.e. the src.yukawa m_nu is conj(M_nu) (conj=True; the choice
    reproduces the paper's delta ~ 130 deg, reproduce_aks.py)."""
    m = aks_Mnu(**p)
    m = m.conj() if conj else m
    return yk.observables(aks_Yu(**p), aks_Yd(**p), aks_Ye(**p), m_nu=m, ordering="NO", details=details)


# ---------------------------------------------------------------- CM (1307.7711)
CM_EQ16 = dict(b=0.00304, c=-0.0172, k=-0.0266, h=0.00426, g=1.45e-5)
CM_EQ58 = dict(xi=0.185, eta=-0.865, etapp=0.200)


def cm_Mu(g, h, k, **_):
    """Eq. (13), M_u / (y_t v_u) (RL; symmetric)."""
    return np.array([[1j * g, (1 - 1j) / 2 * g, 0], [(1 - 1j) / 2 * g, g + h, k], [0, k, 1]], complex)


def cm_X(b, c):
    """Eq. (14): M_d = X(c), M_e^T = X(-3c), divided by y_b v_d phi0 zeta0."""
    return np.array([[0, (1 + 1j) * b, 0], [-(1 - 1j) * b, c, 0], [b, b, 1]], complex)


def cm_Md(b, c, **_):
    return cm_X(b, c)


def cm_Me(b, c, **_):
    return cm_X(b, -3 * c).T


def cm_MD(xi, eta, etapp, **_):
    """Dirac neutrino matrix, p. 8 (lambda_i = 1), divided by zeta0 zeta0' v_u."""
    return np.array([[2 * xi + eta, -xi, -xi + etapp], [-xi, 2 * xi + etapp, -xi + eta],
                     [-xi + etapp, -xi + eta, 2 * xi]], float)


CM_MRR = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)


def cm_Mnu(xi, eta, etapp, **_):
    """Eq. (23)."""
    MD = cm_MD(xi, eta, etapp)
    return -MD @ np.linalg.solve(CM_MRR, MD.T)


def cm_obs(p, rl_transpose=True, conj=False, ordering="NO", details=False):
    """Y_LR = M^T (Eq. 9, RL mass matrices). M_nu is real, so conj only matters through the charged-lepton
    phases; the convention is fixed by reproducing Eq. (43) and delta in Eq. (65) (reproduce_cm.py)."""
    f = (lambda M: M.T) if rl_transpose else (lambda M: M)
    Yu, Yd, Ye = f(cm_Mu(**p)), f(cm_Md(**p)), f(cm_Me(**p))
    if conj:
        Yu, Yd, Ye = Yu.conj(), Yd.conj(), Ye.conj()
    m = cm_Mnu(**p).astype(complex)
    return yk.observables(Yu, Yd, Ye, m_nu=m, ordering=ordering, details=details)

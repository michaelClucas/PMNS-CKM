"""Koide-Nishiura yukawaon + democratic universal seesaw model (arXiv:1510.05370, PRD 92 111301(R)).

Mass matrices, Eqs. (18)-(19), dimensionless (Sec. 4):
    Phi0  = diag(z1, z2, z3),  z_i = sqrt(m_ei) / norm,  sum z_i^2 = 1                    (5)
    X3    = (1/3) ones(3, 3)                                                               (7)
    Y_u   = Phi0 Pbar_u (1 + b_u X3)^-1 Pbar_u Phi0,  Pbar_u = diag(e^{i phi1}, e^{i phi2}, 1)  (18)
    Y_d   = Phi0 (1 + b_d e^{i beta_d} X3)^-1 Phi0                                         (18)
    Y_e   = Y_nu = Phi0 Phi0                                                               (18), (19)
    Y_R   = Phi0 Y_u + Y_u^T Phi0 + xi_R Y_e^T Y_e                                         (19)
    M_nu  = Y_nu Y_R^-1 Y_nu^T                                                             (19)
(1 + b X3)^-1 = 1 - b / (1 + b) X3, since X3^2 = X3.

Convention variants (fixed by reproducing the paper's benchmark, reproduce.py; no data used):
    usym  True:  Y_u as printed with Pbar_u on both sides (symmetric; Y_R then symmetric).
          False: Pbar_u on the left, conj(Pbar_u) on the right (hermitian Y_u, as printed in
          Eq. (18): Pbar_u ... P_u). Y_R = Phi0 Y_u + (Phi0 Y_u)^T is symmetric either way.
    conj  apply complex conjugation to all Yukawas (equivalent to RL vs LR reading), flips CP signs.
    phisign  Pbar_u = diag(e^{phisign i phi1}, e^{phisign i phi2}, 1). Sec. 4 prints phisign = +1,
          Eq. (4) with Pbar = P^* (as stated in 1512.08386 Sec. 3) gives -1; only -1 reproduces the
          published |V_ij| (reproduce.log), so -1 is the default.
    nuconj  conjugate M_nu before the engine (paper Majorana convention = conj of the engine's, as
          in exps 022, 023); needed for the published delta_l sign.
Defaults (usym=False, conj=False, nuconj=True, phisign=-1) reproduce every published number.
Engine: src.yukawa.observables (LR convention, V = U_u^dag U_d, U_PMNS = U_e^dag U_nu). The
engine's m_nu is the Majorana matrix in its own convention; nuconj conjugates M_nu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import yukawa as _yk  # noqa: E402

X3 = np.ones((3, 3)) / 3.0
ONE = np.eye(3)


def inv_democratic(b):
    """(1 + b X3)^-1 for complex b != -1."""
    return ONE - (b / (1.0 + b)) * X3


def phi0_from_ratios(r_e_mu, r_mu_tau):
    """Phi0 = diag(sqrt(m_e), sqrt(m_mu), sqrt(m_tau)), normalised to sum z^2 = 1."""
    m = np.array([r_e_mu * r_mu_tau, r_mu_tau, 1.0])
    z = np.sqrt(m / m.sum())
    return np.diag(z)


def matrices(b_u, b_d, beta_d_deg, phi1_deg, phi2_deg, xi_R, phi0, usym=False, conj=False, nuconj=True, phisign=-1):
    P = np.diag([np.exp(phisign * 1j * np.radians(phi1_deg)), np.exp(phisign * 1j * np.radians(phi2_deg)), 1.0])
    Au = inv_democratic(b_u)
    Yu = phi0 @ P @ Au @ (P if usym else P.conj()) @ phi0
    Yd = phi0 @ inv_democratic(b_d * np.exp(1j * np.radians(beta_d_deg))) @ phi0
    Ye = (phi0 @ phi0).astype(complex)
    Ynu = Ye.copy()
    YR = phi0 @ Yu + Yu.T @ phi0 + xi_R * Ye.T @ Ye
    Mnu = Ynu @ np.linalg.inv(YR) @ Ynu.T
    Mnu = 0.5 * (Mnu + Mnu.T)
    if conj:
        Yu, Yd, Ye, Mnu = Yu.conj(), Yd.conj(), Ye.conj(), Mnu.conj()
    if nuconj:
        Mnu = Mnu.conj()
    return Yu, Yd, Ye, Mnu


def predict(theta, phi0, usym=False, conj=False, nuconj=True, phisign=-1, ordering="auto", details=False):
    """theta: dict b_u, b_d, beta_d, phi1, phi2, xi_R (angles in degrees). Returns src.evaluate keys."""
    Yu, Yd, Ye, Mnu = matrices(theta["b_u"], theta["b_d"], theta["beta_d"], theta["phi1"], theta["phi2"],
                               theta["xi_R"], phi0, usym, conj, nuconj, phisign)
    return _yk.observables(Yu, Yd, Ye, Mnu, ordering=ordering, details=details)

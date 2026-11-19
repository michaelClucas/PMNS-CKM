"""A4 x SU(5) sparse Clebsch textures + CSD3 seesaw (arXiv:1503.03306, eqs. 2.15, 2.23-2.25).

All quantities at the GUT scale. Yukawas in LR convention (Y_ij L_i R_j); left rotations
diagonalise Y Y^dagger. `conj` selects the complex-conjugation convention (fixed by the
pipeline check in reproduce.py, not by fit quality).
"""
import math

import numpy as np

from src.mixing import extract_angles

THETA = math.pi / 3          # Y^d_12 phase (paper)
ETA = 2 * math.pi / 3        # CSD3 seesaw phase (paper, sign chosen a posteriori there)
C11, C22 = 4 / 9, 9 / 2      # Clebsch factors Y^e/Y^d (frozen)
N_CSD = 3
CONJ = False                 # set from reproduce.py result (see README)
NUCONJ = False               # U_nu vs U_nu* (m = U D U^T vs U* D U^dagger); fixed by reproduce.py


def yukawas(u, d, theta=THETA, c11=C11, c22=C22):
    u11, u12, u13, u22, u23, u33 = u
    d11, d12, d22, d33 = d
    Yu = np.array([[1.296e-5 * u11, 2.16e-4 * u12, 3.6e-3 * u13],
                   [2.16e-4 * u12, 3.6e-3 * u22, 6e-2 * u23],
                   [3.6e-3 * u13, 6e-2 * u23, u33]], dtype=complex)
    a, b, c, t = 1.5e-5 * d11, 2e-4 * d12 * np.exp(1j * theta), 6e-4 * d22, 5e-2 * d33
    Yd = np.array([[a, b, 0], [0, c, 0], [0, 0, t]], dtype=complex)
    Ye = np.array([[c11 * a, 0, 0], [b, c22 * c, 0], [0, 0, t]], dtype=complex)
    return Yu, Yd, Ye


def left(Y):
    """Left rotation U (columns = eigenvectors of Y Y^dagger, ascending) and singular values."""
    w, v = np.linalg.eigh(Y @ Y.conj().T)
    return v, np.sqrt(np.clip(w, 0, None))


def mnu(r, eta=ETA, n=N_CSD, ma=1.0):
    pa = np.array([0, 1, 1.0])
    pb = np.array([1, n, n - 2.0])
    return ma * (np.outer(pa, pa) + r * np.exp(1j * eta) * np.outer(pb, pb))


def nu_rot(m):
    """U_nu with m = U* diag(m_i) U^dagger convention: diagonalise m m^dagger (ascending)."""
    w, v = np.linalg.eigh(m @ m.conj().T)
    return v, np.sqrt(np.clip(w, 0, None))


def predict(u, d, r, eta=ETA, theta=THETA, conj=None, nuconj=None, ma=1.0):
    """GUT-scale predictions: CKM angles, mass ratios, PMNS, dm ratio, plus diagnostics."""
    conj = CONJ if conj is None else conj
    nuconj = NUCONJ if nuconj is None else nuconj
    Yu, Yd, Ye = yukawas(u, d, theta)
    m = mnu(r, eta, ma=ma)
    if conj:
        Yu, Yd, Ye, m = Yu.conj(), Yd.conj(), Ye.conj(), m.conj()
    Uu, yu = left(Yu)
    Ud, yd = left(Yd)
    Ue, ye = left(Ye)
    Un, mn = nu_rot(m)
    if nuconj:
        Un = Un.conj()
    V = Uu.conj().T @ Ud
    U = Ue.conj().T @ Un
    q12, q13, q23, qd = extract_angles(V)
    l12, l13, l23, ld = extract_angles(U)
    # charged-lepton left angles (diagnostic): |Ue| in the flavour basis
    e12, e13, e23, _ = extract_angles(Ue)
    return {
        "ckm_s12": float(np.sin(q12)), "ckm_s13": float(np.sin(q13)), "ckm_s23": float(np.sin(q23)),
        "ckm_delta_rad": float(qd),
        "mr_u_c": yu[0] / yu[1], "mr_c_t": yu[1] / yu[2], "mr_d_s": yd[0] / yd[1], "mr_s_b": yd[1] / yd[2],
        "mr_e_mu": ye[0] / ye[1], "mr_mu_tau": ye[1] / ye[2], "mr_b_tau": yd[2] / ye[2],
        "s12sq": float(np.sin(l12) ** 2), "s13sq": float(np.sin(l13) ** 2), "s23sq": float(np.sin(l23) ** 2),
        "dcp_deg": float(np.degrees(ld)),
        "dm_ratio": (mn[1] ** 2 - mn[0] ** 2) / (mn[2] ** 2 - mn[0] ** 2),
        "_dm21": mn[1] ** 2 - mn[0] ** 2, "_dm31": mn[2] ** 2 - mn[0] ** 2, "_m1": mn[0],
        "_y": (yu, yd, ye), "_theta12e": float(e12), "_theta23e": float(e23), "_theta13e": float(e13),
    }


def scored(p):
    """Drop diagnostic keys."""
    return {k: float(v) for k, v in p.items() if not k.startswith("_")}

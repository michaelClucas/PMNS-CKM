"""S4 x SO(10) model of arXiv:1705.01555 (Bjorkeroth, de Anda, King, Perdomo), Eqs. (10)-(17).

All matrices at the GUT scale. Yukawa matrices are sums of fixed rank-1/rank-2 numerical
matrices built from the CSD3 flavon alignments (1,1,3), (0,1,1), (0,0,1):

  Y^u   = y1u e^{i eta} Y11 + y2u Y22 + y3u e^{i eta'} Y33
  m^nu  = mu1 e^{i eta} Y11 + mu2 Y22 + mu3 e^{i eta'} Y33         (seesaw, Eq. 17)
  Y^d   = y12d e^{i eta/2} Y12 + y2d e^{i a_d} Y22 + y3d e^{i b_d} Y33 + yP e^{i g} YP
  Y^e   = y12e e^{i eta/2} Y12 + y2e e^{i a_e} Y22 + y3e e^{i b_e} Y33 + yP e^{i g} YP
  eta = -2 pi / 3 (fixed by the flavon potential); eta', a_d, b_d, a_e, b_e, g free.

Convention: left rotations from Y Y^dag, V_CKM = Uu^dag Ud, light-neutrino matrix in the
charged-lepton basis Ue^T (m^nu)^* Ue, PMNS from its Takagi factorisation. This is the one of
the eight conjugation choices that reproduces the paper's printed delta_q = 69.2 deg AND
theta23_l = 41.96, delta_l = 300.9 deg at its Table 4 point (checked in reproduce.py).
conj=True complex-conjugates every matrix (all CP phases flip; diagnostic only).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mixing import extract_angles  # noqa: E402

Y11 = np.array([[1, 1, 3], [1, 1, 3], [3, 3, 9]], float)
Y22 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], float)
Y33 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], float)
Y12 = np.array([[0, 1, 1], [1, 2, 4], [1, 4, 6]], float)
YP = np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]], float)
ETA = -2 * np.pi / 3

REAL = ("y1u", "y2u", "y3u", "y12d", "y2d", "y3d", "yP", "y12e", "y2e", "y3e", "mu1", "mu2", "mu3")
PHASES = ("etap", "ad", "bd", "ae", "be", "g")

# Table 4 of 1705.01555 (phases in units of pi; mu_i in meV)
PUBLISHED = {"y1u": 3.009e-6, "y2u": 1.491e-3, "y3u": 0.549, "y12d": -1.186e-4, "y2d": 6.980e-4,
             "y3d": 0.137, "yP": 1.243e-4, "y12e": 1.558e-4, "y2e": 2.248e-3, "y3e": 3.318e-2,
             "mu1": 2.413, "mu2": 27.50, "mu3": 2.900,
             "etap": 1.053 * np.pi, "ad": 0.043 * np.pi, "bd": 0.295 * np.pi, "ae": 1.692 * np.pi,
             "be": 1.755 * np.pi, "g": 0.918 * np.pi}


def e(x):
    return np.exp(1j * x)


def matrices(p, eta=ETA):
    Yu = p["y1u"] * e(eta) * Y11 + p["y2u"] * Y22 + p["y3u"] * e(p["etap"]) * Y33
    mnu = p["mu1"] * e(eta) * Y11 + p["mu2"] * Y22 + p["mu3"] * e(p["etap"]) * Y33
    Yd = p["y12d"] * e(eta / 2) * Y12 + p["y2d"] * e(p["ad"]) * Y22 + p["y3d"] * e(p["bd"]) * Y33 \
        + p["yP"] * e(p["g"]) * YP
    Ye = p["y12e"] * e(eta / 2) * Y12 + p["y2e"] * e(p["ae"]) * Y22 + p["y3e"] * e(p["be"]) * Y33 \
        + p["yP"] * e(p["g"]) * YP
    return Yu, Yd, Ye, mnu


def left(Y):
    w, U = np.linalg.eigh(Y @ Y.conj().T)
    return np.sqrt(np.clip(w, 0, None)), U


def takagi_masses_vectors(m):
    """|eigen| masses ascending and the unitary U with U^T m U = diag (up to phases): eigh of m^dag m."""
    w, U = np.linalg.eigh(m.conj().T @ m)
    return np.sqrt(np.clip(w, 0, None)), U


def observables(p, conj=False, eta=ETA):
    Yu, Yd, Ye, mnu = matrices(p, eta)
    if conj:
        Yu, Yd, Ye, mnu = Yu.conj(), Yd.conj(), Ye.conj(), mnu.conj()
    yu, Uu = left(Yu)
    yd, Ud = left(Yd)
    ye, Ue = left(Ye)
    V = Uu.conj().T @ Ud
    q12, q13, q23, qd = extract_angles(V)
    k = Ue.T @ mnu.conj() @ Ue  # paper convention: reproduces its printed delta_l (README)
    m, Un = takagi_masses_vectors(k)
    l12, l13, l23, ld = extract_angles(Un)
    # Majorana phases (deg): rephase so U^T k U is real positive, then PDG invariants
    D = Un.T @ k @ Un
    Uc = Un @ np.diag(np.exp(-0.5j * np.angle(np.diag(D))))
    a21 = np.degrees(2 * np.angle(Uc[0, 1] / Uc[0, 0])) % 360
    a31 = np.degrees(2 * (np.angle(Uc[0, 2] / Uc[0, 0]) + ld)) % 360
    return {
        "yu": yu, "yd": yd, "ye": ye, "mnu_meV": m,
        "ckm": (float(q12), float(q13), float(q23), float(qd)),
        "pmns": (float(l12), float(l13), float(l23), float(ld)),
        "dm21": (m[1] ** 2 - m[0] ** 2) * 1e-6, "dm31": (m[2] ** 2 - m[0] ** 2) * 1e-6,  # eV^2
        "alpha21": float(a21), "alpha31": float(a31),
    }


def pred_dimless(o):
    """src.evaluate keys at the model scale (dimensionless only)."""
    q12, q13, q23, qd = o["ckm"]
    l12, l13, l23, ld = o["pmns"]
    yu, yd, ye = o["yu"], o["yd"], o["ye"]
    return {"s12sq": np.sin(l12) ** 2, "s13sq": np.sin(l13) ** 2, "s23sq": np.sin(l23) ** 2,
            "dcp_deg": np.degrees(ld) % 360, "dm_ratio": o["dm21"] / o["dm31"],
            "ckm_s12": np.sin(q12), "ckm_s13": np.sin(q13), "ckm_s23": np.sin(q23), "ckm_delta_rad": qd,
            "mr_u_c": yu[0] / yu[1], "mr_c_t": yu[1] / yu[2], "mr_d_s": yd[0] / yd[1],
            "mr_s_b": yd[1] / yd[2], "mr_e_mu": ye[0] / ye[1], "mr_mu_tau": ye[1] / ye[2],
            "mr_b_tau": yd[2] / ye[2]}

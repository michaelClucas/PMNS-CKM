"""Exp 041: compressive Pati-Salam four-zero grammar (PS4Z). Structures, enumeration, bit budget.

Every entry has the same continuous parameters (8 real) and differs only in fixed (not fitted)
discrete choices:

  Y_u = [[0, zu, 0], [zu, wu, xu], [0, xu, 1]]                          (real, zu, xu >= 0)
  Y_d = [[0, zd e^{i p1}, 0], [zd e^{i p1}, wd e^{i p2}, xd], [0, xd, 1]]
  (phase placement (z, w): the first implementation put the phases on (z, x) and the quark-only
  fit failed (chi2 310, V_ub -14 sigma); (z, w) fits the quarks exactly, (w, x) gives 2.5, three
  phases (z, w, x) were also tried. This is a data-selected choice among 4 tried options: 2 bits.)
  theta = (lzu, vu, lxu, lzd, vd, lxd, p1, p2), z = 10^lz, x = 10^lx,
  w = x^2 + sign(v) 10^(5 (|v| - 1))  (v in [-1, 1]: signed log of the 2-3 block determinant
  w - x^2, a coordinate choice that straightens the fine-tuned m2/m3 valley; same family as a
  free real w with |w - x^2| in [1e-5, 1])                                  -> k_cont = 8

Quark-lepton relations (Pati-Salam / SO(10)): each of the three texture positions P in (12, 22, 23)
couples through a Higgs representation r_P in {10 ~ (1,2,2), 126 ~ (15,2,2)} with lepton Clebsch
lambda(10) = 1, lambda(126) = -3, applied to BOTH lepton sectors:
  Y_e[P] = lambda(r_P) Y_d[P],   Y_nu[P] = lambda(r_P) Y_u[P],   (3,3) entry: 10 for all sectors.
Fixed sector factors only; no lepton parameter is fitted. 2^3 = 8 representation patterns.

Neutrino options (4):
  dirac   Dirac neutrinos with Yukawa Y_nu
  I1      type-I seesaw, M_R proportional to 1: m_nu = -Y_nu^* Y_nu^dag
  IIFd    type-II, m_nu proportional to the 126 part of Y_d, F_d = sum_{P: r_P = 126} Y_d[P]
          (SO(10) 126-bar dominance; conj for the engine's LR convention, see model())
  IIFu    type-II, m_nu proportional to the 126 part of Y_u (the up-side projection)

Full grammar G32: 8 x 4 = 32 entries, 5 bits, plus 2 bits for the data-selected d-phase placement:
k = 8 + 7 = 15 > N - 2 = 13 (diagnostic). Compressive sub-grammar G8 (in_g8): 8 entries, 3 + 2 bits,
k = 13 = N - 2 for N = 15. Ordering declared NO before fitting (no bit). Overall sector scales are not fitted
(ratios only); mr_b_tau is not scored.

The bit-budget reasoning that fixes this design is in README.md (Method, "Design").
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.yukawa import left_rotation, observables  # noqa: E402

USE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg",
       "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "dm_ratio"]
QUSE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b"]
QEUSE = QUSE + ["mr_e_mu", "mr_mu_tau"]
LUSE = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
FLOOR = 0.02
N_OBS = len(USE)
NU_OPTS = ("dirac", "I1", "IIFd", "IIFu")
POS = ((0, 1), (1, 1), (1, 2))  # 12, 22, 23
LAM = {0: 1.0, 1: -3.0}  # 10 -> 1, 126 -> -3
PARAMS = ("lzu", "vu", "lxu", "lzd", "vd", "lxd", "p1", "p2")
BOUNDS = {"lzu": (-5.0, -1.0), "vu": (-1.0, 1.0), "lxu": (-3.0, 0.3),
          "lzd": (-5.0, -1.0), "vd": (-1.0, 1.0), "lxd": (-3.0, 0.3),
          "p1": (0.0, 2 * np.pi), "p2": (0.0, 2 * np.pi)}
K_CONT = len(PARAMS)


def entries():
    """All grammar entries: (rep pattern tuple (r12, r22, r23), nu option)."""
    return [dict(id=i, reps=reps, nu=nu) for i, (reps, nu) in
            enumerate(itertools.product(itertools.product((0, 1), repeat=3), NU_OPTS))]


ENTRIES = entries()
G = len(ENTRIES)
PHASE_BITS = 2  # data-selected phase placement (4 placements tried on the quark sector)
BITS = math.ceil(math.log2(G)) + PHASE_BITS  # full grammar G32: 5 + 2 = 7 bits
K = K_CONT + BITS  # 15 > N - 2: the full grammar is a diagnostic, not compressive


def in_g8(e):
    """Compressive sub-grammar G8, declared after the quark-only fits and before any lepton
    prediction was computed: Georgi-Jarlskog placement (the (2,2) entry couples through the 126),
    (1,2) and (2,3) free in {10, 126}, neutrinos type-I (M_R ~ 1) or type-II (m_nu ~ F_d)."""
    return e["reps"][1] == 1 and e["nu"] in ("I1", "IIFd")


G8 = [e["id"] for e in ENTRIES if in_g8(e)]
BITS8 = math.ceil(math.log2(len(G8))) + PHASE_BITS  # 3 + 2 = 5
K8 = K_CONT + BITS8  # 13 = N - 2


def k_of(e):
    """(k, k_discrete) under the grammar the entry is scored in: G8 if it is a member, else G32."""
    return (K8, BITS8) if in_g8(e) else (K, BITS)


def name(e):
    return "".join("H" if r else "h" for r in e["reps"]) + "-" + e["nu"]  # h = 10, H = 126 at 12, 22, 23


def w_of(v, x):
    return x ** 2 + np.sign(v) * 10 ** (5 * (np.abs(v) - 1))


def quark_mats(th):
    zu, xu, zd, xd = (10 ** th[k] for k in ("lzu", "lxu", "lzd", "lxd"))
    wu, wd = w_of(th["vu"], xu), w_of(th["vd"], xd)
    Yu = np.array([[0, zu, 0], [zu, wu, xu], [0, xu, 1]], complex)
    a, w = zd * np.exp(1j * th["p1"]), wd * np.exp(1j * th["p2"])
    Yd = np.array([[0, a, 0], [a, w, xd], [0, xd, 1]], complex)
    return Yu, Yd


def lepton_mats(Yu, Yd, reps):
    Ye, Yn = Yd.copy(), Yu.copy()
    Fd, Fu = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
    for (i, j), r in zip(POS, reps):
        lam = LAM[r]
        for (p, q) in {(i, j), (j, i)}:
            Ye[p, q] = lam * Yd[p, q]
            Yn[p, q] = lam * Yu[p, q]
            if r:
                Fd[p, q], Fu[p, q] = Yd[p, q], Yu[p, q]
    return Ye, Yn, Fd, Fu


def dirac_mnu(M):
    """Dirac neutrinos: symmetric matrix whose Takagi factor is the left rotation of M (as exp 039)."""
    U, y = left_rotation(M)
    return U.conj() @ np.diag(y) @ U.conj().T


def mnu_of(Yn, Fd, Fu, nu):
    if nu == "dirac":
        return dirac_mnu(Yn)
    if nu == "I1":
        m = -Yn.conj() @ Yn.conj().T
        return 0.5 * (m + m.T)
    # Y matrices here are the engine's LR Yukawas, i.e. the conjugate of the Weyl SO(10) coupling
    # matrices (src.yukawa docstring, exp 014 map: Y_f = M_f^dag, m_nu = M_nu); a 126-bar coupling
    # F (Weyl) enters Y_d as conj(F) and m_nu as F, so m_nu = conj(126 part of Y_d).
    if nu == "IIFd":
        return Fd.conj()
    if nu == "IIFu":
        return Fu.conj()
    raise ValueError(nu)


def model(th, e, ordering="NO", details=False):
    Yu, Yd = quark_mats(th)
    Ye, Yn, Fd, Fu = lepton_mats(Yu, Yd, e["reps"])
    m = mnu_of(Yn, Fd, Fu, e["nu"])
    if not np.any(m):
        raise ValueError("m_nu vanishes for this entry (no 126 position)")
    return observables(Yu, Yd, Ye, m_nu=m, ordering=ordering, details=details)


# ---------------------------------------------------------------- batched (synthetic draws)

def build_batch(X, reps, nu):
    """X (B, 8) in PARAMS order; reps (B, 3) in {0, 1}; nu (B,) index into NU_OPTS.
    Returns Yu, Yd, Ye, m_nu of shape (B, 3, 3)."""
    B = len(X)
    zu, xu = 10 ** X[:, 0], 10 ** X[:, 2]
    zd, xd = 10 ** X[:, 3], 10 ** X[:, 5]
    wu, wd = w_of(X[:, 1], xu), w_of(X[:, 4], xd)
    a, wdc = zd * np.exp(1j * X[:, 6]), wd * np.exp(1j * X[:, 7])
    Yu = np.zeros((B, 3, 3), complex)
    Yd = np.zeros((B, 3, 3), complex)
    for Y, z, w, x in ((Yu, zu, wu, xu), (Yd, a, wdc, xd)):
        Y[:, 0, 1] = Y[:, 1, 0] = z
        Y[:, 1, 1] = w
        Y[:, 1, 2] = Y[:, 2, 1] = x
        Y[:, 2, 2] = 1
    lam = np.where(reps == 1, -3.0, 1.0)  # (B, 3)
    S = np.ones((B, 3, 3))
    F = np.zeros((B, 3, 3))
    for k, (i, j) in enumerate(POS):
        S[:, i, j] = S[:, j, i] = lam[:, k]
        F[:, i, j] = F[:, j, i] = reps[:, k]
    Ye, Yn = Yd * S, Yu * S
    Fd, Fu = Yd * F, Yu * F
    w, U = np.linalg.eigh(Yn @ np.conj(np.swapaxes(Yn, 1, 2)))
    y = np.sqrt(np.clip(w, 0, None))
    m_dir = np.conj(U) @ (y[:, :, None] * np.conj(np.swapaxes(U, 1, 2)))
    m_I1 = -np.conj(Yn) @ np.conj(np.swapaxes(Yn, 1, 2))
    m = np.select([nu[:, None, None] == 0, nu[:, None, None] == 1, nu[:, None, None] == 2],
                  [m_dir, m_I1, np.conj(Fd)], np.conj(Fu))
    m = 0.5 * (m + np.swapaxes(m, 1, 2))
    return Yu, Yd, Ye, m


def k_table():
    """Bit budget of the grammar (README table)."""
    return dict(N=N_OBS, N_minus_2=N_OBS - 2, k_cont=K_CONT, phase_bits=PHASE_BITS,
                G32=dict(size=G, bits=BITS, k=K, compression_ok_by_count=K <= N_OBS - 2),
                G8=dict(size=len(G8), ids=G8, bits=BITS8, k=K8, compression_ok_by_count=K8 <= N_OBS - 2),
                k_distribution={f"k={K8} (G8 members)": len(G8), f"k={K} (G32 only)": G - len(G8)})


if __name__ == "__main__":
    import json
    print(json.dumps(k_table(), indent=1))
    for e in ENTRIES:
        print(e["id"], name(e))

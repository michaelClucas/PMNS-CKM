"""The minimal 2O quark-lepton model (arXiv:2307.14926 Sec. 5; arXiv:2409.15823 Sec. 2).

Couplings are real (gCP). Mass matrices exactly as printed in 2409.15823 Eqs. (2.3), (2.8).
Convention: superpotential term F^c M f, so M has rows = RH fields, columns = LH doublets; the
LH rotation comes from M^dag M (option transpose=True uses M M^dag instead, to test the
convention). Light neutrinos: M_nu = -M_D^T M_N^-1 M_D (symmetric, LH basis), diagonalised by
Takagi U_nu^T M_nu U_nu = diag(m); PMNS = U_e^dag U_nu (option conj_nu flips nu conjugation).

Parameter vector (dimensionless except the 4 scales):
  tau = re + i im
  leptons: gE1 (scale), rE2 = gE2/gE1, rE3 = gE3/gE1, nu_scale = g^2 v_u^2 / Lambda
  quarks : gu1 (scale), ru2, ru3, gd1 (scale), rd2, rd3, rd4, rd5
"""
import numpy as np

from modforms_2o import Y

S2, S3 = np.sqrt(2.0), np.sqrt(3.0)


def mass_matrices(tau, rE2, rE3, ru2, ru3, rd2, rd3, rd4, rd5, gE1=1.0, gu1=1.0, gd1=1.0, nu_scale=1.0):
    y2p5 = Y("2hp", 5, tau)
    y4h5 = Y("4h", 5, tau)
    y3p4 = Y("3p", 4, tau)
    y22 = Y("2", 2, tau)
    y4h3 = Y("4h", 3, tau)
    y26 = Y("2", 6, tau)
    y16 = Y("1", 6, tau)[0]
    y1p6 = Y("1p", 6, tau)[0]
    g1, g2, g3 = gE1, gE1 * rE2, gE1 * rE3
    a, b = y2p5, y4h5
    ME = np.array([
        [-g1 * a[1] - S2 * g2 * b[2], S3 * g2 * b[0], S2 * g1 * a[0] + g2 * b[3]],
        [-g1 * a[0] + S2 * g2 * b[3], -S2 * g1 * a[1] + g2 * b[2], -S3 * g2 * b[1]],
        [g3 * y3p4[0], g3 * y3p4[2], g3 * y3p4[1]],
    ])
    MD = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    MN = np.array([[-2 * y22[0], 0, 0],
                   [0, S3 * y22[1], y22[0]],
                   [0, y22[0], S3 * y22[1]]])
    Mnu = -nu_scale * MD.T @ np.linalg.inv(MN) @ MD
    u1, u2, u3 = gu1, gu1 * ru2, gu1 * ru3
    c = y4h3
    Mu = np.array([[u1 * c[2], -u1 * c[1], 0],
                   [u1 * c[3], u1 * c[0], 0],
                   [-u2 * y26[1], u2 * y26[0], u3 * y16]])
    d1, d2, d3, d4, d5 = gd1, gd1 * rd2, gd1 * rd3, gd1 * rd4, gd1 * rd5
    Md = np.array([[d1 * y16 - d3 * y26[0], d2 * y1p6 + d3 * y26[1], -d4 * y26[1]],
                   [d3 * y26[1] - d2 * y1p6, d3 * y26[0] + d1 * y16, d4 * y26[0]],
                   [0, 0, d5]])
    return ME, Mnu, Mu, Md


def left(M, transpose=False):
    H = M @ M.conj().T if transpose else M.conj().T @ M
    w, U = np.linalg.eigh(H)
    return np.sqrt(np.clip(w, 0, None)), U


def takagi(Mnu):
    """U^T M U = diag(m >= 0), ascending."""
    w, U = np.linalg.eigh(Mnu.conj().T @ Mnu)
    m = np.sqrt(np.clip(w, 0, None))
    D = U.T @ Mnu @ U
    ph = np.exp(-0.5j * np.angle(np.diag(D)))
    return m, U * ph


def angles(U):
    """PDG angles (s12^2, s13^2, s23^2, delta in [0, 2pi)) of a unitary matrix."""
    s13 = abs(U[0, 2])
    c13 = np.sqrt(max(1 - s13 ** 2, 0))
    s12sq = abs(U[0, 1]) ** 2 / c13 ** 2
    s23sq = abs(U[1, 2]) ** 2 / c13 ** 2
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    s12, c12 = np.sqrt(s12sq), np.sqrt(1 - s12sq)
    s23, c23 = np.sqrt(s23sq), np.sqrt(1 - s23sq)
    # cos(delta) from |U_{tau 1}|^2 (standard), sign of sin(delta) from J
    num = (s12 * s23) ** 2 + (c12 * c23 * s13) ** 2 - abs(U[2, 0]) ** 2
    den = 2 * s12 * s23 * c12 * c23 * s13
    cd = np.clip(num / den, -1, 1) if den > 0 else 1.0
    d = np.arccos(cd)
    if J < 0:
        d = 2 * np.pi - d
    return s12sq, s13 ** 2, s23sq, d, J


def observables(p, transpose=False, conj_nu=False):
    """p: dict with re, im, rE2, rE3, ru2, ru3, rd2, rd3, rd4, rd5 (+ optional scales)."""
    tau = p["re"] + 1j * p["im"]
    ME, Mnu, Mu, Md = mass_matrices(tau, p["rE2"], p["rE3"], p["ru2"], p["ru3"], p["rd2"], p["rd3"],
                                    p["rd4"], p["rd5"], p.get("gE1", 1.0), p.get("gu1", 1.0),
                                    p.get("gd1", 1.0), p.get("nu_scale", 1.0))
    me, Ue = left(ME, transpose)
    mu_, Uu = left(Mu, transpose)
    md, Ud = left(Md, transpose)
    mn, Un = takagi(Mnu.conj() if conj_nu else Mnu)
    U = Ue.conj().T @ Un
    V = Uu.conj().T @ Ud
    s12, s13, s23, dl, Jl = angles(U)
    q12, q13, q23, dq, Jq = angles(V)
    out = {"s12sq": s12, "s13sq": s13, "s23sq": s23, "dl": dl, "Jl": Jl,
           "th12q": np.arcsin(np.sqrt(q12)), "th13q": np.arcsin(np.sqrt(q13)),
           "th23q": np.arcsin(np.sqrt(q23)), "dq": dq, "Jq": Jq,
           "me": me[0], "mmu": me[1], "mtau": me[2], "mu": mu_[0], "mc": mu_[1], "mt": mu_[2],
           "md": md[0], "ms": md[1], "mb": md[2], "m1": mn[0], "m2": mn[1], "m3": mn[2]}
    out["r_emu"] = me[0] / me[1]
    out["r_mutau"] = me[1] / me[2]
    out["r_uc"] = mu_[0] / mu_[1]
    out["r_ct"] = mu_[1] / mu_[2]
    out["r_ds"] = md[0] / md[1]
    out["r_sb"] = md[1] / md[2]
    out["dm21"] = mn[1] ** 2 - mn[0] ** 2
    out["dm31"] = mn[2] ** 2 - mn[0] ** 2
    out["DM2"] = mn[2] ** 2 - 0.5 * (mn[0] ** 2 + mn[1] ** 2)
    out["dm_ratio"] = out["dm21"] / out["dm31"]
    return out

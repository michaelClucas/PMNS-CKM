"""Exp 045: the catalogue of published predictive quark structures (see models.md for sources).

TEXTURES: name -> (Texture factory, stated scale, note)
RELATIONS: name -> (model_fn, bounds, k_discrete, stated scale, note)
HS: Harrison-Scott 2501.18508 (custom parametrisation, see hs_model).
Indices are 0-based (row, col); Hermitian sectors list entries on/above the diagonal.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from lib45 import (Q8, Sector, Texture, alpha_of, ckm_from_std, fx_matrix, quark_obs, std_from_V, V_std,
                   TWO_PI)

H = "herm"
FRITZSCH = {(0, 1): "c", (1, 2): "c", (2, 2): "1"}            # [[0,A,0],[A*,0,B],[0,B*,C]]
GJ_D = {(0, 1): "c", (1, 1): "r", (2, 2): "1"}                # [[0,F,0],[F*,E,0],[0,0,D]]
FOURZERO = {(0, 1): "c", (1, 1): "r", (1, 2): "c", (2, 2): "1"}
U13 = {(0, 2): "c", (1, 1): "r", (2, 2): "1"}                 # [[0,0,C],[0,B,0],[C*,0,A]]
U13_23 = {(0, 2): "c", (1, 1): "r", (1, 2): "c", (2, 2): "1"}  # [[0,0,C],[0,B,B'],[C*,B'*,A]]


def T(name, u, d, ku=H, kd=H):
    return lambda: Texture(name, Sector("u", ku, u), Sector("d", kd, d))


def lg(name, d, u):
    """Ludl-Grimus Table 3 (complex symmetric). Codes: 'p' real positive, 'c' complex; the entry
    marked by a trailing '!' is the per-sector normalisation ('1' if real, 'c1' if complex)."""
    def conv(spec):
        out = {}
        for k, v in spec.items():
            if v.endswith("!"):
                out[k] = "1" if v[0] == "p" else "c1"
            else:
                out[k] = v
        return out
    return lambda: Texture(name, Sector("u", "sym", conv(u)), Sector("d", "sym", conv(d)))


# Belfatto-Berezhiani 2305.00069 Eq. (3.2) with x_u = 1: Y_d = [[0,A,0],[A,0,Bx],[0,B/x,C]] (real,
# general); Y_u Fritzsch Hermitian; the two physical phases are carried by Y_u (Eq. 3.4: F = F_u F_d^*).
BB_D = {(0, 1): "p", (1, 0): ("tie", (0, 1)), (1, 2): "p", (2, 1): "p", (2, 2): "1"}

TEXTURES = {
    # name: (factory, stated scale, note)
    "F6":   (T("F6", FRITZSCH, FRITZSCH), "any (Fritzsch: unspecified); M_Z and GUT tried",
             "Fritzsch six-zero Hermitian (PLB 73B (1978) 317; 85B (1979) 81)"),
    "DHR":  (T("DHR", FRITZSCH, GJ_D), "GUT (MSSM)", "Dimopoulos-Hall-Raby / HRR GJ six-zero = RRR sol. 2 with E'=0"),
    "RRR1": (T("RRR1", {(0, 1): "c", (1, 1): "r", (2, 2): "1"}, FOURZERO), "GUT (MSSM)", "hep-ph/9303320 Table 1 sol. 1"),
    "RRR2": (T("RRR2", FRITZSCH, FOURZERO), "GUT (MSSM)", "hep-ph/9303320 Table 1 sol. 2"),
    "RRR3": (T("RRR3", U13, FOURZERO), "GUT (MSSM)", "hep-ph/9303320 Table 1 sol. 3 (= Verma 1708.08993 case I)"),
    "RRR4": (T("RRR4", FOURZERO, GJ_D), "GUT (MSSM)", "hep-ph/9303320 Table 1 sol. 4"),
    "RRR5": (T("RRR5", U13_23, GJ_D), "GUT (MSSM)",
             "hep-ph/9303320 Table 1 sol. 5 (= Giraldo-Rojas 1511.08858 case II = Verma 1708.08993 case IV)"),
    "GR1":  (T("GR1", U13_23, FRITZSCH), "M_Z", "Giraldo-Rojas 1511.08858 case I (Eq. 16) = 1811.05068 pattern (Eq. 12)"),
    "BB":   (T("BB", FRITZSCH, BB_D, kd="gen"), "1e3-1e16 GeV (SM)", "Belfatto-Berezhiani 2305.00069 Eq. (3.2), x_u = 1"),
    "LG_S8":  (lg("LG_S8", {(0, 2): "p", (1, 1): "c", (1, 2): "c", (2, 2): "c!"},
                  {(0, 2): "p", (1, 1): "p", (1, 2): "p!"}), "M_Z", "Ludl-Grimus 1501.04942 Table 3 S8"),
    "LG_S9":  (lg("LG_S9", {(0, 2): "p", (1, 1): "p", (1, 2): "p", (2, 2): "c!"},
                  {(0, 2): "p", (1, 1): "p", (2, 2): "c!"}), "M_Z", "Ludl-Grimus 1501.04942 Table 3 S9"),
    "LG_S10": (lg("LG_S10", {(0, 2): "p", (1, 1): "p", (1, 2): "p", (2, 2): "c!"},
                  {(0, 1): "p", (1, 1): "c", (2, 2): "p!"}), "M_Z", "Ludl-Grimus 1501.04942 Table 3 S10 (Eq. 11)"),
    "LG_S12": (lg("LG_S12", {(0, 2): "p", (1, 1): "c", (1, 2): "c!"},
                  {(0, 2): "p", (1, 1): "p", (1, 2): "p", (2, 2): "c!"}), "M_Z", "Ludl-Grimus 1501.04942 Table 3 S12"),
    "LG_S13": (lg("LG_S13", {(0, 2): "p", (1, 1): "c", (1, 2): "c!"},
                  {(0, 1): "p", (1, 1): "p", (1, 2): "p", (2, 2): "c!"}), "M_Z", "Ludl-Grimus 1501.04942 Table 3 S13"),
    "LG_S15": (lg("LG_S15", {(0, 2): "p", (1, 1): "p", (2, 2): "c!"},
                  {(0, 1): "p", (1, 1): "p", (1, 2): "p", (2, 2): "c!"}), "M_Z", "Ludl-Grimus 1501.04942 Table 3 S15"),
    # non-predictive references (k_cont >= 8): fitted only as a sanity check of the pipeline
    "F4":   (T("F4", FOURZERO, FOURZERO), "M_Z", "Fritzsch-Xing four-zero, hep-ph/0212195 (reference, k = 8)"),
}


# ------------------------------------------------------------------ relations (closed form)
MR = ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")
MR_B = {"l_uc": (-8.5, -4.5), "l_ct": (-7.0, -4.0), "l_ds": (-4.0, -2.0), "l_sb": (-5.0, -3.0)}


def _mr(th):
    return {"mr_u_c": np.exp(th["l_uc"]), "mr_c_t": np.exp(th["l_ct"]), "mr_d_s": np.exp(th["l_ds"]),
            "mr_s_b": np.exp(th["l_sb"])}


def gst(th):
    """Gatto-Sartori-Tonin: |V_us| = sqrt(m_d/m_s)."""
    p = _mr(th)
    s13 = th["s13"]
    p.update(ckm_from_std(np.sqrt(p["mr_d_s"]) / np.sqrt(1 - s13 ** 2), s13, th["s23"], th["d"]))
    return p


def hall_rasin(th):
    """|V_ub / V_cb| = sqrt(m_u/m_c)."""
    p = _mr(th)
    r, s23 = np.sqrt(p["mr_u_c"]), th["s23"]
    s13 = r * s23 / np.sqrt(1 + (r * s23) ** 2)
    p.update(ckm_from_std(th["s12"], s13, s23, th["d"]))
    return p


def _solve_delta(target_fn, s12, s13, s23):
    """delta in (0, pi) with target_fn(V(delta)) = 0, or the endpoint closest to a root."""
    f = lambda d: target_fn(V_std(s12, s13, s23, d))
    a, b = 1e-6, np.pi - 1e-6
    fa, fb = f(a), f(b)
    if fa * fb < 0:
        return brentq(f, a, b, xtol=1e-12)
    return a if abs(fa) < abs(fb) else b


def td_ts(th):
    """|V_td / V_ts| = sqrt(m_d/m_s); delta solved in (0, pi) (sign of J: 1 bit)."""
    p = _mr(th)
    r = np.sqrt(p["mr_d_s"])
    d = _solve_delta(lambda V: abs(V[2, 0]) / abs(V[2, 1]) - r, th["s12"], th["s13"], th["s23"])
    p.update(ckm_from_std(th["s12"], th["s13"], th["s23"], d))
    return p


def right_triangle(th):
    """alpha = 90 deg; delta solved in (0, pi) (sign of J: 1 bit)."""
    p = _mr(th)
    d = _solve_delta(lambda V: np.cos(alpha_of(V)), th["s12"], th["s13"], th["s23"])
    p.update(ckm_from_std(th["s12"], th["s13"], th["s23"], d))
    return p


def fx(th):
    """Fritzsch-Xing: V of hep-ph/9707215 Eq. (8) with tan th_u = sqrt(m_u/m_c), tan th_d = sqrt(m_d/m_s),
    phi = +90 deg (sign of J: 1 bit)."""
    p = _mr(th)
    V = fx_matrix(np.arctan(np.sqrt(p["mr_u_c"])), np.arctan(np.sqrt(p["mr_d_s"])), th["t"], np.pi / 2)
    p.update(std_from_V(V))
    return p


def d7(th):
    """Blum-Hagedorn-Lindner 0709.3450: |V_us| = |cos(3 pi / 7)| (D7, j = 1, m_u - m_d = 3)."""
    p = _mr(th)
    s13 = th["s13"]
    p.update(ckm_from_std(abs(np.cos(3 * np.pi / 7)) / np.sqrt(1 - s13 ** 2), s13, th["s23"], th["d"]))
    return p


ANG = {"s12": (0.2, 0.25), "s13": (0.002, 0.006), "s23": (0.03, 0.05), "d": (0.0, TWO_PI), "t": (0.03, 0.05)}


def _b(*names):
    b = dict(MR_B)
    b.update({n: ANG[n] for n in names})
    return b


RELATIONS = {
    # name: (fn, bounds, k_discrete, note)
    "GST": (gst, _b("s13", "s23", "d"), 0, "Gatto-Sartori-Tonin |V_us| = sqrt(m_d/m_s)"),
    "HR": (hall_rasin, _b("s12", "s23", "d"), 0, "Hall-Rasin hep-ph/9303303 |V_ub/V_cb| = sqrt(m_u/m_c)"),
    "TDTS": (td_ts, _b("s12", "s13", "s23"), 1, "|V_td/V_ts| = sqrt(m_d/m_s) (Hall-Rasin; RRRV Eq. 1)"),
    "ALPHA90": (right_triangle, _b("s12", "s13", "s23"), 1, "right unitarity triangle alpha = 90 deg"),
    "FX": (fx, _b("t"), 1, "Fritzsch-Xing: tan th_u = sqrt(m_u/m_c), tan th_d = sqrt(m_d/m_s), phi = 90 deg"),
    "D7": (d7, _b("s13", "s23", "d"), 0, "Blum-Hagedorn-Lindner D7: |V_us| = cos(3 pi/7)"),
}


# ------------------------------------------------------------------ Harrison-Scott 2501.18508
T8 = np.tan(np.pi / 8)


def hs_matrices(th, reading="mod"):
    """Eq. (2.1): M_q = [[c_q e^4, b e^3, 0], [., b e^2, A0 e^2], [0, ., 1]] Hermitian, eps_u = -i tan(pi/8) eps_d,
    |eps_d + eps_u| = lambda0 (Eq. 2.2-2.3). reading 'mod': diagonal entries use |eps|^n (Hermitian),
    off-diagonal complex powers; 'cplx': every power complex (then the matrix is not Hermitian; the
    lower triangle is the conjugate of the upper)."""
    A0, b, cu, cd, lam0 = th["A0"], th["b"], th["cu"], th["cd"], th["lam0"]
    ed = lam0 / abs(1 - 1j * T8)
    eu = -1j * T8 * ed
    out = []
    for e, c in ((eu, cu), (ed, cd)):
        a = abs(e)
        d11 = c * a ** 4 if reading == "mod" else c * e ** 4
        d22 = b * a ** 2 if reading == "mod" else b * e ** 2
        M = np.array([[d11, b * e ** 3, 0], [np.conj(b * e ** 3), d22, A0 * e ** 2],
                      [0, np.conj(A0 * e ** 2), 1]], complex)
        out.append(M)
    return out


def hs_model(th, reading="mod"):
    Mu, Md = hs_matrices(th, reading)
    return quark_obs(Mu, Md)


HS_BOUNDS = {"A0": (0.3, 2.0), "b": (0.1, 1.5), "cu": (-2.0, 2.0), "cd": (-2.0, 2.0), "lam0": (0.15, 0.30)}
HS_PUB = {"A0": 0.854, "b": 0.462, "cd": -0.040, "cu": 0.344, "lam0": 0.22646}  # Table 1 caption

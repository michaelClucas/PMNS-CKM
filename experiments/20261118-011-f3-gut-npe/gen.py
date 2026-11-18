"""F3 synthetic-data generator: SO(10) 10_H + 126bar_H (+120_H) and SU(5) Yukawa relations at M_GUT.

Clebsch structure (verified against Dueck & Rodejohann, arXiv:1306.4468, Eqs. (3) and (4)):
    Y_u = r (H + s F + i t_u G)
    Y_d =     H +   F + i     G
    Y_D = r (H - 3 s F + i t_D G)      (Dirac neutrino)
    Y_l =     H - 3 F + i t_l G
    M_R = r_R^-1 F                    (type I),   M_L = v_L F  (type II, Eq. (3), 126bar couples to L L)
H = Y_10, F = Y_126 (complex symmetric), G = Y_120 (antisymmetric); r, r_R real WLOG (their Sec. 2).
Parameter count of the 10+126 model in their Sec. 2.1: 19 real (H 3, F 12, r, s (2), r_R) with type I.
Type-II light mass: m_nu ∝ F = (Y_d - Y_l)/4 (Bajc, Senjanovic, Vissani, hep-ph/0210207).

All matrices here are Weyl ("superpotential") mass matrices, f^c M f, symmetric for 10 and 126.
The src/rg.py Yukawa (Lbar Y e_R) is Y = M^dagger, so the left-handed rotation is the eigenbasis of
M^dagger M, exactly what rg.observables() uses (Y Y^dagger). The light Majorana matrix m_nu = M_D^T M_R^-1 M_D
(up to an unphysical overall sign/phase) is holomorphic and equals rg.py's kappa up to normalisation:
kappa = U^* diag(m) U^dagger, U_nu from eigh(m^dagger m), U_PMNS = U_e^dagger U_nu, V_CKM = U_u^dagger U_d.

Quark-anchored parametrisation of the 10+126 model (used for the datasets)
--------------------------------------------------------------------------
The SO(10) flavour basis change psi -> W psi (M -> W^T M W, W in U(3)) is used to make Y_u real diagonal
(Takagi), instead of H as in 1306.4468. Then, in units of the bottom Yukawa (overall scale dropped):
    Y_u / y_b = diag(u1, u2, u3),
    Y_d / y_b = U^* diag(d1 e^{i phi1}, d2 e^{i phi2}, e^{i phi3}) U^dagger,  U = diag(1, e^{i a2}, e^{i a3}) V_std(CKM),
    F = (Y_u / r - Y_d) / (s - 1),  H = Y_d - F,  Y_l = Y_d - 4F,  Y_D ∝ Y_d - (1 + 3 s) F.
This is a reparametrisation of (H, F, r, s): 3 (Y_u) + 12 (Y_d) + 2 (s) + 1 (r) = 18 real, the same as
1306.4468's 18 before r_R. Dropping the overall Yukawa scale (y_b) leaves 17 dimensionless parameters;
r_R (type I) or v_L (type II) only sets the absolute neutrino scale and also drops out of the ratios.
"""
from __future__ import annotations

import numpy as np

from src.mixing import extract_angles, standard_matrix

TWO_PI = 2 * np.pi
LN10 = np.log(10.0)

# ------------------------------------------------------------------ observable vector
# x (measurement space z, before the (cos, sin) map of the two phases):
Z_NAMES = ["lg_mu_mc", "lg_mc_mt", "lg_md_ms", "lg_ms_mb", "lg_mb_mt", "lg_me_mmu", "lg_mmu_mtau", "lg_mtau_mb",
           "s12q", "lg_s13q", "lg_s23q", "dq",
           "s12sq_l", "s13sq_l", "s23sq_l", "dl", "lg_dm21_dm3l"]
Z_PHASES = (11, 15)   # indices of delta_q, delta_l in z
QL_IDX = list(range(12))          # quark + charged-lepton block (masses + CKM)
MASS_IDX = list(range(8))
PMNS_IDX = list(range(12, 17))


def z_to_x(z):
    """17-dim z -> 19-dim x with each phase replaced by (cos, sin)."""
    cols = []
    for j in range(z.shape[1]):
        if j in Z_PHASES:
            cols += [np.cos(z[:, j]), np.sin(z[:, j])]
        else:
            cols.append(z[:, j])
    return np.stack(cols, 1)


X_NAMES = []
for _j, _n in enumerate(Z_NAMES):
    X_NAMES += [f"cos_{_n}", f"sin_{_n}"] if _j in Z_PHASES else [_n]


# ------------------------------------------------------------------ linear algebra helpers (batched)

def _dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def left_basis(M):
    """Singular values (ascending) and left-handed rotation U with M^dagger M = U diag(m^2) U^dagger."""
    w, U = np.linalg.eigh(_dag(M) @ M)
    return np.sqrt(np.clip(w, 0, None)), U


def observables_from_mats(Mu, Md, Me, Mnu):
    """Batched (..., 3, 3) Weyl mass matrices -> z (N, 17), ordering label (0 NO, 1 IO), ok mask."""
    mu, Uu = left_basis(Mu)
    md, Ud = left_basis(Md)
    me, Ue = left_basis(Me)
    mn, Un = left_basis(Mnu)  # ascending a < b < c
    V = _dag(Uu) @ Ud
    q12, q13, q23, dq = extract_angles(V)
    a2, b2, c2 = mn[:, 0] ** 2, mn[:, 1] ** 2, mn[:, 2] ** 2
    io = (b2 - a2) > (c2 - b2)         # lightest pair is the upper one -> IO
    # NO: (m1, m2, m3) = (a, b, c); IO: (m3, m1, m2) = (a, b, c) -> column order (b, c, a)
    Un_ord = np.where(io[:, None, None], Un[:, :, [1, 2, 0]], Un)
    P = _dag(Ue) @ Un_ord
    l12, l13, l23, dl = extract_angles(P)
    dm21 = np.where(io, c2 - b2, b2 - a2)
    dm3l = c2 - a2
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.stack([
            np.log10(mu[:, 0] / mu[:, 1]), np.log10(mu[:, 1] / mu[:, 2]),
            np.log10(md[:, 0] / md[:, 1]), np.log10(md[:, 1] / md[:, 2]), np.log10(md[:, 2] / mu[:, 2]),
            np.log10(me[:, 0] / me[:, 1]), np.log10(me[:, 1] / me[:, 2]), np.log10(me[:, 2] / md[:, 2]),
            np.sin(q12), np.log10(np.sin(q13)), np.log10(np.sin(q23)), dq,
            np.sin(l12) ** 2, np.sin(l13) ** 2, np.sin(l23) ** 2, dl, np.log10(dm21 / dm3l)], 1)
    ok = np.all(np.isfinite(z), 1)
    z = np.where(np.isfinite(z), z, 0.0)
    z[:, :11] = np.clip(z[:, :11], -14, 3)
    z[:, 16] = np.clip(z[:, 16], -14, 1)
    return z, io.astype(np.int8), ok


# ------------------------------------------------------------------ SO(10) 10 + 126bar, quark-anchored

def gut_reference():
    """Real-data GUT-scale centres used to place the quark-sector prior windows (see README)."""
    import json, pathlib
    p = pathlib.Path(__file__).with_name("obs_gut.json")
    return json.loads(p.read_text())


THETA_SO10 = ["lg_u1_u2", "lg_u2_u3", "lg_u3_d3", "lg_d1_d2", "lg_d2_d3",
              "s12q", "lg_s13q", "lg_s23q", "dq",
              "a2", "a3", "phi1", "phi2", "phi3", "lg_r", "lg_abs_s", "arg_s"]
THETA_MIX = ["lg_abs_w", "arg_w"]  # type I + II mixing weight (variant 'so10_mixed')


def so10_prior_box(ref, variant):
    """(lo, hi) arrays of the uniform prior box. Quark windows centred on the GUT data values."""
    c = ref["z"]
    box = [
        (c[0] - 0.5, c[0] + 0.5), (c[1] - 0.5, c[1] + 0.5), (-c[4] - 0.5, -c[4] + 0.5),
        (c[2] - 0.5, c[2] + 0.5), (c[3] - 0.5, c[3] + 0.5),
        (0.15, 0.30), (c[9] - 0.4, c[9] + 0.4), (c[10] - 0.4, c[10] + 0.4), (0, TWO_PI),
        (0, TWO_PI), (0, TWO_PI), (0, TWO_PI), (0, TWO_PI), (0, TWO_PI),
        (0.0, 4.0), (-2.0, 2.0), (0, TWO_PI)]
    if variant == "so10_mixed":
        box += [(-3.0, 3.0), (0, TWO_PI)]
    b = np.array(box, dtype=float)
    return b[:, 0], b[:, 1]


def so10_mats(th, variant):
    """theta (N, 17[+2]) -> Weyl matrices (Mu, Md, Me, Mnu) for variant so10_I / so10_II / so10_mixed."""
    N = th.shape[0]
    u = 10.0 ** np.stack([th[:, 0] + th[:, 1] + th[:, 2], th[:, 1] + th[:, 2], th[:, 2]], 1)
    d = 10.0 ** np.stack([th[:, 3] + th[:, 4], th[:, 4], np.zeros(N)], 1)
    s12, s13, s23 = th[:, 5], 10.0 ** th[:, 6], 10.0 ** th[:, 7]
    V = standard_matrix(np.arcsin(s12), np.arcsin(s13), np.arcsin(s23), th[:, 8])
    P = np.zeros((N, 3, 3), complex)
    P[:, 0, 0] = 1
    P[:, 1, 1] = np.exp(1j * th[:, 9])
    P[:, 2, 2] = np.exp(1j * th[:, 10])
    U = P @ V
    D = np.zeros((N, 3, 3), complex)
    for i in range(3):
        D[:, i, i] = d[:, i] * np.exp(1j * th[:, 11 + i])
    Md = np.conj(U) @ D @ _dag(U)
    Mu = np.zeros((N, 3, 3), complex)
    for i in range(3):
        Mu[:, i, i] = u[:, i]
    r = 10.0 ** th[:, 14]
    s = 10.0 ** th[:, 15] * np.exp(1j * th[:, 16])
    F = (Mu / r[:, None, None] - Md) / (s - 1)[:, None, None]
    Me = Md - 4 * F
    MD = Md - (1 + 3 * s)[:, None, None] * F          # Y_D / r
    if variant == "so10_II":
        Mnu = F
    else:
        Mnu = MD @ np.linalg.solve(F, MD)              # M_D F^-1 M_D (symmetric; sign/scale unphysical)
        if variant == "so10_mixed":
            nI = np.linalg.norm(Mnu, axis=(1, 2))[:, None, None]
            nII = np.linalg.norm(F, axis=(1, 2))[:, None, None]
            w = (10.0 ** th[:, 17] * np.exp(1j * th[:, 18]))[:, None, None]
            Mnu = Mnu / nI + w * F / nII
    return Mu, Md, Me, Mnu


# ------------------------------------------------------------------ SO(10) with 120_H (generic H-diagonal basis)

THETA_SO10_120 = (["lg_h1", "lg_h2"] + [f"F{i}{j}_{p}" for i in range(3) for j in range(i, 3) for p in ("lgabs", "arg")]
                  + [f"G{i}{j}_{p}" for i, j in ((0, 1), (0, 2), (1, 2)) for p in ("lgabs", "arg")]
                  + ["lg_r", "lg_abs_s", "arg_s", "lg_abs_tu", "arg_tu", "lg_abs_tD", "arg_tD", "lg_abs_tl", "arg_tl"])


def so10_120_prior_box():
    box = [(-6, -3), (-3.5, -1)] + [(-3, 0.5), (0, TWO_PI)] * 6 + [(-4, 0), (0, TWO_PI)] * 3 \
          + [(0, 4), (-2, 2), (0, TWO_PI)] + [(-2, 2), (0, TWO_PI)] * 3
    b = np.array(box, float)
    return b[:, 0], b[:, 1]


def so10_120_mats(th, seesaw="I"):
    """1306.4468 Eq. (4) with general complex s, t_u, t_D, t_l. H = diag(h1, h2, 1) real (overall scale
    dropped). F_ij = c_ij sqrt(h_i h_j) (hierarchical magnitudes), G_ij = g_ij sqrt(h_i h_j)."""
    N = th.shape[0]
    h = np.stack([10.0 ** th[:, 0], 10.0 ** th[:, 1], np.ones(N)], 1)
    sq = np.sqrt(h[:, :, None] * h[:, None, :])
    H = np.zeros((N, 3, 3), complex)
    for i in range(3):
        H[:, i, i] = h[:, i]
    F = np.zeros((N, 3, 3), complex)
    k = 2
    for i in range(3):
        for j in range(i, 3):
            F[:, i, j] = F[:, j, i] = 10.0 ** th[:, k] * np.exp(1j * th[:, k + 1]) * sq[:, i, j]
            k += 2
    G = np.zeros((N, 3, 3), complex)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        g = 10.0 ** th[:, k] * np.exp(1j * th[:, k + 1]) * sq[:, i, j]
        G[:, i, j], G[:, j, i] = g, -g
        k += 2
    c = lambda a, b: (10.0 ** th[:, a] * np.exp(1j * th[:, b]))[:, None, None]
    r = (10.0 ** th[:, k])[:, None, None]
    s, tu, tD, tl = c(k + 1, k + 2), c(k + 3, k + 4), c(k + 5, k + 6), c(k + 7, k + 8)
    Mu = r * (H + s * F + 1j * tu * G)
    Md = H + F + 1j * G
    MD = H - 3 * s * F + 1j * tD * G
    Me = H - 3 * F + 1j * tl * G
    Mnu = F if seesaw == "II" else MD.transpose(0, 2, 1) @ np.linalg.solve(F, MD)
    return Mu, Md, Me, Mnu


# ------------------------------------------------------------------ SU(5) (5_H + 45_H optional) with type-II 15_H

THETA_SU5 = ["lg_u1_u2", "lg_u2_u3", "lg_u3_d3", "lg_d1_d2", "lg_d2_d3", "s12q", "lg_s13q", "lg_s23q", "dq",
             "a2", "a3", "lg_abs_c45", "arg_c45",
             "lg_n1_n3", "lg_n2_n3", "nu_t12", "nu_t13", "nu_t23", "nu_d", "nu_m1", "nu_m2"]


def su5_prior_box(ref):
    lo, hi = so10_prior_box(ref, "so10_I")
    box = list(zip(lo[:11], hi[:11])) + [(-2, 1), (0, TWO_PI), (-4, 0), (-3, 0)] + [(0, np.pi / 2)] * 3 \
        + [(0, TWO_PI)] * 3
    b = np.array(box, float)
    return b[:, 0], b[:, 1]


def su5_mats(th, clebsch="GJ"):
    """Minimal SU(5): Y_u symmetric (10 10 5_H), Y_d = Y_e^T (10 5bar 5bar_H). Basis: Y_u real diagonal,
    Y_d = U D (U = P V_CKM, right rotation absorbed into the 5bar fields).
    Clebsch 'min': Y_e = Y_d^T.  'GJ': Y_d = U (D + c45 d2 E22), Y_e = (U (D - 3 c45 d2 E22))^T, i.e. a 45_H
    contribution in the 22 entry of the down-quark mass basis (Georgi-Jarlskog factor -3).
    Neutrinos: type II from a 15_H with an independent Yukawa (masses n_i, mixing angles, Dirac + 2 Majorana
    phases drawn from the prior, Haar-like in angles only up to the uniform-angle prior)."""
    N = th.shape[0]
    u = 10.0 ** np.stack([th[:, 0] + th[:, 1] + th[:, 2], th[:, 1] + th[:, 2], th[:, 2]], 1)
    d = 10.0 ** np.stack([th[:, 3] + th[:, 4], th[:, 4], np.zeros(N)], 1)
    V = standard_matrix(np.arcsin(th[:, 5]), np.arcsin(10.0 ** th[:, 6]), np.arcsin(10.0 ** th[:, 7]), th[:, 8])
    P = np.zeros((N, 3, 3), complex)
    P[:, 0, 0] = 1
    P[:, 1, 1] = np.exp(1j * th[:, 9])
    P[:, 2, 2] = np.exp(1j * th[:, 10])
    U = P @ V
    D = np.zeros((N, 3, 3), complex)
    for i in range(3):
        D[:, i, i] = d[:, i]
    De = D.copy()
    if clebsch == "GJ":
        c = 10.0 ** th[:, 11] * np.exp(1j * th[:, 12]) * d[:, 1]
        D[:, 1, 1] += c
        De[:, 1, 1] -= 3 * c
    Mu = np.zeros((N, 3, 3), complex)
    for i in range(3):
        Mu[:, i, i] = u[:, i]
    # Weyl: d^c M_d Q. SU(5) Y_d (Q, d^c) = Y_e^T (L, e^c): M_d = (U D)^T, M_e = U De.
    Md = (U @ D).transpose(0, 2, 1)
    Me = U @ De
    n = np.stack([10.0 ** (th[:, 13]), 10.0 ** th[:, 14], np.ones(N)], 1)
    W = standard_matrix(th[:, 15], th[:, 16], th[:, 17], th[:, 18])
    Dn = np.zeros((N, 3, 3), complex)
    Dn[:, 0, 0] = n[:, 0] * np.exp(1j * th[:, 19])
    Dn[:, 1, 1] = n[:, 1] * np.exp(1j * th[:, 20])
    Dn[:, 2, 2] = n[:, 2]
    Mnu = np.conj(W) @ Dn @ _dag(W)
    return Mu, Md, Me, Mnu


# ------------------------------------------------------------------ dispatch

VARIANTS = {
    "so10_I": dict(theta=THETA_SO10, k_dimless=17, k_total=19,
                   note="10+126bar, type I. 17 dimensionless + y_b scale + r_R scale = 19 (= 1306.4468 Sec. 2.1)"),
    "so10_II": dict(theta=THETA_SO10, k_dimless=17, k_total=19,
                    note="10+126bar, type II (m_nu ∝ F). 17 + y_b + v_L"),
    "so10_mixed": dict(theta=THETA_SO10 + THETA_MIX, k_dimless=19, k_total=21,
                       note="10+126bar, type I + II, complex relative weight w"),
    "so10_120": dict(theta=THETA_SO10_120, k_dimless=len(THETA_SO10_120), k_total=len(THETA_SO10_120) + 2,
                     note="10+126bar+120, all complex: H 2 (+scale), F 12, G 6, r, s, t_u, t_D, t_l -> 29 + y scale + r_R"),
    "su5_min": dict(theta=THETA_SU5[:11] + THETA_SU5[13:], k_dimless=19, k_total=21,
                    note="SU(5) 5_H + 15_H: quark 11 (Y_u 3 incl. scale ratio, Y_d 3, CKM 4, 2 phases) + nu 8 (2 ratios, 3 angles, 3 phases) + 2 scales"),
    "su5_gj": dict(theta=THETA_SU5, k_dimless=21, k_total=23, note="SU(5) + 45_H in the 22 entry (complex c45)"),
}


def prior_box(variant, ref=None):
    ref = ref or gut_reference()
    if variant.startswith("so10_") and variant != "so10_120":
        return so10_prior_box(ref, variant)
    if variant == "so10_120":
        return so10_120_prior_box()
    lo, hi = su5_prior_box(ref)
    if variant == "su5_min":
        keep = list(range(11)) + list(range(13, 21))
        return lo[keep], hi[keep]
    return lo, hi


def mats(theta, variant):
    if variant in ("so10_I", "so10_II", "so10_mixed"):
        return so10_mats(theta, variant)
    if variant == "so10_120":
        return so10_120_mats(theta, "I")
    if variant == "su5_min":
        th = np.concatenate([theta[:, :11], np.zeros((len(theta), 1)), np.zeros((len(theta), 1)), theta[:, 11:]], 1)
        th[:, 11] = -30.0   # c45 = 0
        return su5_mats(th, "GJ")
    return su5_mats(theta, "GJ")


def sample_prior(variant, n, rng, ref=None):
    lo, hi = prior_box(variant, ref)
    return lo + (hi - lo) * rng.random((n, len(lo)))


def simulate(theta, variant):
    """theta -> noiseless z (N, 17), IO label, ok mask."""
    return observables_from_mats(*mats(theta, variant))


def add_noise(z, sigma, rng):
    zn = z + sigma * rng.standard_normal(z.shape)
    for j in Z_PHASES:
        zn[:, j] = np.mod(zn[:, j], TWO_PI)
    return zn

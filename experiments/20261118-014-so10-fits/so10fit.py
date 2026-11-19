"""Core of exp 014: SO(10) Yukawa models, observables, fit targets and a batched Levenberg-Marquardt.

Conventions are those of exp 011 gen.py (verified there against src/rg.observables to 2.4e-8 rad):
Weyl mass matrices f^c M f, left rotation from M^dagger M, V_CKM = U_u^dagger U_d, U_PMNS = U_e^dagger U_nu,
type-I light mass m_nu = M_D^T M_R^-1 M_D (overall sign/phase unphysical).

Models (Dueck-Rodejohann arXiv:1306.4468 Eq. (4)):
    Y_u = r (H + s F + i t_u G),  Y_d = H + F + i G,  Y_D = r (H - 3 s F + i t_D G),  Y_l = H - 3 F + i t_l G,
    M_R = F / r_R.
  'MN' : 10 + 126bar, G = 0, quark-anchored parametrisation of exp 011 (exact reparametrisation of H, F, r, s).
  'FN' : 10 + 126bar + 120, all of H, F, G, r, s, t_u, t_D, t_l, r_R real (CP broken only by the i in front
         of the 120 contributions; Grimus-Kuhbock hep-ph/0612132, used by 1306.4468 Sec. 2.3).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.mixing import extract_angles, standard_matrix  # noqa: E402

TWO_PI = 2 * np.pi
V_EW = 174.0           # GeV, 1306.4468 Yukawa normalisation (checked: their H, F give m_b(GUT) = 1.00 GeV)
ART = ROOT / "artifacts" / "so10"


def dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def left_basis(M):
    w, U = np.linalg.eigh(dag(M) @ M)
    return np.sqrt(np.clip(w, 0, None)), U


def raw_obs(Mu, Md, Me, Mnu):
    """Batched mass matrices -> dict of singular values, CKM/PMNS angles, neutrino spectrum, IO flag."""
    mu, Uu = left_basis(Mu)
    md, Ud = left_basis(Md)
    me, Ue = left_basis(Me)
    mn, Un = left_basis(Mnu)
    q12, q13, q23, dq = extract_angles(dag(Uu) @ Ud)
    a2, b2, c2 = mn[:, 0] ** 2, mn[:, 1] ** 2, mn[:, 2] ** 2
    io = (b2 - a2) > (c2 - b2)
    Un = np.where(io[:, None, None], Un[:, :, [1, 2, 0]], Un)
    l12, l13, l23, dl = extract_angles(dag(Ue) @ Un)
    return dict(mu=mu, md=md, me=me, mn=mn, q12=q12, q13=q13, q23=q23, dq=dq, l12=l12, l13=l13, l23=l23, dl=dl,
                dm21=np.where(io, c2 - b2, b2 - a2), dm3l=c2 - a2, io=io)


# ------------------------------------------------------------------ MN: quark-anchored 10 + 126bar
MN_NAMES = ["lg_u1_u2", "lg_u2_u3", "lg_u3_d3", "lg_d1_d2", "lg_d2_d3", "s12q", "lg_s13q", "lg_s23q", "dq",
            "a2", "a3", "phi1", "phi2", "phi3", "lg_r", "lg_abs_s", "arg_s"]
MN_PER = np.array([0] * 8 + [1] * 6 + [0, 0, 1], bool)


def mn_mats(th):
    """th (N, >=17) -> (Mu, Md, Me, Mnu) in units of y_b (exp 011 gen.so10_mats, type I)."""
    N = th.shape[0]
    u = 10.0 ** np.stack([th[:, 0] + th[:, 1] + th[:, 2], th[:, 1] + th[:, 2], th[:, 2]], 1)
    d = 10.0 ** np.stack([th[:, 3] + th[:, 4], th[:, 4], np.zeros(N)], 1)
    V = standard_matrix(np.arcsin(np.clip(th[:, 5], -1, 1)), np.arcsin(np.clip(10.0 ** th[:, 6], 0, 1)),
                        np.arcsin(np.clip(10.0 ** th[:, 7], 0, 1)), th[:, 8])
    P = np.zeros((N, 3, 3), complex)
    P[:, 0, 0] = 1
    P[:, 1, 1] = np.exp(1j * th[:, 9])
    P[:, 2, 2] = np.exp(1j * th[:, 10])
    U = P @ V
    D = np.zeros((N, 3, 3), complex)
    for i in range(3):
        D[:, i, i] = d[:, i] * np.exp(1j * th[:, 11 + i])
    Md = np.conj(U) @ D @ dag(U)
    Mu = np.zeros((N, 3, 3), complex)
    for i in range(3):
        Mu[:, i, i] = u[:, i]
    r = 10.0 ** th[:, 14]
    s = 10.0 ** th[:, 15] * np.exp(1j * th[:, 16])
    F = (Mu / r[:, None, None] - Md) / (s - 1)[:, None, None]
    Me = Md - 4 * F
    MD = Md - (1 + 3 * s)[:, None, None] * F
    with np.errstate(all="ignore"):
        Mnu = MD @ np.linalg.solve(F, MD)
    return Mu, Md, Me, Mnu


def dr_form_mats(H, F, r, s, G=None, tu=0, tD=0, tl=0):
    """1306.4468 Eq. (4) for a single parameter point -> batched (1,3,3) Yukawas and M_D F^-1 M_D / r^2."""
    G = np.zeros((3, 3)) if G is None else G
    Yu = r * (H + s * F + 1j * tu * G)
    Yd = H + F + 1j * G
    YD = r * (H - 3 * s * F + 1j * tD * G)
    Yl = H - 3 * F + 1j * tl * G
    Mnu = YD.T @ np.linalg.solve(F, YD)
    return [np.asarray(M, complex)[None] for M in (Yu, Yd, Yl, Mnu)]


# ------------------------------------------------------------------ FN: real 10 + 126bar + 120
# Dimensionless parameters: H = diag(h1, h2, 1) (largest |h| normalised to 1, overall sign of all Yukawas
# unphysical), F real symmetric (6), G real antisymmetric (3), r > 0 (overall sign of M_u unphysical), s, t_u,
# t_D, t_l. Every real entry x = SC * sinh(u): smooth through 0, logarithmic for |u| > 2.
FN_NAMES = ["h1", "h2", "F11", "F12", "F13", "F22", "F23", "F33", "G12", "G13", "G23", "r", "s", "tu", "tD", "tl"]
FN_PER = np.zeros(16, bool)
FN_SC = np.array([1e-7] * 11 + [1e-3, 1e-4, 1e-4, 1e-4, 1e-4])


def fn_values(th):
    v = FN_SC * np.sinh(th)
    v[:, 11] = np.abs(v[:, 11])
    return v


def fn_mats(th):
    v = fn_values(th)
    N = th.shape[0]
    H = np.zeros((N, 3, 3))
    H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = v[:, 0], v[:, 1], 1.0
    F = np.zeros((N, 3, 3))
    k = 2
    for i in range(3):
        for j in range(i, 3):
            F[:, i, j] = F[:, j, i] = v[:, k]
            k += 1
    G = np.zeros((N, 3, 3))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        G[:, i, j], G[:, j, i] = v[:, k], -v[:, k]
        k += 1
    c = lambda a: v[:, a][:, None, None]
    r, s, tu, tD, tl = c(11), c(12), c(13), c(14), c(15)
    Mu = r * (H + s * F + 1j * tu * G)
    Md = H + F + 1j * G
    MD = H - 3 * s * F + 1j * tD * G
    Me = H - 3 * F + 1j * tl * G
    with np.errstate(all="ignore"):
        Mnu = np.swapaxes(MD, 1, 2) @ np.linalg.solve(F + 0j, MD)
    return Mu, Md, Me, Mnu


# ------------------------------------------------------------------ targets
Z_NAMES = ["lg_mu_mc", "lg_mc_mt", "lg_md_ms", "lg_ms_mb", "lg_mb_mt", "lg_me_mmu", "lg_mmu_mtau", "lg_mtau_mb",
           "s12q", "lg_s13q", "lg_s23q", "dq", "s12sq_l", "s13sq_l", "s23sq_l", "dl", "lg_dm21_dm3l"]
Z_PHASES = (11, 15)
QL_IDX = list(range(12))
PMNS_IDX = list(range(12, 17))


def z_from_raw(o):
    """17 dimensionless observables (exp 011 z vector)."""
    mu, md, me = o["mu"], o["md"], o["me"]
    with np.errstate(all="ignore"):
        z = np.stack([np.log10(mu[:, 0] / mu[:, 1]), np.log10(mu[:, 1] / mu[:, 2]),
                      np.log10(md[:, 0] / md[:, 1]), np.log10(md[:, 1] / md[:, 2]), np.log10(md[:, 2] / mu[:, 2]),
                      np.log10(me[:, 0] / me[:, 1]), np.log10(me[:, 1] / me[:, 2]), np.log10(me[:, 2] / md[:, 2]),
                      np.sin(o["q12"]), np.log10(np.sin(o["q13"])), np.log10(np.sin(o["q23"])), o["dq"],
                      np.sin(o["l12"]) ** 2, np.sin(o["l13"]) ** 2, np.sin(o["l23"]) ** 2, o["dl"],
                      np.log10(o["dm21"] / o["dm3l"])], 1)
    return z


# 1306.4468 Tables 2 and 3: non-SUSY "no RGE" fit at M_GUT = 2e16 GeV (masses in GeV, Δm² in eV², δ in rad).
# These are the published fit inputs, transcribed for the reproduction only (not used as measured data).
DR_NAMES = ["md", "ms", "mb", "mu", "mc", "mt", "s12q", "s23q", "s13q", "dCKM", "dm21", "dm31", "s12sq", "s23sq",
            "s13sq", "me", "mmu", "mtau"]
DR_VAL = np.array([0.00114, 0.022, 1.0, 0.00048, 0.235, 74.0, 0.2246, 0.042, 0.0035, 1.2153, 7.5e-5, 2.47e-3,
                   0.30, 0.41, 0.023, 0.46965e-3, 0.09915, 1.68558])
DR_ERR = np.array([0.000495, 0.0065, 0.04, 0.000185, 0.0345, 3.85, 0.0011, 0.0013, 0.0003, 0.0576, 0.185e-5,
                   0.0685e-3, 0.013, 0.031, 0.0023, 0.02348e-3, 0.00496, 0.08428])
# 1306.4468 Table 6, "MN, no RGE" best-fit column (same order as DR_NAMES)
DR_MN_BEST = np.array([0.00067, 0.02406, 1.00309, 0.00048, 0.24243, 73.6931, 0.22462, 0.04204, 0.00350, 1.21930,
                       7.50e-5, 2.47e-3, 0.30039, 0.40631, 0.02262, 4.697e-4, 9.914e-2, 1.686])


def dr_vec(o, yscale, nuscale):
    """18 DR observables. yscale: GeV per unit of the matrices' singular values; nuscale: eV per unit of Mnu."""
    ys, ns = yscale[:, None], nuscale[:, None]
    md, mu, me = o["md"] * ys, o["mu"] * ys, o["me"] * ys
    mn = o["mn"] * ns
    a2, b2, c2 = mn[:, 0] ** 2, mn[:, 1] ** 2, mn[:, 2] ** 2
    dm21 = np.where(o["io"], c2 - b2, b2 - a2)
    dm31 = np.where(o["io"], -(c2 - a2), c2 - a2)
    return np.stack([md[:, 0], md[:, 1], md[:, 2], mu[:, 0], mu[:, 1], mu[:, 2],
                     np.sin(o["q12"]), np.sin(o["q23"]), np.sin(o["q13"]), o["dq"], dm21, dm31,
                     np.sin(o["l12"]) ** 2, np.sin(o["l23"]) ** 2, np.sin(o["l13"]) ** 2,
                     me[:, 0], me[:, 1], me[:, 2]], 1)


def wrap(d):
    return (d + np.pi) % TWO_PI - np.pi


def load_ours(ordering="NO"):
    ref = json.loads((HERE / "obs_gut.json").read_text())
    k = "" if ordering == "NO" else "_IO"
    return np.array(ref["z" + k]), np.array(ref["sigma" + k])


class Problem:
    """Residual function r(theta) (B, m) for a model / target / observable subset. Penalises the wrong ordering."""

    def __init__(self, model, target, idx=None, ordering="NO", sigma_override=None):
        self.model, self.target, self.ordering = model, target, ordering
        if model == "MN":
            self.base_names, self.per = MN_NAMES, MN_PER
        else:
            self.base_names, self.per = FN_NAMES, FN_PER
        if target == "DR":
            self.val, self.err = DR_VAL, DR_ERR
            self.names = self.base_names + ["lg_yscale", "lg_nuscale"]
            self.per = np.concatenate([self.per, [False, False]])
            self.obs_names = DR_NAMES
            self.phase_idx = (9,)
        else:
            self.val, self.err = load_ours(ordering)
            self.names = list(self.base_names)
            self.obs_names = Z_NAMES
            self.phase_idx = Z_PHASES
        if sigma_override is not None:
            self.err = np.asarray(sigma_override, float)
        self.idx = list(range(len(self.val))) if idx is None else list(idx)
        self.n = len(self.names)

    def mats(self, th):
        return mn_mats(th) if self.model == "MN" else fn_mats(th)

    def predict(self, th):
        o = raw_obs(*self.mats(th[:, :len(self.base_names)]))
        if self.target == "DR":
            y = dr_vec(o, 10.0 ** th[:, -2], 10.0 ** th[:, -1])
        else:
            y = z_from_raw(o)
        return y, o

    def pulls(self, th):
        y, o = self.predict(th)
        d = y - self.val
        for j in self.phase_idx:
            d[:, j] = wrap(d[:, j])
        p = d / self.err
        return p, o

    def residuals(self, th):
        p, o = self.pulls(th)
        r = p[:, self.idx]
        wrong = o["io"] if self.ordering == "NO" else ~o["io"]
        uses_nu = any(self.obs_names[j] in ("dm21", "dm31", "s12sq", "s23sq", "s13sq", "s12sq_l", "s13sq_l",
                                            "s23sq_l", "dl", "lg_dm21_dm3l") for j in self.idx)
        if uses_nu:
            r = np.concatenate([r, (30.0 * wrong)[:, None]], 1)
        r = np.where(np.isfinite(r), r, 1e4)
        return np.clip(r, -1e4, 1e4)

    def chi2(self, th):
        r = self.residuals(th)
        return (r ** 2).sum(1)


def project(X, lo, hi, per):
    X = np.where(per, np.mod(X, TWO_PI), X)
    return np.where(per, X, np.clip(X, lo, hi))


def batched_lm(prob, X0, lo, hi, n_iter=300, h=1e-6, lam0=1e-2, tol=1e-10, verbose=False):
    """Levenberg-Marquardt on B starts at once (forward-difference Jacobian, one trial step per iteration)."""
    X = project(X0.copy(), lo, hi, prob.per)
    B, n = X.shape
    R = prob.residuals(X)
    C = (R ** 2).sum(1)
    lam = np.full(B, lam0)
    stall = np.zeros(B, int)
    eye = np.eye(n)
    for it in range(n_iter):
        act = stall < 25
        if not act.any():
            break
        ia = np.where(act)[0]
        Xa = X[ia]
        Xp = (Xa[:, None, :] + h * eye[None]).reshape(-1, n)
        Rp = prob.residuals(Xp).reshape(len(ia), n, -1)
        J = (Rp - R[ia][:, None, :]) / h            # (b, n, m)
        A = J @ np.swapaxes(J, 1, 2)                 # (b, n, n)
        g = J @ R[ia][:, :, None]                    # (b, n, 1)
        dA = np.einsum("bii->bi", A)
        M = A + (lam[ia][:, None] * (dA + 1e-9))[:, :, None] * eye[None]
        try:
            step = -np.linalg.solve(M, g)[:, :, 0]
        except np.linalg.LinAlgError:
            step = -np.stack([np.linalg.lstsq(M[k], g[k], rcond=None)[0][:, 0] for k in range(len(ia))])
        step = np.where(np.isfinite(step), step, 0.0)
        step = np.clip(step, -1.0, 1.0)
        Xn = project(Xa + step, lo, hi, prob.per)
        Rn = prob.residuals(Xn)
        Cn = (Rn ** 2).sum(1)
        better = Cn < C[ia]
        rel = (C[ia] - Cn) / np.maximum(C[ia], 1e-30)
        ib = ia[better]
        X[ib], R[ib], C[ib] = Xn[better], Rn[better], Cn[better]
        lam[ib] = np.maximum(lam[ib] / 3, 1e-9)
        lam[ia[~better]] = np.minimum(lam[ia[~better]] * 4, 1e9)
        small = (~better) | (rel < tol)
        stall[ia] = np.where(small, stall[ia] + 1, 0)
        if verbose and (it + 1) % 50 == 0:
            print(f"    it {it + 1} best {C.min():.3f} median {np.median(C):.2f} active {act.sum()}", flush=True)
    return X, C


def mn_box(target, ref_z=None):
    """Prior/search box for MN. Quark windows are centred on the target's quark data (as exp 011)."""
    if target == "DR":
        lmu_mc, lmc_mt = np.log10(0.00048 / 0.235), np.log10(0.235 / 74.0)
        lmd_ms, lms_mb, lmb_mt = np.log10(0.00114 / 0.022), np.log10(0.022 / 1.0), np.log10(1.0 / 74.0)
        c9, c10 = np.log10(0.0035), np.log10(0.042)
    else:
        c = ref_z
        lmu_mc, lmc_mt, lmd_ms, lms_mb, lmb_mt, c9, c10 = c[0], c[1], c[2], c[3], c[4], c[9], c[10]
    box = [(lmu_mc - 0.5, lmu_mc + 0.5), (lmc_mt - 0.5, lmc_mt + 0.5), (-lmb_mt - 0.5, -lmb_mt + 0.5),
           (lmd_ms - 0.5, lmd_ms + 0.5), (lms_mb - 0.5, lms_mb + 0.5),
           (0.15, 0.30), (c9 - 0.4, c9 + 0.4), (c10 - 0.4, c10 + 0.4), (0, TWO_PI)] + [(0, TWO_PI)] * 5 + \
          [(0.0, 4.0), (-2.0, 2.0), (0, TWO_PI)]
    if target == "DR":
        box += [(np.log10(V_EW) - 3, np.log10(V_EW) + 1), (-14, 4)]
    b = np.array(box, float)
    return b[:, 0], b[:, 1]


def fn_box(target):
    b = np.array([(-16.6, 16.6)] * 2 + [(-25, 25)] * 9 + [(0, 15), (-15, 15), (-15, 15), (-15, 15), (-15, 15)], float)
    lo, hi = b[:, 0], b[:, 1]
    if target == "DR":
        lo, hi = np.concatenate([lo, [-6, -14]]), np.concatenate([hi, [4, 4]])
    return lo, hi


def mn_start_from_data(prob, rng, B, lo, hi):
    """Random starts: quark block at the target's data (exact in this parametrisation), the rest uniform."""
    X = lo + (hi - lo) * rng.random((B, len(lo)))
    if prob.target == "DR":
        v = DR_VAL
        X[:, 0], X[:, 1] = np.log10(v[3] / v[4]), np.log10(v[4] / v[5])
        X[:, 2] = np.log10(v[5] / v[2])
        X[:, 3], X[:, 4] = np.log10(v[0] / v[1]), np.log10(v[1] / v[2])
        X[:, 5], X[:, 6], X[:, 7], X[:, 8] = v[6], np.log10(v[8]), np.log10(v[7]), v[9]
        X[:, 17] = np.log10(v[2] / V_EW) + np.log10(V_EW)  # yscale = m_b / (y_b-unit singular value 1)
    else:
        z = prob.val
        X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4] = z[0], z[1], -z[4], z[2], z[3]
        X[:, 5], X[:, 6], X[:, 7], X[:, 8] = z[8], z[9], z[10], z[11]
    return X

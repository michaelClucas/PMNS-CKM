"""Exp 020 core: fixed-tau quark fits with precomputed modular forms, batched Levenberg-Marquardt.

Modules modforms.py, models.py, search.py, lead_scan.py are verbatim copies from exp 017.
Scale and conventions as exp 017 (M_Z, SM, no running of the model; Yukawa ratios from src.rg).
Quark mass-ratio errors: PDG relative errors in quadrature, floor 2% on m_u/m_c, m_d/m_s, m_s/m_b
(light-quark ratios) and exp 017's 1% floor on m_c/m_t.
"""
from __future__ import annotations

import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, ROOT]

import numpy as np  # noqa: E402

import models as md  # noqa: E402
import modforms as mf  # noqa: E402

ART = os.path.join(ROOT, "artifacts", "fixedtau")
os.makedirs(ART, exist_ok=True)

LIGHT_FLOOR = 0.02
QKEYS = ("lr_uc", "lr_ct", "lr_ds", "lr_sb")
CKMKEYS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")
LA_LO, LA_HI = -14.0, 4.0
G_LO, G_HI = -6.0, 6.0


@functools.lru_cache(maxsize=None)
def quark_targets():
    """Same as search.targets (exp 017) for the quark sector, with the 2% light-quark floor."""
    import search as se
    from src.nulls import Universe
    T = se.targets(Universe.real("NO"), "NO")
    out = {}
    for k in QKEYS:
        v, e = T["ratio"][k]
        if k != "lr_ct":
            e = max(e, LIGHT_FLOOR)
        out[k] = (v, e, e)
    for k in CKMKEYS:
        out[k] = tuple(T["gauss"][k])
    return out


def row_values(N, qrep, rowtypes, taus):
    """{(w, r): (n_tau, n_c, 3) complex} for the given row types at the tau points."""
    taus = np.atleast_1d(np.asarray(taus, complex))
    return {(w, r): mf.eval_series(mf.yukawa_row_structures(N, w, r, qrep), taus, N)
            for (w, r, n) in rowtypes}


def sector_matrix(rowset, vals, it, la, ex):
    """vals from row_values; it (B,) tau index; la (B, 2); ex (B, n_extra). Returns (B, 3, 3)."""
    B = la.shape[0]
    L = np.concatenate([np.zeros((B, 1)), la], 1)
    rows = []
    ie = 0
    for i, (w, r, n) in enumerate(rowset):
        v = vals[(w, r)][it]  # (B, n, 3)
        row = v[:, 0]
        if n == 2:
            row = row + ex[:, ie:ie + 1] * v[:, 1]
            ie += 1
        rows.append(np.exp(L[:, i:i + 1]) * row)
    return np.stack(rows, 1)


def quark_obs_fixed(a, vals, it, P):
    nu_, nd_ = md.n_extra(a.u), md.n_extra(a.d)
    Mu = sector_matrix(a.u, vals, it, P[:, 0:2], P[:, 4:4 + nu_])
    Md = sector_matrix(a.d, vals, it, P[:, 2:4], P[:, 4 + nu_:4 + nu_ + nd_])
    Uu, mu = md.left_diag(Mu)
    Ud, mdd = md.left_diag(Md)
    s12, s13, s23, d = md.mixing_obs(np.conj(np.swapaxes(Uu, 1, 2)) @ Ud)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"lr_uc": np.log(mu[:, 0] / mu[:, 1]), "lr_ct": np.log(mu[:, 1] / mu[:, 2]),
                "lr_ds": np.log(mdd[:, 0] / mdd[:, 1]), "lr_sb": np.log(mdd[:, 1] / mdd[:, 2]),
                "ckm_s12": s12, "ckm_s13": s13, "ckm_s23": s23, "ckm_delta_rad": d}


def residuals(o, T):
    """(B, 8) pulls: 4 log mass ratios, CKM s12 s13 s23 (asymmetric Gaussian, PDG), delta (wrapped)."""
    out = []
    for k in QKEYS + CKMKEYS:
        v, ep, em = T[k]
        d = o[k] - v
        if k == "ckm_delta_rad":
            d = (d + np.pi) % (2 * np.pi) - np.pi
        out.append(d / np.where(d > 0, ep, em))
    R = np.stack(out, 1)
    return np.nan_to_num(R, nan=1e4, posinf=1e4, neginf=-1e4)


def n_par(a):
    return 4 + md.n_extra(a.u) + md.n_extra(a.d)


def box(a):
    p = n_par(a)
    lo = np.array([LA_LO] * 4 + [G_LO] * (p - 4))
    hi = np.array([LA_HI] * 4 + [G_HI] * (p - 4))
    return lo, hi


def lm_batch(fun, P0, lo, hi, iters=40, h=1e-6, lam0=1e-2):
    """Projected Levenberg-Marquardt on a batch of starts. fun(P (B,p)) -> residuals (B, m)."""
    P = np.clip(P0.copy(), lo, hi)
    B, p = P.shape
    R = fun(P)
    C = (R ** 2).sum(1)
    lam = np.full(B, lam0)
    I = np.eye(p)
    for _ in range(iters):
        Ps = np.repeat(P[:, None, :], p, 1) + h * I[None]
        Rs = fun(Ps.reshape(B * p, p)).reshape(B, p, -1)
        J = (Rs - R[:, None, :]) / h  # (B, p, m) = dR/dP transposed
        A = J @ np.swapaxes(J, 1, 2)
        g = J @ R[:, :, None]
        diag = np.einsum("bii->bi", A)
        Ad = A + (lam[:, None] * (diag + 1e-9))[:, :, None] * I[None]
        try:
            step = -np.linalg.solve(Ad, g)[:, :, 0]
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(Ad.reshape(-1, p), g.reshape(-1), rcond=None)[0].reshape(B, p)
        step = np.nan_to_num(step)
        Pn = np.clip(P + step, lo, hi)
        Rn = fun(Pn)
        Cn = (Rn ** 2).sum(1)
        ok = Cn < C
        P[ok], R[ok], C[ok] = Pn[ok], Rn[ok], Cn[ok]
        lam = np.where(ok, lam / 3, lam * 4)
        lam = np.clip(lam, 1e-9, 1e9)
    return P, C


def fit_fixed_tau(a, vals, n_tau, rng, n_samp=1500, n_starts=12, iters=40):
    """Best chi2 per tau index for assignment a, couplings only. Returns (chi2 (n_tau,), P (n_tau, p))."""
    T = quark_targets()
    lo, hi = box(a)
    p = len(lo)
    best_c = np.full(n_tau, np.inf)
    best_P = np.zeros((n_tau, p))
    for t in range(n_tau):
        Z = rng.uniform(lo, hi, size=(n_samp, p))
        it = np.full(n_samp, t)
        c = (residuals(quark_obs_fixed(a, vals, it, Z), T) ** 2).sum(1)
        S = Z[np.argsort(c)[:n_starts]]
        f = lambda Q: residuals(quark_obs_fixed(a, vals, np.full(len(Q), t), Q), T)
        P, C = lm_batch(f, S, lo, hi, iters=iters)
        j = int(np.argmin(C))
        best_c[t], best_P[t] = C[j], P[j]
    return best_c, best_P


def quark_obs_tau(a, tau, P):
    """Observables at arbitrary tau (B,) (forms evaluated directly)."""
    return md.quark_obs(a, np.asarray(tau, complex), P)

"""Frozen bi-large (Cabibbo-seed) grammar pipeline for the criterion-4 protocol (exp 030, part B).

Grammar (exp 027 PR-1 plus the RMSV forms of exp 027 quickfit.py):
  U_PMNS = U_l(lambda, A, phi)^dag U_nu
  CDSV part (1902.08962 forms I and II generalised as in exp 027 grammar_pilot.py):
    U_l in {form I  (SO(10)): R23(t23q) F R12(t12q) F^*,
            form II (SU(5)):  F R12(t12q)^T F^* R23(t23q)^T},  F = diag(e^{-i phi/2}, e^{i phi/2}, 1),
    sin t12q = lambda, sin t23q = A lambda^2;
    U_nu = R23 U13(delta_nu) R12 with each sin t_ij^nu in {n lambda, 1 - n lambda : n = 1..4} and
    delta_nu in {0, pi}: 8^3 x 2 x 2 = 2048 configurations, one continuous parameter phi.
  RMSV part (1410.3658, types 1-4 as in exp 027 verify_bilarge.rmsv): sin t13^nu = lambda,
    sin t12^nu = sin t23^nu = psi lambda, delta_nu = 0, U_l type 1-4; two continuous parameters
    (psi in [1, 4.4], phi). 4 configurations.
  Total 2052 configurations -> 12 bits; + 1 ordering bit (ordering chosen inside the pipeline).
  Options with any sine outside (0, 1) are invalid on that universe and skipped.

Search (frozen):
  1. Coarse: phi on a 1 deg grid (360 points); RMSV psi on a 0.025 grid in [1, 4.4] (capped where
     psi lambda < 1). Table chi^2 per ordering from the universe's translated NuFIT tables
     (vectorised copy of Universe.pmns_delta_chi2; checked to agree to 1e-9 by test_vector_chi2).
  2. Per ordering, the 6 configurations with the highest coarse p_global (chi^2_global, dof
     4 - k_cont) are polished: CDSV by bounded Brent on phi within +-1 deg of the grid point,
     RMSV by Nelder-Mead on (psi, phi) from the grid point; objective Universe.scorer(...).chi2.
  3. Each polished configuration is scored with Universe.scorer (src.evaluate.Scorer) with
     use = (s12sq, s13sq, s23sq, dcp_deg), inputs = (ckm_lambda, ckm_A), k = k_cont + 12,
     k_discrete = 12, ordering_selected=True.
  4. Selection: the result with the largest p_global over both orderings (ties: lower chi2_global).
Measured numbers are read only from the universe (load_ckm lambda and A; PMNS through its scorer).
"""
from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2 as _chi2dist

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as _data  # noqa: E402
from src.mixing import extract_angles  # noqa: E402

USE = ("s12sq", "s13sq", "s23sq", "dcp_deg")
INPUTS = ("ckm_lambda", "ckm_A")
N_CONFIGS = 2048 + 4
K_BITS = math.ceil(math.log2(N_CONFIGS))  # 12
PHI = np.radians(np.arange(360.0))
PSI = np.arange(1.0, 4.4 + 1e-9, 0.025)
N_POLISH = 6
OPTS = [("l", n) for n in range(1, 5)] + [("1-l", n) for n in range(1, 5)]
CDSV_CONFIGS = [(o12, o13, o23, d) for o12, o13, o23 in itertools.product(OPTS, repeat=3) for d in (0.0, np.pi)]


# ---------------------------------------------------------------- matrices (batched over leading axes)

def _r12(t):
    t = np.asarray(t, float)
    c, s = np.cos(t), np.sin(t)
    M = np.zeros(t.shape + (3, 3), complex)
    M[..., 0, 0] = c; M[..., 0, 1] = s; M[..., 1, 0] = -s; M[..., 1, 1] = c; M[..., 2, 2] = 1
    return M


def _r23(t):
    t = np.asarray(t, float)
    c, s = np.cos(t), np.sin(t)
    M = np.zeros(t.shape + (3, 3), complex)
    M[..., 1, 1] = c; M[..., 1, 2] = s; M[..., 2, 1] = -s; M[..., 2, 2] = c; M[..., 0, 0] = 1
    return M


def _F(phi):
    phi = np.asarray(phi, float)
    M = np.zeros(phi.shape + (3, 3), complex)
    M[..., 0, 0] = np.exp(-0.5j * phi); M[..., 1, 1] = np.exp(0.5j * phi); M[..., 2, 2] = 1
    return M


def _T(M):
    return np.swapaxes(M, -1, -2)


def ul(lam, A, phi, form):
    """Charged-lepton rotation. form: 'I', 'II' (CDSV) or RMSV type 1-4 (exp 027 verify_bilarge)."""
    phi = np.asarray(phi, float)
    t12, t23 = np.arcsin(lam), np.arcsin(A * lam ** 2)
    F = _F(phi)
    R12 = _r12(np.full(phi.shape, t12))
    R23 = _r23(np.full(phi.shape, t23))
    if form in ("I", 2):
        return R23 @ F @ R12 @ F.conj()
    if form in ("II", 4):
        return F @ _T(R12) @ F.conj() @ _T(R23)
    if form == 1:
        return F @ R12 @ F.conj()
    if form == 3:
        return F.conj() @ _T(R12) @ F
    raise ValueError(form)


def unu(s12, s13, s23, d):
    """R23 U13(d) R12 with PDG phase on the (1,3) entry; broadcasts."""
    s12, s13, s23, d = np.broadcast_arrays(*(np.asarray(x, float) for x in (s12, s13, s23, d)))
    t13 = np.arcsin(s13)
    c13 = np.cos(t13)
    U13 = np.zeros(s13.shape + (3, 3), complex)
    U13[..., 0, 0] = c13; U13[..., 1, 1] = 1; U13[..., 2, 2] = c13
    U13[..., 0, 2] = s13 * np.exp(-1j * d); U13[..., 2, 0] = -s13 * np.exp(1j * d)
    return _r23(np.arcsin(s23)) @ U13 @ _r12(np.arcsin(s12))


def _val(o, lam):
    return o[1] * lam if o[0] == "l" else 1 - o[1] * lam


def config_name(cfg):
    if cfg[0] == "RMSV":
        return f"RMSV type {cfg[1]}"
    o12, o13, o23, d, form = cfg
    f = lambda o: f"{o[1]}l" if o[0] == "l" else f"1-{o[1]}l"
    return f"CDSV-{form} s12={f(o12)} s13={f(o13)} s23={f(o23)} d_nu={'pi' if d else '0'}"


def model_U(cfg, lam, A, theta):
    """U_PMNS of one configuration at parameters theta = {phi[, psi]}."""
    if cfg[0] == "RMSV":
        Un = unu(theta["psi"] * lam, lam, theta["psi"] * lam, 0.0)
        Ul = ul(lam, A, theta["phi"], cfg[1])
    else:
        o12, o13, o23, d, form = cfg
        Un = unu(_val(o12, lam), _val(o13, lam), _val(o23, lam), d)
        Ul = ul(lam, A, theta["phi"], form)
    return np.conj(_T(Ul)) @ Un


def obs_of(U):
    t12, t13, t23, d = extract_angles(U)
    return np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2, np.degrees(d)


# ---------------------------------------------------------------- vectorised translated-table chi^2

def vector_chi2(u, ordering, s12, s13, s23, dcp):
    """Universe.pmns_delta_chi2(ordering, s12, s13, s23, dcp) for arrays (same tables, same maps)."""
    m = u.map_to_real(ordering, np.asarray(s12), np.asarray(s13), np.asarray(s23), np.asarray(dcp))
    a = u.analysis
    c = _data.pmns_profile_1d(ordering, "s12sq", m["s12sq"], a) + _data.pmns_profile_1d(ordering, "s13sq", m["s13sq"], a)
    x, y, z = _data._shifted_grid_2d(a, ordering, "T23-DCP")
    uu = np.asarray(m["s23sq"], float)
    vv = (np.asarray(m["dcp_deg"], float) + 180.0) % 360.0 - 180.0
    out = (uu < x[0]) | (uu > x[-1]) | (vv < y[0]) | (vv > y[-1]) | ~np.isfinite(uu)
    i = np.clip(np.searchsorted(x, uu) - 1, 0, len(x) - 2)
    j = np.clip(np.searchsorted(y, vv) - 1, 0, len(y) - 2)
    tx = (uu - x[i]) / (x[i + 1] - x[i])
    ty = (vv - y[j]) / (y[j + 1] - y[j])
    z2 = ((1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i + 1, j]
          + (1 - tx) * ty * z[i, j + 1] + tx * ty * z[i + 1, j + 1])
    return np.where(out, np.inf, c + z2)


# ---------------------------------------------------------------- coarse grid

def _coarse(lam, A):
    """Groups (cfgs, k_cont, thetas, obs) of the valid configurations; obs arrays have shape
    (n_cfg, n_grid) so one table call per ordering scores a whole group."""
    groups = []
    valid = [(c, [_val(o, lam) for o in c[:3]]) for c in CDSV_CONFIGS]
    valid = [(c, s) for c, s in valid if all(0 < v < 1 for v in s)]
    if valid:
        S = np.array([s for _, s in valid])
        D = np.array([c[3] for c, _ in valid])
        Un = unu(S[:, 0], S[:, 1], S[:, 2], D)  # (C, 3, 3)
        cfgs, obs = [], []
        for f in ("I", "II"):
            Ud = np.conj(_T(ul(lam, A, PHI, f)))  # (G, 3, 3)
            o = obs_of(Ud[None] @ Un[:, None])  # (C, G)
            obs.append(o)
            cfgs += [c + (f,) for c, _ in valid]
        obs = tuple(np.concatenate([o[i] for o in obs]) for i in range(4))
        groups.append((cfgs, 1, [{"phi": p} for p in PHI], obs))
    psi = PSI[PSI * lam < 1]
    if lam < 1 and len(psi):
        P, F = np.meshgrid(psi, PHI, indexing="ij")
        Un = unu(P * lam, lam, P * lam, 0.0)
        obs = []
        for t in (1, 2, 3, 4):
            Ud = np.conj(_T(ul(lam, A, F, t)))
            obs.append(tuple(v.ravel() for v in obs_of(Ud @ Un)))
        obs = tuple(np.stack([o[i] for o in obs]) for i in range(4))
        th = [{"psi": a, "phi": b} for a, b in zip(P.ravel(), F.ravel())]
        groups.append(([("RMSV", t) for t in (1, 2, 3, 4)], 2, th, obs))
    return groups


def _pglob(c2, kc):
    return np.where(np.isfinite(c2), _chi2dist.sf(c2, 4 - kc), 0.0)


def search(u, n_polish=N_POLISH, return_all=False):
    """The frozen pipeline body. Returns the selected src.evaluate.EvalResult (or None)."""
    ck = u.load_ckm()
    lam, A = float(ck["lambda"]["value"]), float(ck["A"]["value"])
    groups = _coarse(lam, A)
    results = []
    for o in _data.ORDERINGS:
        sc = u.scorer(USE, o, inputs=INPUTS)
        off = sc.offset
        cand = []
        for cfgs, kc, th, ob in groups:
            c2 = vector_chi2(u, o, *ob)  # (n_cfg, n_grid)
            ib = np.argmin(c2, axis=1)
            cb = c2[np.arange(len(cfgs)), ib]
            for r in np.flatnonzero(np.isfinite(cb)):
                cand.append((float(_pglob(cb[r] + off, kc)), -float(cb[r]), cfgs[r], kc, th[ib[r]]))
        cand.sort(key=lambda r: (r[0], r[1]), reverse=True)
        for _, _, cfg, kc, th0 in cand[:n_polish]:
            def pred(th, cfg=cfg):
                s12, s13, s23, d = obs_of(model_U(cfg, lam, A, th))
                return {"s12sq": float(s12), "s13sq": float(s13), "s23sq": float(s23), "dcp_deg": float(d)}

            def f(th):
                try:
                    c = sc.chi2(pred(th))
                except ValueError:
                    return 1e9
                return c if np.isfinite(c) else 1e9

            if kc == 1:
                p0 = th0["phi"]
                r = minimize_scalar(lambda x: f({"phi": x}), bounds=(p0 - np.radians(1), p0 + np.radians(1)),
                                    method="bounded", options={"xatol": 1e-7})
                th = {"phi": float(r.x)}
                if f(th) > f(th0):
                    th = th0
            else:
                psi_hi = min(4.4, (1 - 1e-9) / lam)

                def g(x):
                    if not (1.0 <= x[0] <= psi_hi):
                        return 1e9
                    return f({"psi": x[0], "phi": x[1]})
                r = minimize(g, [th0["psi"], th0["phi"]], method="Nelder-Mead",
                             options={"xatol": 1e-7, "fatol": 1e-9, "initial_simplex":
                                      [[th0["psi"], th0["phi"]], [th0["psi"] + 0.0125, th0["phi"]],
                                       [th0["psi"], th0["phi"] + np.radians(0.5)]]})
                th = {"psi": float(r.x[0]), "phi": float(r.x[1])} if r.fun <= f(th0) else th0
            res = sc(pred(th), k=kc + K_BITS, k_discrete=K_BITS, ordering_selected=True)
            results.append((res, cfg, th))
    if not results:
        return None
    results.sort(key=lambda r: (-(r[0].p_global if np.isfinite(r[0].p_global) else -1), r[0].chi2_global))
    if return_all:
        return results
    return results[0][0]


def pipeline(u):
    """Module-level pipeline for src.nulls (picklable)."""
    return search(u)

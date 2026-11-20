"""Exp 043 generator G43 (claims.md), proxy residuals, fitter and official scoring.

Structure spec (dict, JSON-serialisable):
  Z      list of positions from P_ALL names ("11", "12", "13", "22", "23"); 33 always present
  e      "S" | "P" | "F"; ce: list of len |Z|+1 (order Z..., "33"), floats in CSET or null (continuous)
  nu     None (Y1 fits) | "D" | "I1" | "IR" | "II" | "F"
  cnu    list len |Z|+1 (D, I1, IR, II), cR list len |Z|+1 (IR only)
  extra  list of positions (names from NU_POS) for P_nu (lepton-only complex entries)
Parameter vector: [u (|Z|), log10|d| (|Z|), arg d (|Z|), charged-lepton continuous Clebsches (signed log),
                   neutrino lepton-only parameters]
"""
from __future__ import annotations

import os
import sys
import pathlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import least_squares, minimize

from src import data as _data
from src import evaluate as ev
from src.yukawa import observables_batch

POS = {"11": (0, 0), "12": (0, 1), "13": (0, 2), "22": (1, 1), "23": (1, 2), "33": (2, 2)}
P_ALL = ("11", "12", "13", "22", "23")
NU_POS = ("11", "12", "13", "22", "23", "33")
CSET = np.array([s * v for v in (0.5, 1, 1.5, 2, 3, 4.5, 6) for s in (1, -1)], float)

QUARK_KEYS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")
Y1 = QUARK_KEYS + ("mr_e_mu", "mr_mu_tau")
Y2 = Y1 + ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")
MR6 = ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau")
FLOOR = 0.02
BAD = 30.0

# ---------------------------------------------------------------- data targets (from data/ via src)
_CKM = _data.load_ckm()
_PM = _data.load_pmns("NO")
_DMB = ev._dm_best("NO", _data.DEFAULT_ANALYSIS)["dm_ratio"]
_DMS = float(ev._dm_sigma("NO", _data.DEFAULT_ANALYSIS)["dm_ratio"])
_MRV, _MRN, _MRC = ev.mass_ratio_targets()


def _mr_whitener(keys):
    idx = [_MRN.index(k) for k in keys]
    C = np.asarray(_MRC)[np.ix_(idx, idx)] + FLOOR ** 2 * np.eye(len(idx))
    L = np.linalg.cholesky(C)
    return np.linalg.inv(L), np.log([_MRV[k] for k in keys])


def _wrap(x, per):
    return (x + per / 2) % per - per / 2


class Proxy:
    """Residual vector for a key set (claims.md 'Fitting and pass rule')."""

    def __init__(self, keys):
        self.keys = tuple(keys)
        self.mr = tuple(k for k in self.keys if k in MR6)
        self.Winv, self.mrlog = _mr_whitener(self.mr) if self.mr else (None, None)

    def __call__(self, o):
        """o: dict of (B,) arrays -> (B, R) residuals."""
        res = []
        for k in self.keys:
            if k.startswith("ckm_"):
                r = _CKM[k[4:]]
                p = o[k]
                if k == "ckm_delta_rad":
                    d = _wrap(p - r["value"], 2 * np.pi)
                else:
                    d = p - r["value"]
                res.append(d / np.where(d > 0, r["err_plus"], r["err_minus"]))
            elif k in ("s12sq", "s13sq", "s23sq"):
                r = _PM[k]
                d = o[k] - r["value"]
                res.append(d / np.where(d > 0, r["err_plus"], r["err_minus"]))
            elif k == "dcp_deg":
                r = _PM[k]
                d = _wrap(o[k] - r["value"], 360.0)
                res.append(d / np.where(d > 0, r["err_plus"], r["err_minus"]))
            elif k == "dm_ratio":
                with np.errstate(all="ignore"):
                    res.append((np.log(np.abs(o[k])) - np.log(_DMB)) / (_DMS / _DMB))
        if self.mr:
            with np.errstate(all="ignore"):
                z = np.log(np.stack([o[k] for k in self.mr], 1)) - self.mrlog
            res.extend(list((z @ self.Winv.T).T))
        R = np.stack(res, 1)
        return np.where(np.isfinite(R), R, BAD)


_SC = {}


def scorer(keys):
    keys = tuple(keys)
    if keys not in _SC:
        _SC[keys] = ev.Scorer(keys, "NO", mass_floor=FLOOR)
    return _SC[keys]


# ---------------------------------------------------------------- structures

def n_lepton(spec):
    nz = len(spec["Z"])
    le = sum(c is None for c in spec["ce"])
    nu = spec.get("nu")
    if nu is None:
        ln = 0
    elif nu == "F":
        ln = 2 * nz
    else:
        ln = 2 * len(spec.get("extra") or [])
    return le, ln


def layout(spec):
    nz = len(spec["Z"])
    le, ln = n_lepton(spec)
    lo = [-7.5] * nz + [-7.0] * nz + [-2 * np.pi] * nz + [-3.0] * le
    hi = [7.5] * nz + [0.5] * nz + [2 * np.pi] * nz + [3.0] * le
    if spec.get("nu") == "F":
        lo += [-6.0] * nz + [-2 * np.pi] * nz
        hi += [1.5] * nz + [2 * np.pi] * nz
    elif ln:
        J = ln // 2
        lo += [-5.0] * J + [-2 * np.pi] * J
        hi += [1.0] * J + [2 * np.pi] * J
    return np.array(lo), np.array(hi)


def k_cont(spec):
    return len(layout(spec)[0])


def _slog(v, off):
    return np.sign(v) * 10.0 ** (np.abs(v) - off)


def _place(B, vals, Z, diag33=None):
    M = np.zeros((B, 3, 3), complex)
    for j, z in enumerate(Z):
        a, b = POS[z]
        M[:, a, b] = vals[:, j]
        M[:, b, a] = vals[:, j]
    if diag33 is not None:
        M[:, 2, 2] = diag33
    return M


def _dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def build(spec, X):
    """X: (B, k) -> Yu, Yd, Ye, m_nu (B, 3, 3)."""
    X = np.atleast_2d(X)
    B = len(X)
    Z = list(spec["Z"])
    nz = len(Z)
    u = _slog(X[:, :nz], 7.0)
    d = 10.0 ** X[:, nz:2 * nz] * np.exp(1j * X[:, 2 * nz:3 * nz])
    Yu = _place(B, u.astype(complex), Z, 1.0)
    Yd = _place(B, d, Z, 1.0)
    i = 3 * nz
    ce = []
    for c in spec["ce"]:
        if c is None:
            ce.append(_slog(X[:, i], 2.0))
            i += 1
        else:
            ce.append(np.full(B, float(c)))
    ce = np.stack(ce, 1)
    ZZ = Z + ["33"]
    Ce = _place(B, ce.astype(complex), ZZ)
    Ye = Ce * Yd
    nu = spec.get("nu")
    if nu is None:
        mnu = np.tile(np.diag([1e-3, 1e-2, 1.0]).astype(complex), (B, 1, 1))
        return Yu, Yd, Ye, mnu
    with np.errstate(all="ignore"):
        if nu == "F":
            mod = 10.0 ** X[:, i:i + nz]
            ph = X[:, i + nz:i + 2 * nz]
            mnu = _place(B, mod * np.exp(1j * ph), Z, 1.0)
            return Yu, Yd, Ye, mnu
        Cn = _place(B, np.tile(np.asarray(spec["cnu"], complex), (B, 1)), ZZ)
        if nu == "II":
            mnu = Cn * Yd
        else:
            Yn = Cn * Yu
            if nu == "D":
                w, U = np.linalg.eigh(Yn @ _dag(Yn))
                s = np.sqrt(np.clip(w, 0, None))
                mnu = U @ (s[:, :, None] * np.swapaxes(U, 1, 2))
            elif nu == "I1":
                mnu = -np.conj(Yn) @ _dag(Yn)
            elif nu == "IR":
                CR = _place(B, np.tile(np.asarray(spec["cR"], complex), (B, 1)), ZZ)
                MR = CR * Yd
                try:
                    mnu = -np.conj(Yn) @ np.linalg.solve(MR, _dag(Yn))
                except np.linalg.LinAlgError:
                    mnu = np.full((B, 3, 3), np.nan, complex)
            else:
                raise ValueError(nu)
        extra = spec.get("extra") or []
        if extra:
            sv = np.linalg.norm(mnu, ord=2, axis=(1, 2))
            mnu = mnu / np.where(sv > 0, sv, 1.0)[:, None, None]
            J = len(extra)
            w = 10.0 ** X[:, i:i + J] * np.exp(1j * X[:, i + J:i + 2 * J])
            mnu = mnu + _place(B, w, extra)
        mnu = 0.5 * (mnu + np.swapaxes(mnu, 1, 2))
    return Yu, Yd, Ye, mnu


def _angles(U):
    """PDG angles (sines) and delta in [0, 2pi) from |U| and J (rephasing invariant; J and the |U31|
    relation of src.mixing's convention)."""
    A = np.abs(U)
    s13 = np.clip(A[:, 0, 2], 0, 1)
    c13 = np.sqrt(np.clip(1 - s13 ** 2, 1e-300, None))
    s12 = np.clip(A[:, 0, 1] / c13, 0, 1)
    s23 = np.clip(A[:, 1, 2] / c13, 0, 1)
    c12, c23 = np.sqrt(1 - s12 ** 2), np.sqrt(1 - s23 ** 2)
    J = np.imag(U[:, 0, 0] * U[:, 1, 1] * np.conj(U[:, 0, 1]) * np.conj(U[:, 1, 0]))
    den = c12 * s12 * c23 * s23 * c13 ** 2 * s13
    a = s12 * s23 * c12 * c23 * s13
    with np.errstate(all="ignore"):
        cosd = (s12 ** 2 * s23 ** 2 + c12 ** 2 * c23 ** 2 * s13 ** 2 - A[:, 2, 0] ** 2) / (2 * a)
        sind = J / den
    d = np.mod(np.arctan2(sind, cosd), 2 * np.pi)
    return s12, s13, s23, d


def _lrot(M):
    w, U = np.linalg.eigh(M @ _dag(M))
    return U, np.sqrt(np.clip(w, 0, None))


def fast_observables(Yu, Yd, Ye, m):
    """Same keys/conventions as observables_batch(ordering="NO") for the scored keys."""
    with np.errstate(all="ignore"):
        Uu, yu = _lrot(Yu)
        Ud, yd = _lrot(Yd)
        Ue, ye = _lrot(Ye)
        w, Un = np.linalg.eigh(_dag(m) @ m)  # columns of the Takagi factor up to phases
        mn = np.sqrt(np.clip(w, 0, None))
        V = _dag(Uu) @ Ud
        U = _dag(Ue) @ Un
        q12, q13, q23, qd = _angles(V)
        l12, l13, l23, ld = _angles(U)
        o = {"ckm_s12": q12, "ckm_s13": q13, "ckm_s23": q23, "ckm_delta_rad": qd,
             "s12sq": l12 ** 2, "s13sq": l13 ** 2, "s23sq": l23 ** 2, "dcp_deg": np.degrees(ld),
             "mr_u_c": yu[:, 0] / yu[:, 1], "mr_c_t": yu[:, 1] / yu[:, 2], "mr_d_s": yd[:, 0] / yd[:, 1],
             "mr_s_b": yd[:, 1] / yd[:, 2], "mr_e_mu": ye[:, 0] / ye[:, 1], "mr_mu_tau": ye[:, 1] / ye[:, 2],
             "dm_ratio": (mn[:, 1] ** 2 - mn[:, 0] ** 2) / np.abs(mn[:, 2] ** 2 - mn[:, 0] ** 2)}
    return o


def observe(spec, X, exact=False):
    Yu, Yd, Ye, m = build(spec, X)
    if not exact:
        return fast_observables(Yu, Yd, Ye, m)
    with np.errstate(all="ignore"):
        return observables_batch(Yu, Yd, Ye, m, ordering="NO", sym_tol=np.inf)


# ---------------------------------------------------------------- fitting

def _fd_jac(spec, proxy, x, h=1e-6):
    k = len(x)
    X = np.tile(x, (k + 1, 1))
    X[1:] += h * np.eye(k)
    R = proxy(observe(spec, X))
    return (R[1:] - R[0]).T / h


def fit_one(spec, keys, x0, max_nfev=400):
    proxy = Proxy(keys)
    lo, hi = layout(spec)
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    f = lambda x: proxy(observe(spec, x[None]))[0]
    j = lambda x: _fd_jac(spec, proxy, x)
    try:
        r = least_squares(f, x0, jac=j, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev,
                          xtol=1e-10, ftol=1e-10, gtol=1e-10)
        return r.x, float(2 * r.cost), float(np.max(np.abs(r.fun)))
    except Exception:
        return x0, float("inf"), float("inf")


def pred_of(spec, x):
    o = observe(spec, np.asarray(x)[None], exact=True)
    return {k: float(v[0]) for k, v in o.items()}


def official(spec, keys, x):
    p = pred_of(spec, x)
    sc = scorer(keys)
    try:
        r = sc(p, k=min(k_cont(spec), 10 ** 6))
        return {"chi2": float(r.chi2), "max_pull": float(r.max_abs_pull), "fit_ok": bool(r.fit_ok),
                "p_global": float(r.p_global), "pulls": {k: float(v) for k, v in r.pulls.items()},
                "pred": {k: p[k] for k in keys}}
    except Exception as e:  # non-finite prediction
        return {"chi2": float("inf"), "max_pull": float("inf"), "fit_ok": False, "p_global": 0.0,
                "pulls": {}, "pred": {k: p.get(k) for k in keys}, "error": str(e)[:200]}


def polish(spec, keys, x, maxfev=2500):
    sc = scorer(keys)
    lo, hi = layout(spec)

    def g(z):
        z = np.clip(z, lo, hi)
        try:
            return sc.chi2_soft(pred_of(spec, z), include_finite=True)
        except Exception:
            return 1e12
    r = minimize(g, x, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-8, "fatol": 1e-8,
                                                       "adaptive": True})
    return np.clip(r.x, lo, hi)


def random_lepton(spec, rng):
    """Random start values for the non-quark parameters."""
    nz = len(spec["Z"])
    lo, hi = layout(spec)
    k = len(lo)
    out = []
    le, ln = n_lepton(spec)
    for _ in range(le):
        out.append(rng.choice([-1, 1]) * (rng.uniform(np.log10(0.3), 1.0) + 2.0))
    if spec.get("nu") == "F":
        out += list(rng.uniform(-3, 0.5, nz)) + list(rng.uniform(-np.pi, np.pi, nz))
    elif ln:
        J = ln // 2
        out += list(rng.uniform(-3, 0, J)) + list(rng.uniform(-np.pi, np.pi, J))
    assert 3 * nz + len(out) == k
    return np.array(out)


def fit_structure(spec, keys, bank, n_starts, rng, quark_starts=None):
    """Multi-start proxy fit; official decision; polish near-passes. Returns a record."""
    best = None
    starts = []
    for s in range(n_starts):
        q = quark_starts[s % len(quark_starts)] if quark_starts is not None else bank[rng.integers(len(bank))]
        x0 = np.concatenate([np.asarray(q, float), random_lepton(spec, rng)])
        x, c, mr = fit_one(spec, keys, x0)
        starts.append(c)
        if best is None or c < best[1]:
            best = (x, c, mr)
    x, c, mr = best
    off = official(spec, keys, x)
    polished = False
    if mr < 3 and not off["max_pull"] < 2:
        x2 = polish(spec, keys, x)
        off2 = official(spec, keys, x2)
        polished = True
        if (off2["max_pull"], off2["chi2"]) < (off["max_pull"], off["chi2"]):
            x, off = x2, off2
    le, ln = n_lepton(spec)
    return {"spec": spec, "keys": "Y1" if tuple(keys) == Y1 else ("Y2" if tuple(keys) == Y2 else list(keys)),
            "k_cont": k_cont(spec), "l_e": le, "l_nu": ln, "proxy_chi2": c, "proxy_maxres": mr,
            "start_chi2": starts, "polished": polished, "x": list(map(float, x)),
            "chi2": off["chi2"], "max_pull": off["max_pull"], "fit_ok": off["fit_ok"],
            "p_global": off["p_global"], "pass": bool(np.isfinite(off["chi2"]) and off["max_pull"] < 2),
            "pulls": off["pulls"], "pred": off["pred"]}

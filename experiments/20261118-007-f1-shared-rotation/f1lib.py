"""F1 shared-rotation search: U_PMNS = A Phi B, Phi = diag(1, e^{i alpha}, e^{i beta}).

Every composition in the catalog reduces to that form with (A, B) built from V_CKM and a
fixed matrix X:
  left forms   W Phi X          W in {V, V^dag, V^T, V^*}
  right forms  X Phi W          W in {V, V^dag, V^T, V^*}
  sandwiches   V Phi X V^dag, V X Phi V^dag, V^dag Phi X V, V^dag X Phi V,
               V^T Phi X V^*, V^* Phi X V^T
So U_ij(alpha, beta) = sum_k A_ik phi_k B_kj = sum_k T_ijk phi_k with T_ijk = A_ik B_kj,
which makes grid evaluation one einsum per configuration batch.

Physical observables (s12^2, s13^2, s23^2, delta) are invariant under left/right diagonal
rephasing, so Phi on the outside of any product is unphysical and is not enumerated.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
from scipy.optimize import minimize

from src import data
from src.mixing import extract_angles

R2, R3, R6 = np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0)
GOLD = (1 + np.sqrt(5.0)) / 2


def _std_real(t12, t13=0.0, t23=np.pi / 4):
    """Real PDG standard matrix (delta = 0)."""
    s12, c12, s13, c13, s23, c23 = (np.sin(t12), np.cos(t12), np.sin(t13), np.cos(t13),
                                    np.sin(t23), np.cos(t23))
    return np.array([[c12 * c13, s12 * c13, s13],
                     [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
                     [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13]],
                    dtype=complex)


def _rot(i, j, t):
    R = np.eye(3, dtype=complex)
    R[i, i] = R[j, j] = np.cos(t)
    R[i, j] = np.sin(t)
    R[j, i] = -np.sin(t)
    return R


TM_THETA = np.pi / 12  # fixed by hand before looking at the fit; not tuned (see README)


def x_catalog():
    """Finite catalog of symmetry-motivated X (name -> 3x3 unitary)."""
    w = np.exp(2j * np.pi / 3)
    tbm = _std_real(np.arcsin(1 / R3))
    cat = {
        "I": np.eye(3, dtype=complex),                      # with row/col perms: all 6 permutation matrices
        "TBM": tbm,                                         # S4 / A4 (Harrison-Perkins-Scott)
        "BM": _std_real(np.pi / 4),                         # S4 bimaximal
        "GR1": _std_real(np.arctan(1 / GOLD)),              # A5 golden ratio, tan t12 = 1/phi
        "GR2": _std_real(np.arccos(GOLD / 2)),              # cos t12 = phi/2 (t12 = 36 deg)
        "HEX": _std_real(np.pi / 6),                        # hexagonal / D12
        "TM1": tbm @ _rot(1, 2, TM_THETA),                  # keeps TBM column 1
        "TM2": tbm @ _rot(0, 2, TM_THETA),                  # keeps TBM column 2
        "DEM": np.array([[1 / R2, 1 / R6, 1 / R3],           # diagonalizes the democratic matrix
                         [-1 / R2, 1 / R6, 1 / R3],
                         [0, -2 / R6, 1 / R3]], dtype=complex),
        "UW": np.array([[1, 1, 1], [1, w, w * w], [1, w * w, w]], dtype=complex) / R3,  # A4 / Delta(27) magic (Cabibbo-Wolfenstein)
    }
    for k, X in cat.items():
        assert np.allclose(X @ X.conj().T, np.eye(3), atol=1e-12), k
    return cat


PERMS = list(itertools.permutations(range(3)))


def _pmat(p):
    P = np.zeros((3, 3), dtype=complex)
    P[np.arange(3), list(p)] = 1
    return P


def _forms(V, X):
    Vd, VT, Vc = V.conj().T, V.T, V.conj()
    W = {"V": V, "Vd": Vd, "VT": VT, "Vc": Vc}
    out = {}
    for n, M in W.items():
        out[f"{n}.P.X"] = (M, X)
    for n, M in W.items():
        out[f"X.P.{n}"] = (X, M)
    out["V.P.X.Vd"] = (V, X @ Vd)
    out["V.X.P.Vd"] = (V @ X, Vd)
    out["Vd.P.X.V"] = (Vd, X @ V)
    out["Vd.X.P.V"] = (Vd @ X, V)
    out["VT.P.X.Vc"] = (VT, X @ Vc)
    out["Vc.P.X.VT"] = (Vc, X @ VT)
    return out


FORM_NAMES = list(_forms(np.eye(3), np.eye(3)).keys())


def raw_configs():
    """List of (form, xname, row_perm_idx, col_perm_idx). X_eff = P_r X P_c."""
    cat = x_catalog()
    return [(f, x, r, c) for f in FORM_NAMES for x in cat for r in range(6) for c in range(6)]


def build_T(V, configs):
    """T tensors (C, 3, 3, 3) for the given V and configuration list."""
    cat = x_catalog()
    P = [_pmat(p) for p in PERMS]
    T = np.empty((len(configs), 3, 3, 3), dtype=complex)
    cache = {}
    for n, (f, x, r, c) in enumerate(configs):
        key = (x, r, c)
        if key not in cache:
            cache[key] = _forms(V, P[r] @ cat[x] @ P[c])
        A, B = cache[key][f]
        T[n] = A[:, None, :] * B.T[None, :, :]  # T_ijk = A_ik B_kj
    return T


def phase_grid(n):
    a = 2 * np.pi * np.arange(n) / n
    aa, bb = np.meshgrid(a, a, indexing="ij")
    return np.stack([aa.ravel(), bb.ravel()], -1)


def phis(ab):
    ab = np.atleast_2d(ab)
    return np.stack([np.ones(len(ab)), np.exp(1j * ab[:, 0]), np.exp(1j * ab[:, 1])], -1)


def observables(U):
    """(s12sq, s13sq, s23sq, dcp_deg) arrays from (..., 3, 3), via src.mixing.extract_angles."""
    t12, t13, t23, d = extract_angles(U)
    return np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2, np.degrees(d)


def fast_observables(e1, e2, e3, m1, m2, m3, tol=1e-12, delta_tol=1e-15):
    """Same as observables() from the first two rows only (same formulas as
    extract_angles; agreement checked in check_scorer.py)."""
    a = [np.abs(z) ** 2 for z in (e1, e2, e3, m1, m3)]
    E1, E2, E3, M1, M3 = a
    row = E1 + E2
    generic = row > tol * tol
    c13sq = np.where(generic, row, 1.0)
    s13 = E3
    s12 = np.where(generic, E2 / c13sq, 0.0)
    s23 = np.where(generic, M3 / c13sq, M1 / np.maximum(M1 + np.abs(m2) ** 2, 1e-300))
    c12, c23 = 1 - s12, 1 - s23
    J = np.imag(e1 * m2 * np.conj(e2) * np.conj(m1))
    n_sin = np.where(generic, J / c13sq, 0.0)
    n_cos = 0.5 * (M1 - s12 * c23 - c12 * s23 * s13)
    D = np.sqrt(np.clip(s12 * c12 * s23 * c23 * s13, 0, None))
    d = np.degrees(np.mod(np.arctan2(n_sin, n_cos), 2 * np.pi))
    d = np.where(D > delta_tol, d, 0.0)
    return s12, s13, s23, d


# ---------------------------------------------------------------- dedup

def fingerprints(T, n_grid=12, decimals=6):
    """Hash of the multiset of rephasing invariants (|U_ij|^2, J) over an n_grid^2 phase grid.

    Two configurations related by an integer-affine reparametrization of (alpha, beta)
    (e.g. Phi P = P Phi', sign flips = pi shifts) have the same multiset on a grid whose
    step divides pi/2, so they get the same hash. Used only to count distinct trials.
    """
    ph = phis(phase_grid(n_grid))
    out = []
    for s in range(0, len(T), 256):
        U = np.einsum("cijk,gk->cgij", T[s:s + 256], ph)
        a2 = np.abs(U) ** 2
        J = np.imag(U[..., 0, 0] * U[..., 1, 1] * np.conj(U[..., 0, 1]) * np.conj(U[..., 1, 0]))
        feat = np.concatenate([a2.reshape(*a2.shape[:2], 9), J[..., None]], -1)
        feat = np.round(feat, decimals) + 0.0
        for f in feat:
            f = f[np.lexsort(f.T[::-1])]
            out.append(hashlib.sha1(f.tobytes()).hexdigest())
    return out


# ---------------------------------------------------------------- scoring

class TableChi2:
    """Vectorized copy of src.data.pmns_delta_chi2(ordering, s12sq, s13sq, s23sq, dcp_deg)
    (method='auto', relative_to='local'): 2D (s23sq, dcp) table + 1D s12sq, s13sq profiles.
    Checked against the scalar loader function in check_scorer.py."""

    def __init__(self, ordering, analysis=data.DEFAULT_ANALYSIS):
        self.ordering = ordering
        x, y, z = data._grid_2d(analysis, ordering, "T23-DCP")
        self.x, self.y, self.z = x, y, z - z.min()
        self.p1 = {}
        for t in ("T12", "T13"):
            a, b = data._profile_1d(analysis, ordering, t)
            self.p1[t] = (a, b - b.min())

    def __call__(self, s12, s13, s23, dcp):
        s12, s13, s23, dcp = np.broadcast_arrays(*(np.asarray(v, float) for v in (s12, s13, s23, dcp)))
        d = (dcp + 180.0) % 360.0 - 180.0
        x, y, z = self.x, self.y, self.z
        i = np.clip(np.searchsorted(x, s23) - 1, 0, len(x) - 2)
        j = np.clip(np.searchsorted(y, d) - 1, 0, len(y) - 2)
        tx = (s23 - x[i]) / (x[i + 1] - x[i])
        ty = (d - y[j]) / (y[j + 1] - y[j])
        out = ((1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i + 1, j]
               + (1 - tx) * ty * z[i, j + 1] + tx * ty * z[i + 1, j + 1])
        bad = (s23 < x[0]) | (s23 > x[-1]) | (d < y[0]) | (d > y[-1])
        for t, v in (("T12", s12), ("T13", s13)):
            a, b = self.p1[t]
            out = out + np.interp(v, a, b)
            bad |= (v < a[0]) | (v > a[-1])
        return np.where(bad, np.inf, out)


class GaussChi2:
    """Gaussian chi^2 with asymmetric NuFIT 1 sigma widths of `ordering`, centred on `target`
    (s12sq, s13sq, s23sq, dcp_deg). delta difference wrapped to [-180, 180)."""

    def __init__(self, target, ordering):
        P = data.load_pmns(ordering)
        self.t = np.asarray(target, float)
        self.ep = np.array([P[k]["err_plus"] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg")])
        self.em = np.array([P[k]["err_minus"] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg")])

    def __call__(self, s12, s13, s23, dcp):
        tot = 0.0
        for n, v in enumerate((s12, s13, s23, dcp)):
            r = np.asarray(v, float) - self.t[n]
            if n == 3:
                r = (r + 180.0) % 360.0 - 180.0
            tot = tot + (r / np.where(r >= 0, self.ep[n], self.em[n])) ** 2
        return tot


# ---------------------------------------------------------------- search

STRUCT_VALUES = np.radians([0, 90, 180, 270, 120, 240])  # Z4 and Z3 phases


def struct_points():
    return np.array([(a, b) for a in STRUCT_VALUES for b in STRUCT_VALUES])


def grid_eval(T, scorers, ab, chunk=128):
    """chi^2 on phase points ab for all configs: dict name -> (C, G)."""
    phT = np.ascontiguousarray(phis(ab).T)
    T6 = np.ascontiguousarray(T[:, :2].reshape(len(T), 6, 3))
    out = {k: np.empty((len(T), len(ab))) for k in scorers}
    for s in range(0, len(T), chunk):
        U = T6[s:s + chunk] @ phT  # (c, 6, G): rows e1 e2 e3 m1 m2 m3
        obs = fast_observables(*(U[:, i] for i in range(6)))
        for k, f in scorers.items():
            out[k][s:s + chunk] = f(*obs)
    return out


def predict(Tc, ab):
    U = Tc @ phis(ab)[0]
    return tuple(float(v) for v in fast_observables(*U[:2].ravel()))


def polish(Tc, scorer, starts):
    """Nelder-Mead from each start; returns (chi2, alpha, beta) of the best."""
    def f(x):
        return float(scorer(*predict(Tc, x)))
    best = (np.inf, np.nan, np.nan)
    for x0 in starts:
        if not np.isfinite(f(x0)):
            continue
        r = minimize(f, x0, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 600})
        if r.fun < best[0]:
            best = (float(r.fun), float(np.mod(r.x[0], 2 * np.pi)), float(np.mod(r.x[1], 2 * np.pi)))
    return best


def grid_starts(row, ab, n_starts, sep=np.radians(30)):
    """Up to n_starts grid points by ascending chi^2, at least `sep` apart on the torus."""
    order = np.argsort(row)
    chosen = []
    for g in order:
        if not np.isfinite(row[g]):
            break
        if all(np.max(np.abs(np.angle(np.exp(1j * (ab[g] - ab[c]))))) >= sep for c in chosen):
            chosen.append(g)
        if len(chosen) == n_starts:
            break
    return [ab[g] for g in chosen]


# Search settings shared by the real-data run and every fake universe.
N_GRID = 48          # 7.5 deg phase grid
TOP_POLISH = 20      # configs polished per universe and scorer (by grid chi^2)
N_STARTS = 3         # Nelder-Mead starts per polished config


def search(T, scorers, Ts_struct=None):
    """The pipeline applied identically to real data and to every fake universe.

    T: (C,3,3,3) for the deduplicated continuous-phase configs.
    Ts_struct: (C',3,3,3) for the raw configs used by the structured-phase variant.
    Returns dict scorer -> {best_cont, cfg_cont, ab_cont, best_struct, cfg_struct, pt_struct}.
    """
    ab = phase_grid(N_GRID)
    G = grid_eval(T, scorers, ab)
    res = {}
    for k, f in scorers.items():
        gmin = G[k].min(1)
        top = np.argsort(gmin)[:TOP_POLISH]
        best = (np.inf, -1, (np.nan, np.nan))
        for c in top:
            if not np.isfinite(gmin[c]):
                continue
            chi2, a, b = polish(T[c], f, grid_starts(G[k][c], ab, N_STARTS))
            if chi2 < best[0]:
                best = (chi2, int(c), (a, b))
        res[k] = {"best_cont": best[0], "cfg_cont": best[1], "ab_cont": best[2]}
    if Ts_struct is not None:
        sp = struct_points()
        S = grid_eval(Ts_struct, scorers, sp)
        for k in scorers:
            c, g = np.unravel_index(np.argmin(S[k]), S[k].shape)
            res[k].update(best_struct=float(S[k][c, g]), cfg_struct=int(c), pt_struct=int(g))
    return res


# ---------------------------------------------------------------- CKM

def ckm_central():
    c = data.load_ckm()
    from src.mixing import wolfenstein_matrix
    return wolfenstein_matrix(c["lambda"]["value"], c["A"]["value"],
                              c["rhobar"]["value"], c["etabar"]["value"])


def ckm_samples(rng, n):
    from src.haar import split_normal
    from src.mixing import wolfenstein_matrix
    c = data.load_ckm()
    p = [split_normal(rng, c[k]["value"], c[k]["err_plus"], c[k]["err_minus"], n)
         for k in ("lambda", "A", "rhobar", "etabar")]
    return wolfenstein_matrix(*p)


def ckm_like_prior(rng):
    """Hierarchy-preserving fake CKM: Wolfenstein params from broad uniform priors."""
    from src.mixing import wolfenstein_matrix
    lam = rng.uniform(0.10, 0.35)
    A = rng.uniform(0.5, 1.2)
    rb, eb = rng.uniform(-0.5, 0.5, 2)
    return wolfenstein_matrix(lam, A, rb, eb), (lam, A, rb, eb)

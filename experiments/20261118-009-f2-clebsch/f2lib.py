"""F2 Clebsch search: U_PMNS = V_e^dag . D(phi) . U_nu,  D(phi) = diag(1, e^{i phi}, e^{i phi}).

V_e is a CKM-like charged-lepton rotation whose angles are tied to the quark angles by
GUT Clebsch-type factors:
    th12e = c12 * theta_C,  th23e = c23 * theta_cb,  th13e = c13 * theta_ub,  delta_e = delta_q
    c in C0 = {0} u C,  C = {1/6, 2/9, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 9/2, 6}
C = the SU(5)/Pati-Salam Yukawa-ratio Clebsch factors {1, 3/2, 2, 3, 9/2, 6} of
Antusch-Spinrath arXiv:0902.4644 plus their inverses (messenger-mass Clebsches enter
inversely, Antusch-King-Spinrath arXiv:1311.0877). The set is closed under c -> 1/c, so
th = theta_q * c and th = theta_q / c enumerate the same lattice.
Product order of V_e: "std" = R23 U13(delta_e) R12 (PDG order, = src.mixing.standard_matrix)
or "rev" = R12 U13(delta_e) R23.
U_nu: LO patterns TBM, BM, GR1, GR2, HEX (src.baselines.lo_matrix = R23(pi/4) R12(th12nu)),
or TM-type U_TBM R23(th_nu, phi_nu) (TM1) / U_TBM R13(th_nu, phi_nu) (TM2) with th_nu
fitted and phi_nu in {0, 90, 180, 270 deg} (CP-symmetry values).

U_ij(phi) = P_ij + e^{i phi} Q_ij with P_ij = A_i0 B_0j, Q_ij = sum_{k=1,2} A_ik B_kj
(A = V_e^dag, B = U_nu), so a phase scan is cheap. Only the 7 entries needed for the
observables are kept: e1 e2 e3 m1 m2 m3 t3.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import pathlib
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import data  # noqa: E402
from src.baselines import lo_matrix, r12, r13, r23  # noqa: E402
from src.mixing import extract_angles  # noqa: E402

C_SET = [Fraction(1, 6), Fraction(2, 9), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1),
         Fraction(3, 2), Fraction(2), Fraction(3), Fraction(9, 2), Fraction(6)]
C0 = [Fraction(0)] + C_SET
PATTERNS = ["TBM", "BM", "GR1", "GR2", "HEX"]
TM_TYPES = [(t, p) for t in ("TM1", "TM2") for p in (0.0, 90.0, 180.0, 270.0)]
STRUCT_PHI = np.radians([0.0, 90.0, 180.0, 270.0])
ENTRIES = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)]
TH_NU_GRID = np.linspace(0.0, 0.5, 101)  # TM th_nu grid (0.005 rad)
GUT_SCALE = 2e16


# ---------------------------------------------------------------- configs

def configs():
    """List of config dicts. Tiers (nested searches):
    A   th12e only (c23 = c13 = 0), 5 LO patterns              60
    B   th12e (c in C0) + th23e (c in C), 2 V_e orders          1320
    C   th12e, th23e (C0) + th13e (c in C), 2 orders           15840
    TM  TM-type U_nu (8) with th12e only                        96
    """
    out = []
    for nu in PATTERNS:
        for c12 in C0:
            out.append(dict(tier="A", nu=nu, c12=c12, c23=Fraction(0), c13=Fraction(0), order="std"))
    for nu in PATTERNS:
        for c12 in C0:
            for c23 in C_SET:
                for order in ("std", "rev"):
                    out.append(dict(tier="B", nu=nu, c12=c12, c23=c23, c13=Fraction(0), order=order))
    for nu in PATTERNS:
        for c12 in C0:
            for c23 in C0:
                for c13 in C_SET:
                    for order in ("std", "rev"):
                        out.append(dict(tier="C", nu=nu, c12=c12, c23=c23, c13=c13, order=order))
    for t, ph in TM_TYPES:
        for c12 in C0:
            out.append(dict(tier="TM", nu=f"{t}@{int(ph)}", c12=c12, c23=Fraction(0), c13=Fraction(0),
                            order="std"))
    for i, c in enumerate(out):
        c["id"] = i
        c["name"] = f"{c['nu']}|c12={c['c12']}|c23={c['c23']}|c13={c['c13']}|{c['order']}"
    return out


TIER_SETS = {"A": ("A",), "AB": ("A", "B"), "ABC": ("A", "B", "C"), "ALL": ("A", "B", "C", "TM")}


def tier_counts(cfgs):
    return {k: sum(c["tier"] in v for c in cfgs) for k, v in TIER_SETS.items()}


# ---------------------------------------------------------------- matrices

def tm_matrix(kind, th, phi_deg):
    R = r23 if kind == "TM1" else r13
    return lo_matrix("TBM") @ R(th, np.radians(phi_deg))


def u_nu(nu, th_nu=None):
    if nu in PATTERNS:
        return lo_matrix(nu)
    kind, ph = nu.split("@")
    return tm_matrix(kind, th_nu, float(ph))


def v_e(t12, t13, t23, delta, order):
    if order == "std":
        return r23(t23) @ r13(t13, delta) @ r12(t12)
    return r12(t12) @ r13(t13, delta) @ r23(t23)


def e_angles(cfg, ckm):
    """ckm = (theta_C, theta_ub, theta_cb, delta_q) radians."""
    tC, tub, tcb, dq = ckm
    return float(cfg["c12"]) * tC, float(cfg["c13"]) * tub, float(cfg["c23"]) * tcb, dq


def full_matrix(cfg, ckm, phi, th_nu=None):
    t12, t13, t23, d = e_angles(cfg, ckm)
    A = v_e(t12, t13, t23, d, cfg["order"]).conj().T
    D = np.diag([1.0, np.exp(1j * phi), np.exp(1j * phi)])
    return A @ D @ u_nu(cfg["nu"], th_nu)


def build_PQ_slow(cfgs, ckm):
    """Rows: one per LO config, and len(TH_NU_GRID) per TM config.
    Returns P, Q (R, 7) complex, row_cfg (R,) config index, row_thnu (R,) (nan for LO)."""
    P, Q, rc, rt = [], [], [], []
    for c in cfgs:
        t12, t13, t23, d = e_angles(c, ckm)
        A = v_e(t12, t13, t23, d, c["order"]).conj().T
        ths = TH_NU_GRID if c["tier"] == "TM" else [np.nan]
        for th in ths:
            B = u_nu(c["nu"], th)
            Pm = np.outer(A[:, 0], B[0, :])
            Qm = A[:, 1:] @ B[1:, :]
            P.append([Pm[i, j] for i, j in ENTRIES])
            Q.append([Qm[i, j] for i, j in ENTRIES])
            rc.append(c["id"])
            rt.append(th)
    return np.array(P), np.array(Q), np.array(rc), np.array(rt)


def _brot(i, j, t, ph=None):
    n = len(t)
    R = np.zeros((n, 3, 3), complex)
    k = 3 - i - j
    R[:, k, k] = 1
    c, s = np.cos(t), np.sin(t)
    e = np.exp(-1j * ph) if ph is not None else 1.0
    R[:, i, i] = c
    R[:, j, j] = c
    R[:, i, j] = s * e
    R[:, j, i] = -s * np.conj(e)
    return R


_STRUCT_CACHE = {}


def _structure(cfgs):
    key = id(cfgs)
    if key not in _STRUCT_CACHE:
        rc, rt, c12, c13, c23, rev, B = [], [], [], [], [], [], []
        for c in cfgs:
            ths = TH_NU_GRID if c["tier"] == "TM" else [np.nan]
            for th in ths:
                rc.append(c["id"])
                rt.append(th)
                c12.append(float(c["c12"]))
                c13.append(float(c["c13"]))
                c23.append(float(c["c23"]))
                rev.append(c["order"] == "rev")
                B.append(u_nu(c["nu"], th))
        _STRUCT_CACHE[key] = tuple(np.array(a) for a in (rc, rt, c12, c13, c23, rev, B))
    return _STRUCT_CACHE[key]


def build_PQ(cfgs, ckm):
    """Vectorized build_PQ_slow (same rows, same order; equality checked in check_pipeline.py)."""
    rc, rt, c12, c13, c23, rev, B = _structure(cfgs)
    tC, tub, tcb, dq = ckm
    R12, R13, R23 = _brot(0, 1, c12 * tC), _brot(0, 2, c13 * tub, np.full(len(rc), dq)), _brot(1, 2, c23 * tcb)
    Ve = np.where(rev[:, None, None], R12 @ R13 @ R23, R23 @ R13 @ R12)
    A = np.conj(np.swapaxes(Ve, 1, 2))
    Pm = A[:, :, 0, None] * B[:, None, 0, :]
    Qm = A[:, :, 1:] @ B[:, 1:, :]
    ii = [e[0] for e in ENTRIES]
    jj = [e[1] for e in ENTRIES]
    return Pm[:, ii, jj], Qm[:, ii, jj], rc, rt


# ---------------------------------------------------------------- observables

def fast_obs(E):
    """E (..., 7) complex entries e1 e2 e3 m1 m2 m3 t3 -> s12sq, s13sq, s23sq, dcp_deg.
    Same formulas as src.mixing.extract_angles (checked in check_pipeline.py)."""
    e1, e2, e3, m1, m2, m3, t3 = (E[..., k] for k in range(7))
    E1, E2, E3, M1, M3, T3 = (np.abs(z) ** 2 for z in (e1, e2, e3, m1, m3, t3))
    row = E1 + E2
    c13sq = np.maximum(row, 1e-300)
    s13 = E3
    s12 = E2 / c13sq
    s23 = M3 / np.maximum(M3 + T3, 1e-300)
    c12, c23 = 1 - s12, 1 - s23
    J = np.imag(e1 * m2 * np.conj(e2) * np.conj(m1))
    n_sin = J / c13sq
    n_cos = 0.5 * (M1 - s12 * c23 - c12 * s23 * s13)
    D = s12 * c12 * s23 * c23 * s13
    d = np.degrees(np.mod(np.arctan2(n_sin, n_cos), 2 * np.pi))
    d = np.where(D > 1e-15, d, 0.0)
    return s12, s13, s23, d


def obs_exact(U):
    t12, t13, t23, d = extract_angles(U)
    return float(np.sin(t12) ** 2), float(np.sin(t13) ** 2), float(np.sin(t23) ** 2), float(np.degrees(d))


# ---------------------------------------------------------------- chi^2

class TableChi2:
    """Vectorized copy of src.data.pmns_delta_chi2(ordering, s12sq, s13sq, s23sq, dcp_deg)
    (method='auto', relative_to='local'): 2D (s23sq, dcp) table + 1D s12sq, s13sq profiles.
    Same code as exp 007 f1lib.TableChi2; checked against the loader in check_pipeline.py."""

    def __init__(self, ordering, analysis=data.DEFAULT_ANALYSIS):
        self.ordering = ordering
        x, y, z = data._grid_2d(analysis, ordering, "T23-DCP")
        self.x, self.y, self.z = x, y, z - z.min()
        self.dx, self.dy = (x[-1] - x[0]) / (len(x) - 1), (y[-1] - y[0]) / (len(y) - 1)
        assert np.allclose(np.diff(x), self.dx) and np.allclose(np.diff(y), self.dy)
        self.p1 = {}
        for t in ("T12", "T13"):
            a, b = data._profile_1d(analysis, ordering, t)
            self.p1[t] = (a, b - b.min())

    def __call__(self, s12, s13, s23, dcp):
        s12, s13, s23, dcp = np.broadcast_arrays(*(np.asarray(v, float) for v in (s12, s13, s23, dcp)))
        d = (dcp + 180.0) % 360.0 - 180.0
        x, y, z = self.x, self.y, self.z
        # the 2D table is a uniform grid (checked in check_pipeline.py): direct index
        i = np.clip(np.floor((s23 - x[0]) / self.dx).astype(np.int64), 0, len(x) - 2)
        j = np.clip(np.floor((d - y[0]) / self.dy).astype(np.int64), 0, len(y) - 2)
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
    """Gaussian chi^2 with the asymmetric NuFIT 1 sigma widths of `ordering`, centred on
    `target` (s12sq, s13sq, s23sq, dcp_deg); delta difference wrapped. As exp 007 null (a)."""

    def __init__(self, target, ordering):
        P = data.load_pmns(ordering)
        keys = ("s12sq", "s13sq", "s23sq", "dcp_deg")
        self.ordering = ordering
        self.t = np.asarray(target, float)
        self.ep = np.array([P[k]["err_plus"] for k in keys])
        self.em = np.array([abs(P[k]["err_minus"]) for k in keys])

    def __call__(self, s12, s13, s23, dcp):
        tot = 0.0
        for n, v in enumerate((s12, s13, s23, dcp)):
            r = np.asarray(v, float) - self.t[n]
            if n == 3:
                r = (r + 180.0) % 360.0 - 180.0
            tot = tot + (r / np.where(r >= 0, self.ep[n], self.em[n])) ** 2
        return tot


# ---------------------------------------------------------------- CKM inputs

def ckm_mz():
    """(theta_C, theta_ub, theta_cb, delta_q) at M_Z from data/ckm.json (PDG standard)."""
    c = data.load_ckm()
    return (float(np.arcsin(c["s12"]["value"])), float(np.arcsin(c["s13"]["value"])),
            float(np.arcsin(c["s23"]["value"])), float(c["delta_rad"]["value"]))


def gut_running(cache=HERE / "gut_ckm.json"):
    """SM running of the CKM angles M_Z -> 2e16 GeV with src.rg (computed once, cached).
    Returns dict with GUT angles and the ratios r12, r13, r23 and delta shift."""
    if cache.exists():
        return json.loads(cache.read_text())
    from src import rg
    lo = rg.ckm_at(rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"])
    hi = rg.ckm_at(GUT_SCALE)
    out = {"mu_gut": GUT_SCALE, "model": "SM",
           "mz": [lo["theta12"], lo["theta13"], lo["theta23"], lo["delta"]],
           "gut": [hi["theta12"], hi["theta13"], hi["theta23"], hi["delta"]]}
    out["r12"], out["r13"], out["r23"] = (hi[k] / lo[k] for k in ("theta12", "theta13", "theta23"))
    out["ddelta"] = hi["delta"] - lo["delta"]
    cache.write_text(json.dumps(out, indent=1))
    return out


def to_gut(ckm, run):
    """Apply the SM running ratios to an (theta_C, theta_ub, theta_cb, delta) tuple.
    The ratios are driven by y_t, y_b and gauge couplings and depend negligibly on the CKM
    angles themselves, so they are reused for fake CKMs (approximation, documented)."""
    tC, tub, tcb, d = ckm
    return tC * run["r12"], tub * run["r13"], tcb * run["r23"], d + run["ddelta"]


def ckm_at_scale(scale):
    if scale == "MZ":
        return ckm_mz()
    g = gut_running()["gut"]  # (t12, t13, t23, delta) from src.rg
    return (g[0], g[1], g[2], g[3])


def ckm_like_prior(rng):
    """Hierarchy-preserving fake CKM (as exp 007): Wolfenstein lambda ~ U[0.10, 0.35],
    A ~ U[0.5, 1.2], rhobar, etabar ~ U[-0.5, 0.5]. Returns (theta_C, theta_ub, theta_cb, delta)."""
    from src.mixing import wolfenstein_to_standard
    lam = rng.uniform(0.10, 0.35)
    A = rng.uniform(0.5, 1.2)
    rb, eb = rng.uniform(-0.5, 0.5, 2)
    t12, t13, t23, d = wolfenstein_to_standard(lam, A, rb, eb)
    return (float(t12), float(t13), float(t23), float(d)), (lam, A, rb, eb)


# ---------------------------------------------------------------- search pipeline

def _obs_at(P, Q, phi):
    E = P + np.exp(1j * phi)[..., None] * Q
    return fast_obs(E)


def grid_scan(P, Q, scorers, nphi=360, chunk=500):
    """Best chi^2 over a phi grid per row, per scorer. Returns dict name -> (best, argphi)."""
    phis = np.arange(nphi) * (2 * np.pi / nphi)
    R = len(P)
    out = {k: (np.empty(R), np.empty(R)) for k in scorers}
    ph = np.exp(1j * phis)[None, :, None]
    for a in range(0, R, chunk):
        b = min(a + chunk, R)
        E = P[a:b, None, :] + ph * Q[a:b, None, :]
        o = fast_obs(E)
        for k, s in scorers.items():
            c = s(*o)
            i = np.argmin(c, axis=1)
            out[k][0][a:b] = c[np.arange(b - a), i]
            out[k][1][a:b] = phis[i]
    return out


def golden_refine(P, Q, scorer, phi0, half=np.radians(1.0), iters=40):
    """Vectorized golden-section refinement of phi in [phi0 - half, phi0 + half]."""
    g = (np.sqrt(5) - 1) / 2
    a, b = phi0 - half, phi0 + half
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = scorer(*_obs_at(P, Q, x1)), scorer(*_obs_at(P, Q, x2))
    for _ in range(iters):
        left = f1 < f2
        a = np.where(left, a, x1)
        b = np.where(left, x2, b)
        xn = np.where(left, b - g * (b - a), a + g * (b - a))
        fn = scorer(*_obs_at(P, Q, xn))
        x1, f1, x2, f2 = (np.where(left, xn, x2), np.where(left, fn, f2),
                          np.where(left, x1, xn), np.where(left, f1, fn))
    x = np.where(f1 < f2, x1, x2)
    f = np.minimum(f1, f2)
    f0 = scorer(*_obs_at(P, Q, phi0))
    better = f0 < f
    return np.mod(np.where(better, phi0, x), 2 * np.pi), np.where(better, f0, f)


def struct_scan(P, Q, scorers):
    """chi^2 at the 4 structured phases: dict name -> (R, 4)."""
    E = P[:, None, :] + np.exp(1j * STRUCT_PHI)[None, :, None] * Q[:, None, :]
    o = fast_obs(E)
    return {k: s(*o) for k, s in scorers.items()}


def search(P, Q, row_cfg, row_tier, scorers, nphi=180, n_refine=100):
    """The pipeline used identically for real data and fakes.
    Fitted phase: grid over phi, then golden refinement of the n_refine best rows (per
    scorer). Structured: exact chi^2 at phi in {0, 90, 180, 270}.
    Returns {scorer: {"fit": {tier: best}, "struct": {tier: best}}}."""
    g = grid_scan(P, Q, scorers, nphi)
    st = struct_scan(P, Q, scorers)
    res = {}
    for k, s in scorers.items():
        best, arg = g[k]
        top = np.argsort(best)[:n_refine]
        _, fr = golden_refine(P[top], Q[top], s, arg[top], half=2 * np.pi / nphi)
        best = best.copy()
        best[top] = np.minimum(best[top], fr)
        sb = st[k].min(axis=1)
        r = {"fit": {}, "struct": {}}
        for tname, tiers in TIER_SETS.items():
            m = np.isin(row_tier, tiers)
            r["fit"][tname] = float(best[m].min())
            r["struct"][tname] = float(sb[m].min())
        res[k] = r
    return res


def obs_cache(P, Q, nphi=180, chunk=500):
    """Grid observables (4, R, nphi) float32 and structured-phase observables (4, R, 4)."""
    phis = np.arange(nphi) * (2 * np.pi / nphi)
    R = len(P)
    G = np.empty((4, R, nphi), np.float32)
    ph = np.exp(1j * phis)[None, :, None]
    for a in range(0, R, chunk):
        b = min(a + chunk, R)
        o = fast_obs(P[a:b, None, :] + ph * Q[a:b, None, :])
        for k in range(4):
            G[k, a:b] = o[k]
    E = P[:, None, :] + np.exp(1j * STRUCT_PHI)[None, :, None] * Q[:, None, :]
    S = np.array(fast_obs(E))
    return phis, G, S


def search_cached(cache, P, Q, row_tier, scorers, n_refine=100, chunk=4000):
    """Same as search() but with precomputed grid observables (for fixed CKM, many targets)."""
    phis, G, S = cache
    R, nphi = G.shape[1], G.shape[2]
    masks = {t: np.isin(row_tier, v) for t, v in TIER_SETS.items()}
    res = {}
    for k, s in scorers.items():
        best, arg = np.empty(R), np.empty(R)
        for a in range(0, R, chunk):
            b = min(a + chunk, R)
            c = s(*(G[j, a:b].astype(float) for j in range(4)))
            i = np.argmin(c, axis=1)
            best[a:b] = c[np.arange(b - a), i]
            arg[a:b] = phis[i]
        top = np.argsort(best)[:n_refine]
        _, fr = golden_refine(P[top], Q[top], s, arg[top], half=2 * np.pi / nphi)
        best[top] = np.minimum(best[top], fr)
        sb = s(*S).min(axis=1)
        res[k] = {"fit": {t: float(best[m].min()) for t, m in masks.items()},
                  "struct": {t: float(sb[m].min()) for t, m in masks.items()}}
    return res

"""Exp 044 grammar G44 (grammar.md): predictive quark textures, builder, proxy, fitter, official scoring.

Texture spec (JSON-serialisable dict):
  t     "H" (Hermitian) | "S" (complex symmetric) | "G" (general complex); same type for u and d
  Z     [u positions, d positions], 1-based strings "ij" (i <= j for H/S); "33" always present
  ties  list of [q, p, r]: |Y_q| = r |Y_p|; q, p entry names "u12", "d23", ...; r in RLIST (string)
  ph    {entry: code} for every physical phase slot: "f" (free) or a code in FCODES
        (H diagonal slots take "0" or "pi" only). Gauge-fixed entries have phase 0 and no slot.
Moduli: the (3,3) entry of each sector is 1 (sector scales are not fitted). Every other nonzero entry
has a free modulus (log10, fit box [-6, 1]) unless it is the q of a tie.
Parameter vector: [log10 moduli of free entries (canonical entry order)] + [free phases (slot order)].
"""
from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

SECT = ("u", "d")
HS_POS = ("11", "12", "13", "22", "23")          # (3,3) always present
G_POS = ("11", "12", "13", "21", "22", "23", "31", "32")
FCODES = ("0", "pi", "+pi/2", "-pi/2", "+2pi/3", "-2pi/3")
FVAL = {"0": 0.0, "pi": math.pi, "+pi/2": math.pi / 2, "-pi/2": -math.pi / 2,
        "+2pi/3": 2 * math.pi / 3, "-2pi/3": -2 * math.pi / 3}
RLIST = ("1", "2", "3", "1/2", "1/3")
KMAX = 6
MAX_TIES = 1
G_MAX_ENTRIES = 4   # per sector, including (3,3)
QUARK_KEYS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")


# ---------------------------------------------------------------- patterns and canonicalisation

def _ij(p):
    return int(p[0]) - 1, int(p[1]) - 1


def _name(i, j):
    return f"{i + 1}{j + 1}"


def _sym(p):
    i, j = _ij(p)
    return _name(min(i, j), max(i, j))


SWAP12 = {0: 1, 1: 0, 2: 2}


def _perm_pos(p, rows, cols, sym):
    i, j = _ij(p)
    q = _name(rows[i], cols[j])
    return _sym(q) if sym else q


def _generic_ok(t, Zu, Zd, rng):
    """Structural check (no data): each sector generically rank 3 with distinct singular values,
    and the generic CKM has all |V_ij| > 0."""
    for _ in range(2):
        Ys = [_rand_matrix(t, Z, rng) for Z in (Zu, Zd)]
        Us = []
        for Y in Ys:
            s = np.linalg.svd(Y, compute_uv=False)
            if s[-1] < 1e-6 * s[0] or np.min(np.diff(np.sort(s))) < 1e-6 * s[0]:
                return False
            w, U = np.linalg.eigh(Y @ Y.conj().T)
            Us.append(U)
        V = np.abs(Us[0].conj().T @ Us[1])
        if np.min(V) < 1e-8:
            return False
    return True


def _rand_matrix(t, Z, rng):
    Y = np.zeros((3, 3), complex)
    for p in Z:
        i, j = _ij(p)
        v = rng.uniform(0.3, 1.0) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        if t == "H":
            if i == j:
                Y[i, i] = v.real if abs(v.real) > 0.05 else 0.3
            else:
                Y[i, j], Y[j, i] = v, np.conj(v)
        elif t == "S":
            Y[i, j] = Y[j, i] = v
        else:
            Y[i, j] = v
    return Y


def sector_patterns(t):
    pos = HS_POS if t in "HS" else G_POS
    out = []
    for n in range(0, len(pos) + 1):
        for c in itertools.combinations(pos, n):
            Z = tuple(sorted(c + ("33",)))
            if t == "G" and len(Z) > G_MAX_ENTRIES:
                continue
            out.append(Z)
    return out


def canon_pair(t, Zu, Zd):
    """Canonical representative of a pattern pair under the simultaneous 1<->2 row swap (H, S: rows
    and columns) and, for G, independent column permutations per sector (right-handed rotations are
    unobservable). Representatives must keep (3,3) nonzero in both sectors."""
    best = None
    rowperms = [{0: 0, 1: 1, 2: 2}, SWAP12]
    colperms = [dict(enumerate(p)) for p in itertools.permutations(range(3))] if t == "G" else [None]
    for rp in rowperms:
        for cu in colperms:
            for cd in colperms:
                if t == "G":
                    a = tuple(sorted(_perm_pos(p, rp, cu, False) for p in Zu))
                    b = tuple(sorted(_perm_pos(p, rp, cd, False) for p in Zd))
                else:
                    a = tuple(sorted(_perm_pos(p, rp, rp, True) for p in Zu))
                    b = tuple(sorted(_perm_pos(p, rp, rp, True) for p in Zd))
                if "33" not in a or "33" not in b:
                    continue
                key = (a, b)
                if best is None or key < best:
                    best = key
    return best


def pattern_pairs(t, seed=440):
    rng = np.random.default_rng(seed)
    pats = sector_patterns(t)
    seen = set()
    out = []
    for Zu in pats:
        for Zd in pats:
            c = canon_pair(t, Zu, Zd)
            if c in seen:
                continue
            seen.add(c)
            if _generic_ok(t, c[0], c[1], rng):
                out.append(c)
    return out, len(seen)


# ---------------------------------------------------------------- gauge fixing (phase slots)

ENTRY_ORDER = ("33", "23", "32", "22", "13", "31", "12", "21", "11")


def entries(spec_or_t, Z=None):
    """Canonical entry order: u then d, positions in ENTRY_ORDER."""
    if Z is None:
        Z = spec_or_t["Z"]
    out = []
    for s, Zs in zip(SECT, Z):
        for p in ENTRY_ORDER:
            if p in Zs:
                out.append(s + p)
    return out


def _gauge_vec(t, e):
    """Linear action of the observable-preserving rephasings on the phase of entry e.
    H: P Y P^dag with common P (3 angles). S: P Y P with common P plus a free overall phase per sector
    (3 + 2). G: P_L Y Q_s with common P_L and per-sector Q (3 + 3 + 3)."""
    s, (i, j) = e[0], _ij(e[1:])
    si = SECT.index(s)
    if t == "H":
        v = np.zeros(3)
        v[i] += 1
        v[j] -= 1
    elif t == "S":
        v = np.zeros(5)
        v[i] += 1
        v[j] += 1
        v[3 + si] += 1
    else:
        v = np.zeros(9)
        v[i] += 1
        v[3 + 3 * si + j] += 1
    return v


def phase_slots(t, Z):
    return _phase_slots_cached(t, (tuple(Z[0]), tuple(Z[1])))


import functools


@functools.lru_cache(maxsize=None)
def _phase_slots_cached(t, Z):
    """(gauged entries, complex slots, real-sign slots). Greedy gauge fixing in canonical entry
    order: an entry's phase is set to 0 if its gauge vector is independent of those already chosen."""
    es = entries(t, Z)
    chosen = []
    gauged, cslots, rslots = [], [], []
    for e in es:
        i, j = _ij(e[1:])
        if t == "H" and i == j:
            if e[1:] != "33":
                rslots.append(e)
            continue
        v = _gauge_vec(t, e)
        M = np.array(chosen + [v])
        if np.linalg.matrix_rank(M, tol=1e-9) > len(chosen):
            chosen.append(v)
            gauged.append(e)
        else:
            cslots.append(e)
    return gauged, cslots, rslots


# ---------------------------------------------------------------- ties

def tie_options(t, Z):
    """Single ties allowed by grammar.md: within a sector |Y_q| = r |Y_p| for two non-(3,3) entries
    (q after p in canonical order) or |Y_q| = r (tie to (3,3)); across sectors |Y_dP| = r |Y_uP| at the
    same position P (both present, P != 33). G textures take no ties."""
    if t == "G":
        return []
    out = []
    for s, Zs in zip(SECT, Z):
        ne = [s + p for p in ENTRY_ORDER if p in Zs and p != "33"]
        for a in range(len(ne)):
            q = ne[a]
            for r in RLIST:
                out.append((q, s + "33", r))
            for b in range(a):
                for r in RLIST:
                    out.append((q, ne[b], r))
    for p in HS_POS:
        if p in Z[0] and p in Z[1]:
            for r in RLIST:
                out.append(("d" + p, "u" + p, r))
    return out


def tie_sets(t, Z):
    """Tie sets of size 0..MAX_TIES; each entry is the q of at most one tie, and a q is never a p."""
    one = tie_options(t, Z)
    sets = [()]
    for n in range(1, MAX_TIES + 1):
        for comb in itertools.combinations(one, n):
            qs = [c[0] for c in comb]
            ps = [c[1] for c in comb]
            if len(set(qs)) < n or set(qs) & set(ps):
                continue
            sets.append(comb)
    return sets


def n_free_moduli(Z, ties):
    n = len(Z[0]) + len(Z[1]) - 2
    return n - len(ties)


GCODES = ("0", "pi", "+pi/2", "+2pi/3")   # grammar values; the global CP flag negates them
NONREAL = ("+pi/2", "+2pi/3")
CONJ = {"0": "0", "pi": "pi", "+pi/2": "-pi/2", "+2pi/3": "-2pi/3", "f": "f"}


def count_entries(t, Z, ties):
    """Number of grammar entries with k_cont <= KMAX for this (pattern pair, tie set)."""
    _, cs, rs = phase_slots(t, Z)
    return count_nm(len(cs), len(rs), n_free_moduli(Z, ties))


@functools.lru_cache(maxsize=None)
def count_nm(c, nr, nm):
    """Slots: c complex (free or one of GCODES), nr real signs. The CP flag doubles an entry iff at
    least one complex slot holds a non-real fixed value."""
    if nm > KMAX:
        return 0
    tot = 0
    for f in range(0, min(c, KMAX - nm) + 1):
        nfix = c - f
        real_only = 2 ** nfix
        tot += math.comb(c, f) * (real_only + 2 * (len(GCODES) ** nfix - real_only))
    return tot * 2 ** nr


def iter_phase_assignments(t, Z, ties):
    _, cs, rs = phase_slots(t, Z)
    nm = n_free_moduli(Z, ties)
    copts = ("f",) + GCODES
    for cv in itertools.product(copts, repeat=len(cs)):
        if nm + sum(v == "f" for v in cv) > KMAX:
            continue
        cps = (1, -1) if any(v in NONREAL for v in cv) else (1,)
        for cp in cps:
            cvv = cv if cp == 1 else tuple(CONJ[v] for v in cv)
            for rv in itertools.product(("0", "pi"), repeat=len(rs)):
                ph = dict(zip(cs, cvv))
                ph.update(zip(rs, rv))
                yield ph


def make_spec(t, Z, ties, ph):
    return {"t": t, "Z": [list(Z[0]), list(Z[1])], "ties": [list(x) for x in ties], "ph": dict(ph)}


def spec_id(spec):
    z = ",".join("".join(p for p in Zs if p != "33") or "-" for Zs in spec["Z"])
    ti = ";".join(f"{q}={r}{p}" for q, p, r in spec["ties"])
    ph = ";".join(f"{e}:{v}" for e, v in sorted(spec["ph"].items()))
    return f"{spec['t']}|{z}|{ti}|{ph}"


# ---------------------------------------------------------------- builder

def layout(spec):
    """(free-modulus entries, free-phase entries, lo, hi)."""
    tied = {q for q, _, _ in spec["ties"]}
    fm = [e for e in entries(spec) if e[1:] != "33" and e not in tied]
    fp = [e for e, v in spec["ph"].items() if v == "f"]
    fp = [e for e in entries(spec) if e in fp]
    lo = [-6.0] * len(fm) + [-np.pi] * len(fp)
    hi = [1.0] * len(fm) + [np.pi] * len(fp)
    return fm, fp, np.array(lo), np.array(hi)


def k_cont(spec):
    fm, fp, _, _ = layout(spec)
    return len(fm) + len(fp)


def _frac(r):
    return float(Fraction(r))


def build(spec, X):
    """X: (B, k) -> Yu, Yd (B, 3, 3)."""
    X = np.atleast_2d(np.asarray(X, float))
    B = len(X)
    t = spec["t"]
    fm, fp, _, _ = layout(spec)
    mod = {}
    for a, e in enumerate(fm):
        mod[e] = 10.0 ** X[:, a]
    for s in SECT:
        mod[s + "33"] = np.ones(B)
    pending = list(spec["ties"])
    while pending:  # ties never chain (tie_sets), one pass suffices; loop kept for safety
        q, p, r = pending.pop(0)
        if p in mod:
            mod[q] = _frac(r) * mod[p]
        else:
            pending.append((q, p, r))
    ph = {}
    for b, e in enumerate(fp):
        ph[e] = X[:, len(fm) + b]
    for e, v in spec["ph"].items():
        if v != "f":
            ph[e] = np.full(B, FVAL[v])
    Ys = []
    for si, s in enumerate(SECT):
        Y = np.zeros((B, 3, 3), complex)
        for p in spec["Z"][si]:
            e = s + p
            i, j = _ij(p)
            v = mod[e] * np.exp(1j * ph.get(e, np.zeros(B)))
            if t == "H":
                if i == j:
                    Y[:, i, i] = v.real
                else:
                    Y[:, i, j] = v
                    Y[:, j, i] = np.conj(v)
            elif t == "S":
                Y[:, i, j] = v
                Y[:, j, i] = v
            else:
                Y[:, i, j] = v
        Ys.append(Y)
    return Ys[0], Ys[1]


def _dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def _angles(V):
    A = np.abs(V)
    s13 = np.clip(A[:, 0, 2], 0, 1)
    c13 = np.sqrt(np.clip(1 - s13 ** 2, 1e-300, None))
    s12 = np.clip(A[:, 0, 1] / c13, 0, 1)
    s23 = np.clip(A[:, 1, 2] / c13, 0, 1)
    c12, c23 = np.sqrt(1 - s12 ** 2), np.sqrt(1 - s23 ** 2)
    J = np.imag(V[:, 0, 0] * V[:, 1, 1] * np.conj(V[:, 0, 1]) * np.conj(V[:, 1, 0]))
    den = c12 * s12 * c23 * s23 * c13 ** 2 * s13
    a = s12 * s23 * c12 * c23 * s13
    with np.errstate(all="ignore"):
        cosd = (s12 ** 2 * s23 ** 2 + c12 ** 2 * c23 ** 2 * s13 ** 2 - A[:, 2, 0] ** 2) / (2 * a)
        sind = J / den
    d = np.mod(np.arctan2(sind, cosd), 2 * np.pi)
    return s12, s13, s23, d, J


def quark_obs(Yu, Yd):
    with np.errstate(all="ignore"):
        wu, Uu = np.linalg.eigh(Yu @ _dag(Yu))
        wd, Ud = np.linalg.eigh(Yd @ _dag(Yd))
        yu, yd = np.sqrt(np.clip(wu, 0, None)), np.sqrt(np.clip(wd, 0, None))
        V = _dag(Uu) @ Ud
        s12, s13, s23, d, J = _angles(V)
        return {"ckm_s12": s12, "ckm_s13": s13, "ckm_s23": s23, "ckm_delta_rad": d,
                "mr_u_c": yu[:, 0] / yu[:, 1], "mr_c_t": yu[:, 1] / yu[:, 2],
                "mr_d_s": yd[:, 0] / yd[:, 1], "mr_s_b": yd[:, 1] / yd[:, 2], "J": J}


def observe(spec, X):
    return quark_obs(*build(spec, X))


def observe_exact(spec, x):
    """Official engine (src.yukawa.observables) for one point."""
    from src.yukawa import observables
    Yu, Yd = build(spec, np.asarray(x)[None])
    Ye = np.diag([1e-3, 1e-1, 1.0]).astype(complex)
    o = observables(Yu[0], Yd[0], Ye, np.diag([1e-3, 1e-2, 1.0]).astype(complex), ordering="NO")
    return {k: float(o[k]) for k in QUARK_KEYS}


# ---------------------------------------------------------------- data, scales, proxy

SCALES = {
    "MZ": None,
    "GUT10": {"mu": 2e16, "model": "MSSM", "tan_beta": 10},
    "GUT30": {"mu": 2e16, "model": "MSSM", "tan_beta": 30},
}
FLOOR = 0.02
BAD = 30.0
_SC = {}


def scorer(scale="MZ"):
    if scale not in _SC:
        from src import evaluate as ev
        _SC[scale] = ev.Scorer(QUARK_KEYS, "NO", mass_floor=FLOOR, scale=SCALES[scale])
    return _SC[scale]


def _wrap(x, per):
    return (x + per / 2) % per - per / 2


class Proxy:
    """Vectorised residuals equal to the official scorer's (CKM asymmetric Gaussian, correlated mass
    block with the 2% floor, high-scale translation as in src.evaluate.Scorer._prep)."""

    def __init__(self, scale="MZ"):
        from src import data as _data
        sc = scorer(scale)
        self.sc = sc
        ck = _data.load_ckm()
        self.ckm = {k: ck[k[4:]] for k in QUARK_KEYS[:4]}
        self.mr = QUARK_KEYS[4:]
        self.Linv = np.linalg.inv(sc._mr_L)
        self.mrv = sc._mr_value
        self.shift = dict(sc.scale_shift)
        self.slope = dict(sc.scale_slope)
        from src.evaluate import SCALE_ADDITIVE
        self.add = SCALE_ADDITIVE

    def low(self, o):
        p = dict(o)
        for n, s in self.shift.items():
            if n in self.slope:
                p[n] = self.slope[n] * p[n] - s
            else:
                p[n] = p[n] - s if n in self.add else p[n] * s
        return p

    def __call__(self, o):
        p = self.low(o)
        res = []
        for k in QUARK_KEYS[:4]:
            r = self.ckm[k]
            d = p[k] - r["value"]
            if k == "ckm_delta_rad":
                d = _wrap(d, 2 * np.pi)
            res.append(d / np.where(d > 0, r["err_plus"], r["err_minus"]))
        m = np.stack([p[k] for k in self.mr], 1) - self.mrv
        res.extend(list((m @ self.Linv.T).T))
        R = np.stack(res, 1)
        return np.where(np.isfinite(R), R, BAD)


_PX = {}


def proxy(scale):
    if scale not in _PX:
        _PX[scale] = Proxy(scale)
    return _PX[scale]


def official(spec, x, scale="MZ"):
    p = observe_exact(spec, x)
    sc = scorer(scale)
    k = k_cont(spec)
    try:
        r = sc(p, k=k)
        return {"chi2": float(r.chi2), "max_pull": float(r.max_abs_pull), "p": float(r.p_global),
                "fit_ok": bool(r.fit_ok), "pulls": {a: float(b) for a, b in r.pulls.items()}, "pred": p}
    except Exception as e:
        return {"chi2": float("inf"), "max_pull": float("inf"), "p": 0.0, "fit_ok": False, "pulls": {},
                "pred": p, "error": str(e)[:200]}


# ---------------------------------------------------------------- filters and fitting
from scipy.optimize import least_squares, minimize  # noqa: E402
from scipy.stats import chi2 as _chi2  # noqa: E402


def chi2_crit(k):
    return float(_chi2.ppf(0.95, 8 - k))


def cp_dead(spec, rng, n=64):
    """Filter 1: J = 0 at n random points."""
    _, _, lo, hi = layout(spec)
    X = rng.uniform(lo, hi, (n, len(lo)))
    J = observe(spec, X)["J"]
    return bool(np.all(np.abs(J) < 1e-12))


def _box(spec):
    fm, fp, lo, hi = layout(spec)
    lo = lo.copy()
    hi = hi.copy()
    lo[len(fm):] = -2 * np.pi
    hi[len(fm):] = 2 * np.pi
    return lo, hi


class SectorMass:
    """Filter 2: marginal 2-ratio chi^2 of one sector (official covariance block, floor, scale)."""

    def __init__(self, scale):
        px = proxy(scale)
        self.px = px
        C = px.sc._mr_cov
        self.idx = {"u": [0, 1], "d": [2, 3]}
        self.Linv = {s: np.linalg.inv(np.linalg.cholesky(C[np.ix_(i, i)])) for s, i in self.idx.items()}
        self.val = {s: px.mrv[i] for s, i in self.idx.items()}
        self.keys = {"u": ("mr_u_c", "mr_c_t"), "d": ("mr_d_s", "mr_s_b")}

    def resid(self, s, o):
        p = self.px.low(o)
        m = np.stack([p[k] for k in self.keys[s]], 1) - self.val[s]
        R = m @ self.Linv[s].T
        return np.where(np.isfinite(R), R, BAD)


_SM = {}


def sector_mass(scale):
    if scale not in _SM:
        _SM[scale] = SectorMass(scale)
    return _SM[scale]


def sector_params(spec, s):
    """Indices of the parameters that change sector s's matrix (own moduli and phases, tie sources)."""
    fm, fp, _, _ = layout(spec)
    src = {p for q, p, r in spec["ties"] if q[0] == s}
    idx = [a for a, e in enumerate(fm) if e[0] == s or e in src]
    idx += [len(fm) + b for b, e in enumerate(fp) if e[0] == s]
    return idx


def sector_min(spec, s, scale, rng, n_rand=512, n_lm=4):
    sm = sector_mass(scale)
    lo, hi = _box(spec)
    idx = sector_params(spec, s)
    x0 = 0.5 * (lo + hi)
    if not idx:
        return float((sm.resid(s, observe(spec, x0[None])) ** 2).sum())
    X = np.tile(x0, (n_rand, 1))
    X[:, idx] = rng.uniform(lo[idx], hi[idx], (n_rand, len(idx)))
    c = (sm.resid(s, observe(spec, X)) ** 2).sum(1)
    best = float(np.min(c))
    for i in np.argsort(c)[:n_lm]:
        def f(z, xb=X[i]):
            x = xb.copy()
            x[idx] = z
            return sm.resid(s, observe(spec, x[None]))[0]
        try:
            r = least_squares(f, X[i, idx], bounds=(lo[idx], hi[idx]), method="trf", max_nfev=200,
                              xtol=1e-9, ftol=1e-9)
            best = min(best, float(2 * r.cost))
        except Exception:
            pass
        if best < 1e-6:
            break
    return best


def _jac(spec, px, x, h=1e-6):
    k = len(x)
    X = np.tile(x, (k + 1, 1))
    X[1:] += h * np.eye(k)
    R = px(observe(spec, X))
    return (R[1:] - R[0]).T / h


def fit_entry(spec, scale, rng, n_rand=2048, n_lm=6, max_nfev=300):
    px = proxy(scale)
    lo, hi = _box(spec)
    k = len(lo)
    X = rng.uniform(lo, hi, (n_rand, k))
    c = (px(observe(spec, X)) ** 2).sum(1)
    best = None
    for i in np.argsort(c)[:n_lm]:
        f = lambda x: px(observe(spec, x[None]))[0]
        j = lambda x: _jac(spec, px, x)
        try:
            r = least_squares(f, X[i], jac=j, bounds=(lo, hi), method="trf", x_scale="jac",
                              max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            cand = (float(2 * r.cost), r.x)
        except Exception:
            cand = (float(c[i]), X[i])
        if best is None or cand[0] < best[0]:
            best = cand
        if best[0] < 1e-3:
            break
    pc, x = best
    off = official(spec, x, scale)
    polished = False
    if off["max_pull"] < 3 and not off["fit_ok"]:
        x2 = polish(spec, x, scale)
        off2 = official(spec, x2, scale)
        polished = True
        if off2["chi2"] < off["chi2"]:
            x, off = x2, off2
    return {"proxy_chi2": pc, "x": [float(v) for v in x], "polished": polished, **off}


def polish(spec, x, scale, maxfev=1500):
    sc = scorer(scale)
    lo, hi = _box(spec)

    def g(z):
        z = np.clip(z, lo, hi)
        try:
            return sc.chi2_soft(observe_exact(spec, z), include_finite=True)
        except Exception:
            return 1e12
    r = minimize(g, x, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-9, "fatol": 1e-9,
                                                       "adaptive": True})
    return np.clip(r.x, lo, hi)


def fingerprint(spec, n=3, seed=4499):
    lo, hi = _box(spec)
    rng = np.random.default_rng(seed + len(lo))
    X = rng.uniform(lo, hi, (n, len(lo)))
    o = observe(spec, X)
    v = np.concatenate([o[k] for k in QUARK_KEYS])
    return tuple(np.round(v, 8))


# ---------------------------------------------------------------- batched Levenberg-Marquardt (fast path)

def blm(resid, X0, lo, hi, iters=80, h=1e-6):
    """Vectorised LM over S starts. resid: (B, k) -> (B, m). Returns (X, chi2) per start."""
    X = np.clip(np.array(X0, float), lo, hi)
    S, k = X.shape
    R = resid(X)
    C = (R ** 2).sum(1)
    lam = np.full(S, 1e-3)
    E = np.eye(k)
    for _ in range(iters):
        P = (X[:, None, :] + h * E[None]).reshape(-1, k)
        RP = resid(P).reshape(S, k, -1)
        Jt = (RP - R[:, None, :]) / h
        A = Jt @ np.swapaxes(Jt, 1, 2)
        g = np.einsum("skm,sm->sk", Jt, R)
        D = np.einsum("skk->sk", A) + 1e-12
        M = A + lam[:, None, None] * (D[:, :, None] * E[None])
        try:
            step = -np.linalg.solve(M, g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = -g * 1e-3
        Xn = np.clip(X + step, lo, hi)
        Rn = resid(Xn)
        Cn = (Rn ** 2).sum(1)
        acc = Cn < C
        X[acc], R[acc], C[acc] = Xn[acc], Rn[acc], Cn[acc]
        lam = np.where(acc, lam / 3, lam * 5)
        lam = np.clip(lam, 1e-9, 1e9)
        if np.all(lam > 1e8) or np.min(C) < 1e-10:
            break
    return X, C


def sector_min_fast(spec, s, scale, rng, n_rand=512, n_lm=12):
    sm = sector_mass(scale)
    lo, hi = _box(spec)
    idx = sector_params(spec, s)
    x0 = 0.5 * (lo + hi)
    if not idx:
        return float((sm.resid(s, observe(spec, x0[None])) ** 2).sum())
    X = np.tile(x0, (n_rand, 1))
    X[:, idx] = rng.uniform(lo[idx], hi[idx], (n_rand, len(idx)))
    c = (sm.resid(s, observe(spec, X)) ** 2).sum(1)
    top = np.argsort(c)[:n_lm]

    def res(Z):
        XX = np.tile(x0, (len(Z), 1))
        XX[:, idx] = Z
        return sm.resid(s, observe(spec, XX))
    _, C = blm(res, X[top][:, idx], lo[idx], hi[idx], iters=60)
    return float(min(np.min(C), np.min(c)))


def fit_entry_fast(spec, scale, rng, n_rand=2048, n_lm=16, iters=100, do_polish=True):
    px = proxy(scale)
    lo, hi = _box(spec)
    k = len(lo)
    X = rng.uniform(lo, hi, (n_rand, k))
    c = (px(observe(spec, X)) ** 2).sum(1)
    top = np.argsort(c)[:n_lm]
    res = lambda Z: px(observe(spec, Z))
    XF, C = blm(res, X[top], lo, hi, iters=iters)
    i = int(np.argmin(C))
    x = XF[i]
    rec = {"proxy_chi2": float(C[i]), "x": [float(v) for v in x], "polished": False,
           "n_basin": int(np.sum(C < C[i] + 1.0))}
    if not do_polish:
        return rec
    off = official(spec, x, scale)
    if off["max_pull"] < 3 and not off["fit_ok"]:
        x2 = polish(spec, x, scale)
        off2 = official(spec, x2, scale)
        rec["polished"] = True
        if off2["chi2"] < off["chi2"]:
            x, off = x2, off2
            rec["x"] = [float(v) for v in x]
    rec.update(off)
    return rec


# ---------------------------------------------------------------- sector solutions (starts for full fits)

def sector_key(spec, s):
    """Everything that defines sector s's matrix family (pattern, own phase codes, ties into s)."""
    Zs = spec["Z"][SECT.index(s)]
    ph = tuple(sorted((e, v) for e, v in spec["ph"].items() if e[0] == s))
    ties = tuple(sorted(tuple(x) for x in spec["ties"] if x[0][0] == s or x[1][0] == s))
    return (spec["t"], s, tuple(Zs), ph, ties)


def sector_solutions(spec, s, scale, rng, n_rand=512, n_lm=24, keep=8):
    """(min chi2, list of sector parameter vectors with chi2 < 4) for sector s (filter 2)."""
    sm = sector_mass(scale)
    lo, hi = _box(spec)
    idx = sector_params(spec, s)
    x0 = 0.5 * (lo + hi)
    if not idx:
        return float((sm.resid(s, observe(spec, x0[None])) ** 2).sum()), []
    X = np.tile(x0, (n_rand, 1))
    X[:, idx] = rng.uniform(lo[idx], hi[idx], (n_rand, len(idx)))
    c = (sm.resid(s, observe(spec, X)) ** 2).sum(1)
    top = np.argsort(c)[:n_lm]

    def res(Z):
        XX = np.tile(x0, (len(Z), 1))
        XX[:, idx] = Z
        return sm.resid(s, observe(spec, XX))
    Z, C = blm(res, X[top][:, idx], lo[idx], hi[idx], iters=60)
    o = np.argsort(C)
    sols = []
    for i in o:
        if C[i] > 4:
            break
        if all(np.max(np.abs(Z[i] - z)) > 0.05 for z in sols):
            sols.append(Z[i])
        if len(sols) >= keep:
            break
    return float(min(np.min(C), np.min(c))), [list(map(float, z)) for z in sols]


def fit_from_sectors(spec, scale, sols, rng, n_start=16, iters=80, do_polish=True):
    """Full fit started from combinations of sector solutions (dict s -> list of vectors)."""
    px = proxy(scale)
    lo, hi = _box(spec)
    k = len(lo)
    fm, fp, _, _ = layout(spec)
    idx = {s: sector_params(spec, s) for s in SECT}
    ph_idx = list(range(len(fm), k))
    X = rng.uniform(lo, hi, (n_start, k))
    for a in range(n_start):
        for s in SECT:
            if sols.get(s):
                z = sols[s][rng.integers(len(sols[s]))]
                X[a, idx[s]] = z
        if a % 2 and ph_idx:
            X[a, ph_idx] = rng.uniform(-np.pi, np.pi, len(ph_idx))
    res = lambda Z: px(observe(spec, Z))
    XF, C = blm(res, X, lo, hi, iters=iters)
    i = int(np.argmin(C))
    x = XF[i]
    rec = {"proxy_chi2": float(C[i]), "x": [float(v) for v in x], "polished": False,
           "n_basin": int(np.sum(C < C[i] + 1.0))}
    if not do_polish:
        return rec
    off = official(spec, x, scale)
    if off["max_pull"] < 3 and not off["fit_ok"]:
        x2 = polish(spec, x, scale)
        off2 = official(spec, x2, scale)
        rec["polished"] = True
        if off2["chi2"] < off["chi2"]:
            x, off = x2, off2
            rec["x"] = [float(v) for v in x]
    rec.update(off)
    return rec


def fit_thorough(spec, scale, rng, n_start=64, iters=150):
    """Validation / near-pass refit: fresh sector solutions, 64 batched-LM starts (half from sector
    solutions, half uniform), scipy least_squares from the best 5, Nelder-Mead polish on the official chi2."""
    px = proxy(scale)
    lo, hi = _box(spec)
    k = len(lo)
    sols = {s: sector_solutions(spec, s, scale, rng, n_rand=2048, n_lm=48, keep=16)[1] for s in SECT}
    fm, fp, _, _ = layout(spec)
    idx = {s: sector_params(spec, s) for s in SECT}
    X = rng.uniform(lo, hi, (n_start, k))
    for a in range(n_start // 2):
        for s in SECT:
            if sols.get(s):
                X[a, idx[s]] = sols[s][rng.integers(len(sols[s]))]
        if a % 2 and fp:
            X[a, len(fm):] = rng.uniform(-np.pi, np.pi, len(fp))
    res = lambda Z: px(observe(spec, Z))
    XF, C = blm(res, X, lo, hi, iters=iters)
    best = (float(np.min(C)), XF[int(np.argmin(C))])
    for i in np.argsort(C)[:5]:
        f = lambda x: px(observe(spec, x[None]))[0]
        try:
            r = least_squares(f, XF[i], jac=lambda x: _jac(spec, px, x), bounds=(lo, hi), method="trf",
                              x_scale="jac", max_nfev=400, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if 2 * r.cost < best[0]:
                best = (float(2 * r.cost), r.x)
        except Exception:
            pass
    x = best[1]
    off = official(spec, x, scale)
    if not off["fit_ok"]:
        x2 = polish(spec, x, scale, maxfev=3000)
        off2 = official(spec, x2, scale)
        if off2["chi2"] < off["chi2"]:
            x, off = x2, off2
    return {"proxy_chi2": best[0], "x": [float(v) for v in x], **off}

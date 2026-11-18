"""Grammar, bottom-up enumeration, forward-mode evaluation and scoring for exp 010.

Grammar (fixed before looking at any score, see README):
  variable leaves (16): CKM angles t12q, t13q, t23q (rad), CKM phase dq (rad), mass ratios
      r_uc = m_u/m_c, r_ct = m_c/m_t, r_ds = m_d/m_s, r_sb = m_s/m_b, r_em = m_e/m_mu,
      r_mt = m_mu/m_tau, and the square roots of the six ratios (q_uc, ..., q_mt)
  constant leaves (10): 1, 2, 3, 1/2, 1/3, sqrt2, sqrt3, pi/4, pi/2, pi
  unary: neg, sqrt, sin, cos, asin, atan  (no trig/inverse-trig applied directly to a trig/
      inverse-trig output; no neg(neg))
  binary: + - * /
  complexity = number of nodes.
An expression's value is read as an angle in radians: s_ij^2 = sin^2(value) for the three
PMNS angles, delta = value mod 2 pi for the PMNS phase. One expression predicts one observable.

Functional dedup: every expression is evaluated on FP_POINTS random input points drawn from the
fake-input prior (data-independent seed); expressions with identical fingerprints are the same
function and only the first (lowest complexity) is kept. The number of distinct functions is
the trials count.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as _data  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "exp010"

BASE = ["t12q", "t13q", "t23q", "dq", "r_uc", "r_ct", "r_ds", "r_sb", "r_em", "r_mt"]
RATIOS = BASE[4:]
VAR_LEAVES = BASE + ["q" + r[1:] for r in RATIOS]  # q_uc = sqrt(r_uc) ...
CONSTS = [("1", 1.0), ("2", 2.0), ("3", 3.0), ("1/2", 0.5), ("1/3", 1 / 3), ("sqrt2", np.sqrt(2)),
          ("sqrt3", np.sqrt(3)), ("pi/4", np.pi / 4), ("pi/2", np.pi / 2), ("pi", np.pi)]
LEAF_NAMES = VAR_LEAVES + [c[0] for c in CONSTS]
N_VAR = len(VAR_LEAVES)
N_LEAF = len(LEAF_NAMES)

# op codes: 0 leaf; unary 1..6; binary 7..10
UNARY = {1: "neg", 2: "sqrt", 3: "sin", 4: "cos", 5: "asin", 6: "atan"}
BINARY = {7: "+", 8: "-", 9: "*", 10: "/"}
TRIG = {3, 4, 5, 6}
OPNAMES = {**UNARY, **BINARY}

FP_POINTS = 8
FP_SEED = 1234
TARGETS = ["s12sq", "s13sq", "s23sq", "dcp_deg"]


# ------------------------------------------------------------------ input priors

def wolfenstein_angles(lam, A, rho, eta):
    from src.mixing import wolfenstein_to_standard
    t12, t13, t23, d = wolfenstein_to_standard(lam, A, rho, eta)
    return np.stack([t12, t13, t23, d], axis=-1)


def fake_base_inputs(rng, real_ratios, n):
    """Null (b) / fingerprint prior: CKM from exp-007 Wolfenstein priors, ratios log-uniform
    over one decade centred on the real value at the scale in use."""
    lam = rng.uniform(0.10, 0.35, n)
    A = rng.uniform(0.5, 1.2, n)
    rho = rng.uniform(-0.5, 0.5, n)
    eta = rng.uniform(-0.5, 0.5, n)
    ang = wolfenstein_angles(lam, A, rho, eta)
    r = np.asarray(real_ratios)[None, :] * 10 ** rng.uniform(-0.5, 0.5, (n, len(RATIOS)))
    return np.concatenate([ang, r], axis=1)  # (n, 10)


def leaf_values(base):
    """base (..., 10) -> leaf values (..., N_LEAF)."""
    base = np.asarray(base, dtype=float)
    q = np.sqrt(base[..., 4:])
    c = np.broadcast_to(np.array([v for _, v in CONSTS]), base.shape[:-1] + (len(CONSTS),))
    return np.concatenate([base, q, c], axis=-1)


def leaf_tangents(base, dirs):
    """Forward-mode seeds: dirs (S, 10) are 1-sigma shifts of the base inputs.
    Returns (N_LEAF, S)."""
    base = np.asarray(base, dtype=float)
    S = dirs.shape[0]
    T = np.zeros((N_LEAF, S))
    T[:10] = dirs.T
    T[10:16] = (0.5 / np.sqrt(base[4:]))[:, None] * dirs[:, 4:].T
    return T


# ------------------------------------------------------------------ enumeration

def _apply_unary(op, x):
    with np.errstate(all="ignore"):
        if op == 1:
            return -x
        if op == 2:
            return np.sqrt(x)
        if op == 3:
            return np.sin(x)
        if op == 4:
            return np.cos(x)
        if op == 5:
            return np.arcsin(x)
        if op == 6:
            return np.arctan(x)
    raise ValueError(op)


def _apply_binary(op, x, y):
    with np.errstate(all="ignore"):
        if op == 7:
            return x + y
        if op == 8:
            return x - y
        if op == 9:
            return x * y
        if op == 10:
            return x / y
    raise ValueError(op)


_MULT = None


def _hash_rows(V):
    """uint64 hash of fingerprint rows (relative precision ~1e-11); non-finite -> sentinel."""
    global _MULT
    if _MULT is None:
        _MULT = np.random.default_rng(99).integers(1, 2 ** 62, FP_POINTS, dtype=np.uint64) | np.uint64(1)
    fin = np.isfinite(V)
    m, e = np.frexp(np.where(fin, V, 0.0))
    q = (np.round(m * 2.0 ** 34).astype(np.int64) + (e.astype(np.int64) << 40))
    q = np.where(fin, q, np.int64(-7777777)).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (q * _MULT[None, :]).sum(axis=1, dtype=np.uint64)
        h ^= h >> np.uint64(29)
    return h


class Enum:
    """Bottom-up enumeration with functional dedup.

    leaf_set / unary_set / binary_set restrict the alphabet (used by the NN-guided search).
    """

    def __init__(self, cmax, leaf_set=None, unary_set=None, binary_set=None, fp_base=None,
                 max_level=None, verbose=False):
        self.leaf_set = list(range(N_LEAF)) if leaf_set is None else list(leaf_set)
        self.unary_set = list(UNARY) if unary_set is None else list(unary_set)
        self.binary_set = list(BINARY) if binary_set is None else list(binary_set)
        if fp_base is None:
            fp_base = default_fp_base()
        self.fpX = leaf_values(fp_base)  # (P, N_LEAF)
        self.cmax = cmax
        self.max_level = max_level or 10 ** 9
        self.verbose = verbose
        self._build()

    def _build(self):
        ops, A, B, comp, V = [], [], [], [], []
        level_idx = {}
        n = 0
        # leaves
        Vl = self.fpX.T[self.leaf_set]  # (L, P)
        h = _hash_rows(Vl)
        _, keep = np.unique(h, return_index=True)
        keep = np.sort(keep)
        seen = np.sort(h[keep])
        ls = np.array(self.leaf_set)[keep]
        ops.append(np.zeros(len(ls), np.int8)); A.append(ls.astype(np.int32)); B.append(np.full(len(ls), -1, np.int32))
        comp.append(np.ones(len(ls), np.int8)); V.append(Vl[keep])
        level_idx[1] = np.arange(n, n + len(ls)); n += len(ls)
        self.counts_raw = {1: len(self.leaf_set)}
        self.counts = {1: len(ls)}
        allV = [Vl[keep]]
        allops = [np.zeros(len(ls), np.int8)]
        for c in range(2, self.cmax + 1):
            cand_op, cand_a, cand_b, cand_V = [], [], [], []
            Vcat = np.concatenate(allV)
            opcat = np.concatenate(allops)
            # unary on level c-1
            prev = level_idx[c - 1]
            for u in self.unary_set:
                child_ops = opcat[prev]
                if u in TRIG:
                    m = ~np.isin(child_ops, list(TRIG))
                elif u == 1:
                    m = child_ops != 1
                else:
                    m = np.ones(len(prev), bool)
                idx = prev[m]
                if len(idx):
                    cand_op.append(np.full(len(idx), u, np.int8)); cand_a.append(idx); cand_b.append(np.full(len(idx), -1))
                    cand_V.append(_apply_unary(u, Vcat[idx]))
            # binary
            for i in range(1, c - 1):
                j = c - 1 - i
                Li, Lj = level_idx[i], level_idx[j]
                for bop in self.binary_set:
                    if bop in (7, 9) and i > j:
                        continue  # commutative: only i <= j
                    # chunk over Li
                    step = max(1, 4_000_000 // max(1, len(Lj)))
                    for s in range(0, len(Li), step):
                        ia = Li[s:s + step]
                        aa = np.repeat(ia, len(Lj))
                        bb = np.tile(Lj, len(ia))
                        vv = _apply_binary(bop, Vcat[aa], Vcat[bb])
                        hh = _hash_rows(vv)
                        # early dedup within chunk vs seen to bound memory
                        _, first = np.unique(hh, return_index=True)
                        first = np.sort(first)
                        sel = first[~np.isin(hh[first], seen)]
                        anyfin = np.isfinite(vv[sel]).any(axis=1)
                        sel = sel[anyfin]
                        cand_op.append(np.full(len(sel), bop, np.int8)); cand_a.append(aa[sel]); cand_b.append(bb[sel])
                        cand_V.append(vv[sel])
            if not cand_op:
                break
            cop = np.concatenate(cand_op); ca = np.concatenate(cand_a); cb = np.concatenate(cand_b)
            cV = np.concatenate(cand_V)
            self.counts_raw[c] = int(len(cop))
            fin = np.isfinite(cV).any(axis=1)
            hh = _hash_rows(cV)
            _, first = np.unique(hh, return_index=True)
            first = np.sort(first)
            nov = ~np.isin(hh[first], seen)
            sel = first[nov & fin[first]]
            if len(sel) > self.max_level:
                sel = sel[:self.max_level]
            seen = np.union1d(seen, hh[sel])
            ops.append(cop[sel]); A.append(ca[sel].astype(np.int32)); B.append(cb[sel].astype(np.int32))
            comp.append(np.full(len(sel), c, np.int8))
            allV.append(cV[sel]); allops.append(cop[sel])
            level_idx[c] = np.arange(n, n + len(sel)); n += len(sel)
            self.counts[c] = int(len(sel))
            if self.verbose:
                print(f"  level {c}: {len(cop)} candidates, {len(sel)} new distinct, total {n}", flush=True)
        self.op = np.concatenate(ops); self.a = np.concatenate(A); self.b = np.concatenate(B)
        self.comp = np.concatenate(comp)
        self.level_idx = level_idx

    def __len__(self):
        return len(self.op)

    def save(self, path):
        np.savez_compressed(path, op=self.op, a=self.a, b=self.b, comp=self.comp,
                            counts=json.dumps(self.counts), counts_raw=json.dumps(self.counts_raw))

    # ---- evaluation
    def evaluate(self, base, dirs=None):
        return evaluate(self.op, self.a, self.b, self.comp, base, dirs)

    def expr(self, i):
        return expr_str(self.op, self.a, self.b, i)


class Loaded:
    def __init__(self, path, cmax=None):
        """cmax truncates to complexity <= cmax (nodes are stored in complexity order)."""
        z = np.load(path)
        self.op, self.a, self.b, self.comp = z["op"], z["a"], z["b"], z["comp"]
        self.counts = {int(k): v for k, v in json.loads(str(z["counts"])).items()}
        self.counts_raw = {int(k): v for k, v in json.loads(str(z["counts_raw"])).items()}
        if cmax is not None:
            n = int(np.searchsorted(self.comp, cmax + 1))
            self.op, self.a, self.b, self.comp = self.op[:n], self.a[:n], self.b[:n], self.comp[:n]
            self.counts = {k: v for k, v in self.counts.items() if k <= cmax}

    def __len__(self):
        return len(self.op)

    def evaluate(self, base, dirs=None):
        return evaluate(self.op, self.a, self.b, self.comp, base, dirs)

    def expr(self, i):
        return expr_str(self.op, self.a, self.b, i)


def default_fp_base():
    real = real_inputs("MZ")["base"]
    rng = np.random.default_rng(FP_SEED)
    return fake_base_inputs(rng, real[4:], FP_POINTS)


def evaluate(op, a, b, comp, base, dirs=None):
    """Value (N,) and, if dirs (S, 10) given, forward-mode tangents (N, S) at one input point.

    Nodes are stored in complexity order, children always earlier, so one pass per level works.
    """
    N = len(op)
    X = leaf_values(base)
    v = np.empty(N)
    T = None if dirs is None else np.empty((N, dirs.shape[0]))
    TL = None if dirs is None else leaf_tangents(base, dirs)
    leaf = op == 0
    v[leaf] = X[a[leaf]]
    if T is not None:
        T[leaf] = TL[a[leaf]]
    with np.errstate(all="ignore"):
        for c in range(2, int(comp.max()) + 1):
            lv = np.nonzero(comp == c)[0]
            if not len(lv):
                continue
            o = op[lv]
            for code in np.unique(o):
                idx = lv[o == code]
                x = v[a[idx]]
                if code <= 6:
                    v[idx] = _apply_unary(code, x)
                    if T is not None:
                        tx = T[a[idx]]
                        if code == 1:
                            d = -np.ones_like(x)
                        elif code == 2:
                            d = 0.5 / np.sqrt(x)
                        elif code == 3:
                            d = np.cos(x)
                        elif code == 4:
                            d = -np.sin(x)
                        elif code == 5:
                            d = 1 / np.sqrt(1 - x * x)
                        else:
                            d = 1 / (1 + x * x)
                        T[idx] = d[:, None] * tx
                else:
                    y = v[b[idx]]
                    v[idx] = _apply_binary(code, x, y)
                    if T is not None:
                        tx, ty = T[a[idx]], T[b[idx]]
                        if code == 7:
                            T[idx] = tx + ty
                        elif code == 8:
                            T[idx] = tx - ty
                        elif code == 9:
                            T[idx] = y[:, None] * tx + x[:, None] * ty
                        else:
                            T[idx] = tx / y[:, None] - (x / y ** 2)[:, None] * ty
    return v, T


def expr_str(op, a, b, i):
    o = int(op[i])
    if o == 0:
        return LEAF_NAMES[int(a[i])]
    if o <= 6:
        s = expr_str(op, a, b, int(a[i]))
        return f"-({s})" if o == 1 else f"{UNARY[o]}({s})"
    return f"({expr_str(op, a, b, int(a[i]))} {BINARY[o]} {expr_str(op, a, b, int(b[i]))})"


def expr_features(op, a, b, i, acc=None):
    """Multi-hot features of expression i: leaves used and ops used."""
    if acc is None:
        acc = (set(), set())
    o = int(op[i])
    if o == 0:
        acc[0].add(int(a[i]))
    else:
        acc[1].add(o)
        expr_features(op, a, b, int(a[i]), acc)
        if o > 6:
            expr_features(op, a, b, int(b[i]), acc)
    return acc


# ------------------------------------------------------------------ real inputs and errors

_REAL_CACHE = {}


def _patched_yukawas(mass_shift=None, alpha_shift=0.0):
    """M_Z Yukawas with one input record shifted (in-process monkeypatch, src untouched)."""
    import copy
    from src import rg
    ms = copy.deepcopy(_data.load_masses())
    if mass_shift:
        k, dv = mass_shift
        ms[k]["value"] += dv
    sm = copy.deepcopy(rg.load_sm_inputs())
    sm["pdg2026"]["alpha_s_MZ"]["value"] += alpha_shift
    orig_lm, orig_sm, orig_q = rg._data.load_masses, rg.load_sm_inputs, rg.quark_masses_mz
    try:
        rg._data.load_masses = lambda: ms
        rg.load_sm_inputs = lambda: sm
        q = rg.quark_masses_mz.__wrapped__()
        rg.quark_masses_mz = lambda: q
        y = dict(rg._derived_yukawas_tuple.__wrapped__())
    finally:
        rg._data.load_masses, rg.load_sm_inputs, rg.quark_masses_mz = orig_lm, orig_sm, orig_q
    return y


def _base_at(mu, yuk=None, ckm=None):
    from src import rg
    r = rg.ckm_at(mu, yukawas=yuk, ckm=ckm)
    rr = r["ratios"]
    return np.array([r["theta12"], r["theta13"], r["theta23"], r["delta"] % (2 * np.pi),
                     rr["mu_over_mc"], rr["mc_over_mt"], rr["md_over_ms"], rr["ms_over_mb"],
                     rr["me_over_mmu"], rr["mmu_over_mtau"]])


SCALES = {"MZ": None, "GUT": 2e16}


def real_inputs(scale):
    """Real base inputs at a scale and the 1-sigma shift directions of every error source.

    Sources: m_u, m_d, m_s, m_c, m_b, m_t (direct), alpha_s(M_Z), CKM s12, s13, s23, delta.
    Each direction = (x(+sigma) - x(-sigma)) / 2 with sigma = mean of the asymmetric errors,
    computed through the full src.rg chain (QCD running to M_Z, SM RGEs to the scale).
    Charged-lepton mass errors are < 1e-7 relative and omitted.
    """
    if scale in _REAL_CACHE:
        return _REAL_CACHE[scale]
    import json as _j
    cache = ART / f"real_inputs_{scale}.json"
    if cache.exists():
        d = _j.loads(cache.read_text())
        out = {"base": np.array(d["base"]), "dirs": np.array(d["dirs"]), "sources": d["sources"]}
        _REAL_CACHE[scale] = out
        return out
    from src import rg
    mz = rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"]
    mu = mz if scale == "MZ" else SCALES[scale]
    ms = _data.load_masses()
    ck = _data.load_ckm()
    y0 = rg.derived_yukawas_mz()
    ckm0 = (np.arcsin(ck["s12"]["value"]), np.arcsin(ck["s13"]["value"]), np.arcsin(ck["s23"]["value"]),
            ck["delta_rad"]["value"])
    base = _base_at(mu, y0, ckm0)
    dirs, sources = [], []
    for k in ["m_u", "m_d", "m_s", "m_c", "m_b", "m_t_direct"]:
        s = 0.5 * (ms[k]["err_plus"] + ms[k]["err_minus"])
        xp = _base_at(mu, _patched_yukawas((k, s)), ckm0)
        xm = _base_at(mu, _patched_yukawas((k, -s)), ckm0)
        dirs.append((xp - xm) / 2); sources.append(k)
    sa = rg.load_sm_inputs()["pdg2026"]["alpha_s_MZ"]
    s = 0.5 * (sa["err_plus"] + sa["err_minus"])
    xp = _base_at(mu, _patched_yukawas(None, s), ckm0)
    xm = _base_at(mu, _patched_yukawas(None, -s), ckm0)
    dirs.append((xp - xm) / 2); sources.append("alpha_s")
    for i, k in enumerate(["s12", "s13", "s23", "delta_rad"]):
        s = 0.5 * (ck[k]["err_plus"] + ck[k]["err_minus"])
        cp, cm = list(ckm0), list(ckm0)
        if k == "delta_rad":
            cp[i] += s; cm[i] -= s
        else:
            cp[i] = np.arcsin(ck[k]["value"] + s); cm[i] = np.arcsin(ck[k]["value"] - s)
        xp = _base_at(mu, y0, tuple(cp)); xm = _base_at(mu, y0, tuple(cm))
        dirs.append((xp - xm) / 2); sources.append("ckm_" + k)
    out = {"base": base, "dirs": np.array(dirs), "sources": sources}
    ART.mkdir(parents=True, exist_ok=True)
    cache.write_text(_j.dumps({"base": base.tolist(), "dirs": np.array(dirs).tolist(), "sources": sources}))
    _REAL_CACHE[scale] = out
    return out


def fake_dirs(real, base_fake):
    """Error directions for a fake universe: same relative errors as the real inputs
    (absolute for the CKM phase)."""
    rel = real["base"].copy()
    rel[3] = 1.0
    bf = np.asarray(base_fake).copy()
    bf[3] = 1.0
    return real["dirs"] / rel[None, :] * bf[None, :]


# ------------------------------------------------------------------ scoring

def _sig_max():
    p = _data.load_pmns("NO")
    return {t: 0.5 * (p[t]["err_plus"] + p[t]["err_minus"]) for t in TARGETS}


SIG_MAX = None


def predictions(v, T, target):
    """Expression values -> (prediction p, prediction sigma) for a target observable.

    Validity rules (applied identically to real and fake universes; added after the first
    pass showed aliasing, see README): a mixing-angle expression must take a value in
    [0, pi/2] (no sin^2 aliasing of large angles); a delta expression a value in [-2 pi, 2 pi];
    and the propagated input error sigma_pred must not exceed the NuFIT 1-sigma width of the
    target (NO widths), otherwise the expression makes no prediction. Invalid -> p = nan.
    """
    global SIG_MAX
    if SIG_MAX is None:
        SIG_MAX = _sig_max()
    sig_v = None if T is None else np.sqrt((T ** 2).sum(axis=1))
    with np.errstate(all="ignore"):
        if target == "dcp_deg":
            ok = (v >= -2 * np.pi) & (v <= 2 * np.pi)
            p = np.degrees(np.mod(v, 2 * np.pi))
            s = None if sig_v is None else np.degrees(sig_v)
        else:
            ok = (v >= 0) & (v <= np.pi / 2)
            p = np.sin(v) ** 2
            s = None if sig_v is None else np.abs(np.sin(2 * v)) * sig_v
        if s is not None:
            s = np.where(np.isfinite(s), s, np.inf)
            ok &= s <= SIG_MAX[target]
        p = np.where(ok, p, np.nan)
    return p, s


class ProfileScorer:
    """1D NuFIT profile convolved with a Gaussian prediction error (profiled nuisance):
        dchi2(p, sigma) = min_t [ prof(t) + (t - p)^2 / sigma^2 ]
    prof(t) = src.data.pmns_delta_chi2(ordering, **{target: t}), relative to the ordering's
    local minimum (the official loader).
    sin^2 targets: tabulated on a 2001 x 51 (p, sigma) grid over the region where prof < 64
    (8 sigma), bilinear interpolation, exact prof(p) at sigma = 0. Outside that region a
    Gaussian approximation (p - best)^2 / (w^2 + sigma^2) with the NuFIT 1-sigma width w of
    that side is used (only matters for |pull| > ~8).
    delta: periodic, full circle, 0.5 deg grid.
    pull = sign(p - best fit) * sqrt(dchi2).
    """

    def __init__(self, target, ordering):
        self.target, self.ordering = target, ordering
        rec = _data.load_pmns(ordering)[target]
        self.ep, self.em = rec["err_plus"], rec["err_minus"]
        if target == "dcp_deg":
            t = np.linspace(0, 360, 1441)[:-1]
            prof = np.array([_data.pmns_delta_chi2(ordering, dcp_deg=x) for x in t])
            self.p_grid = np.linspace(0, 360, 721)
            self.periodic = True
            sig = np.concatenate([[0], np.logspace(-1, np.log10(400), 50)])
            self.lo, self.hi = -np.inf, np.inf
        else:
            self.periodic = False
            f = lambda x: _data.pmns_delta_chi2(ordering, **{target: float(x)})
            tc = np.linspace(0, 1, 4001)
            pc = np.array([f(x) for x in tc])
            ok = np.nonzero(np.isfinite(pc) & (pc < 64))[0]
            self.lo, self.hi = tc[max(ok[0] - 1, 0)], tc[min(ok[-1] + 1, len(tc) - 1)]
            t = np.linspace(self.lo, self.hi, 2001)
            prof = np.array([f(x) for x in t])
            prof = np.where(np.isfinite(prof), prof, 1e4)
            self.p_grid = t
            sig = np.concatenate([[0], np.logspace(-5, 0, 50)])
        self.t, self.prof = t, prof
        self.best = t[np.argmin(prof)]
        self.sig_grid = sig
        Z = np.empty((len(self.p_grid), len(sig)))
        for j, s_ in enumerate(sig):
            if s_ == 0:
                Z[:, j] = self._prof(self.p_grid)
            else:
                d = self.p_grid[:, None] - t[None, :]
                if self.periodic:
                    d = (d + 180) % 360 - 180
                Z[:, j] = (prof[None, :] + (d / s_) ** 2).min(axis=1)
        self.Z = Z

    def _prof(self, p):
        if self.periodic:
            return np.interp(np.mod(p, 360), np.append(self.t, 360), np.append(self.prof, self.prof[0]))
        return np.interp(p, self.t, self.prof)

    def dchi2(self, p, s):
        p = np.asarray(p, float)
        s = np.zeros_like(p) if s is None else np.broadcast_to(np.asarray(s, float), p.shape)
        out = np.full(p.shape, np.inf)
        ok = np.isfinite(p) & np.isfinite(s)
        if self.periodic:
            p = np.mod(p, 360)
        inside = ok & (p >= self.lo) & (p <= self.hi)
        outside = ok & ~inside
        if outside.any():
            d = p[outside] - self.best
            w = np.where(d > 0, self.ep, self.em)
            out[outside] = d ** 2 / (w ** 2 + s[outside] ** 2)
        pp, ss = p[inside], s[inside]
        pg, sg = self.p_grid, self.sig_grid
        fi = np.clip((pp - pg[0]) / (pg[1] - pg[0]), 0, len(pg) - 1 - 1e-9)
        i0 = fi.astype(int); ti = fi - i0
        i1 = np.minimum(i0 + 1, len(pg) - 1)
        lg = np.log(sg[1:])
        fj = np.interp(np.log(np.clip(ss, sg[1], sg[-1])), lg, np.arange(len(lg))) + 1
        j0 = np.clip(fj.astype(int), 1, len(sg) - 2); tj = fj - j0
        Z = self.Z
        zs = ((1 - ti) * (1 - tj) * Z[i0, j0] + ti * (1 - tj) * Z[i1, j0]
              + (1 - ti) * tj * Z[i0, j0 + 1] + ti * tj * Z[i1, j0 + 1])
        z0 = self._prof(pp)
        z1 = (1 - ti) * Z[i0, 1] + ti * Z[i1, 1]
        small = ss < sg[1]
        zs = np.where(small, z0 + (z1 - z0) * (ss / sg[1]), zs)
        out[inside] = np.minimum(zs, z0)
        return out

    def pull(self, p, s):
        d = self.dchi2(p, s)
        diff = np.asarray(p) - self.best
        if self.periodic:
            diff = (diff + 180) % 360 - 180
        return np.sign(diff) * np.sqrt(np.maximum(d, 0))


class GaussScorer:
    """Gaussian score with NuFIT asymmetric 1-sigma widths of an ordering, centred anywhere
    (null (a) and its like-for-like real comparison). Prediction error in quadrature."""

    def __init__(self, target, ordering, center=None):
        rec = _data.load_pmns(ordering)[target]
        self.target = target
        self.center = rec["value"] if center is None else center
        self.ep, self.em = rec["err_plus"], rec["err_minus"]
        self.periodic = target == "dcp_deg"

    def pull(self, p, s, center=None):
        c = self.center if center is None else center
        d = np.asarray(p) - c
        if self.periodic:
            d = (d + 180) % 360 - 180
        w = np.where(d > 0, self.ep, self.em)
        s = 0 if s is None else s
        return d / np.sqrt(w ** 2 + s ** 2)


def best_by_level(absPull, comp, cmax):
    """Cumulative best |pull| for complexity <= c, c = 1..cmax."""
    out = np.full(cmax, np.inf)
    for c in range(1, cmax + 1):
        m = comp == c
        if m.any():
            x = absPull[m]
            x = x[np.isfinite(x)]
            if len(x):
                out[c - 1] = x.min()
    return np.minimum.accumulate(out)


def count_by_level(absPull, comp, cmax, thr):
    return np.array([int(((absPull < thr) & (comp <= c)).sum()) for c in range(1, cmax + 1)])

"""Exp 037 library: per-relation exact-likelihood statistic for the frozen exp 029 list.

T(data, ckm) = -1/2 min_theta chi2_PMNS(pred(theta, ckm); data), profiled over the relation's own
continuous parameters theta (k_cont). chi2 is the official within-ordering NuFIT 6.1 Delta chi^2
(src.data tables through src.nulls.Universe, translated tables for shifted data) or, for the
projected-precision worlds, a Gaussian with the exp 029 future sigmas.

Profile procedure (identical for real data, injected worlds and every fake):
  coarse grid over theta (shared by all data sets for one CKM), then for each data set the
  N_START best coarse points are refined by REFINE_STAGES nested local grids (n_ref points per
  dimension, half-width = previous step, periodic dims wrapped, bounded dims clipped).
Symbolic relations (k_cont = 0) are evaluated in closed form, vectorised over fakes.

Every measured number is read through src (Universe records, src.data tables, NuFIT best fit).
Model code is imported from exps 009, 010, 027 and src.baselines (import only).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import re
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments" / "20261119-027-verified-scout", ROOT / "experiments" / "20261118-009-f2-clebsch",
          ROOT / "experiments" / "20261118-010-symbolic-masses"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src import data as _data  # noqa: E402
from src import evaluate as _ev  # noqa: E402
from src import nulls as _nl  # noqa: E402
from src.baselines import lo_matrix, km_phase  # noqa: E402
from src.mixing import extract_angles, standard_matrix  # noqa: E402
from vlib import r12 as v_r12, r23 as v_r23  # noqa: E402  (exp 027, import only)
from f2lib import v_e  # noqa: E402  (exp 009, import only)

FROZEN = ROOT / "experiments" / "20261119-029-forecast-freeze" / "frozen_relations.csv"
FROZEN_SHA = "04d6574e93619ec122bc2c2b7cc48b7200283e51958e4bdac2f3af7113734217"
SENS = ROOT / "experiments" / "20261119-029-forecast-freeze" / "sensitivities.json"
PMNS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
ORDS = ("NO", "IO")
TWO_PI = 2 * np.pi
NULL_CODE = {"N1": 1, "N2": 2, "N1lam": 5}
SEED = 37


# ---------------------------------------------------------------- frozen list

def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_frozen():
    h = sha256(FROZEN)
    if h != FROZEN_SHA:
        raise RuntimeError(f"frozen_relations.csv hash mismatch: {h}")
    rows = list(csv.DictReader(open(FROZEN, newline="")))
    for r in rows:
        r["params"] = json.loads(r["params_today"])
        r["pred"] = {k: float(r[f"pred_{k}"]) for k in PMNS if r[f"pred_{k}"] not in ("", "nan")}
        r["keys"] = tuple(r["predicts"].split())
    return rows


# ---------------------------------------------------------------- CKM contexts

def ctx_from_records(ck: dict) -> dict:
    t12, t13, t23 = (float(np.arcsin(ck[k]["value"])) for k in ("s12", "s13", "s23"))
    d = float(ck["delta_rad"]["value"])
    return {"t12q": t12, "t13q": t13, "t23q": t23, "dq": d, "lam": float(ck["lambda"]["value"]),
            "A": float(ck["A"]["value"]), "V": standard_matrix(t12, t13, t23, d)}


def real_ctx():
    return ctx_from_records(_nl.Universe.real().load_ckm())


def fake_bank(null: str, n: int, start: int = 0):
    """Fake CKM contexts j = start..start+n-1 for N1 / N1lam (exact src.nulls generators,
    rng = default_rng([37, code, j])); for N2 the fake PMNS centres per ordering."""
    base = _nl.Universe.real()
    out = []
    for j in range(start, start + n):
        rng = np.random.default_rng([SEED, NULL_CODE[null], j])
        u = _nl.make_fake(null, base, rng)
        out.append(ctx_from_records(u.load_ckm()) if null != "N2" else u.pmns_centre)
    return out


# ---------------------------------------------------------------- data sets (chi2 providers)

class TableData:
    """Official within-ordering NuFIT Delta chi^2 of a Universe (translated tables if shifted)."""
    kind = "table"

    def __init__(self, universe, ordering):
        self.u, self.o = universe, ordering
        self.a = universe.analysis
        self.grid = _data._shifted_grid_2d(self.a, ordering, "T23-DCP")

    def chi2(self, obs: dict, keys) -> np.ndarray:
        m = self.u.map_to_real(self.o, **{k: np.asarray(obs[k], float) for k in keys})
        if len(keys) == 1:
            k = keys[0]
            return _data.pmns_profile_1d(self.o, k, m[k], self.a)
        c = _data.pmns_profile_1d(self.o, "s12sq", m["s12sq"], self.a) + \
            _data.pmns_profile_1d(self.o, "s13sq", m["s13sq"], self.a)
        x, y, z = self.grid
        uu = np.asarray(m["s23sq"], float)
        vv = (np.asarray(m["dcp_deg"], float) + 180.0) % 360.0 - 180.0
        out = (uu < x[0]) | (uu > x[-1]) | (vv < y[0]) | (vv > y[-1]) | ~np.isfinite(uu) | ~np.isfinite(vv)
        uu = np.where(np.isfinite(uu), uu, x[0])
        vv = np.where(np.isfinite(vv), vv, y[0])
        i = np.clip(np.searchsorted(x, uu) - 1, 0, len(x) - 2)
        j = np.clip(np.searchsorted(y, vv) - 1, 0, len(y) - 2)
        tx = (uu - x[i]) / (x[i + 1] - x[i])
        ty = (vv - y[j]) / (y[j + 1] - y[j])
        z2 = ((1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i + 1, j]
              + (1 - tx) * ty * z[i, j + 1] + tx * ty * z[i + 1, j + 1])
        return np.where(out, np.inf, c + z2)


def future_sigma(ordering, scenario):
    """Exp 029 projected 1 sigma per PMNS key (s13sq: current symmetrized NuFIT)."""
    s = json.loads(SENS.read_text())["scenarios"][scenario]
    p = _data.load_pmns(ordering)
    out = {}
    for k in PMNS:
        v = s[k]["sigma"]
        if v == "current":
            v = 0.5 * (p[k]["err_plus"] + p[k]["err_minus"])
        elif isinstance(v, dict):
            v = v[ordering]
        out[k] = float(v)
    return out


class GaussData:
    """Gaussian future likelihood centred on `centre` with sigmas `sig` (delta residual wrapped)."""
    kind = "gauss"

    def __init__(self, centre, sig, ordering):
        self.c, self.s, self.o = dict(centre), dict(sig), ordering

    def chi2(self, obs: dict, keys) -> np.ndarray:
        tot = 0.0
        for k in keys:
            d = np.asarray(obs[k], float) - self.c[k]
            if k == "dcp_deg":
                d = (d + 180.0) % 360.0 - 180.0
            tot = tot + (d / self.s[k]) ** 2
        return np.where(np.isfinite(tot), tot, np.inf)


def translated_chi2_many(o, key, value, centres, analysis=_data.DEFAULT_ANALYSIS):
    """Single-key official chi2 of one prediction `value` under many N2 fake centres
    (vectorised Universe.map_to_real octant rule + 1D profile)."""
    t = _ev.pmns_best(o, analysis)
    c = np.asarray(centres, float)
    if key == "s23sq":
        same = (c - 0.5) * (t["s23sq"] - 0.5) >= 0
        m = np.where(same, value - c + t["s23sq"], t["s23sq"] - (value - c))
    else:
        m = value - c + t[key]
    return _data.pmns_profile_1d(o, key, m, analysis)


# ---------------------------------------------------------------- observables

def obs_of(U) -> dict:
    t12, t13, t23, d = extract_angles(U)
    return {"s12sq": np.sin(t12) ** 2, "s13sq": np.sin(t13) ** 2, "s23sq": np.sin(t23) ** 2,
            "dcp_deg": np.degrees(d)}


def pdiag(*ph):
    """Batched diag(e^{i ph_1}, e^{i ph_2}, e^{i ph_3}); ph broadcast to a common shape."""
    ph = np.broadcast_arrays(*(np.asarray(x, float) for x in ph))
    M = np.zeros(ph[0].shape + (3, 3), complex)
    for i in range(3):
        M[..., i, i] = np.exp(1j * ph[i])
    return M


def dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


# ---------------------------------------------------------------- relations

class Fitted:
    """Relation with continuous parameters. dims: list of (lo, hi, periodic, n_coarse)."""
    keys = PMNS
    n_ref = 11
    stages = 3
    n_start = 2

    def __init__(self, rid, dims, names):
        self.id, self.dims, self.names = rid, dims, names
        axes = []
        for lo, hi, per, n in dims:
            axes.append(np.linspace(lo, hi, n, endpoint=not per))
        self.steps = [(hi - lo) / (n if per else n - 1) for lo, hi, per, n in dims]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.P0 = np.stack([m.ravel() for m in mesh], axis=1)

    def valid(self, ctx) -> bool:
        return True

    def bounds_now(self, ctx):
        return [(lo, hi) for lo, hi, _, _ in self.dims]

    def U(self, P, ctx):  # P (n, d) -> (n, 3, 3)
        raise NotImplementedError

    def obs(self, P, ctx):
        o = obs_of(self.U(P, ctx))
        bad = self.mask_bad(P, ctx)
        if bad is not None:
            for k in o:
                o[k] = np.where(bad, np.nan, o[k])
        return o

    def mask_bad(self, P, ctx):
        return None

    def _template(self):
        if not hasattr(self, "_tmpl"):
            g = np.linspace(-1.0, 1.0, self.n_ref)
            mesh = np.meshgrid(*([g] * len(self.dims)), indexing="ij")
            self._tmpl = np.stack([m.ravel() for m in mesh], axis=1)  # (nl, d)
        return self._tmpl

    def _local_many(self, X0, half, ctx):
        """Local grids around each row of X0 (m, d) -> (m, nl, d)."""
        P = X0[:, None, :] + self._template()[None] * np.asarray(half)[None, None, :]
        bnds = self.bounds_now(ctx)
        for d, (lo, hi, per, _) in enumerate(self.dims):
            if per:
                P[..., d] = lo + np.mod(P[..., d] - lo, hi - lo)
            else:
                P[..., d] = np.clip(P[..., d], bnds[d][0], bnds[d][1])
        return P

    def profile(self, ctx, datasets, coarse=None):
        """min chi2 and argmin theta for each data set. coarse: precomputed (P0 obs) to reuse.
        Starts: the n_start best coarse points that are > 1.5 coarse steps apart (searched among the 4 n_start
        best); each start is refined by `stages` local grids (n_ref per dim, half-width = previous step),
        keeping the best point; the result is the minimum over the starts and the coarse grid."""
        nd = len(datasets)
        if not self.valid(ctx):
            return np.full(nd, np.inf), [None] * nd
        obs0 = coarse if coarse is not None else self.obs(self.P0, ctx)
        best, arg = np.full(nd, np.inf), [None] * nd
        steps = np.array(self.steps, float)
        pairs, xs = [], []
        for i, D in enumerate(datasets):
            c0 = D.chi2(obs0, self.keys)
            c0 = np.where(np.isfinite(c0), c0, np.inf)
            order = np.argsort(c0)
            if not np.isfinite(c0[order[0]]):
                continue
            best[i], arg[i] = c0[order[0]], self.P0[order[0]]
            starts = []
            for idx in order[: 4 * self.n_start]:
                if not np.isfinite(c0[idx]):
                    break
                p = self.P0[idx]
                if all(np.max(np.abs(p - q) / steps) > 1.5 for q in starts):
                    starts.append(p)
                if len(starts) >= self.n_start:
                    break
            for x in starts:
                pairs.append(i)
                xs.append(x)
        if not pairs:
            return best, arg
        pairs = np.array(pairs)
        xb = np.array(xs, float)
        cb = np.full(len(pairs), np.inf)
        half = steps.copy()
        nl = len(self._template())
        for _ in range(self.stages):
            P = self._local_many(xb, half, ctx)  # (m, nl, d)
            o = self.obs(P.reshape(-1, P.shape[-1]), ctx)
            cl = np.empty((len(pairs), nl))
            for i in np.unique(pairs):
                sel = np.flatnonzero(pairs == i)
                idx = (sel[:, None] * nl + np.arange(nl)[None]).ravel()
                cl[sel] = datasets[i].chi2({k: o[k][idx] for k in self.keys}, self.keys).reshape(len(sel), nl)
            cl = np.where(np.isfinite(cl), cl, np.inf)
            j = np.argmin(cl, axis=1)
            cj = cl[np.arange(len(pairs)), j]
            upd = np.isfinite(cj) & (cj <= cb)
            xb[upd] = P[np.flatnonzero(upd), j[upd]]
            cb[upd] = cj[upd]
            half = half * 2.0 / (self.n_ref - 1)
        for m, i in enumerate(pairs):
            if cb[m] < best[i]:
                best[i], arg[i] = cb[m], xb[m]
        return best, arg


class A4(Fitted):
    """K-A4a-X: U = V^dag diag(1, e^ia, e^ib) X ; K-A4b-BM: U = V diag(1, e^ia, e^ib) X V^dag."""

    def __init__(self, rid, X, order):
        super().__init__(rid, [(0.0, TWO_PI, True, 48), (0.0, TWO_PI, True, 48)], ("alpha", "beta"))
        self.X, self.order = lo_matrix(X).astype(complex), order

    def U(self, P, ctx):
        D = pdiag(np.zeros(len(P)), P[:, 0], P[:, 1])
        V = ctx["V"]
        if self.order == "Vdag_X":
            return dag(V) @ D @ self.X
        return V @ D @ self.X @ dag(V)


class E1(Fitted):
    """K-E1: PMNS = KM(t1, t2, t3, d_l), d_l = -delta_KM(V_CKM) mod 2 pi."""
    n_ref = 7
    stages = 6
    n_start = 3

    def __init__(self, rid):
        H = np.pi / 2
        super().__init__(rid, [(0.0, H, False, 19)] * 3, ("t1", "t2", "t3"))

    def U(self, P, ctx):
        d = (-km_phase(ctx["V"])) % TWO_PI
        t1, t2, t3 = P[:, 0], P[:, 1], P[:, 2]
        c1, s1, c2, s2, c3, s3 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), np.cos(t3), np.sin(t3)
        e = np.exp(1j * d)
        M = np.empty((len(P), 3, 3), complex)
        M[:, 0, 0] = c1; M[:, 0, 1] = -s1 * c3; M[:, 0, 2] = -s1 * s3
        M[:, 1, 0] = s1 * c2; M[:, 1, 1] = c1 * c2 * c3 - s2 * s3 * e; M[:, 1, 2] = c1 * c2 * s3 + s2 * c3 * e
        M[:, 2, 0] = s1 * s2; M[:, 2, 1] = c1 * s2 * c3 + c2 * s3 * e; M[:, 2, 2] = c1 * s2 * s3 - c2 * c3 * e
        return M


class F2LO(Fitted):
    """F2LO-X: U = V_e^dag diag(1, e^iphi, e^iphi) X, V_e = R23(theta_cb) R12(theta_C) (exp 009 v_e)."""

    def __init__(self, rid, X):
        super().__init__(rid, [(0.0, TWO_PI, True, 360)], ("phi",))
        self.X = lo_matrix(X).astype(complex)
        self.n_start = 1

    def U(self, P, ctx):
        Ve = v_e(ctx["t12q"], 0.0, ctx["t23q"], ctx["dq"], "std")
        D = pdiag(np.zeros(len(P)), P[:, 0], P[:, 0])
        return dag(Ve) @ D @ self.X


def _ul(lam, A, phi, form):
    """Exp 027 verify_bilarge U_l forms, batched over phi."""
    R12, R23 = v_r12(np.arcsin(lam)), v_r23(np.arcsin(A * lam ** 2))
    F = pdiag(-phi / 2, phi / 2, np.zeros_like(phi))
    if form in ("I", 2):
        return R23 @ F @ R12 @ dag(F)
    if form in ("II", 4):
        return F @ R12.T @ dag(F) @ R23.T
    if form == 1:
        return F @ R12 @ dag(F)
    raise ValueError(form)


class CDSV(Fitted):
    def __init__(self, rid, form):
        super().__init__(rid, [(0.0, TWO_PI, True, 360)], ("phi",))
        self.form = form
        self.n_start = 1

    def valid(self, ctx):
        lam = ctx["lam"]
        s23 = 1 - lam if self.form == "I" else 3 * lam
        return 0 < 2 * lam < 1 and 0 < s23 < 1 and 0 < lam < 1 and A_ok(ctx)

    def U(self, P, ctx):
        lam, A = ctx["lam"], ctx["A"]
        s23 = 1 - lam if self.form == "I" else 3 * lam
        Unu = standard_matrix(np.arcsin(2 * lam), np.arcsin(lam), np.arcsin(s23), np.pi)
        return dag(_ul(lam, A, P[:, 0], self.form)) @ Unu


def A_ok(ctx):
    return 0 < ctx["A"] * ctx["lam"] ** 2 < 1


class RMSV(Fitted):
    def __init__(self, rid, utype):
        super().__init__(rid, [(1.0, 4.4, False, 69), (0.0, TWO_PI, True, 72)], ("psi", "phi"))
        self.utype = utype

    def valid(self, ctx):
        return 0 < ctx["lam"] < 1 and A_ok(ctx)

    def bounds_now(self, ctx):
        return [(1.0, min(4.4, (1 - 1e-9) / ctx["lam"])), (0.0, TWO_PI)]

    def mask_bad(self, P, ctx):
        return P[:, 0] * ctx["lam"] >= 1

    def U(self, P, ctx):
        lam, A = ctx["lam"], ctx["A"]
        s = np.clip(P[:, 0] * lam, 0, 1)
        Unu = standard_matrix(np.arcsin(s), np.full(len(P), np.arcsin(lam)), np.arcsin(s), np.zeros(len(P)))
        return dag(_ul(lam, A, P[:, 1], self.utype)) @ Unu


# ---- symbolic (exp 010 grammar)

_CONST = {"pi/4": np.pi / 4, "pi/2": np.pi / 2, "1/3": 1 / 3, "1/2": 0.5, "sqrt2": np.sqrt(2), "sqrt3": np.sqrt(3),
          "pi": np.pi}
_TOKEN = {"pi/4": "C_PI4", "pi/2": "C_PI2", "1/3": "C_THIRD", "1/2": "C_HALF", "sqrt2": "C_SQ2", "sqrt3": "C_SQ3"}


def _mass_leaves():
    import symlib as S
    b = S.real_inputs("MZ")["base"]
    out = {}
    for i, n in enumerate(S.BASE):
        if i >= 4:
            out[n] = float(b[i])
            out["q" + n[1:]] = float(np.sqrt(b[i]))
    return out


class Symbolic:
    """One exp 010 expression: value read as an angle (rad); s^2 = sin^2(value) for angles in [0, pi/2],
    delta = value mod 2 pi for values in [-2 pi, 2 pi]; otherwise no prediction (invalid)."""

    def __init__(self, rid, formula, key):
        self.id, self.key, self.keys = rid, key, (key,)
        expr = formula.split("=", 1)[1].rsplit("(rad", 1)[0].strip()
        self.expr = expr
        e = expr
        for tok, name in sorted(_TOKEN.items(), key=lambda t: -len(t[0])):
            e = e.replace(tok, name)
        e = re.sub(r"\bpi\b", "C_PI", e)
        self.code = compile(e, rid, "eval")
        self.mass = _mass_leaves()

    def value(self, t12q, t13q, t23q, dq):
        ns = {"sin": np.sin, "cos": np.cos, "asin": np.arcsin, "atan": np.arctan, "sqrt": np.sqrt,
              "C_PI4": np.pi / 4, "C_PI2": np.pi / 2, "C_THIRD": 1 / 3, "C_HALF": 0.5, "C_SQ2": np.sqrt(2),
              "C_SQ3": np.sqrt(3), "C_PI": np.pi, "t12q": t12q, "t13q": t13q, "t23q": t23q, "dq": dq, **self.mass}
        with np.errstate(all="ignore"):
            return eval(self.code, {"__builtins__": {}}, ns)

    def predict(self, t12q, t13q, t23q, dq):
        v = np.asarray(self.value(t12q, t13q, t23q, dq), float)
        with np.errstate(all="ignore"):
            if self.key == "dcp_deg":
                ok = (v >= -2 * np.pi) & (v <= 2 * np.pi)
                p = np.degrees(np.mod(v, 2 * np.pi))
            else:
                ok = (v >= 0) & (v <= np.pi / 2)
                p = np.sin(v) ** 2
        return np.where(ok & np.isfinite(v), p, np.nan)

    def predict_ctx(self, ctx):
        return self.predict(ctx["t12q"], ctx["t13q"], ctx["t23q"], ctx["dq"])


def make_relation(row):
    rid = row["id"]
    if rid.startswith("K-A4a-"):
        return A4(rid, rid.split("-")[-1], "Vdag_X")
    if rid == "K-A4b-BM":
        return A4(rid, "BM", "V_X_Vdag")
    if rid == "K-E1":
        return E1(rid)
    if rid.startswith("F2LO-"):
        return F2LO(rid, rid.split("-")[-1])
    if rid.startswith("BL-CDSV-"):
        return CDSV(rid, rid.split("-")[-1])
    if rid.startswith("BL-RMSV-"):
        return RMSV(rid, int(rid.split("-")[-1]))
    if rid.startswith("S-"):
        return Symbolic(rid, row["formula"], row["keys"][0])
    raise ValueError(rid)


# ---------------------------------------------------------------- statistics

N_MIN_REAL = 1000
N_MIN_WORLD = 300
N_MAX = 50_000
K_STOP = 2          # pass needs k <= 1 hits at N_MAX (p_hat = (k+1)/(N_MAX+1) <= 4.0e-5)
GLOBAL_ALPHA = 0.01
REL_TOL = 1e-9


def sidak_local(m, alpha=GLOBAL_ALPHA):
    return 1 - (1 - alpha) ** (1 / m)


def is_hit(chi2_fake, chi2_data):
    """Fake at least as good: chi2_fake <= chi2_data (T_fake >= T_data), relative tolerance."""
    chi2_data = np.asarray(chi2_data, float)
    return np.isfinite(chi2_fake) & (chi2_fake <= chi2_data + REL_TOL * np.maximum(1.0, np.abs(chi2_data)))


def cp_upper_one_sided(k, n, cl=0.95):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(cl, k + 1, n - k))


# ---------------------------------------------------------------- fake banks (cached arrays)

BANK_DIR = ROOT / "artifacts" / "037"
CKM_FIELDS = ("t12q", "t13q", "t23q", "dq", "lam", "A")


def build_bank(null: str, n: int = None):
    """Arrays of the first n fakes of a null (rng = default_rng([37, code, j])); cached in artifacts/037."""
    n = N_MAX if n is None else n
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    f = BANK_DIR / f"bank_{null}_{n}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    items = fake_bank(null, n)
    if null == "N2":
        out = {f"{o}_{k}": np.array([c[o][k] for c in items]) for o in ORDS for k in PMNS}
    else:
        out = {k: np.array([c[k] for c in items]) for k in CKM_FIELDS}
    np.savez(f, **out)
    return out


def ctx_from_bank(b, j):
    c = {k: float(b[k][j]) for k in CKM_FIELDS}
    c["V"] = standard_matrix(c["t12q"], c["t13q"], c["t23q"], c["dq"])
    return c


def n2_fake_universe(b, j, base=None):
    from dataclasses import replace
    base = base or _nl.Universe.real()
    centres = {o: {k: float(b[f"{o}_{k}"][j]) for k in PMNS} for o in ORDS}
    return replace(base, kind="N2", pmns_centre=centres, truth={"pmns_centre": centres})

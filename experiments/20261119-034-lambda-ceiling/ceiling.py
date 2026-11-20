"""Exp 034: detectability ceiling for Cabibbo-scale CKM -> PMNS links under the plain N1 null.

Usage:
  python ceiling.py qcheck                         quadrature accuracy check (design, no pipeline)
  python ceiling.py time                           one-world timing (oracle only)
  python ceiling.py pipe <task> [<task> ...]       pipeline + oracle, task = <set>:<label>:<i>
  python ceiling.py orc <task> [<task> ...]        oracle only (today physical/Gaussian, nominal, optimistic)
Sets: L2 (PMNS = f(lambda, A) + noise), L4 (PMNS = f(lambda, A, rhobar, etabar) + noise), S0 (size check),
K13 (sin th13 = lambda / sqrt 2, exact), KQLC (K13 plus th12 = pi/4 - th_C), KR (K13 / KQLC at the measured
lambda, A; oracle only, label 13 or QLC).
Writes artifacts/034/{pipe,orc}/<set>_<label>_<i>.json. Measured values only through src loaders.
The exp 028 networks and statistic are used through exp 032's inject.py (imported, not copied).
"""
from __future__ import annotations

import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import importlib.util
import json
import math
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E032 = ROOT / "experiments" / "20261119-032-nn-power"
E029 = ROOT / "experiments" / "20261119-029-forecast-freeze"
ART = ROOT / "artifacts" / "034"

_spec = importlib.util.spec_from_file_location("inject032", E032 / "inject.py")
I32 = importlib.util.module_from_spec(_spec)
sys.modules["inject032"] = I32
_spec.loader.exec_module(I32)  # puts exp 028 and the repo root on sys.path

from src import evaluate as ev  # noqa: E402
from src import nulls  # noqa: E402

LN2 = math.log(2)
WKEYS = ("lambda", "A", "rhobar", "etabar")
LO = np.array([nulls.N1_PRIOR[k][0] for k in WKEYS])
HI = np.array([nulls.N1_PRIOR[k][1] for k in WKEYS])
N_FAKES = 2000
BLOCK = int(os.environ.get("EXP034_BLOCK", 100))  # stopping check interval; detection outcome does not depend on it
MAX_HITS = nulls.max_hits_to_pass(N_FAKES)  # 11
SCALE_G, N_FEAT, OMEGA_SD = 0.3, 8, 1.5
N_MC = 500
N_BISECT = 20  # max regula-falsi steps
SOBOL_M = 16
CODES = {"L2": 340, "L4": 440, "S0": 500, "K13": 613, "KQLC": 614, "KR": 700}
SCEN = ("today_gauss", "nominal", "optimistic")


def code_of(s, label):
    if s in ("L2", "L4"):
        return CODES[s] + int(round(4 * float(label)))
    if s == "KR":
        return CODES[s] + (1 if label == "13" else 2)
    return CODES[s]


def to_u(w):
    return 2 * (np.asarray(w, float) - LO) / (HI - LO) - 1


def from_u(u):
    return LO + (np.asarray(u, float) + 1) * (HI - LO) / 2


# ---------------------------------------------------------------- quadrature over the N1 prior box (u in [-1, 1]^4)
def quad(dims, m=SOBOL_M):
    if dims == 2:  # exp 032 grid, coordinates 2, 3 irrelevant
        g = np.linspace(-1, 1, 151)
        X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        return np.concatenate([X, np.zeros((len(X), 2))], 1)
    from scipy.stats import qmc
    return qmc.Sobol(d=4, scramble=True, seed=3404).random_base2(m) * 2 - 1


_Q = {}


def Q(dims):
    if dims not in _Q:
        _Q[dims] = quad(dims)
    return _Q[dims]


def target_rows(s12, s13, s23, dcp_deg):
    n = len(s13)
    rows = np.stack([s12, s13, s23, np.radians(dcp_deg) % (2 * np.pi), np.zeros(n)], 1)
    return I32.to_target(rows)[:, :4]


class Link:
    """Random smooth map of the active normalized coordinates (exp 032 family), mean = y_ref + g(u)."""

    def __init__(self, rng, active, y_ref):
        self.act = list(active)
        self.dims = 2 if len(self.act) == 2 else 4
        self.y_ref = y_ref
        self.w = rng.standard_normal((4, N_FEAT))
        self.om = rng.standard_normal((4, N_FEAT, len(self.act))) * OMEGA_SD
        self.ph = rng.uniform(0, 2 * np.pi, (4, N_FEAT))
        raw = self._raw(Q(self.dims))
        self.mu, self.sd = raw.mean(0), raw.std(0)

    def _raw(self, u):
        x = np.atleast_2d(u)[:, self.act]
        arg = np.einsum("kmd,nd->nkm", self.om, x) + self.ph[None]
        return np.einsum("km,nkm->nk", self.w, np.cos(arg))

    def mean(self, u):
        return self.y_ref + SCALE_G * (self._raw(u) - self.mu) / self.sd


class Named:
    """Known-physics-style exact links through lambda: K13 sin th13 = lambda / sqrt 2; KQLC adds th12 = pi/4 - th_C."""
    dims = 2

    def __init__(self, which, best):
        self.which, self.best = which, best

    def mean(self, u):
        lam = from_u(np.atleast_2d(u))[:, 0]
        n = len(lam)
        s13 = lam ** 2 / 2
        s12 = (np.sin(np.pi / 4 - np.arcsin(lam)) ** 2 if self.which == "QLC" else np.full(n, self.best["s12sq"]))
        return target_rows(s12, s13, np.full(n, self.best["s23sq"]), np.full(n, self.best["dcp_deg"]))


def info_bits(mean_fn, dims, cov, xs, eps):
    """I(y; u) in bits, y = mean(u) + N(0, cov), u ~ U[-1, 1]^4: MC over (xs, eps), quadrature for p(y)."""
    L = np.linalg.cholesky(cov)
    Li = np.linalg.inv(L)
    gQ = (mean_fn(Q(dims)) @ Li.T).astype(np.float32)
    nQ = np.sum(gQ.astype(np.float64) ** 2, 1).astype(np.float32)
    y = mean_fn(xs) @ Li.T + eps
    lc = -0.5 * np.sum(eps ** 2, 1)
    lm = np.empty(len(y))
    for a in range(0, len(y), 125):
        yy = y[a:a + 125]
        q = (yy.astype(np.float32) @ gQ.T) - 0.5 * nQ[None]  # + const(y), added back below
        mx = q.max(1, keepdims=True)
        lm[a:a + 125] = (mx[:, 0] + np.log(np.mean(np.exp(q - mx), 1, dtype=np.float64))
                         - 0.5 * np.sum(yy ** 2, 1))
    return float(np.mean(lc - lm) / LN2)


def mc_draws(code, i):
    r = np.random.default_rng([34, code, i, 7])
    return r.uniform(-1, 1, (N_MC, 4)), r.standard_normal((N_MC, 4))


def tune_sigma(model, level, xs, eps):
    """log sigma such that I = level: 6 bisection steps on [-8, 3], then Illinois regula falsi to 0.005 bits."""
    f = lambda t: info_bits(model.mean, model.dims, math.exp(2 * t) * np.eye(4), xs, eps) - level
    a, b = -8.0, 3.0
    for _ in range(6):
        m = 0.5 * (a + b)
        a, b = (m, b) if f(m) > 0 else (a, m)
    fa, fb, side = f(a), f(b), 0
    t, ft = b, fb
    for _ in range(N_BISECT):
        t = (a * fb - b * fa) / (fb - fa)
        ft = f(t)
        if abs(ft) < 0.005:
            break
        if ft > 0:
            a, fa = t, ft
            if side == 1:
                fb /= 2
            side = 1
        else:
            b, fb = t, ft
            if side == -1:
                fa /= 2
            side = -1
    return math.exp(t), ft + level


# ---------------------------------------------------------------- measurement covariances (target space)
def covariances(best):
    p = I32.ART / "meas_cov.json"
    if not p.exists():
        I32.meas_cov()
    cm = np.array(json.load(open(p))["cov"])
    sd_t = np.sqrt(np.diag(cm))
    sens = json.load(open(E029 / "sensitivities.json"))["scenarios"]
    out = {"today_gauss": cm}
    for sc in ("nominal", "optimistic"):
        S = sens[sc]
        fut = sd_t.copy()
        for j, k in enumerate(("s12sq", "s13sq", "s23sq")):
            v = S[k]["sigma"]
            if not isinstance(v, str):
                fut[j] = v / (best[k] * (1 - best[k]))
        d = S["dcp_deg"]["sigma"]["NO"]
        pd = best["dcp_deg"] / 360
        fut[3] = (d / 360) / (pd * (1 - pd))
        D = np.diag(np.minimum(fut / sd_t, 1.0))
        out[sc] = D @ cm @ D
    return out


# ---------------------------------------------------------------- worlds
def build(s, label, i, base, covs):
    code = code_of(s, label)
    rng = np.random.default_rng([34, code, i])
    y_ref = I32.target4(base["best"])
    W = dict(set=s, label=label, i=i, code=code)
    if s == "KR":
        w_true = np.array(base["w_real"], float)
    else:
        w_true = from_u(rng.uniform(-1, 1, 4))
    u_true = to_u(w_true)
    xs, eps = mc_draws(code, i)
    if s in ("L2", "L4"):
        model = Link(rng, [0, 1] if s == "L2" else [0, 1, 2, 3], y_ref)
        sig, I_link = tune_sigma(model, float(label), xs, eps)
        y = model.mean(u_true)[0] + sig * rng.standard_normal(4)
    elif s == "S0":
        model = Link(rng, [0, 1], y_ref)
        sig, I_link = None, 0.0
        y = model.mean(rng.uniform(-1, 1, (1, 4)))[0] + SCALE_G * rng.standard_normal(4)
    else:
        model = Named("QLC" if (s == "KQLC" or label == "QLC") else "13", base["best"])
        sig, I_link = 0.0, None
        y = model.mean(u_true)[0]
    pmns_true = I32.phys_from_target(y)
    w_meas, lnm, pm = I32.measure(rng, base, list(w_true), base["lnm"], pmns_true)
    W.update(model=model, sigma=sig, I_link=I_link, w_true=w_true.tolist(), w_meas=list(map(float, w_meas)),
             lnm=lnm, pmns_true=pmns_true, pmns_meas=pm, dm_ratio=base["dm_ratio"], y_true=y)
    if s != "S0":
        s2 = (sig or 0.0) ** 2
        W["I_meas"] = {sc: info_bits(model.mean, model.dims, s2 * np.eye(4) + covs[sc], xs, eps) for sc in SCEN}
    return W


def fakes_w(code, i):
    F = np.empty((N_FAKES, 4))
    for j in range(N_FAKES):
        r = np.random.default_rng([34, code, i, 1, j])
        F[j] = [r.uniform(*nulls.N1_PRIOR[k]) for k in WKEYS]  # same draw order as src.nulls.n1_universe
    return F


def oracle(W, covs, FW):
    """Oracle ranks with the exact link density, today's physical noise and Gaussian scenarios."""
    if W["set"] == "S0":
        return {}
    model, s2 = W["model"], (W["sigma"] or 0.0) ** 2
    uw, uf = to_u(np.array([W["w_meas"]])), to_u(FW)
    mw, mf = model.mean(uw)[0], model.mean(uf)
    out = {}
    r9 = np.random.default_rng([34, W["code"], W["i"], 9])
    for sc in ("today_phys",) + SCEN:
        C = s2 * np.eye(4) + covs["today_gauss" if sc == "today_phys" else sc]
        P = np.linalg.inv(C)
        if sc == "today_phys":
            yhat = I32.target4(W["pmns_meas"])
        else:
            yhat = W["y_true"] + r9.multivariate_normal(np.zeros(4), covs[sc])
        dw, df = yhat - mw, yhat - mf
        tw = -0.5 * dw @ P @ dw
        tf = -0.5 * np.einsum("na,ab,nb->n", df, P, df)
        k = int(np.sum(tf >= tw))
        out[sc] = dict(k=k, rank=k / N_FAKES, detected=bool(nulls.decide(k, N_FAKES)["passed"]))
    return out


def summary(W, extra):
    keep = ("set", "label", "i", "sigma", "I_link", "I_meas", "w_true", "w_meas", "pmns_true", "pmns_meas", "dm_ratio")
    out = {k: W[k] for k in keep if k in W}
    out["lnm"] = np.asarray(W["lnm"]).tolist()
    out.update(extra)
    return out


def run_pipe(W, pipe, FW):
    m = np.array([W["lnm"] / np.log(10)])
    ys = I32.conv_points(W["pmns_meas"], W["dm_ratio"])
    T_w = float(pipe.T(np.array([I32.ctx_from_w(W["w_meas"])]), m, ys)[0])
    Tf, k = [], 0
    for b in range(0, N_FAKES, BLOCK):
        ctx = np.array([I32.ctx_from_w(w) for w in FW[b:b + BLOCK]])
        Tf.extend(pipe.T(ctx, m, ys).tolist())
        k = int(np.sum(~(np.array(Tf) < T_w)))  # NaN fake counts as >= (against detection)
        if k > MAX_HITS:
            break
    Tf = np.array(Tf)
    n = len(Tf)
    return dict(T_world=T_w, n_fakes_eval=n, k_ge=k, rank_eval=k / n, stopped_early=n < N_FAKES,
                detected=bool(n == N_FAKES and nulls.decide(k, N_FAKES)["passed"]),
                T_fake_median=float(np.nanmedian(Tf)), n_nan_fakes=int(np.isnan(Tf).sum()),
                T_fakes=np.round(Tf, 4).tolist())


def main(argv):
    mode = argv[0]
    base = I32.real_base()
    covs = covariances(base["best"])
    if mode == "qcheck":
        return qcheck(base)
    if mode == "time":
        t = time.time()
        W = build("L4", "8", 0, base, covs)
        t1 = time.time()
        o = oracle(W, covs, fakes_w(W["code"], 0))
        print("L4 8 build", round(t1 - t, 1), "s, oracle", round(time.time() - t1, 1), "s", W["I_link"], W["I_meas"], o)
        return
    sub = ART / ("pipe" if mode == "pipe" else "orc")
    sub.mkdir(parents=True, exist_ok=True)
    pipe = None
    for task in argv[1:]:
        s, label, i = task.split(":")
        i = int(i)
        path = sub / f"{s}_{label}_{i}.json"
        if path.exists():
            continue
        t0 = time.time()
        W = build(s, label, i, base, covs)
        FW = fakes_w(W["code"], i)
        extra = dict(oracle=oracle(W, covs, FW))
        if mode == "pipe":
            if pipe is None:
                import torch
                torch.set_num_threads(int(os.environ.get("EXP034_THREADS", 2)))
                pipe = I32.Pipe()
            extra.update(run_pipe(W, pipe, FW))
        extra["seconds"] = round(time.time() - t0, 1)
        json.dump(summary(W, extra), open(path, "w"))
        o = extra["oracle"].get("today_phys", {})
        print(f"{time.strftime('%H:%M:%S')} {mode} {task}: I_link {W['I_link']} k {extra.get('k_ge')}/"
              f"{extra.get('n_fakes_eval')} det {extra.get('detected')} oracle k {o.get('k')} ({extra['seconds']} s)",
              flush=True)


def qcheck(base):
    """Information at the tuned sigma with 2^16 vs 2^18 Sobol points (L4) and 151^2 vs 301^2 grid (L2)."""
    y_ref = I32.target4(base["best"])
    for s, lv in (("L4", 8), ("L4", 12), ("L2", 12)):
        for i in range(3):
            code = code_of(s, lv)
            rng = np.random.default_rng([34, code, i])
            rng.uniform(-1, 1, 4)
            model = Link(rng, [0, 1] if s == "L2" else [0, 1, 2, 3], y_ref)
            xs, eps = mc_draws(code, i)
            sig, I = tune_sigma(model, lv, xs, eps)
            dims = model.dims
            old = _Q.pop(dims)
            if dims == 4:
                _Q[4] = quad(4, 18)
            else:
                g = np.linspace(-1, 1, 301)
                X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
                _Q[2] = np.concatenate([X, np.zeros((len(X), 2))], 1)
            I_fine = info_bits(model.mean, dims, sig * sig * np.eye(4), xs, eps)
            _Q[dims] = old
            print(f"{s} {lv} bits world {i}: sigma {sig:.4g} I(coarse) {I:.3f} I(fine) {I_fine:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

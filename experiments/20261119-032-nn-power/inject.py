"""Exp 032: power of the exp 028 NN information test (see README pre-registration).

Usage:
  python inject.py meas_cov                 -> artifacts/032/meas_cov.json (PMNS noise covariance, target space)
  python inject.py run <task> [<task> ...]  task = OOM:<level>:<i> | IM:<family>:<i>
Each world writes artifacts/032/inj/<kind>_<label>_<i>.json (skipped if present).
Measured values only through src loaders. Exp 028 networks and statistic are imported unchanged.
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
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E028 = ROOT / "experiments" / "20261119-028-nn-info-decisive"
E025 = ROOT / "experiments" / "20261119-025-nn-ckm-info"
ART = ROOT / "artifacts" / "032"
sys.path[:0] = [str(E028), str(ROOT)]

from src import data as sd  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import nulls  # noqa: E402
from src.mixing import standard_matrix, standard_to_wolfenstein  # noqa: E402
import analyze as A28  # noqa: E402  (exp 028 analysis functions)
from train import load, to_target  # noqa: E402

LN2 = math.log(2)
N_FAKES = 2000
BLOCK = 100
MAX_HITS = nulls.max_hits_to_pass(N_FAKES)  # 11
SCALE_G = 0.3
N_FEAT = 8
OMEGA_SD = 1.5
GRID = 151
N_MC = 1000
WKEYS = ("lambda", "A", "rhobar", "etabar")
CODES = {"MN": 901, "F2": 902, "MNu": 903}


def code_of(kind, label):
    return int(round(100 + 10 * float(label))) if kind == "OOM" else CODES[label]


def split_normal(rng, rec, centre):
    z = rng.standard_normal()
    return centre + z * (rec["err_plus"] if z > 0 else abs(rec["err_minus"]))


def ctx_from_w(w):
    v = ev.ckm_from_wolfenstein(*w)
    return [np.log10(v["s12"]), np.log10(v["s13"]), np.log10(v["s23"]), np.cos(v["delta_rad"]), np.sin(v["delta_rad"])]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def phys_from_target(y4):
    s = sigmoid(np.asarray(y4[:3]))
    return dict(s12sq=float(s[0]), s13sq=float(s[1]), s23sq=float(s[2]), dcp_deg=float(360 * sigmoid(y4[3])))


def target4(p):
    """(s12sq, s13sq, s23sq, dcp_deg) -> 4-dim exp 028 target (no dm ratio)."""
    row = [[p["s12sq"], p["s13sq"], p["s23sq"], np.radians(p["dcp_deg"]) % (2 * np.pi), 0.0]]
    return to_target(np.array(row))[0, :4]


def clip_meas(p):
    out = {k: float(np.clip(p[k], 1e-6, 1 - 1e-6)) for k in ("s12sq", "s13sq", "s23sq")}
    out["dcp_deg"] = float(p["dcp_deg"] % 360.0)
    return out


# ---------------------------------------------------------------- PMNS measurement-noise covariance
def meas_cov(n=4000):
    best = ev.pmns_best("NO")
    t0 = target4(best)
    rng = np.random.default_rng(3201)
    D = []
    for _ in range(n):
        off = nulls.pmns_noise("NO", rng)
        D.append(target4(clip_meas({k: best[k] - off[k] for k in best})) - t0)
    D = np.array(D)
    out = dict(n=n, mean=D.mean(0).tolist(), cov=np.cov(D.T).tolist(), sd=D.std(0).tolist())
    ART.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ART / "meas_cov.json", "w"), indent=1)
    return out


# ---------------------------------------------------------------- OOM link and its information
_G = np.linspace(-1, 1, GRID)
GRID_X = np.stack(np.meshgrid(_G, _G, indexing="ij"), -1).reshape(-1, 2)


class Link:
    def __init__(self, rng):
        self.w = rng.standard_normal((4, N_FEAT))
        self.om = rng.standard_normal((4, N_FEAT, 2)) * OMEGA_SD
        self.ph = rng.uniform(0, 2 * np.pi, (4, N_FEAT))
        raw = self._raw(GRID_X)
        self.mu, self.sd = raw.mean(0), raw.std(0)

    def _raw(self, x):
        x = np.atleast_2d(x)
        arg = np.einsum("kmd,nd->nkm", self.om, x) + self.ph[None]
        return np.einsum("km,nkm->nk", self.w, np.cos(arg))

    def __call__(self, x):
        return SCALE_G * (self._raw(x) - self.mu) / self.sd


def info_bits(link, cov, xs, eps):
    """I(y; x) in bits, y = g(x) + N(0, cov), x ~ U[-1,1]^2: MC over (xs, eps), grid quadrature for p(y)."""
    L = np.linalg.cholesky(cov)
    Li = np.linalg.inv(L)
    gG = link(GRID_X) @ Li.T  # whitened
    gx = link(xs) @ Li.T
    y = gx + eps  # whitened y = L^-1 (g(x) + L eps)
    lc = -0.5 * np.sum(eps ** 2, 1)  # log N(y | g(x)) up to the common constant
    q = -0.5 * (np.sum(y ** 2, 1)[:, None] + np.sum(gG ** 2, 1)[None] - 2 * y @ gG.T)
    m = q.max(1, keepdims=True)
    lm = m[:, 0] + np.log(np.mean(np.exp(q - m), 1))
    return float(np.mean(lc - lm) / LN2)


def tune_sigma(link, level, cm, rng_mc):
    xs = rng_mc.uniform(-1, 1, (N_MC, 2))
    eps = rng_mc.standard_normal((N_MC, 4))
    lo, hi = -8.0, 3.0  # log sigma bracket
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        I = info_bits(link, np.exp(2 * mid) * np.eye(4), xs, eps)
        lo, hi = (mid, hi) if I > level else (lo, mid)
    s = math.exp(0.5 * (lo + hi))
    return s, info_bits(link, s * s * np.eye(4), xs, eps), info_bits(link, s * s * np.eye(4) + cm, xs, eps)


# ---------------------------------------------------------------- worlds
def real_base():
    ck = sd.load_ckm()
    mr, names, cov_log = ev.mass_ratio_targets()
    idx = [names.index(k) for k in A28.MR_KEYS]
    p = sd.load_pmns("NO")
    r = abs(p["dm21sq_1e-5eV2"]["value"] * 1e-5 / (p["dm3l_sq_1e-3eV2"]["value"] * 1e-3))
    return dict(ck=ck, w_real=[ck[k]["value"] for k in WKEYS],
                lnm=np.array([np.log(mr[k]) for k in A28.MR_KEYS]), cov_log=np.asarray(cov_log)[np.ix_(idx, idx)],
                dm_ratio=r, best=ev.pmns_best("NO"))


def measure(rng, base, w_true, lnm_true, pmns_true):
    """Today's measurement noise on CKM (Wolfenstein), mass ratios, PMNS."""
    w_meas = [split_normal(rng, base["ck"][k], w_true[j]) for j, k in enumerate(WKEYS)]
    lnm = lnm_true + rng.multivariate_normal(np.zeros(len(lnm_true)), base["cov_log"])
    off = nulls.pmns_noise("NO", rng)
    pm = clip_meas({k: pmns_true[k] + off[k] for k in pmns_true})
    return w_meas, lnm, pm


def oom_world(level, i, base, cm):
    rng = np.random.default_rng([32, code_of("OOM", level), i])
    x0 = rng.uniform(-1, 1, 2)
    link = Link(rng)
    y_ref = target4(base["best"])
    rng_mc = np.random.default_rng([32, code_of("OOM", level), i, 7])
    if level > 0:
        sig, I_link, I_meas = tune_sigma(link, level, cm, rng_mc)
        y = y_ref + link(x0)[0] + sig * rng.standard_normal(4)
    else:
        sig, I_link, I_meas = None, 0.0, 0.0
        y = y_ref + link(rng.uniform(-1, 1, 2))[0] + SCALE_G * rng.standard_normal(4)
    pmns_true = phys_from_target(y)
    w_true = [base["w_real"][0], base["w_real"][1], float(x0[0]), float(x0[1])]
    w_meas, lnm, pm = measure(rng, base, w_true, base["lnm"], pmns_true)
    W = dict(kind="OOM", label=level, i=i, w_true=w_true, w_meas=w_meas, lnm=lnm, pmns_true=pmns_true,
             pmns_meas=pm, dm_ratio=base["dm_ratio"], sigma=sig, I_link=I_link, I_meas_gauss=I_meas)
    if level > 0:
        cov = sig * sig * np.eye(4) + cm
        P, ld = np.linalg.inv(cov), np.linalg.slogdet(cov)[1]
        yhat = target4(pm)

        def oracle(rho, eta):
            d = yhat - y_ref - link(np.stack([rho, eta], 1))
            return -0.5 * np.einsum("na,ab,nb->n", d, P, d) - 0.5 * ld
        W["oracle"] = oracle
    return W


_GEN = None


def gen025():
    global _GEN
    if _GEN is None:
        spec = importlib.util.spec_from_file_location("gen025", E025 / "gen.py")
        _GEN = importlib.util.module_from_spec(spec)
        sys.modules["gen025"] = _GEN
        spec.loader.exec_module(_GEN)
    return _GEN


PMNS_WIN = dict(s12sq=(0.15, 0.5), s13sq=(0.005, 0.06), s23sq=(0.25, 0.75))  # "real-like" PMNS window
LAM_WIN = (0.15, 0.3)


def im_world(label, i, base, cm):
    """MN / F2: first draw passing the exp 025 filter, the PMNS window, lambda in LAM_WIN and (rhobar, etabar)
    in the null's box. MNu: MN with only the CKM conditions; its PMNS noise is applied in target space."""
    g = gen025()
    fam = "MN" if label == "MNu" else label
    rng = np.random.default_rng([32, code_of("IM", label), i])
    tries, batch = 0, 20000 if fam == "MN" else 2000
    while True:
        o = g.GENS[fam](rng, batch)
        c, m, y, io, ok = g.features(o)
        tries += batch
        if ok.any():
            c, m, y = c[ok], m[ok], y[ok]
            V = standard_matrix(np.arcsin(10 ** c[:, 0]), np.arcsin(10 ** c[:, 1]), np.arcsin(10 ** c[:, 2]),
                                np.arctan2(c[:, 4], c[:, 3]))
            lam, A, rho, eta = standard_to_wolfenstein(V)
            acc = (lam > LAM_WIN[0]) & (lam < LAM_WIN[1]) & (np.abs(rho) <= 1) & (np.abs(eta) <= 1)
            if label != "MNu":
                for j, (lo, hi) in enumerate(PMNS_WIN.values()):
                    acc &= (y[:, j] > lo) & (y[:, j] < hi)
            if acc.any():
                j = int(np.flatnonzero(acc)[0])
                break
    c, m, y = c[j], m[j], y[j]
    w_true = [float(lam[j]), float(A[j]), float(rho[j]), float(eta[j])]
    pmns_true = dict(s12sq=float(y[0]), s13sq=float(y[1]), s23sq=float(y[2]), dcp_deg=float(np.degrees(y[3]) % 360))
    w_meas, lnm, pm = measure(rng, base, w_true, m * np.log(10), pmns_true)
    W = dict(kind="IM", label=label, i=i, w_true=w_true, w_meas=w_meas, lnm=lnm, pmns_true=pmns_true, pmns_meas=pm,
             dm_ratio=float(10 ** y[4]), draws=tries, ckm_ctx_true=c.tolist())
    if label == "MNu":  # target-space Gaussian noise (physical NuFIT offsets are meaningless far from the real point)
        mc = json.load(open(ART / "meas_cov.json"))
        W["y_meas_target"] = (target4(pmns_true) + rng.multivariate_normal(mc["mean"], mc["cov"])).tolist()
        W["pmns_meas"] = phys_from_target(W["y_meas_target"])
    return W


def conv_points(pm, r):
    rng = np.random.default_rng(2525)  # exp 028 offsets
    pts = [[pm["s12sq"], pm["s13sq"], pm["s23sq"], np.radians(pm["dcp_deg"]) % (2 * np.pi), np.log10(r)]]
    for _ in range(A28.N_CONV):
        off = nulls.pmns_noise("NO", rng)
        q = clip_meas({k: pm[k] - off[k] for k in pm})
        pts.append([q["s12sq"], q["s13sq"], q["s23sq"], np.radians(q["dcp_deg"]) % (2 * np.pi), np.log10(r)])
    return to_target(np.array(pts))


def conv_points_target(y4, r):
    """Same 256 exp 028 offsets, applied as target-space shifts measured at the NuFIT best fit."""
    best = ev.pmns_best("NO")
    t0 = target4(best)
    rng = np.random.default_rng(2525)
    pts = [np.append(y4, np.log10(r))]
    for _ in range(A28.N_CONV):
        off = nulls.pmns_noise("NO", rng)
        d = t0 - target4(clip_meas({k: best[k] - off[k] for k in best}))
        pts.append(np.append(np.asarray(y4) - d, np.log10(r)))
    return np.array(pts)


# ---------------------------------------------------------------- pipeline (exp 028 statistic)
class Pipe:
    def __init__(self):
        D = load("NO")
        self.F = len(D["fams"])
        ptrain = np.bincount(D["fam"][D["split"] == 0], minlength=self.F) / np.sum(D["split"] == 0)
        self.lw = -np.log(ptrain) + np.log(1.0 / self.F)  # equal weights
        self.M = A28.models("NO")
        assert [m[0] for m in self.M] == [0, 1, 2]

    def T(self, ctx_ckm, m, ys):
        """T_conv (bits) for each CKM context row, world's masses m (1, 6) and PMNS points ys."""
        n, J, F = len(ctx_ckm), len(ys), self.F
        c1 = np.concatenate([ctx_ckm, np.repeat(m, n, 0)], 1)
        L1, LP1, L0, LP0 = [], [], [], []
        for s, q1, q0, k1, k0 in self.M:
            L1.append(A28.all_family_logq(q1, np.tile(ys, (n, 1)), np.repeat(c1, J, 0), F).reshape(n, J, F).astype(np.float32))
            LP1.append(k1.log_proba(c1))
            L0.append(A28.all_family_logq(q0, ys, np.repeat(m, J, 0), F))
            LP0.append(k0.log_proba(m)[0])
        E = dict(L1=np.stack(L1), LP1=np.stack(LP1), L0=np.stack(L0), LP0=np.stack(LP0))
        conv, _ = A28.statistic(E, self.lw)
        return conv


def fake_w(rng, base, w_meas):
    lam = split_normal(rng, base["ck"]["lambda"], w_meas[0])
    A = split_normal(rng, base["ck"]["A"], w_meas[1])
    return [lam, A, float(rng.uniform(*nulls.N1_PRIOR["rhobar"])), float(rng.uniform(*nulls.N1_PRIOR["etabar"]))]


def run_world(W, base, pipe):
    t0 = time.time()
    code = code_of(W["kind"], W["label"])
    m = np.array([W["lnm"] / np.log(10)])
    ys = (conv_points_target(W["y_meas_target"], W["dm_ratio"]) if "y_meas_target" in W
          else conv_points(W["pmns_meas"], W["dm_ratio"]))
    FW = np.array([fake_w(np.random.default_rng([32, code, W["i"], 1, j]), base, W["w_meas"]) for j in range(N_FAKES)])
    T_real = float(pipe.T(np.array([ctx_from_w(W["w_meas"])]), m, ys)[0])
    Tf = []
    k = 0
    for b in range(0, N_FAKES, BLOCK):
        ctx = np.array([ctx_from_w(w) for w in FW[b:b + BLOCK]])
        Tf.extend(pipe.T(ctx, m, ys).tolist())
        k = int(np.sum(~(np.array(Tf) < T_real)))  # a NaN fake counts as >= (against detection)
        if k > MAX_HITS:
            break
    Tf = np.array(Tf)
    n_eval = len(Tf)
    det = bool(n_eval == N_FAKES and nulls.decide(k, N_FAKES)["passed"])
    k300 = int(np.sum(~(Tf[:300] < T_real)))
    det300 = bool(n_eval >= 300 and k300 == 0)
    out = {kk: v for kk, v in W.items() if kk not in ("oracle", "lnm")}
    out.update(lnm=W["lnm"].tolist(), T_real=T_real, n_fakes_eval=n_eval, k_ge=k, rank_eval=k / n_eval,
               stopped_early=n_eval < N_FAKES, detected=det, k300=k300, detected_300=det300,
               T_fake_median=float(np.nanmedian(Tf)), n_nan_fakes=int(np.isnan(Tf).sum()), T_fakes=np.round(Tf, 4).tolist(),
               fakes_rho_eta_eval=np.round(FW[:n_eval, 2:], 4).tolist())
    if "oracle" in W:
        To = W["oracle"](np.array([W["w_meas"][2]]), np.array([W["w_meas"][3]]))[0]
        Tof = W["oracle"](FW[:, 2], FW[:, 3])
        ko = int(np.sum(Tof >= To))
        out.update(oracle_k_ge=ko, oracle_rank=ko / N_FAKES, oracle_detected=bool(nulls.decide(ko, N_FAKES)["passed"]))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv):
    if argv[0] == "meas_cov":
        print(meas_cov())
        return
    torch.set_num_threads(int(os.environ.get("EXP032_THREADS", 1)))
    (ART / "inj").mkdir(parents=True, exist_ok=True)
    base = real_base()
    cm = np.array(json.load(open(ART / "meas_cov.json"))["cov"])
    pipe = None
    for task in argv[1:]:
        kind, label, i = task.split(":")
        i = int(i)
        path = ART / "inj" / f"{kind}_{label}_{i}.json"
        if path.exists():
            continue
        if pipe is None:
            pipe = Pipe()
        W = oom_world(float(label), i, base, cm) if kind == "OOM" else im_world(label, i, base, cm)
        res = run_world(W, base, pipe)
        json.dump(res, open(path, "w"))
        print(f"{time.strftime('%H:%M:%S')} {task}: I_link {res.get('I_link')} T {res['T_real']:.3f} "
              f"k {res['k_ge']}/{res['n_fakes_eval']} det {res['detected']} det300 {res['detected_300']} "
              f"oracle {res.get('oracle_rank')} ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

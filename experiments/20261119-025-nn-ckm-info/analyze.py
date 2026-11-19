"""Information gain, real-data log ratio, N1 calibration and family-drop sensitivity (exp 025).

Usage: python analyze.py [n_fakes]   -> results.json (+ printed tables)
Measured values only through src loaders: CKM global fit (src.data.load_ckm), M_Z mass ratios
(src.evaluate.mass_ratio_targets), NuFIT NO best fit and likelihood shape (src.evaluate.pmns_best,
src.nulls.pmns_noise), dm ratio (src.data.load_pmns). N1 fakes from src.nulls.n1_universe.
"""
from __future__ import annotations

import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import math
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from src import data as sd  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import nulls  # noqa: E402
from src.infogain import cp_interval, logmeanexp, null_info, pmns_data_bits  # noqa: E402
from train import ART, Model, load, to_target  # noqa: E402
from gen import FAMILIES, MR_KEYS  # noqa: E402

LN2 = math.log(2)
ORD = "NO"
N_CONV = 256  # PMNS points drawn from the NuFIT likelihood shape for the convolved ratio


def real_inputs():
    c = sd.load_ckm()
    s12, s13, s23, d = (c[k]["value"] for k in ("s12", "s13", "s23", "delta_rad"))
    ckm = np.array([[np.log10(s12), np.log10(s13), np.log10(s23), np.cos(d), np.sin(d)]])
    mr, _, _ = ev.mass_ratio_targets()
    m = np.array([[np.log10(mr[k]) for k in MR_KEYS]])
    p = sd.load_pmns(ORD)
    r = p["dm21sq_1e-5eV2"]["value"] * 1e-5 / (p["dm3l_sq_1e-3eV2"]["value"] * 1e-3)
    best = ev.pmns_best(ORD)
    y = np.array([[best["s12sq"], best["s13sq"], best["s23sq"], np.radians(best["dcp_deg"]) % (2 * np.pi),
                   np.log10(r)]])
    rng = np.random.default_rng(2525)
    ys = []
    for _ in range(N_CONV):
        off = nulls.pmns_noise(ORD, rng)
        ys.append([best["s12sq"] - off["s12sq"], best["s13sq"] - off["s13sq"], best["s23sq"] - off["s23sq"],
                   np.radians(best["dcp_deg"] - off["dcp_deg"]) % (2 * np.pi), np.log10(r)])
    return ckm, m, to_target(y), to_target(np.array(ys)), dict(s12=s12, s13=s13, s23=s23, delta=d, r=r, **best)


def n1_ckms(n, seed=(25, 1)):
    base = nulls.Universe.real(ORD)
    out, w = [], []
    for i in range(n):
        u = nulls.n1_universe(base, np.random.default_rng([*seed, i]))
        c = u.load_ckm()
        s12, s13, s23, d = (c[k]["value"] for k in ("s12", "s13", "s23", "delta_rad"))
        out.append([np.log10(s12), np.log10(s13), np.log10(s23), np.cos(d), np.sin(d)])
        w.append([u.truth["wolfenstein"][k] for k in ("lambda", "A", "rhobar", "etabar")])
    return np.array(out), np.array(w)


def hpd_rank(model, y, c, m=4000):
    """Fraction of q samples with higher density than y (0 = mode, 1 = far tail), per row."""
    s = model.sample(c, m)  # (m, n, 5) standardised
    n = c.shape[0]
    ls = []
    for j in range(n):
        ss = s[:, j] * model.ys + model.ym
        ls.append(model.log_prob(ss, np.repeat(c[j:j + 1], m, 0)))
    ls = np.array(ls)  # (n, m)
    ly = model.log_prob(y, c)
    return (ls > ly[:, None]).mean(1)


def calibration(q, D, mask, rng, n=1000, m=300):
    idx = rng.choice(np.flatnonzero(mask), n, replace=False)
    c = (D["c1"] if q.cm.size == 11 else D["c0"])[idx]
    r = hpd_rank(q, D["y"][idx], c, m)
    return {f"{lv:.2f}": float(np.mean(r <= lv)) for lv in (0.5, 0.68, 0.9, 0.95)}


def predictive_summary(q, c, m=20000):
    s = q.sample(c, m)[:, 0] * q.ys + q.ym
    sig = lambda x: 1 / (1 + np.exp(-x))
    cols = dict(s12sq=sig(s[:, 0]), s13sq=sig(s[:, 1]), s23sq=sig(s[:, 2]), dcp_deg=360 * sig(s[:, 3]),
                log10_r=s[:, 4])
    return {k: [float(np.quantile(v, p)) for p in (0.16, 0.5, 0.84)] for k, v in cols.items()}


def stat(q1, q0, ckm, m, y, ys):
    """T_point = log q1 - log q0 at the NuFIT best fit; T_conv the same for the predictive densities
    averaged over N_CONV points from the NuFIT likelihood shape (nats). ckm may hold many rows."""
    k = len(ckm)
    c1 = np.concatenate([ckm, np.repeat(m, k, 0)], 1)
    lp1 = q1.log_prob(np.repeat(y, k, 0), c1)
    lp0 = q0.log_prob(y, m)[0]
    yy = np.tile(ys, (k, 1))
    cc = np.repeat(c1, len(ys), 0)
    l1 = q1.log_prob(yy, cc).reshape(k, len(ys))
    l0 = q0.log_prob(ys, np.repeat(m, len(ys), 0))
    conv1 = np.array([logmeanexp(r) for r in l1])
    return lp1 - lp0, conv1 - logmeanexp(l0), lp1, conv1


def support_mask(D, ckm_fakes, mask):
    """N1 fakes whose CKM context lies inside the training CKM-context cloud: 10-NN distance (in
    training-standardised units) <= the 99th percentile of the training points' own 10-NN distance."""
    from sklearn.neighbors import NearestNeighbors
    X = D["c1"][mask][:, :5]
    rng = np.random.default_rng(7)
    X = X[rng.choice(len(X), min(60000, len(X)), replace=False)]
    mu, sd_ = X.mean(0), X.std(0)
    nn = NearestNeighbors(n_neighbors=11).fit((X - mu) / sd_)
    dref = nn.kneighbors((X[:5000] - mu) / sd_)[0][:, -1]
    thr = np.quantile(dref, 0.99)
    d = nn.kneighbors((ckm_fakes - mu) / sd_, n_neighbors=10)[0][:, -1]
    return d <= thr, float(thr), d


def one_tag(tag, D, real, fakes, rng, full=False):
    q1 = Model.load(ART / "flows" / f"{tag}_q1.pt")
    q0 = Model.load(ART / "flows" / f"{tag}_q0.pt")
    ckm, m, y, ys, _ = real
    te = D["split"] == 2
    lp1 = q1.log_prob(D["y"][te], D["c1"][te])
    lp0 = q0.log_prob(D["y"][te], D["c0"][te])
    g = (lp1 - lp0) / LN2
    fam_te = D["fam"][te]
    info = {}
    for i, f in enumerate(D["fams"]):
        gi = g[fam_te == i]
        info[f] = dict(I_bits=float(gi.mean()), se=float(gi.std() / np.sqrt(len(gi))), n=int(len(gi)))
    info["ALL"] = dict(I_bits=float(g.mean()), se=float(g.std() / np.sqrt(len(g))), n=int(len(g)))
    T_pt, T_cv, _, _ = stat(q1, q0, ckm, m, y, ys)
    F_pt, F_cv, _, _ = stat(q1, q0, fakes["ckm"], m, y, ys)
    out = dict(info=info, T_point_bits=float(T_pt[0] / LN2), T_conv_bits=float(T_cv[0] / LN2))
    for name, Tr, Tf in (("point", T_pt[0], F_pt), ("conv", T_cv[0], F_cv)):
        for sub, msk in (("all", np.ones(len(Tf), bool)), ("in_support", fakes["support"])):
            tf = Tf[msk]
            h = int(np.sum(tf >= Tr))
            lo, hi = cp_interval(h, len(tf))
            ni = null_info(-2 * Tr, -2 * tf)  # chi2-like = -2 T: B_ckm = log2 q1(real CKM) / mean q1(fake CKM)
            out[f"rank_{name}_{sub}"] = dict(n=int(len(tf)), n_fakes_ge_real=h, frac=h / len(tf), cp95=[lo, hi],
                                             B_ckm_bits=ni["B_ckm"], S_rank_bits=ni["S_rank"],
                                             fake_T_bits_median=float(np.median(tf) / LN2),
                                             fake_T_bits_q05_q95=[float(np.quantile(tf, .05) / LN2),
                                                                  float(np.quantile(tf, .95) / LN2)])
    if full:
        c1 = np.concatenate([ckm, m], 1)
        out["hpd_rank_real_q1"] = float(hpd_rank(q1, y, c1, 20000)[0])
        out["hpd_rank_real_q0"] = float(hpd_rank(q0, y, m, 20000)[0])
        out["pred_q1"] = predictive_summary(q1, c1)
        out["pred_q0"] = predictive_summary(q0, m)
        out["calib_q1"] = calibration(q1, D, te, rng)
        out["calib_q0"] = calibration(q0, D, te, rng)
        out["fake_T_point_bits"] = (F_pt / LN2).round(4).tolist()
    return out


def main(n_fakes=2000):
    torch.set_num_threads(4)
    rng = np.random.default_rng(250)
    D = load("main")
    real = real_inputs()
    ckm_f, w_f = n1_ckms(n_fakes)
    sup, thr, dist = support_mask(D, ckm_f, D["split"] == 0)
    fakes = dict(ckm=ckm_f, support=sup)
    res = dict(ordering=ORD, n_fakes=n_fakes, n_conv=N_CONV, real=real[4],
               pmns_data_bits=pmns_data_bits(ORD),
               support=dict(threshold=thr, frac_in_support=float(sup.mean())),
               real_ckm_knn_dist=float(support_mask(D, real[0], D["split"] == 0)[2][0]))
    res["main"] = one_tag("main", D, real, fakes, rng, full=True)
    print(json.dumps({k: v for k, v in res["main"].items() if k != "fake_T_point_bits"}, indent=1), flush=True)
    res["drop"] = {}
    for f in FAMILIES:
        p = ART / "flows" / f"drop_{f}_q1.pt"
        if p.exists() and (ART / "flows" / f"drop_{f}_q0.pt").exists():
            Dd = load(f"drop_{f}")
            res["drop"][f] = one_tag(f"drop_{f}", Dd, real, fakes, rng)
            r = res["drop"][f]
            print(f"drop {f}: T_point {r['T_point_bits']:.2f} T_conv {r['T_conv_bits']:.2f} "
                  f"rank_point {r['rank_point_all']['frac']:.3f} rank_conv {r['rank_conv_all']['frac']:.3f}", flush=True)
    for extra in ("seed1",):
        if (ART / "flows" / f"{extra}_q1.pt").exists():
            res[extra] = one_tag(extra, load("main"), real, fakes, rng)
    res["n1_wolfenstein_first10"] = w_f[:10].tolist()
    json.dump(res, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)

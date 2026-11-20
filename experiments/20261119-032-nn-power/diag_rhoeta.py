"""Exp 032 post-hoc diagnostic (not pre-registered): how much PMNS information do (rhobar, etabar) carry
inside the F2 and MN generators, at fixed lambda, A and fixed other model parameters?

For each base draw from the family prior, (rhobar, etabar) is scanned on a 25 x 25 grid over [-1, 1]^2
(the lambda-matched null's box) with everything else fixed, giving PMNS(x) in exp 028 target space.
Reported per base: s.d. of each target coordinate over the grid, and the oracle information
I(y_meas; rhobar, etabar) for y_meas = PMNS(x) + today's measurement noise (Gaussian, artifacts/032/meas_cov.json),
x uniform on the box (grid quadrature, 2000 MC draws). This is the most a perfect estimator could extract
with the lambda-matched null for that world.
Usage: python diag_rhoeta.py [n_base]  -> diag_rhoeta.json
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

import inject as I
from src.mixing import extract_angles, wolfenstein_to_standard

G = np.linspace(-1, 1, 25)
GX = np.stack(np.meshgrid(G, G, indexing="ij"), -1).reshape(-1, 2)


def target_from(s12sq, s13sq, s23sq, dcp):
    lg = lambda p: np.log(p) - np.log1p(-p)
    d = np.clip(np.mod(dcp, 2 * np.pi) / (2 * np.pi), 1e-6, 1 - 1e-6)
    return np.stack([lg(s12sq), lg(s13sq), lg(s23sq), lg(d)], 1)


def info_tab(Y, cm, rng, n=2000):
    """I(y; x) bits, y = Y[x] + N(0, cm), x uniform over the grid rows of Y."""
    L = np.linalg.cholesky(cm)
    Li = np.linalg.inv(L)
    Yw = Y @ Li.T
    idx = rng.integers(len(Y), size=n)
    eps = rng.standard_normal((n, 4))
    y = Yw[idx] + eps
    lc = -0.5 * np.sum(eps ** 2, 1)
    q = -0.5 * (np.sum(y ** 2, 1)[:, None] + np.sum(Yw ** 2, 1)[None] - 2 * y @ Yw.T)
    m = q.max(1, keepdims=True)
    lm = m[:, 0] + np.log(np.mean(np.exp(q - m), 1))
    return float(np.mean(lc - lm) / math.log(2))


def angles_grid(lam, A):
    return np.array([wolfenstein_to_standard(lam, A, r, e) for r, e in GX], float)  # (625, 4)


def f2_base(rng, g, f2, cfgs):
    lam, A = rng.uniform(0.15, 0.3), rng.uniform(0.3, 2.0)
    cfg = cfgs[rng.integers(len(cfgs))]
    th_nu = rng.uniform(0, 0.5) if cfg["tier"] == "TM" else None
    phi = rng.uniform(0, 2 * np.pi)
    ang = angles_grid(lam, A)
    out = []
    for a in ang:
        U = f2.full_matrix(cfg, tuple(float(x) for x in a), phi, th_nu)
        l12, l13, l23, dl = (float(x) for x in extract_angles(U))
        out.append([np.sin(l12) ** 2, np.sin(l13) ** 2, np.sin(l23) ** 2, dl])
    o = np.array(out)
    return dict(tier=cfg["tier"], lam=lam, A=A), o


def mn_base(rng, g, so):
    lo, hi = g.mn_box_broad()
    while True:
        th = rng.uniform(lo, hi)
        lam, A = rng.uniform(0.15, 0.3), rng.uniform(0.3, 2.0)
        ang = angles_grid(lam, A)
        T = np.repeat(th[None], len(ang), 0)
        T[:, 5] = np.sin(ang[:, 0])
        T[:, 6] = np.log10(np.sin(ang[:, 1]))
        T[:, 7] = np.log10(np.sin(ang[:, 2]))
        T[:, 8] = ang[:, 3]
        ob = g.obs_from_mats(*so.mn_mats(T))
        c, m, y, io, ok = g.features(ob)
        if ok.mean() > 0.9:  # base passes the exp 025 filter over (almost) the whole box
            o = np.stack([ob["s12sq"], ob["s13sq"], ob["s23sq"], ob["dcp"]], 1)
            return dict(lam=lam, A=A, frac_ok=float(ok.mean())), o[ok]


def summarize(rows):
    I_ = np.array([r["I_meas"] for r in rows])
    return dict(n=len(rows), I_meas_q10_50_90=np.quantile(I_, [.1, .5, .9]).round(3).tolist(),
                frac_I_ge_4=float(np.mean(I_ >= 4)), frac_I_ge_7=float(np.mean(I_ >= 7)),
                sd_target_median=np.median([r["sd"] for r in rows], 0).round(4).tolist())


def main(n=100):
    g = I.gen025()
    f2 = g._load("f2lib", I.ROOT / "experiments" / "20261118-009-f2-clebsch" / "f2lib.py")
    so = g._so10()
    cfgs = f2.configs()
    cm = np.array(json.load(open(I.ART / "meas_cov.json"))["cov"])
    rng = np.random.default_rng(3299)
    res = {}
    for fam in ("F2", "MN"):
        rows, skipped = [], 0
        while len(rows) < n:
            meta, o = f2_base(rng, g, f2, cfgs) if fam == "F2" else mn_base(rng, g, so)
            ok = np.all(np.isfinite(o), 1) & np.all((o[:, :3] > 1e-6) & (o[:, :3] < 1 - 1e-6), 1)
            if ok.sum() < 300:  # degenerate base (e.g. s13 = 0 exactly in an LO F2 pattern): skipped, counted
                skipped += 1
                continue
            Y = target_from(*o[ok].T)
            centre = np.median(o[ok], 0)
            reallike = bool((0.15 < centre[0] < 0.5) and (0.005 < centre[1] < 0.06) and (0.25 < centre[2] < 0.75))
            rows.append(dict(meta, sd=Y.std(0).tolist(), I_meas=info_tab(Y, cm, rng), reallike=reallike))
        res[fam] = dict(skipped_degenerate=skipped, all=summarize(rows), reallike=summarize([r for r in rows if r["reallike"]]) if any(
            r["reallike"] for r in rows) else None, meas_sd=np.sqrt(np.diag(cm)).round(4).tolist())
        print(fam, json.dumps(res[fam]), flush=True)
    json.dump(res, open(I.HERE / "diag_rhoeta.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)

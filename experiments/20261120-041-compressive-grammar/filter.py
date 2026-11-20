"""Exp 041 step 2: pre-search filter from the quark sector (exp 039's lesson).

The 8 continuous parameters are shared by all 32 entries and the 8 quark observables depend on
them alone (not on the lepton choices). fit_ok needs total chi2 <= chi2_crit(p = 0.05, dof 7) =
14.07 and every |pull| < 2, so a passing point must have chi2_quark <= 14.07 and sin^2(theta23)
inside the NuFIT 6.1 NO 1D 2 sigma range. Rule (fixed before any lepton number was looked at):

  kill an entry if no quark-allowed point (chi2_quark <= 14.07) gives s23sq in the 2 sigma range.

Quark-allowed points: quark-only DE fits from several seeds (all solution branches found), then
Gaussian (Laplace) clouds around each branch at covariance scales 1, 4, 16 x the inverse
half-Hessian, keeping points with chi2_quark <= 14.07. Lepton predictions per entry come from the
batched builder. Sampling-limited (tails of the quark region are not covered); the exact fits of
every entry (fit41.py) check the filter.
Output: filter.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fit41 as F  # noqa: E402
import grammar as g  # noqa: E402
from scipy.stats import chi2 as chi2d  # noqa: E402
from src import evaluate as E  # noqa: E402
from src.yukawa import observables_batch  # noqa: E402

CRIT = float(chi2d.ppf(0.95, g.N_OBS - g.K_CONT))  # 14.07


def s23_range(level=4.0):
    sc = E.Scorer(["s23sq"], "NO")
    xs = np.linspace(0.3, 0.7, 4001)
    c = np.array([sc.chi2({"s23sq": x}) for x in xs])
    ok = xs[c <= level]
    return float(ok.min()), float(ok.max())


def hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H, f0


def main(seeds=4, n_per=4000):
    t0 = time.time()
    lo2, hi2 = s23_range(4.0)
    lo3, hi3 = s23_range(9.0)
    print(f"s23sq NO 2sigma [{lo2:.4f}, {hi2:.4f}], 3sigma [{lo3:.4f}, {hi3:.4f}]; crit {CRIT:.2f}", flush=True)
    qfits = []
    for s in range(1, seeds + 1):
        r = F.fit(0, use=g.QUSE, seed=s, popsize=15, maxiter=250)
        qfits.append(r)
        print("quark fit seed", s, "chi2", round(r["chi2"], 4), {k: round(v, 4) for k, v in r["params"].items()},
              f"{time.time() - t0:.0f}s", flush=True)
    fq = F.objective(0, g.QUSE)
    rng = np.random.default_rng(41)
    pts, chis = [], []
    branches = []
    for r in qfits:
        x0 = np.array([r["params"][p] for p in g.PARAMS])
        if any(np.allclose(x0, b, atol=1e-2) for b in branches):
            continue
        branches.append(x0)
        H, f0 = hessian(fq, x0)
        w, V = np.linalg.eigh(0.5 * H)
        w = np.clip(w, 1e-6, None)
        C = (V / w) @ V.T
        L = np.linalg.cholesky(C + 1e-12 * np.eye(len(x0)))
        for scale in (1.0, 4.0, 16.0):
            X = x0 + np.sqrt(scale) * rng.standard_normal((n_per, len(x0))) @ L.T
            X = np.clip(X, F.LO, F.HI)
            c = np.array([fq(x) for x in X])
            keep = c <= CRIT
            pts.append(X[keep])
            chis.append(c[keep])
            print(f"branch {len(branches)} scale {scale}: {keep.sum()}/{n_per} quark-allowed", flush=True)
    X = np.concatenate(pts)
    cq = np.concatenate(chis)
    out = dict(crit=CRIT, s23_2sigma=[lo2, hi2], s23_3sigma=[lo3, hi3], n_points=int(len(X)),
               quark_fits=[dict(seed=r["seed"], chi2=r["chi2"], params=r["params"], pred=r["pred"]) for r in qfits],
               n_branches=len(branches), entries=[])
    sc_l = E.Scorer(g.LUSE + ["mr_e_mu", "mr_mu_tau"], "NO", mass_floor=g.FLOOR)
    for e in g.ENTRIES:
        B = len(X)
        reps = np.tile(np.array(e["reps"]), (B, 1))
        nu = np.full(B, g.NU_OPTS.index(e["nu"]))
        with np.errstate(all="ignore"):
            o = observables_batch(*g.build_batch(X, reps, nu), ordering="NO")
        s23 = o["s23sq"]
        fin = np.isfinite(s23)
        in2 = fin & (s23 >= lo2) & (s23 <= hi2)
        # cheapest total chi2 among the quark-allowed points (lepton + charged-lepton terms, soft)
        tot = np.full(B, np.inf)
        idx = np.where(fin)[0]
        for i in idx[np.argsort(-np.abs(s23[idx] - 0.5))[::-1][:400]] if len(idx) else []:
            p = {k: o[k][i] for k in g.LUSE + ["mr_e_mu", "mr_mu_tau"]}
            try:
                tot[i] = cq[i] + sc_l.chi2_soft(p, include_finite=True)
            except ValueError:
                pass
        rec = dict(eid=e["id"], entry=g.name(e), n_finite=int(fin.sum()),
                   s23_min=float(np.nanmin(s23)) if fin.any() else None,
                   s23_max=float(np.nanmax(s23)) if fin.any() else None,
                   n_in_2sigma=int(in2.sum()), killed=bool(not in2.any()),
                   min_total_soft_chi2_sampled=float(np.min(tot)) if np.isfinite(tot).any() else None)
        out["entries"].append(rec)
        print(rec, flush=True)
    out["n_killed"] = int(sum(r["killed"] for r in out["entries"]))
    out["secs"] = time.time() - t0
    json.dump(out, open(HERE / "filter.json", "w"), indent=1)
    print("killed", out["n_killed"], "of", g.G, flush=True)


if __name__ == "__main__":
    main()

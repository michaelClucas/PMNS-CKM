"""Global fit: many random starts of batched Levenberg-Marquardt + basin hopping around the best minima.

Usage: python global_fit.py <model MN|FN> <target DR|ours> <n_starts> <seed> <n_proc> <tag> [subset all|ql|nodl] [errs]
  subset 'ql' fits only the 12 quark + charged-lepton observables (target 'ours' only); 'nodl' drops the PMNS
  Dirac phase (1306.4468 did not fit it).
  errs: 'own' (default) or 'drrel' (our central values with 1306.4468's relative errors, diagnostic).
Writes artifacts/so10/fit_<tag>.npz (all minima) and fit_<tag>.json (best point, pulls, minima histogram).
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import so10fit as S  # noqa: E402

CHUNK = 64


def dr_rel_sigma():
    """Our z vector with errors converted from 1306.4468's relative errors (Table 3 / Table 2)."""
    v, e = S.DR_VAL, S.DR_ERR
    rel = e / v
    lr = lambda a, b: np.hypot(rel[a], rel[b]) / np.log(10)
    # md ms mb mu mc mt (0-5), me mmu mtau (15-17)
    return np.array([lr(3, 4), lr(4, 5), lr(0, 1), lr(1, 2), lr(2, 5), lr(15, 16), lr(16, 17), lr(17, 2),
                     e[6], rel[8] / np.log(10), rel[7] / np.log(10), e[9],
                     e[12], e[14], e[13], 0.375, np.hypot(rel[10], rel[11]) / np.log(10)])


def make_problem(model, target, subset, errs):
    idx = {"ql": S.QL_IDX, "nodl": [j for j in range(17) if j != 15]}.get(subset)
    sig = dr_rel_sigma() if errs == "drrel" else None
    return S.Problem(model, "DR" if target == "DR" else "ours", idx=idx, sigma_override=sig)


def box(prob):
    return S.mn_box(prob.target, prob.val) if prob.model == "MN" else S.fn_box(prob.target)


def fix_nuscale(prob, X):
    if prob.target != "DR":
        return X
    o = S.raw_obs(*prob.mats(X[:, :len(prob.base_names)]))
    with np.errstate(all="ignore"):
        X[:, -1] = 0.5 * np.log10(S.DR_VAL[11] / np.maximum(o["mn"][:, 2] ** 2 - o["mn"][:, 0] ** 2, 1e-300))
    X[:, -1] = np.where(np.isfinite(X[:, -1]), X[:, -1], 0.0)
    return X


def starts(prob, rng, B):
    lo, hi = box(prob)
    if prob.model == "MN":
        X = S.mn_start_from_data(prob, rng, B, lo, hi)
    else:
        X = lo + (hi - lo) * rng.random((B, len(lo)))
        # hierarchical starts: |h1| << |h2| << 1 typical, other entries log-uniform over many decades
        X[:, :11] = np.sign(rng.standard_normal((B, 11))) * rng.uniform(5, 16.6, (B, 11))
        if prob.target == "DR":
            X[:, -2] = rng.uniform(-3, 1, B)
    return fix_nuscale(prob, X)


def work(args):
    model, target, subset, errs, X0, n_iter = args
    prob = make_problem(model, target, subset, errs)
    lo, hi = box(prob)
    X, C = S.batched_lm(prob, X0, lo, hi, n_iter=n_iter)
    return X, C


def run_batches(pool, model, target, subset, errs, X0, n_iter):
    jobs = [(model, target, subset, errs, X0[i:i + CHUNK], n_iter) for i in range(0, len(X0), CHUNK)]
    res = pool.map(work, jobs)
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def perturb(prob, X, rng, scale):
    lo, hi = box(prob)
    P = X + scale * rng.standard_normal(X.shape) * np.where(prob.per, 1.0, 0.15 * np.minimum(hi - lo, 4.0))
    return S.project(P, lo, hi, prob.per)


def main():
    model, target, n, seed, nproc, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), \
        int(sys.argv[5]), sys.argv[6]
    subset = sys.argv[7] if len(sys.argv) > 7 else "all"
    errs = sys.argv[8] if len(sys.argv) > 8 else "own"
    rng = np.random.default_rng(seed)
    prob = make_problem(model, target, subset, errs)
    t0 = time.time()
    X0 = starts(prob, rng, n)
    allX, allC = [], []
    with Pool(nproc) as pool:
        X, C = run_batches(pool, model, target, subset, errs, X0, 400)
        allX.append(X), allC.append(C)
        q = np.quantile(C, [0, 0.01, 0.1, 0.5])
        print(f"[{tag}] stage A: {n} starts, {time.time() - t0:.0f}s, chi2 min/1%/10%/50% = {np.round(q, 2)}", flush=True)
        # basin hopping around the best minima
        for rnd in range(4):
            o = np.argsort(C)[:32]
            base = np.repeat(X[o], 12, 0)
            Xp = perturb(prob, base, rng, scale=[0.5, 0.3, 0.15, 0.05][rnd])
            Xp = fix_nuscale(prob, Xp)
            Xb, Cb = run_batches(pool, model, target, subset, errs, Xp, 500)
            allX.append(Xb), allC.append(Cb)
            X, C = np.concatenate([X, Xb]), np.concatenate([C, Cb])
            print(f"[{tag}] hop {rnd}: best {C.min():.3f}, 10 best {np.round(np.sort(C)[:10], 2)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        # final polish of the 16 best
        o = np.argsort(C)[:16]
        Xf, Cf = run_batches(pool, model, target, subset, errs, X[o], 3000)
        X, C = np.concatenate([X, Xf]), np.concatenate([C, Cf])
    S.ART.mkdir(parents=True, exist_ok=True)
    np.savez(S.ART / f"fit_{tag}.npz", theta=X, chi2=C, stageA_chi2=allC[0])
    ib = int(np.argmin(C))
    p, ob = prob.pulls(X[ib:ib + 1])
    used = prob.idx
    summary = dict(model=model, target=target, subset=subset, errs=errs, n_starts=n, seed=seed,
                   n_params=prob.n, n_obs=len(used), best_chi2=float(C[ib]),
                   best_theta=dict(zip(prob.names, map(float, X[ib]))),
                   best_pulls={prob.obs_names[j]: float(p[0, j]) for j in range(len(prob.obs_names))},
                   best_pred={prob.obs_names[j]: float(prob.predict(X[ib:ib + 1])[0][0, j]) for j in range(len(prob.obs_names))},
                   best_IO=bool(ob["io"][0]),
                   stageA_quantiles=dict(zip(["min", "q01", "q10", "q50"], map(float, np.quantile(allC[0], [0, .01, .1, .5])))),
                   n_minima_below=dict({str(t): int((C < t).sum()) for t in (2, 5, 10, 20, 50, 100)}),
                   wall_s=time.time() - t0)
    (S.HERE / f"fit_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"[{tag}] best chi2 {C[ib]:.4f} (N={len(used)}, k={prob.n}) IO={summary['best_IO']}  pulls: "
          + " ".join(f"{k}={v:+.2f}" for k, v in summary["best_pulls"].items()), flush=True)


if __name__ == "__main__":
    main()

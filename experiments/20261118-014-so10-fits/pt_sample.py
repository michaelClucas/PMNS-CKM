"""Parallel-tempering DE-MC for p(theta | data subset) in the SO(10) models (target 'ours', GUT scale, NO vector).

Usage: python pt_sample.py <model MN|FN> <subset ql|all> <n_iter> <seed> <tag> [n_temps] [walkers] [own|drrel]
Prior: uniform on the search box (MN: so10fit.mn_box, quark windows centred on the data as in exp 011, phases
uniform, lg r in [0,4], lg|s| in [-2,2]; FN: uniform in the sinh-coordinates of so10fit.fn_box).
Ladder: n_temps levels, beta geometric from 1 to 3e-5. Within a level: DE-MC proposals (ter Braak 2006) with a
random-subspace mask, Gaussian jitter, and 10% "snooker-free" big jumps (gamma = 1). Adjacent-level swaps every
step (random pairing). All levels start from independent prior draws, so the cold chain's mode weights come from
tempering, not from seeds. Likelihood 'ql' = 12 quark + charged-lepton observables: PMNS never enters.
Writes artifacts/so10/pt_<tag>.npz (cold-level samples of the second half, their z vectors and IO flag).
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import so10fit as S  # noqa: E402


def main():
    model, subset, n_iter, seed, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    W = int(sys.argv[7]) if len(sys.argv) > 7 else 24
    rng = np.random.default_rng(seed)
    idx = S.QL_IDX if subset == "ql" else None
    errs = sys.argv[8] if len(sys.argv) > 8 else "own"
    if errs == "drrel":   # our central values, 1306.4468 relative errors (diagnostic)
        from global_fit import dr_rel_sigma
        prob = S.Problem(model, "ours", idx=idx, sigma_override=dr_rel_sigma())
    else:
        prob = S.Problem(model, "ours", idx=idx)
    lo, hi = (S.mn_box("ours", prob.val) if model == "MN" else S.fn_box("ours"))
    per = prob.per
    d = len(lo)
    betas = np.geomspace(1.0, 3e-5, K)
    bvec = np.repeat(betas, W)                       # level-major layout: level k = rows k*W .. k*W+W-1
    X = lo + (hi - lo) * rng.random((K * W, d))
    if model == "MN":
        X = S.mn_start_from_data(prob, rng, K * W, lo, hi)
    c2 = prob.chi2(X)
    g0 = 2.38 / np.sqrt(2 * d)
    keep_th, keep_c2 = [], []
    acc = np.zeros(K)
    sw_acc, sw_n = np.zeros(K - 1), np.zeros(K - 1)
    t0 = time.time()
    lev = np.arange(K * W) // W
    for it in range(n_iter):
        # DE-MC within each level
        a = lev * W + rng.integers(W, size=K * W)
        b = lev * W + (a - lev * W + 1 + rng.integers(W - 1, size=K * W)) % W
        big = rng.random(K * W) < 0.1
        gam = np.where(big, 1.0, g0 * np.exp(0.5 * rng.standard_normal(K * W)))[:, None]
        diff = X[a] - X[b]
        diff = np.where(per, S.wrap(diff), diff)
        mask = rng.random(X.shape) < np.where(rng.random((K * W, 1)) < 0.5, 1.0, 0.25)
        # hot levels also get prior-scale jitter so they can cross between distant basins
        jit = (1e-5 + 5e-3 * (bvec < 1e-2))[:, None] * (hi - lo)
        prop = X + mask * (gam * diff + jit * rng.standard_normal(X.shape))
        prop = np.where(per, np.mod(prop, 2 * np.pi), prop)
        inb = np.all(per | ((prop >= lo) & (prop <= hi)), 1)
        c2p = np.full(K * W, np.inf)
        if inb.any():
            c2p[inb] = prob.chi2(prop[inb])
        ok = np.log(rng.random(K * W)) < -0.5 * bvec * (c2p - c2)
        X[ok], c2[ok] = prop[ok], c2p[ok]
        acc += np.bincount(lev[ok], minlength=K) / W
        # swaps between adjacent levels (even or odd pairs alternately, random walker pairing)
        for k in range(it % 2, K - 1, 2):
            i = k * W + rng.permutation(W)
            j = (k + 1) * W + np.arange(W)
            la = (betas[k] - betas[k + 1]) * (-0.5) * (c2[j] - c2[i])
            sw = np.log(rng.random(W)) < la
            sw_acc[k] += sw.mean()
            sw_n[k] += 1
            ii, jj = i[sw], j[sw]
            X[ii], X[jj] = X[jj].copy(), X[ii].copy()
            c2[ii], c2[jj] = c2[jj].copy(), c2[ii].copy()
        if it >= n_iter // 2 and it % 20 == 0:
            keep_th.append(X[:W].copy())
            keep_c2.append(c2[:W].copy())
        if (it + 1) % max(1, n_iter // 20) == 0:
            print(f"  [{tag}] it {it + 1}/{n_iter} cold chi2 med {np.median(c2[:W]):.2f} min {c2[:W].min():.2f} | "
                  f"acc cold {acc[0] / (it + 1):.3f} hot {acc[-1] / (it + 1):.3f} | swap acc min "
                  f"{(sw_acc / np.maximum(sw_n, 1)).min():.2f} | t={time.time() - t0:.0f}s", flush=True)
    th = np.concatenate(keep_th)
    z, o = prob.predict(th)
    S.ART.mkdir(parents=True, exist_ok=True)
    np.savez(S.ART / f"pt_{tag}.npz", theta=th, chi2=np.concatenate(keep_c2), z=z, io=o["io"],
             walker=np.tile(np.arange(W), len(keep_th)), betas=betas,
             swap_acc=sw_acc / np.maximum(sw_n, 1), acc=acc / n_iter)
    print(f"[{tag}] saved {len(th)} cold samples; swap acceptance per pair "
          f"{np.round(sw_acc / np.maximum(sw_n, 1), 2)}", flush=True)


if __name__ == "__main__":
    main()

"""Multi-start least-squares fits of the SO(10) variants to the GUT-scale data vector.

target 'ql':  12 quark + charged-lepton observables only (PMNS held out)
target 'all': all 17 observables (NO)
Writes artifacts/f3/fits_<variant>_<target>.npz (theta, chi2) and prints a summary.
Usage: python fit_ms.py <variant> <target> <n_starts> <seed> [fromql|random] [n_proc]
'fromql' starts from jittered DE-MC ql-posterior samples (artifacts/f3/mcmc_<variant>_ql.npz) and saves
fits_<variant>_<target>_fromql.npz.
"""
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402

ART = HERE.parents[1] / "artifacts" / "f3"
REF = gen.gut_reference()
Z0, S0 = np.array(REF["z"]), np.array(REF["sigma"])


def pulls(th, variant, idx):
    z, io, ok = gen.simulate(th[None], variant)
    d = z[0] - Z0
    for j in gen.Z_PHASES:
        d[j] = (d[j] + np.pi) % (2 * np.pi) - np.pi
    p = d / S0
    if 15 in idx:  # IO solutions are scored against the NO data vector here; flag them with a penalty
        p = np.append(p[idx], 10.0 * io[0])
        return p
    return p[idx]


START = None
if len(sys.argv) > 5 and sys.argv[5] == "fromql":
    START = np.load(ART / f"mcmc_{sys.argv[1]}_ql.npz")["theta"].astype(float)


def one(args):
    variant, idx, seed = args
    rng = np.random.default_rng(seed)
    lo, hi = gen.prior_box(variant, REF)
    x0 = lo + (hi - lo) * rng.random(len(lo))
    if START is not None:  # start from a random ql-posterior sample (DE-MC), jittered
        x0 = START[rng.integers(len(START))] + 0.01 * (hi - lo) * rng.standard_normal(len(lo))
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    try:
        r = least_squares(pulls, x0, bounds=(lo, hi), args=(variant, idx), method="trf", x_scale=(hi - lo) / 10,
                          max_nfev=4000)
        return r.x, 2 * r.cost
    except Exception:
        return x0, np.inf


def main():
    variant, target, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    idx = list(range(12)) if target == "ql" else list(range(17))
    t0 = time.time()
    nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    with Pool(nproc) as pool:
        res = pool.map(one, [(variant, idx, seed * 100000 + i) for i in range(n)], chunksize=4)
    th = np.array([r[0] for r in res])
    c2 = np.array([r[1] for r in res])
    ART.mkdir(parents=True, exist_ok=True)
    tag = "_fromql" if START is not None else ""
    np.savez(ART / f"fits_{variant}_{target}{tag}.npz", theta=th, chi2=c2)
    o = np.argsort(c2)
    print(f"{variant} {target}: {n} starts in {time.time() - t0:.0f}s; chi2 quantiles "
          f"{np.round(np.quantile(c2[np.isfinite(c2)], [0, 0.05, 0.25, 0.5]), 2)}; n(chi2<{len(idx) // 2})="
          f"{(c2 < len(idx) // 2).sum()}")
    for i in o[:3]:
        z, io, _ = gen.simulate(th[i][None], variant)
        pl = (z[0] - Z0) / S0
        for j in gen.Z_PHASES:
            pl[j] = ((z[0, j] - Z0[j] + np.pi) % (2 * np.pi) - np.pi) / S0[j]
        print(f"  chi2 {c2[i]:.2f} IO={io[0]} pulls: " + " ".join(f"{gen.Z_NAMES[j]}={pl[j]:+.1f}" for j in range(17)))


if __name__ == "__main__":
    main()

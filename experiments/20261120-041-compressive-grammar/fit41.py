"""Exp 041 exact fits with the official scorer (src.evaluate): NuFIT 6.1 default tables, PDG 2026
masses with the 2% floor, M_Z (no running), ordering NO (declared).

  fit(eid, use, seed, ...)   scipy DE on Scorer.chi2_soft(include_finite=True) + L-BFGS-B polish,
                             optional extra polish from a given start (emulator optimum); official
                             evaluate at the optimum with k = 8 + 5 bits (k_discrete = 5).
Usage:
  python fit41.py all [seeds] [workers]   exact fits of every grammar entry -> fits.jsonl
  python fit41.py emu [workers]           polish from the emulator optima (search.json) -> fits_emu.jsonl
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import grammar as g  # noqa: E402
from src import evaluate as E  # noqa: E402

LO = np.array([g.BOUNDS[p][0] for p in g.PARAMS])
HI = np.array([g.BOUNDS[p][1] for p in g.PARAMS])


def objective(eid, use, scale=None):
    sc = E.Scorer(use, "NO", mass_floor=g.FLOOR, scale=scale)
    e = g.ENTRIES[eid]

    def f(x):
        try:
            with np.errstate(all="ignore"):
                c = sc.chi2_soft(g.model(dict(zip(g.PARAMS, map(float, x))), e), include_finite=True)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e9
        return c if np.isfinite(c) else 1e9
    return f


def official(eid, th, use, scale=None):
    k, kd = g.k_of(g.ENTRIES[eid]) if list(use) == g.USE else (len(th), 0)
    pred = g.model(th, g.ENTRIES[eid])
    r = E.evaluate(pred, "NO", use=use, k=k, k_discrete=kd, mass_floor=g.FLOOR, scale=scale)
    return dict(chi2=r.chi2, chi2_global=r.chi2_global, N=r.N, k=r.k, k_cont=r.k_continuous,
                k_discrete=r.k_discrete, dof=r.dof, p=r.p, p_global=r.p_global, max_pull=r.max_abs_pull,
                compression_ok=bool(r.compression_ok), fit_ok=bool(r.fit_ok), pulls=r.pulls,
                pred={k2: pred[k2] for k2 in g.USE})


def fit(eid, use=None, seed=1, popsize=15, maxiter=250, start=None, scale=None, de=True):
    from scipy.optimize import differential_evolution, minimize
    use = list(use or g.USE)
    t0 = time.time()
    f = objective(eid, use, scale)
    best_x, best_f, de_f = None, np.inf, None
    if de:
        r = differential_evolution(f, list(zip(LO, HI)), seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-10,
                                   init="sobol", polish=False)
        p = minimize(f, r.x, method="L-BFGS-B", bounds=list(zip(LO, HI)), options=dict(maxfun=4000))
        best_x, best_f = (p.x, p.fun) if p.fun < r.fun else (r.x, r.fun)
        de_f = float(r.fun)
    if start is not None:
        x0 = np.clip(start, LO, HI)
        p = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(LO, HI)), options=dict(maxfun=3000))
        q = minimize(f, p.x, method="Nelder-Mead", bounds=list(zip(LO, HI)), options=dict(maxfev=3000))
        for c in (p, q):
            if c.fun < best_f:
                best_x, best_f = c.x, c.fun
    th = dict(zip(g.PARAMS, map(float, best_x)))
    out = dict(eid=eid, entry=g.name(g.ENTRIES[eid]), seed=seed, use=use, params=th, de_chi2=de_f,
               soft_chi2=float(best_f), secs=time.time() - t0, from_start=start is not None)
    try:
        out.update(official(eid, th, use, scale))
    except Exception as ex:  # noqa: BLE001
        out["error"] = repr(ex)
    if not np.isfinite(out.get("chi2", np.inf)):
        out["gauss_chi2"] = float(best_f - E.CHI2_CAP) if best_f >= E.CHI2_CAP else None
    return out


def _run(args):
    eid, seed, start, de = args
    try:
        return fit(eid, seed=seed, start=start, de=de, popsize=12, maxiter=200)
    except Exception as ex:  # noqa: BLE001
        return dict(eid=eid, seed=seed, error=repr(ex))


def main():
    cmd = sys.argv[1]
    if cmd == "all":
        seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        jobs = [(e["id"], s, None, True) for s in range(1, seeds + 1) for e in g.ENTRIES]
        fn = HERE / "fits.jsonl"
    elif cmd == "emu":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        S = json.load(open(HERE / "search.json"))
        jobs = [(r["eid"], 0, np.array(r["x"]), False) for r in S]
        fn = HERE / "fits_emu.jsonl"
    else:
        raise SystemExit(__doc__)
    done = set()
    if fn.exists():
        for line in open(fn):
            d = json.loads(line)
            done.add((d["eid"], d.get("seed")))
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} jobs, {workers} workers", flush=True)
    with mp.Pool(workers) as pool, open(fn, "a") as fh:
        for out in pool.imap_unordered(_run, jobs):
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            print(out.get("eid"), out.get("entry"), "seed", out.get("seed"), "chi2", round(out.get("chi2", np.nan), 2),
                  "soft", round(out.get("soft_chi2", np.nan), 1), "p_glob", out.get("p_global"), "maxpull",
                  round(out.get("max_pull", np.nan), 2), out.get("error", ""), f"{out.get('secs', 0):.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()

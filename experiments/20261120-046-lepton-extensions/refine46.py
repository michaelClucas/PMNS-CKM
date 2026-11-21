"""Exp 046 refinement rounds (after fit46.py): the first rounds disagreed between seeds for some options
(the PMNS term is a plateau outside the NuFIT tables), so every (option, cell) is refitted with
L-BFGS-B multistart from
  * every optimum already found for the same option in ANY cell, by N 15 or PMNS-only fits (warm starts),
  * a fine grid over the lepton-only parameters (ln x, ln y step 0.5 in [-34, 34]; for BB the 3 best
    ln x_e from a charged-lepton-only scan) at the exp 045 quark optimum: best n_grid points,
  * n_sobol Sobol points (seeded).
Usage: python refine46.py <model|all> <seeds> [n_proc] [n_sobol] [n_grid] [N15|PMNS|N11]
Appends to fits.jsonl / fits_<use>.jsonl with "tag": "refine".
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
import fit46 as F  # noqa: E402
import lib46 as L  # noqa: E402
from src import evaluate as E  # noqa: E402


def prior_points(m, a, y, mr, b):
    pts = []
    for fn in ("fits.jsonl", "fits_PMNS.jsonl", "fits_N11.jsonl"):
        p = HERE / fn
        if not p.exists():
            continue
        for line in open(p):
            d = json.loads(line)
            if "error" in d or (d["model"], d["assign"], d["ynu"], d["mr"]) != (m, a, y, mr):
                continue
            if set(d["params"]) == set(b):
                pts.append(d["params"])
    return pts


def grid_starts(fn, b, keys, scale, quark, n_grid):
    sc = E.Scorer(keys, "NO", mass_floor=0.02, scale=scale)
    base = dict(quark)
    if "lxe" in b:  # charged-lepton-only scan for x_e
        scl = E.Scorer(list(L.CL), "NO", mass_floor=0.02, scale=scale)
        xs = np.linspace(-5, 5, 201)
        cs = []
        for v in xs:
            th = dict(base, lxe=v, lx=0.0, ly=0.0)
            cs.append(scl.chi2(fn(th)))
        lxes = list(xs[np.argsort(cs)[:3]])
    else:
        lxes = [None]
    if "lx" not in b:
        out = []
        for v in lxes:
            out.append(dict(base, **({} if v is None else {"lxe": v})))
        return out
    ax = np.linspace(-34, 34, 137)
    cand = []
    for v in lxes:
        for gx in ax:
            for gy in ax:
                th = dict(base, lx=gx, ly=gy)
                if v is not None:
                    th["lxe"] = v
                try:
                    c = sc.chi2_soft(fn(th), include_finite=True)
                except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
                    continue
                if np.isfinite(c):
                    cand.append((c, th))
    cand.sort(key=lambda t: t[0])
    return [t for _, t in cand[:n_grid]]


def run(job, n_sobol, n_grid, use):
    m, a, y, mr, sc, eb, el, seed = job
    t0 = time.time()
    keys = list(F.USES[use])
    fn, b, per, qn = L.ext(m, a, y, mr)
    scale = dict(L.SCALES[sc])
    if m == "RRR5":
        scale.update(eta_b=eb, eta_l=el)
    warm = prior_points(m, a, y, mr, b)
    for q in F.quark_starts(m, sc, eb):
        warm += grid_starts(fn, b, keys, scale, q, n_grid)
    fr = E.fit_and_evaluate(fn, b, "NO", keys, periodic=per, n_starts=n_sobol, seed=100 + seed,
                            extra_starts=warm, mass_floor=0.02, scale=scale, method="lbfgsb",
                            k_discrete=L.BITS[m])
    r = fr.result
    k_cont = len(b) + L.EXTRA_CONT[m]
    pred = fn(fr.params)
    soft = E.Scorer(keys, "NO", mass_floor=0.02, scale=scale).chi2_soft(pred, include_finite=True)
    return {"use": use, "tag": "refine", "model": m, "assign": a, "ynu": y, "mr": mr, "scale": sc, "eta_b": eb,
            "eta_l": el, "seed": 100 + seed, "N": r.N, "k_cont": k_cont, "bits": L.BITS[m],
            "k": k_cont + L.BITS[m], "chi2": r.chi2, "chi2_soft": soft,
            "p_global": r.p_global if np.isfinite(r.chi2) else 0.0, "max_pull": r.max_abs_pull,
            "pulls": r.pulls, "params": fr.params, "pred": {k: pred[k] for k in L.N15}, "starts": fr.n_starts,
            "start_chi2_sorted": sorted(fr.start_chi2)[:5], "nfev": fr.nfev, "sec": time.time() - t0}


def _worker(args):
    job, ns, ng, use = args
    try:
        return run(job, ns, ng, use)
    except Exception as ex:  # logged, not hidden
        return {"job": list(job), "error": repr(ex)}


if __name__ == "__main__":
    models = list(L.OPTIONS) if sys.argv[1] == "all" else sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ns = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    ng = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    use = sys.argv[6] if len(sys.argv) > 6 else "N15"
    J = F.jobs(models, seeds, "D" if use != "N15" else None)
    out = HERE / ("fits.jsonl" if use == "N15" else f"fits_{use}.jsonl")
    print(f"refine {use}: {len(J)} jobs, {nproc} processes, sobol {ns}, grid {ng}", flush=True)
    with mp.Pool(nproc) as pool, open(out, "a") as fh:
        for r in pool.imap_unordered(_worker, [(j, ns, ng, use) for j in J]):
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r, flush=True)
            else:
                print(f"refine {use} {r['model']} {r['assign']} Ynu={r['ynu']} MR={r['mr']} {r['scale']} "
                      f"eb {r['eta_b']:+.2f} el {r['eta_l']:+.2f} s{r['seed']} chi2 {r['chi2']:.4g} "
                      f"soft {r['chi2_soft']:.4g} maxpull {r['max_pull']:.2f} {r['sec']:.0f}s", flush=True)

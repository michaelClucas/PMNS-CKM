"""Exp 045 fits with the official scorer (src.evaluate.fit_and_evaluate, method "de", PDG 2026, 2% floor).

Usage: python fit45.py <filter-substring or 'all'> [n_proc]   -> appends JSON lines to fits.jsonl
Job = (structure, scale tag, seed). Scale tags:
  MZ          no running (data scale)
  G10, G30    mu 2e16, MSSM, tan beta 10 / 30, M_SUSY 3 TeV, no thresholds (eta = 0)
  SM<e>       mu 10^e GeV, SM running (Harrison-Scott and Belfatto-Berezhiani scale scans)
k_cont = number of fitted parameters = Jacobian rank (lib45.reduce_params); k_discrete as stated.
For the SM-scale scans the scale itself is a data-selected continuous choice: +1 is added in summarize.py.
"""
from __future__ import annotations

import functools
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

import structures as S  # noqa: E402
from lib45 import Q8, reduce_params  # noqa: E402
from src import evaluate as E  # noqa: E402

SCALES = {"MZ": None,
          "G10": {"mu": 2e16, "model": "MSSM", "tan_beta": 10.0},
          "G30": {"mu": 2e16, "model": "MSSM", "tan_beta": 30.0}}
for e in (3, 5, 7, 9, 12, 16):
    SCALES[f"SM{e}"] = {"mu": 10.0 ** e, "model": "SM"}
DE_OPTS = {"popsize": 20, "maxiter": 800}


def get_model(name):
    """-> (model_fn, bounds, k_discrete, periodic)."""
    if name in S.TEXTURES:
        tex = S.TEXTURES[name][0]()
        reduce_params(tex)
        b = tex.bounds()
        return tex.model, b, 0, [k for k in b if k.startswith("f")]
    if name in S.RELATIONS:
        fn, b, kd, _ = S.RELATIONS[name]
        return fn, b, kd, [k for k in b if k == "d"]
    if name == "HS":
        return functools.partial(S.hs_model, reading="mod"), S.HS_BOUNDS, 0, []
    raise KeyError(name)


def run(job, use=Q8, tag=""):
    name, sc, seed = job
    t0 = time.time()
    fn, b, kd, per = get_model(name)
    fr = E.fit_and_evaluate(fn, b, "NO", list(use), k_discrete=kd, periodic=per, seed=seed,
                            mass_floor=0.02, scale=SCALES[sc], method="de", optimizer_options=DE_OPTS)
    r = fr.result
    pred = fn(fr.params)
    out = {"name": name, "scale": sc, "seed": seed, "use": list(use), "tag": tag, "N": r.N, "k": r.k,
           "k_discrete": r.k_discrete, "chi2": r.chi2, "p": r.p, "pulls": r.pulls,
           "params": fr.params, "pred": {k: pred[k] for k in Q8}, "nfev": fr.nfev, "sec": time.time() - t0}
    return out


def jobs_all():
    J = []
    for n in S.TEXTURES:
        for sc in ("MZ", "G10", "G30"):
            for seed in ((1,) if sc == "G10" else (1, 2)):
                J.append((n, sc, seed))
    for n in S.RELATIONS:
        J += [(n, "MZ", 1), (n, "MZ", 2)]
    for e in (3, 5, 7, 9, 12, 16):
        J += [("HS", f"SM{e}", 1), ("BB", f"SM{e}", 1)]
    J.append(("HS", "MZ", 1))
    return J


def _worker(job):
    try:
        return run(job)
    except Exception as ex:  # logged, not hidden
        return {"name": job[0], "scale": job[1], "seed": job[2], "error": repr(ex)}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else "all"
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    J = [j for j in jobs_all() if flt == "all" or any(f == j[0] or f == f"{j[0]}@{j[1]}" for f in flt.split(","))]
    done = set()
    if (HERE / "fits.jsonl").exists():  # resume: skip jobs already logged without error
        for line in open(HERE / "fits.jsonl"):
            d = json.loads(line)
            if "error" not in d and not d.get("tag"):
                done.add((d["name"], d["scale"], d["seed"]))
    J = [j for j in J if j not in done]
    print(f"{len(J)} jobs, {nproc} processes", flush=True)
    with mp.Pool(nproc) as pool, open(HERE / "fits.jsonl", "a") as fh:
        for r in pool.imap_unordered(_worker, J):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r, flush=True)
            else:
                print(f"{r['name']:8s} {r['scale']:5s} s{r['seed']} k {r['k']} chi2 {r['chi2']:.3f} p {r['p']:.3g} "
                      f"{r['sec']:.0f}s", flush=True)

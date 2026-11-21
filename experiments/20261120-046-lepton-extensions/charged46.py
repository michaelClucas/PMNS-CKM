"""Exp 046 charged-lepton stage: fit Q8 + mr_e_mu + mr_mu_tau (N 10) with every quark parameter (and BB's
x_e) free; the neutrino sector does not enter these 10 observables. A failure here bounds every N 15 option
of that model from below, whatever the neutrino sector. L-BFGS-B multistart, seeds 1-3, warm start from
exp 045's quark optimum (x_e from a 1D scan). Output: charged.json, printed log.
Usage: python charged46.py [n_proc]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit46 as F  # noqa: E402
import lib46 as L  # noqa: E402
from src import evaluate as E  # noqa: E402

KEYS = list(L.Q8) + list(L.CL)
CELLS = ([("BB", "XE", s, 0.0, 0.0) for s in ("SM3", "SM6", "SM16")]
         + [("RRR5", "GJ", "G30", -0.25, el) for el in (-0.15, 0.0, 0.15)]
         + [("HS", "SU5", "SM7", 0.0, 0.0), ("HS", "PS", "SM7", 0.0, 0.0)])


def run(args):
    (m, a, sc, eb, el), seed = args
    fn, b, per, _ = L.ext(m, a, "u", "I")
    scale = dict(L.SCALES[sc])
    if m == "RRR5":
        scale.update(eta_b=eb, eta_l=el)
    warm = []
    for q in F.quark_starts(m, sc, eb):
        if "lxe" in b:
            warm += [dict(q, lxe=v) for v in np.linspace(-5, 5, 11)]
        else:
            warm.append(dict(q))
    fr = E.fit_and_evaluate(fn, b, "NO", KEYS, periodic=per, n_starts=16, seed=seed, extra_starts=warm,
                            mass_floor=0.02, scale=scale, method="lbfgsb")
    r = fr.result
    return {"cell": [m, a, sc, eb, el], "seed": seed, "N": r.N, "chi2": r.chi2, "max_pull": r.max_abs_pull,
            "pulls": r.pulls, "params": fr.params}


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    jobs = [(c, s) for c in CELLS for s in (1, 2, 3)]
    out = []
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(run, jobs):
            out.append(r)
            top = sorted(r["pulls"].items(), key=lambda t: -abs(t[1]))[:3]
            print(f"{' '.join(map(str, r['cell']))} s{r['seed']}: N10 chi2 {r['chi2']:.4g} max pull "
                  f"{r['max_pull']:.3g} {[(k, round(v, 1)) for k, v in top]}", flush=True)
            json.dump(out, open(HERE / "charged.json", "w"), indent=1, default=float)

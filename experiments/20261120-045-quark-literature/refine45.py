"""Exp 045 refit of the textures: the single-seed DE fits of fit45.py disagree between seeds for several
textures (multimodal landscape over a wide log box), so every (texture, scale) is refitted with
L-BFGS-B multi-start (src.evaluate.fit_and_evaluate method "lbfgsb") from
  * every parameter point found by fit45.py for the same texture at ANY scale (warm starts), plus
  * n_sobol Sobol points in the fit box.
Appends to fits.jsonl with seed = 100 (the summary takes the minimum over all runs).
Usage: python refine45.py <names|all> [n_proc] [n_sobol] [scales, default MZ,G10,G30]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fit45 as F  # noqa: E402
import structures as S  # noqa: E402
from lib45 import Q8  # noqa: E402
from src import evaluate as E  # noqa: E402

N_SOBOL = 8


def starts_for(name):
    out = []
    for line in open(HERE / "fits.jsonl"):
        d = json.loads(line)
        if d.get("name") == name and "error" not in d and not d.get("tag"):
            out.append(d["params"])
    return out


def job(arg):
    name, sc, n_sobol = arg
    t0 = time.time()
    fn, b, kd, per = F.get_model(name)
    st = [s for s in starts_for(name) if set(s) == set(b)]
    if name == "HS":
        st.append(dict(S.HS_PUB))
    fr = E.fit_and_evaluate(fn, b, "NO", list(Q8), k_discrete=kd, periodic=per, n_starts=n_sobol, seed=100,
                            extra_starts=st, mass_floor=0.02, scale=F.SCALES[sc], method="lbfgsb")
    r = fr.result
    pred = fn(fr.params)
    return {"name": name, "scale": sc, "seed": 100, "use": list(Q8), "tag": "", "N": r.N, "k": r.k,
            "k_discrete": r.k_discrete, "chi2": r.chi2, "p": r.p, "pulls": r.pulls, "params": fr.params,
            "pred": {k: pred[k] for k in Q8}, "nfev": fr.nfev, "n_starts": fr.n_starts, "sec": time.time() - t0,
            "method": "lbfgsb multistart (refine45)"}


if __name__ == "__main__":
    names = list(S.TEXTURES) if sys.argv[1] == "all" else sys.argv[1].split(",")
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else N_SOBOL
    scales = sys.argv[4].split(",") if len(sys.argv) > 4 else ("MZ", "G10", "G30")
    J = [(n, sc, ns) for n in names for sc in scales]
    print(f"refine: {len(J)} jobs", flush=True)
    with mp.Pool(nproc) as pool, open(HERE / "fits.jsonl", "a") as fh:
        for r in pool.imap_unordered(job, J):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"refine {r['name']:8s} {r['scale']:4s} k {r['k']} chi2 {r['chi2']:.3f} starts {r['n_starts']} "
                  f"{r['sec']:.0f}s", flush=True)

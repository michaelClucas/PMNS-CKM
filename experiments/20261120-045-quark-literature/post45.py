"""Exp 045 post-fits (run after fit45.py).

  python post45.py eta  <names|all> [n_proc]   eta_b profile at GUT, MSSM tan beta 30: local Nelder-Mead refits
                                               from the best G30 point on an eta_b grid (eta_q = 0; for the Q8
                                               keys only (1+eta_q)/(1+eta_b) matters). -> eta.jsonl
  (env ETA_GRID="a,b,..." overrides the eta_b grid)
  python post45.py miss <name@scale,...> [n_proc]  dominant-miss profile: drop the worst-pull key, refit the other 7
                                               (L-BFGS-B, 16 Sobol + all earlier optima as starts), score the dropped key at the 7-key optimum. -> miss.jsonl
"""
from __future__ import annotations

import json
import os
import multiprocessing as mp
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

import fit45 as F  # noqa: E402
from lib45 import Q8  # noqa: E402
from src import evaluate as E  # noqa: E402

ETA_GRID = (-0.45, -0.3, -0.15, 0.15, 0.3, 0.45)


def best_fits():
    best = {}
    for line in open(HERE / "fits.jsonl"):
        d = json.loads(line)
        if "error" in d or d.get("tag"):
            continue
        key = (d["name"], d["scale"])
        if key not in best or d["chi2"] < best[key]["chi2"]:
            best[key] = d
    return best


def _grid():
    return tuple(float(x) for x in os.environ["ETA_GRID"].split(",")) if os.environ.get("ETA_GRID") else ETA_GRID


def eta_job(name):
    b = best_fits()[(name, "G30")]
    fn, bounds, kd, per = F.get_model(name)
    out = []
    start = dict(b["params"])
    for eta in _grid():
        sc = dict(F.SCALES["G30"], eta_b=eta)
        starts = [b["params"]] + ([start] if start is not b["params"] else [])
        fr = E.fit_and_evaluate(fn, bounds, "NO", list(Q8), k_discrete=kd, periodic=per, n_starts=0,
                                extra_starts=starts, mass_floor=0.02, scale=sc, method="nelder-mead")
        start = fr.params
        out.append({"name": name, "eta_b": eta, "chi2": fr.result.chi2, "k_fit": fr.result.k,
                    "pulls": fr.result.pulls, "params": fr.params})
    return out


def miss_job(spec):
    name, scale = spec.split("@")
    b = best_fits()[(name, scale)]
    worst = max(b["pulls"], key=lambda k: abs(b["pulls"][k]))
    use7 = [k for k in Q8 if k != worst]
    t0 = time.time()
    fn, bounds, kd, per = F.get_model(name)
    from refine45 import starts_for
    st = [s for s in starts_for(name) if set(s) == set(bounds)]
    fr = E.fit_and_evaluate(fn, bounds, "NO", use7, k_discrete=kd, periodic=per, n_starts=16, seed=7,
                            extra_starts=st, mass_floor=0.02, scale=F.SCALES[scale], method="lbfgsb")
    pred = fn(fr.params)
    full = E.evaluate(pred, "NO", use=list(Q8), mass_floor=0.02, scale=F.SCALES[scale])
    return {"name": name, "scale": scale, "dropped": worst, "pull_full_fit": b["pulls"][worst],
            "chi2_7": fr.result.chi2, "k": fr.result.k, "pred_dropped": pred[worst],
            "pull_dropped_at_7fit": full.pulls[worst], "pulls_7fit": full.pulls, "sec": time.time() - t0}


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    if mode == "eta":
        import structures as S
        names = list(S.TEXTURES) if arg == "all" else arg.split(",")
        with mp.Pool(nproc) as pool, open(HERE / "eta.jsonl", "a") as fh:
            for res in pool.imap_unordered(eta_job, names):
                for r in res:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                m = min(res, key=lambda r: r["chi2"])
                print(f"eta {res[0]['name']:8s} min chi2 {m['chi2']:.2f} at eta_b {m['eta_b']}", flush=True)
    elif mode == "miss":
        with mp.Pool(nproc) as pool, open(HERE / "miss.jsonl", "a") as fh:
            for r in pool.imap_unordered(miss_job, arg.split(",")):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"miss {r['name']}@{r['scale']} drop {r['dropped']} chi2_7 {r['chi2_7']:.2f} "
                      f"pull of dropped {r['pull_dropped_at_7fit']:+.1f} (full fit {r['pull_full_fit']:+.1f})", flush=True)

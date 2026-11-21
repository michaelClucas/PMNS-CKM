"""Exp 046 N 15 fits (extensions.md, Tests 2): official scorer, L-BFGS-B multistart, rounds = seeds.

Usage: python fit46.py <model|all> <round seeds, e.g. 1,2,3> [n_proc] [n_sobol] [n_warm] [N15|PMNS|N11]
  PMNS: fit only the 4 PMNS parameters + dm_ratio over all parameters (a lower bound on their N 15 terms);
  N11: Q8 + charged leptons + dm_ratio (held-out PMNS). Both only for M_R = diag(1, x, y).
Appends JSON lines to fits.jsonl. One job = (model, assign, ynu, mr, scale, eta_b, eta_l, seed).
Starts: n_sobol Sobol points over the full box (fit_and_evaluate) + n_warm warm starts = exp 045's quark
optimum for that structure/scale (nearest eta_b for RRR5) with the new lepton parameters drawn at random
(seeded). RRR5's eta_b, eta_l are on a grid (each grid point is its own scorer).
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
import lib46 as L  # noqa: E402
from src import evaluate as E  # noqa: E402

ETA_B = (-0.25,)  # exp 045 optimum; GJ R bound (bounds46) is eta-independent
ETA_L = (-0.15, 0.0, 0.15)
USES = {"N15": L.N15, "PMNS": L.PM + ("dm_ratio",), "N11": tuple(L.Q8) + L.CL + ("dm_ratio",)}
USE = "N15"
SCALES = {"RRR5": ["G30"], "BB": ["SM3", "SM6", "SM16"], "HS": ["SM7"]}


def quark_starts(model, sc, eta_b=None):
    out = []
    if model == "RRR5":
        best = {}
        for line in open(L.E45 / "eta.jsonl"):
            d = json.loads(line)
            if d["name"] == "RRR5" and abs(d["eta_b"] - eta_b) < 1e-9:
                if d["eta_b"] not in best or d["chi2"] < best[d["eta_b"]]["chi2"]:
                    best[d["eta_b"]] = d
        out = [b["params"] for b in best.values()]
        if not out:  # nearest grid value
            ds = [json.loads(l) for l in open(L.E45 / "eta.jsonl")]
            ds = [d for d in ds if d["name"] == "RRR5"]
            d = min(ds, key=lambda d: (abs(d["eta_b"] - eta_b), d["chi2"]))
            out = [d["params"]]
        return out
    name, tag = model, sc
    best = None
    for line in open(L.E45 / "fits.jsonl"):
        d = json.loads(line)
        if d.get("name") == name and d.get("scale") == tag and "chi2" in d and not d.get("tag"):
            if best is None or d["chi2"] < best["chi2"]:
                best = d
    if best is None and tag.startswith("SM"):  # scale not fitted in exp 045 (BB 1e6 GeV): nearest SM scale
        avail = {json.loads(l).get("scale") for l in open(L.E45 / "fits.jsonl")}
        sm = sorted((abs(int(t[2:]) - int(tag[2:])), t) for t in avail if t and t.startswith("SM"))
        for _, t in sm:
            got = quark_starts(model, t)
            if got:
                return got
    return [best["params"]] if best else []


def jobs(models, seeds, only_mr=None):
    J = []
    for m in models:
        for (a, y, mr) in L.OPTIONS[m]:
            if only_mr and mr != only_mr:
                continue
            for sc in SCALES[m]:
                etas = [(eb, el) for eb in ETA_B for el in ETA_L] if m == "RRR5" else [(0.0, 0.0)]
                for eb, el in etas:
                    for s in seeds:
                        J.append((m, a, y, mr, sc, eb, el, s))
    return J


def run(job, n_sobol, n_warm, use="N15"):
    m, a, y, mr, sc, eb, el, seed = job
    keys = list(USES[use])
    t0 = time.time()
    fn, b, per, qn = L.ext(m, a, y, mr)
    scale = dict(L.SCALES[sc])
    if m == "RRR5":
        scale.update(eta_b=eb, eta_l=el)
    rng = np.random.default_rng(1000 * seed + 7)
    warm = []
    for q in quark_starts(m, sc, eb):
        for _ in range(n_warm):
            th = dict(q)
            for k in b:
                if k not in th:
                    lo, hi = b[k]
                    th[k] = rng.uniform(-12, 12) if k in ("lx", "ly") else rng.uniform(lo, hi)
            warm.append(th)
    # grid scan of the lepton-only parameters at each quark warm start (cheap global search)
    lep = [k for k in ("lxe", "lx", "ly") if k in b]
    if lep:
        sc_ = E.Scorer(keys, "NO", mass_floor=0.02, scale=scale)
        axes = {"lxe": np.linspace(-4, 4, 17), "lx": np.linspace(-34, 34, 35), "ly": np.linspace(-34, 34, 35)}
        grid = np.array(np.meshgrid(*[axes[k] for k in lep], indexing="ij")).reshape(len(lep), -1).T
        cand = []
        for q in quark_starts(m, sc, eb):
            for z in grid:
                th = dict(q)
                th.update(dict(zip(lep, z)))
                try:
                    c = sc_.chi2_soft(fn(th), include_finite=True)
                except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
                    continue
                if np.isfinite(c):
                    cand.append((c, th))
        cand.sort(key=lambda t: t[0])
        warm += [t for _, t in cand[:n_warm]]
        # jitter the grid optimum per seed so rounds are not identical
        warm += [{k: v + (rng.normal(0, 0.5) if k in lep else 0.0) for k, v in cand[0][1].items()}] if cand else []
    fr = E.fit_and_evaluate(fn, b, "NO", keys, periodic=per, n_starts=n_sobol, seed=seed,
                            extra_starts=warm, mass_floor=0.02, scale=scale, method="lbfgsb",
                            k_discrete=L.BITS[m])
    r = fr.result
    k_cont = len(b) + L.EXTRA_CONT[m]
    pred = fn(fr.params)
    soft = E.Scorer(keys, "NO", mass_floor=0.02, scale=scale).chi2_soft(pred, include_finite=True)
    return {"use": use, "model": m, "assign": a, "ynu": y, "mr": mr, "scale": sc, "eta_b": eb, "eta_l": el, "seed": seed,
            "N": r.N, "k_cont": k_cont, "bits": L.BITS[m], "k": k_cont + L.BITS[m],
            "chi2": r.chi2, "chi2_soft": soft, "p_global": r.p_global if np.isfinite(r.chi2) else 0.0,
            "dof_note": "p uses dof = N - k_cont incl. extra continuous (recomputed in summarize)",
            "max_pull": r.max_abs_pull, "pulls": r.pulls, "params": fr.params,
            "pred": {k: pred[k] for k in L.N15}, "starts": fr.n_starts,
            "start_chi2_sorted": sorted(fr.start_chi2)[:5], "nfev": fr.nfev, "sec": time.time() - t0}


def _worker(args):
    job, ns, nw, use = args
    try:
        return run(job, ns, nw, use)
    except Exception as ex:  # logged, not hidden
        return {"job": list(job), "error": repr(ex)}


if __name__ == "__main__":
    models = list(L.OPTIONS) if sys.argv[1] == "all" else sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ns = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    use = sys.argv[6] if len(sys.argv) > 6 else "N15"
    J = jobs(models, seeds, "D" if use != "N15" else None)
    out = HERE / ("fits.jsonl" if use == "N15" else f"fits_{use}.jsonl")
    print(f"{len(J)} jobs, {nproc} processes, sobol {ns}, warm {nw}", flush=True)
    with mp.Pool(nproc) as pool, open(out, "a") as fh:
        for r in pool.imap_unordered(_worker, [(j, ns, nw, use) for j in J]):
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r, flush=True)
            else:
                print(f"{r['model']} {r['assign']} Ynu={r['ynu']} MR={r['mr']} {r['scale']} eb {r['eta_b']:+.2f} "
                      f"el {r['eta_l']:+.2f} s{r['seed']} chi2 {r['chi2']:.4g} soft {r['chi2_soft']:.4g} "
                      f"maxpull {r['max_pull']:.2f} {r['sec']:.0f}s", flush=True)

"""Exp 031: global (differential evolution) rerun of exp 015's BSV type-II fit stage.

Exp 015 used multistart local least squares (5 starts per sign pattern) on its own GUT-level
residual vector (common.residuals: 10 nuisance pulls + 3 charged-lepton GUT residuals + NuFIT
PMNS residuals + Delta m^2 ratio). That objective never used src.evaluate.Scorer.chi2_soft, so the
exp 026 landscape bug cannot apply; this rerun checks whether the local search was the limit.
Same objective, scipy differential_evolution over the model box plus nuisance pulls in [-60, 60],
then a least-squares polish. Compares with the best fit-stage chi2_gut in exp 015's fits_*.jsonl.

usage: python rerun_bsv_de.py <variant R0|R0eta|C> <scenario SM|5|10|...> <seed> [maxiter] [popsize]
Run from the repo root. Output: artifacts/031/bsv_<variant>_<scen>_s<seed>.json (+ summary line on stdout).
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, least_squares

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "20261118-015-bsv-type2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))
import common as B  # noqa: E402

BOX = {"x": (-8, 8), "yhat": (-3, 3), "eta": (-0.3, 0.8), "xr": (-8, 8), "xi": (-8, 8),
       "a2": (0, 2 * np.pi), "a3": (0, 2 * np.pi), "p1": (0, 2 * np.pi), "p2": (0, 2 * np.pi), "p3": (0, 2 * np.pi)}
ZMAX = 60.0


def committed_best(variant, scen, ordering="NO"):
    best = None
    for fn in ("fits_R0.jsonl", "fits_R0eta.jsonl", "fits_C.jsonl"):
        p = EXP / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["variant"] == variant and r["scenario"] == scen and r["ordering"] == ordering:
                if best is None or r["chi2_gut"] < best:
                    best = r["chi2_gut"]
    return best


def main():
    variant, scen_s, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    maxiter = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    popsize = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    model, tb = ("SM", None) if scen_s == "SM" else ("MSSM", int(scen_s))
    ms = 3000.0
    ref = B.gut_reference(model, tb, ms)
    scen = B.scenario_name(model, tb, ms)
    names, _ = B.VARIANTS[variant]
    n = len(names)
    lo = np.r_[[BOX[k][0] for k in names], -ZMAX * np.ones(10)]
    hi = np.r_[[BOX[k][1] for k in names], ZMAX * np.ones(10)]
    signs = B.SIGNS if variant != "C" else [(0.0, 0.0)]
    t0 = time.time()
    out = []
    np.seterr(all="ignore")
    for sg in signs:
        def res(q, sg=sg):
            return B.residuals(ref, variant, q[:n], q[n:], "NO", sg)

        def f(q):
            r = res(q)
            return float(np.dot(r, r))

        d = differential_evolution(f, list(zip(lo, hi)), seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-10,
                                   init="sobol", polish=False)
        s = least_squares(res, d.x, method="trf", max_nfev=4000, bounds=(lo, hi))
        c = float(2 * s.cost)
        x = s.x if c < d.fun else d.x
        out.append({"sign": list(sg), "chi2_de": float(d.fun), "chi2_polished": min(c, float(d.fun)),
                    "nfev": int(d.nfev), "p": x[:n].tolist(), "z": x[n:].tolist()})
        print(f"{scen} {variant} NO sign={np.round(sg, 2).tolist()} DE {d.fun:.2f} polished {min(c, d.fun):.2f} "
              f"nfev {d.nfev} ({time.time() - t0:.0f}s)", flush=True)
    best = min(out, key=lambda r: r["chi2_polished"])
    summ = {"variant": variant, "scenario": scen, "ordering": "NO", "seed": seed, "maxiter": maxiter, "popsize": popsize,
            "zmax": ZMAX, "best_chi2_gut_de": best["chi2_polished"],
            "committed_best_chi2_gut": committed_best(variant, scen), "runs": out, "wall_s": time.time() - t0}
    pred = B.predict(ref, variant, np.array(best["p"]), np.array(best["z"]), "NO", tuple(best["sign"]))
    summ["best_pmns_gut"] = {k: float(v) for k, v in pred["pmns"].items()}
    summ["best_max_abs_nuisance_pull"] = float(np.max(np.abs(best["z"])))
    art = ROOT / "artifacts" / "031"
    art.mkdir(parents=True, exist_ok=True)
    (art / f"bsv_{variant}_{scen_s}_s{seed}.json").write_text(json.dumps(summ, indent=1))
    print("SUMMARY", json.dumps({k: summ[k] for k in ("variant", "scenario", "seed", "best_chi2_gut_de",
                                                     "committed_best_chi2_gut", "best_max_abs_nuisance_pull")}),
          flush=True)


if __name__ == "__main__":
    main()

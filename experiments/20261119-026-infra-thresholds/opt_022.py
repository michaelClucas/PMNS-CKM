"""Global/gradient optimizer test on exp 022's 9-D texture (SU(5) + CSD3, primary setting).

Does fit_and_evaluate(method=...) reach exp 022's chi2 22.40 from random starts only (the
paper's benchmark start is NOT given)? Exp 022 found that 48 Nelder-Mead Sobol starts stall at
chi2 > 3000 and a hand-rolled DE stalled at 13777.

Usage (from this folder): ../../.venv/Scripts/python opt_022.py <method> <seed> [budget]
  method: lbfgsb | de | de_log
  de_log: method "de" on a log-magnitude reparametrisation of the same model (|u_ij|, |d_ij|
          in [1e-2, 10] as log10 plus a sign bit per entry, encoded as a continuous variable
          in [-1, 1] whose sign is used); same chi2, same bounds on the physical parameters.
Writes opt_<method>_s<seed>.json. Single process.
"""
import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = Path(__file__).resolve().parent
EXP22 = HERE.parent / "20261119-022-su5-csd3"
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(EXP22))

from src import evaluate as ev  # noqa: E402
import fit as F  # noqa: E402  (exp 022's fit.py: model, bounds, observables, scale)

TARGET = 22.401884762527306  # exp 022 fit_primary.json chi2
LOGN = ("u11", "u12", "u13", "u22", "u23", "d11", "d12", "d22")


def main(method, seed, budget):
    model = F.make_model()
    bounds, fn = dict(F.BOUNDS), model
    if method == "de_log":
        bounds = {**{f"l{n}": (-2.0, 1.0) for n in LOGN}, **{f"s{n}": (-1.0, 1.0) for n in LOGN}, "r": (0.0, 1.0)}

        def fn(t):
            th = {n: (1.0 if t[f"s{n}"] >= 0 else -1.0) * 10 ** t[f"l{n}"] for n in LOGN}
            th["r"] = t["r"]
            return model(th)
    kw = dict(k_discrete=F.K_DISCRETE, seed=seed, mass_floor=F.FLOOR, scale=F.SCALE)
    t0 = time.time()
    if method == "lbfgsb":
        # budget = number of Sobol starts; each L-BFGS-B run is capped at 3000 evaluations
        fr = ev.fit_and_evaluate(fn, bounds, "NO", F.USE, n_starts=budget, method="lbfgsb",
                                 optimizer_options={"maxfun": 3000}, **kw)
    else:
        fr = ev.fit_and_evaluate(fn, bounds, "NO", F.USE, method="de",
                                 optimizer_options={"maxfev": budget, "popsize": 15, "maxiter": 100000}, **kw)
    r = fr.result
    out = {"method": method, "seed": seed, "budget": budget, "chi2": r.chi2, "target": TARGET,
           "reached": bool(r.chi2 <= TARGET + 0.01), "nfev": fr.nfev, "seconds": time.time() - t0,
           "start_chi2_sorted": sorted(fr.start_chi2)[:10], "n_starts": fr.n_starts, "params": fr.params,
           "max_abs_pull": r.max_abs_pull}
    (HERE / f"opt_{method}_s{seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "params"}, default=float))


if __name__ == "__main__":
    m = sys.argv[1]
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    b = int(sys.argv[3]) if len(sys.argv) > 3 else (48 if m == "lbfgsb" else 150000)
    main(m, s, b)

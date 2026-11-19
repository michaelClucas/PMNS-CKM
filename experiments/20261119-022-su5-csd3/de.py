"""Global-search cross-check of fit.py: scipy differential evolution (the paper's own minimiser
type) on the same Scorer, same bounds, then fit_and_evaluate polished from the DE optimum.
Usage: python de.py [tag] [seed]
"""
import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from src import evaluate as ev  # noqa: E402
import fit as F  # noqa: E402
import model as M  # noqa: E402


def main(tag="primary", seed=1):
    use = list(F.USE)
    if tag == "v1":
        use.insert(7, "mr_s_b")
    model_fn = F.make_model()
    sc = ev.Scorer(use, "NO", mass_floor=F.FLOOR, scale=F.SCALE)
    names = list(F.BOUNDS)

    def f(x):
        try:
            c = sc.chi2_soft(model_fn(dict(zip(names, x))))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return 1e5
        return c if np.isfinite(c) else 1e5

    t0 = time.time()
    r = differential_evolution(f, [F.BOUNDS[n] for n in names], seed=seed, popsize=20, maxiter=3000,
                               tol=1e-10, mutation=(0.5, 1.0), recombination=0.7, polish=True, init="sobol")
    best = dict(zip(names, map(float, r.x)))
    fr = ev.fit_and_evaluate(model_fn, F.BOUNDS, "NO", use, k_discrete=F.K_DISCRETE, n_starts=1, seed=seed,
                             extra_starts=[best], mass_floor=F.FLOOR, scale=F.SCALE, leak_check=True)
    res = fr.result
    out = {"tag": tag, "seed": seed, "de_fun": float(r.fun), "de_nfev": int(r.nfev), "chi2": res.chi2,
           "p_global": res.p_global, "max_abs_pull": res.max_abs_pull, "pulls": res.pulls, "params": fr.params,
           "seconds": time.time() - t0}
    Path(f"de_{tag}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print(f"[DE {tag} seed {seed}] DE chi2 {r.fun:.3f} -> polished {res.chi2:.3f} p_global {res.p_global:.3g} "
          f"max|pull| {res.max_abs_pull:.2f} ({time.time() - t0:.0f} s, DE nfev {r.nfev})")
    print("   params", json.dumps({k: round(v, 4) for k, v in fr.params.items()}))
    print("   pulls", json.dumps({k: round(v, 2) for k, v in res.pulls.items()}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "primary", int(sys.argv[2]) if len(sys.argv) > 2 else 1)

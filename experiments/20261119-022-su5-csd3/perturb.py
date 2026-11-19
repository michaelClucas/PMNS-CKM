"""Basin search around the primary best fit: Nelder-Mead from log-normal perturbations
(sigma 0.7 in log|x|, 20% sign flips) of fit_primary.json, then fit_and_evaluate polish.
Usage: python perturb.py seed n_starts
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
from scipy.optimize import minimize  # noqa: E402

from src import evaluate as ev  # noqa: E402
import fit as F  # noqa: E402

seed, n = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(seed)
names = list(F.BOUNDS)
lo = np.array([F.BOUNDS[k][0] for k in names]); hi = np.array([F.BOUNDS[k][1] for k in names])
x0 = np.array([json.loads(Path("fit_primary.json").read_text())["params"][k] for k in names])
model_fn = F.make_model()
sc = ev.Scorer(F.USE, "NO", mass_floor=F.FLOOR, scale=F.SCALE)


def f(x):
    try:
        c = sc.chi2_soft(model_fn(dict(zip(names, x))))
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return 1e5
    return c if np.isfinite(c) else 1e5


t0 = time.time()
finals = []
best = (f(x0), x0)
for i in range(n):
    x = x0 * np.exp(rng.normal(0, 0.7, x0.size)) * np.where(rng.random(x0.size) < 0.2, -1, 1)
    x[-1] = abs(x[-1])
    x = np.clip(x, lo, hi)
    r = minimize(f, x, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                 options=dict(xatol=1e-9, fatol=1e-9, maxfev=8000, adaptive=True))
    finals.append(float(r.fun))
    if r.fun < best[0]:
        best = (float(r.fun), r.x)
fr = ev.fit_and_evaluate(model_fn, F.BOUNDS, "NO", F.USE, k_discrete=F.K_DISCRETE, n_starts=1, seed=seed,
                         extra_starts=[dict(zip(names, best[1]))], mass_floor=F.FLOOR, scale=F.SCALE)
out = {"seed": seed, "n": n, "start_chi2": f(x0), "finals_sorted": sorted(finals)[:20],
       "n_below_30": int(sum(c < 30 for c in finals)), "best_polished": fr.result.chi2,
       "params": fr.params, "pulls": fr.result.pulls, "seconds": time.time() - t0}
Path(f"perturb_s{seed}.json").write_text(json.dumps(out, indent=1))
print(f"[perturb seed {seed}] {n} starts, {out['n_below_30']} end < 30; best polished chi2 {fr.result.chi2:.3f} "
      f"(primary {out['start_chi2']:.3f}); lowest finals {[round(c, 2) for c in sorted(finals)[:5]]} ({time.time() - t0:.0f} s)")

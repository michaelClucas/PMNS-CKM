"""Refit exp 023's eb08 variant (S4 x SO(10), eta_b = -0.8 fixed) with the full threshold treatment
(src.evaluate scale={..., "eta_b": -0.8}) instead of exp 023's linear model-side rescaling.

Same model, parameters, residuals and least_squares loop as exp 023 fit_ours.py; starts: the
committed eb08 best point (seed 0) plus log-normal / phase perturbations of the published point
(seeds 1..n-1, exp 023's start recipe). Usage (from this folder):
  ../../.venv/Scripts/python refit_023.py [n_starts]
Writes refit_023_eb08.json. Single process.
"""
import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = Path(__file__).resolve().parent
E23 = HERE.parent / "20261119-023-s4so10"
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(E23))

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import fitlib as FL  # noqa: E402
from model import observables, pred_dimless  # noqa: E402
from src.evaluate import Scorer  # noqa: E402

EB = -0.8
SC_FULL = Scorer(FL.USE_ALL, "NO", mass_floor=FL.FLOOR, scale=dict(FL.SCALE, eta_b=EB))
PAR = FL.Param(free_phases=FL.PHASES, eta_b=EB)


def pred_raw(x):
    p, _ = PAR.unpack(np.asarray(x))
    return pred_dimless(observables(p))


def resid(x):
    try:
        return FL.residuals(pred_raw(x), SC_FULL)
    except (ValueError, np.linalg.LinAlgError):
        return np.full(n_res, 1e3)


committed = json.loads((E23 / "fit_eb08.json").read_text())
n_res = len(FL.residuals(pred_raw(committed["x"]), SC_FULL))


def start(seed):
    if seed == 0:
        return np.array(committed["x"], float)
    rng = np.random.default_rng(seed)
    x0 = PAR.pack(FL.PUBLISHED)
    nr = len(FL.REAL)
    x0[:nr] *= np.exp(rng.normal(0, 0.7, nr)) * rng.choice([-1, 1], nr, p=[0.3, 0.7])
    x0[nr:nr + len(PAR.free)] = rng.uniform(0, 2 * np.pi, len(PAR.free))
    return x0


def one(seed):
    x0, best = start(seed), None
    for _ in range(3):
        r = least_squares(resid, x0, method="trf", max_nfev=600, x_scale="jac")
        c = float(np.sum(r.fun ** 2))
        if best is not None and c > best[0] - 1e-3:
            best = min(best, (c, r.x), key=lambda t: t[0])
            break
        best = (c, r.x) if best is None or c < best[0] else best
        x0 = r.x
    return best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    res = sorted(((s,) + one(s) for s in range(n)), key=lambda t: t[1])
    s, c, x = res[0]
    ev = SC_FULL(pred_raw(x), k=17)
    at_committed = SC_FULL(pred_raw(committed["x"]), k=17)
    out = {"eta_b": EB, "chi2_linear_committed": committed["chi2"], "chi2_full_at_committed_point": at_committed.chi2,
           "chi2_full_refit": ev.chi2, "pulls_full_refit": ev.pulls, "max_abs_pull": ev.max_abs_pull,
           "starts_chi2": [round(t[1], 3) for t in res], "best_seed": s, "x": list(map(float, x)),
           "seconds": time.time() - t0}
    (HERE / "refit_023_eb08.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "x"}, indent=1, default=float))

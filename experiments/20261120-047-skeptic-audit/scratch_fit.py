"""Exp 047 (a): from-scratch fits of the three textures (no warm start from exp 044).

Stage 1: mass-ratio residuals only (whitened 4-ratio block) by trust-region least squares from random
log-uniform points in [-4, 0] (phases uniform); stage 2: all 8 residuals; stage 3: L-BFGS-B on the official
chi^2. Reports the best chi^2 and how many starts end within 0.5 of it.
Usage: python scratch_fit.py [n_starts]
"""
import json
import sys
import warnings

import numpy as np
from scipy.optimize import least_squares, minimize

import textures as T

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = np.random.default_rng(4703)
    out = {}
    for name, scale in (("A6842", "MZ"), ("A6624", "GUT10"), ("A47977", "MZ")):
        tex = T.TEX[name]
        ob = T.Obj(tex, scale)
        lo, hi = T.box(tex)
        nm = tex["n_mod"]
        finals, xs = [], []
        for _ in range(n):
            x0 = np.concatenate([rng.uniform(-4, 0, nm), rng.uniform(-np.pi, np.pi, tex["k"] - nm)])
            x1 = least_squares(lambda x: ob.resid(x)[4:], x0, bounds=(lo, hi), method="trf", max_nfev=300).x
            x2 = least_squares(ob.resid, x1, bounds=(lo, hi), method="trf", max_nfev=400).x
            r = minimize(ob.chi2, x2, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 500})
            finals.append(float(r.fun))
            xs.append(r.x)
        finals = np.array(finals)
        i = int(np.argmin(finals))
        rr, _ = ob.full(xs[i])
        out[name] = {"scale": scale, "n_starts": n, "best_chi2": rr.chi2, "fit_ok": rr.fit_ok, "p": rr.p_global,
                     "max_pull": rr.max_abs_pull, "x": xs[i].tolist(),
                     "n_within_0.5": int((finals < finals.min() + 0.5).sum()),
                     "n_fit_ok_region_chi2_lt_5": int((finals < 5).sum()),
                     "quantiles": np.quantile(finals, [0.1, 0.5, 0.9]).tolist()}
        print(name, scale, json.dumps({k: v for k, v in out[name].items() if k != "x"}), flush=True)
    (T.HERE / "scratch_fit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

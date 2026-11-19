"""Step 4: the six free phases (eta', alpha_d, beta_d, alpha_e, beta_e, gamma) fixed to Z_n roots.

n = 3: each phase in {0, 2pi/3, 4pi/3}; 3^6 = 729 assignments, k_discrete = ceil(log2 729) = 10 bits.
Couplings stay real with either sign (a sign is part of a continuous coupling, not a bit), so the
effective phase of a free-phase term lies in Z_6; eta = -2pi/3 as in the paper.
Every assignment is fitted (warm start from the free-phase best fit `fit_<base>.json`), the best 20
are refitted from 6 extra random-sign starts. Writes fit_z3_<base>.json; all assignments' chi^2 go to
artifacts/023/z3_<base>_all.json.
Usage: python fit_zn.py <base variant: ebfree|eb0|eb08> [n_proc]
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from fitlib import HERE, ROOT, PHASES, REAL, Param, USE_ALL, make_resid, predict, scorer

N_ROOT = 3


def par_for(combo, base):
    eb = {"ebfree": "free", "eb0": "fixed0", "eb08": -0.8}[base]
    fixed = {k: 2 * np.pi * c / N_ROOT for k, c in zip(PHASES, combo)}
    return Param(free_phases=(), fixed_phases=fixed, eta_b=eb)


def fit_combo(args):
    combo, base, x_warm, seeds = args
    par = par_for(combo, base)
    f = make_resid(par)
    best = None
    for s in seeds:
        x0 = np.array(x_warm, float)
        if s > 0:
            rng = np.random.default_rng(s * 7919 + sum(c * 3 ** i for i, c in enumerate(combo)))
            x0[:13] *= np.exp(rng.normal(0, 0.3, 13)) * rng.choice([-1, 1], 13)
        r = least_squares(f, x0, method="trf", max_nfev=300, x_scale="jac")
        r = least_squares(f, r.x, method="trf", max_nfev=300, x_scale="jac")
        c = float(np.sum(r.fun ** 2))
        if best is None or c < best[0]:
            best = (c, r.x.tolist())
    return tuple(combo), best[0], best[1]


def main():
    base = sys.argv[1]
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    fit = json.loads((HERE / f"fit_{base}.json").read_text())
    x_full = np.array(fit["x"])
    x_warm = list(x_full[:13]) + ([x_full[-1]] if base == "ebfree" else [])
    combos = list(itertools.product(range(N_ROOT), repeat=6))
    t0 = time.time()
    with Pool(n_proc) as pool:
        out = pool.map(fit_combo, [(c, base, x_warm, (0,)) for c in combos], chunksize=8)
        out.sort(key=lambda t: t[1])
        top = pool.map(fit_combo, [(c, base, x, (0, 1, 2, 3, 4, 5, 6)) for c, _, x in out[:20]])
    top.sort(key=lambda t: t[1])
    combo, c, x = top[0]
    par = par_for(combo, base)
    p, eb = par.unpack(np.array(x))
    pred, o = predict(p, eb)
    k_cont = 10 + (1 if base in ("ebfree", "eb08") else 0)
    k_d = int(np.ceil(np.log2(N_ROOT ** 6)))
    res = scorer(USE_ALL)(pred, k=k_cont + k_d, k_discrete=k_d)
    summary = {"base": base, "n_root": N_ROOT, "n_assignments": len(combos), "k_cont": k_cont, "k_discrete_phase_bits": k_d,
               "best_combo_units_2pi_over_n": list(combo), "chi2": res.chi2, "N": res.N, "dof": res.dof, "p": res.p,
               "max_abs_pull": res.max_abs_pull, "pulls": res.pulls, "eta_b": eb,
               "params": {k: float(v) for k, v in p.items()}, "pred_gut": {k: float(v) for k, v in pred.items()},
               "top20_after_refine": [[list(cc), round(cv, 3)] for cc, cv, _ in top],
               "n_assign_chi2_below": {str(t): int(sum(1 for _, cv, _ in out if cv < t)) for t in (20, 30, 50, 100)},
               "x": x, "wall_s": time.time() - t0}
    (HERE / f"fit_z3_{base}.json").write_text(json.dumps(summary, indent=1, default=float))
    (ROOT / "artifacts" / "023" / f"z3_{base}_all.json").write_text(json.dumps([[list(cc), cv] for cc, cv, _ in out]))
    print("Z3", base, "best", combo, "chi2", round(res.chi2, 3), "N", res.N, "k_cont", k_cont, "bits", k_d,
          "p", res.p, "max|pull|", round(res.max_abs_pull, 2))
    print({k: round(v, 2) for k, v in res.pulls.items()})


if __name__ == "__main__":
    main()

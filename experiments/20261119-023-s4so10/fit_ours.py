"""Step 2 (and 4): refit 1705.01555 to our data with the official scorer.

Variants (argv[1]):
  eb0      free phases, eta_b = 0 (pure src.rg MSSM running)                  k_cont 16
  ebfree   free phases, eta_b fitted                                          k_cont 17
  eb08     free phases, eta_b = -0.8 (paper's empirically chosen value)       k_cont 17 (eta_b counted)
  real_eb0 / real_ebfree   all six free phases = 0, signed couplings (CP only from eta = -2pi/3)
Writes fit_<variant>.json. Usage: python fit_ours.py <variant> [n_starts] [n_proc]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from fitlib import HERE, PUBLISHED, PHASES, REAL, Param, USE_ALL, make_resid, predict, scorer

VARIANTS = {
    "eb0": dict(free_phases=PHASES, eta_b="fixed0"),
    "ebfree": dict(free_phases=PHASES, eta_b="free"),
    "eb08": dict(free_phases=PHASES, eta_b=-0.8),
    "real_eb0": dict(free_phases=(), fixed_phases={k: 0.0 for k in PHASES}, eta_b="fixed0"),
    "real_ebfree": dict(free_phases=(), fixed_phases={k: 0.0 for k in PHASES}, eta_b="free"),
}


def one(args):
    variant, seed, use = args
    par = Param(**VARIANTS[variant])
    f = make_resid(par, use)
    rng = np.random.default_rng(seed)
    x0 = par.pack(PUBLISHED, -0.8 if par.eb_mode == "free" else 0.0)
    if seed > 0:
        nr = len(REAL)
        x0[:nr] *= np.exp(rng.normal(0, 0.7, nr)) * rng.choice([-1, 1], nr, p=[0.3, 0.7])
        x0[nr:nr + len(par.free)] = rng.uniform(0, 2 * np.pi, len(par.free))
        if par.eb_mode == "free":
            x0[-1] = rng.uniform(-0.85, 0.3)
    best = None
    for _ in range(3):  # restart from the result: trf stalls on the sqrt(PMNS) residual
        r = least_squares(f, x0, method="trf", max_nfev=600, x_scale="jac")
        c = float(np.sum(r.fun ** 2))
        if best is not None and c > best[0] - 1e-3:
            best = min(best, (c, r.x), key=lambda t: t[0])
            break
        best = (c, r.x) if best is None or c < best[0] else best
        x0 = r.x
    return seed, best[0], best[1].tolist()


def official(variant, x, use=USE_ALL, k_cont=None, k_discrete=0):
    par = Param(**VARIANTS[variant])
    p, eb = par.unpack(np.asarray(x))
    pred, o = predict(p, eb)
    sc = scorer(use)
    res = sc(pred, k=k_cont + k_discrete, k_discrete=k_discrete)
    return res, p, eb, o, pred


def k_of(variant):
    par = Param(**VARIANTS[variant])
    # 13 reals minus 3 overall scales that no scored observable sees (Y_u, m_nu, common (Y_d, Y_e))
    k = 10 + len(par.free) + (1 if par.eb_mode == "free" else 0) + (1 if isinstance(par.eb_mode, float) else 0)
    return k


def main():
    variant = sys.argv[1]
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    t0 = time.time()
    with Pool(n_proc) as pool:
        out = pool.map(one, [(variant, s, USE_ALL) for s in range(n_starts)])
    out.sort(key=lambda t: t[1])
    seed, c, x = out[0]
    kc = k_of(variant)
    res, p, eb, o, pred = official(variant, x, k_cont=kc)
    summary = {
        "variant": variant, "use": list(USE_ALL), "N": res.N, "k_cont": kc,
        "chi2": res.chi2, "chi2_global": res.chi2_global, "dof": res.dof, "p": res.p,
        "max_abs_pull": res.max_abs_pull, "pulls": res.pulls, "terms": res.terms,
        "pred_gut": {k: float(v) for k, v in pred.items()}, "eta_b": eb,
        "params": {k: float(v) for k, v in p.items()},
        "phases_over_pi": {k: float((p[k] / np.pi) % 2) for k in PHASES},
        "nu_masses_meV_rel": [float(v) for v in o["mnu_meV"] / o["mnu_meV"][2]],
        "all_starts_chi2": [round(t[1], 3) for t in out], "best_seed": seed, "x": x,
        "wall_s": time.time() - t0,
    }
    (HERE / f"fit_{variant}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(variant, "chi2", round(res.chi2, 3), "N", res.N, "k_cont", kc, "eta_b", round(eb, 3),
          "max|pull|", round(res.max_abs_pull, 2), "starts", [round(t[1], 1) for t in out[:8]])
    print({k: round(v, 2) for k, v in res.pulls.items()})


if __name__ == "__main__":
    main()

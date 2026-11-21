"""Exp 046 conditional PMNS test (M_R = diag(1, x, y) options).

Quark parameters frozen at exp 045's quark optimum for the structure and scale (Q8 fitted, chi2 <= 2);
BB's x_e set by a charged-lepton-only scan (m_e/m_mu, m_mu/m_tau). Then the only free parameters are
ln x, ln y: scan them on a fine grid (step 0.25 in [-34, 34]) and polish the best 10 points with L-BFGS-B,
scoring only the PMNS block (4 PMNS + dm_ratio, N 5) at the model scale. The result is the best PMNS
fit the extension can give once its quark sector is fitted: Y_nu and U_e are then fixed, and only the 2
M_R parameters act on 5 observables. Output: cond.json, printed log.
Usage: python cond46.py [n_proc]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit46 as F  # noqa: E402
import lib46 as L  # noqa: E402
from src import evaluate as E  # noqa: E402

PMK = list(L.PM) + ["dm_ratio"]


def cells():
    out = []
    for m in ("RRR5", "BB", "HS"):
        for (a, y, mr) in L.OPTIONS[m]:
            if mr != "D":
                continue
            for sc in F.SCALES[m]:
                out.append((m, a, y, sc, -0.25 if m == "RRR5" else 0.0, 0.0))
    return out


def run(cell):
    m, a, y, sc, eb, el = cell
    fn, b, per, qn = L.ext(m, a, y, "D")
    scale = dict(L.SCALES[sc])
    if m == "RRR5":
        scale.update(eta_b=eb, eta_l=el)
    q = F.quark_starts(m, sc, eb)[0]
    # refit the quark parameters on Q8 alone at this scale (exp 045 optimum or nearest-scale start)
    qfn, qb, qper = L.quark_part(m)
    from lib45 import quark_obs
    fq = E.fit_and_evaluate(lambda t: quark_obs(*qfn(t)), qb, "NO", list(L.Q8), periodic=qper, n_starts=4,
                            seed=1, extra_starts=[q], mass_floor=0.02, scale=scale, method="lbfgsb")
    base = dict(fq.params)
    note = {"q8_chi2": fq.result.chi2}
    if "lxe" in b:
        scl = E.Scorer(list(L.CL), "NO", mass_floor=0.02, scale=scale)
        xs = np.linspace(-5, 5, 2001)
        cs = [scl.chi2(fn(dict(base, lxe=v, lx=0.0, ly=0.0))) for v in xs]
        base["lxe"] = float(xs[int(np.argmin(cs))])
        note["cl_chi2_at_best_xe"] = float(min(cs))
    sc5 = E.Scorer(PMK, "NO", mass_floor=0.02, scale=scale)
    scq = E.Scorer(list(L.Q8) + list(L.CL), "NO", mass_floor=0.02, scale=scale)
    ax = np.arange(-34, 34.001, 0.25)
    cand = []
    n_in = 0
    for gx in ax:
        for gy in ax:
            try:
                p = fn(dict(base, lx=gx, ly=gy))
                c = sc5.chi2(p)
                cs_ = sc5.chi2_soft(p, include_finite=True)
            except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
                continue
            n_in += np.isfinite(c)
            cand.append((cs_, gx, gy))
    cand.sort()
    best = None
    for _, gx, gy in cand[:10]:
        f = lambda z: sc5.chi2_soft(fn(dict(base, lx=z[0], ly=z[1])), include_finite=True)
        r = minimize(f, [gx, gy], method="L-BFGS-B", bounds=[L.LNMR, L.LNMR])
        if best is None or r.fun < best.fun:
            best = r
    th = dict(base, lx=float(best.x[0]), ly=float(best.x[1]))
    p = fn(th)
    res = sc5(p, k=2)
    rq = scq(p)
    return {"cell": list(cell), "pmns_chi2": res.chi2, "pmns_soft": float(best.fun), "pmns_pulls": res.pulls,
            "grid_points_inside_tables": int(n_in), "grid_points": len(ax) ** 2,
            "quark_cl_chi2": rq.chi2, "quark_cl_pulls": rq.pulls, "params": th,
            "pred": {k: p[k] for k in L.N15}, **note}


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = []
    with mp.Pool(nproc) as pool:
        for r in pool.imap(run, cells()):
            out.append(r)
            json.dump(out, open(HERE / "cond.json", "w"), indent=1, default=float)
    for r in out:
        print(f"{' '.join(map(str, r['cell']))}: PMNS chi2 {r['pmns_chi2']:.4g} (soft {r['pmns_soft']:.5g}); "
              f"Q8 refit {r['q8_chi2']:.3g}; grid in-table {r['grid_points_inside_tables']}/{r['grid_points']}; Q8+CL chi2 {r['quark_cl_chi2']:.4g}; "
              f"pred {{{', '.join(f'{k}: {r['pred'][k]:.4g}' for k in PMK + list(L.CL))}}}", flush=True)
    json.dump(out, open(HERE / "cond.json", "w"), indent=1, default=float)

"""Exp 035 (b), staged fit of the Dirac SO(10) structure in dirac.py (same model, N, k, scale).

Plain DE over 17 parameters did not reach the NuFIT tables (dirac_fits.jsonl). The quark sector
(Y_u = S + D, Y_d = r2 S + e^{iw} r1 D) does not depend on (c_e, c_nu), so:
  stage 1  quark-only fits (use = CKM 4 + mr_u_c, mr_c_t, mr_d_s, mr_s_b at 2e16 GeV, official Scorer):
           DE (popsize 10, maxiter 250) for seeds 1..S; the final DE populations' best members with
           quark chi2 < Q_MAX are kept as starts (deduplicated);
  stage 2  for each Clebsch pair: L-BFGS-B (maxfun 3000) on the full N = 16 chi2_soft(include_finite=True)
           from the best N_PER = 3 starts of each seed, then the official src.evaluate.evaluate at the best point.
Usage: python dirac_staged.py stage1 <seed>        -> artifacts/035/quark_starts_<seed>.npy (gitignored)
       python dirac_staged.py stage2 <c_e> <c_nu> <ordering>  -> JSON line on stdout
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import dirac as M  # noqa: E402
from src import evaluate as E  # noqa: E402
from src.stats import discrete_bits  # noqa: E402

ART = ROOT / "artifacts" / "035"
QUSE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b"]
NAMES = list(M.BOUNDS)
LO = np.array([M.BOUNDS[n][0] for n in NAMES])
HI = np.array([M.BOUNDS[n][1] for n in NAMES])
Q_MAX = 30.0
N_PER = 3  # starts per stage-1 seed used in stage 2 (time budget)


def stage1(seed):
    sc = E.Scorer(QUSE, "NO", scale=M.SCALE)
    model = M.model_factory(1.0, 1.0, "NO")  # leptons irrelevant here
    pop = []

    def f(x):
        try:
            c = sc.chi2(model(dict(zip(NAMES, x))))
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e9
        if np.isfinite(c):
            pop.append((c, np.array(x)))
        return c if np.isfinite(c) else 1e9

    t0 = time.time()
    r = differential_evolution(f, list(zip(LO, HI)), seed=seed, popsize=10, maxiter=250, tol=1e-10, init="sobol",
                               polish=True)
    pop.sort(key=lambda t: t[0])
    keep = []
    for c, x in pop:
        if c > Q_MAX or len(keep) >= 12:
            break
        if all(np.max(np.abs(x - y)) > 0.3 for y in keep):
            keep.append(x)
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / f"quark_starts_{seed}.npy", np.array(keep))
    print(json.dumps(dict(seed=seed, quark_chi2=float(r.fun), n_starts=len(keep), secs=time.time() - t0)), flush=True)


def stage2(c_e_s, c_nu_s, ordering):
    # per seed file the starts are sorted by quark chi2; take the best N_PER of each seed
    starts = np.concatenate([np.load(p)[:N_PER] for p in sorted(ART.glob("quark_starts_*.npy"))])
    model = M.model_factory(M.CLEBSCH[c_e_s], M.CLEBSCH[c_nu_s], ordering)
    sc = E.Scorer(M.USE, ordering, scale=M.SCALE)

    def f(x):
        try:
            c = sc.chi2_soft(model(dict(zip(NAMES, x))), include_finite=True)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e9
        return c if np.isfinite(c) else 1e9

    t0 = time.time()
    best = None
    finals = []
    for x0 in starts:
        r = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(LO, HI)), options=dict(maxfun=3000))
        finals.append(float(r.fun))
        if best is None or r.fun < best.fun:
            best = r
    th = dict(zip(NAMES, map(float, best.x)))
    kd = discrete_bits(len(M.CLEBSCH) ** 2)
    res = E.evaluate(model(th), ordering, use=M.USE, k=len(NAMES) + kd, k_discrete=kd, scale=M.SCALE)
    print(json.dumps(dict(c_e=c_e_s, c_nu=c_nu_s, ordering=ordering, chi2=res.chi2, chi2_global=res.chi2_global,
                          N=res.N, k=res.k, k_cont=res.k_continuous, dof=res.dof, p_global=res.p_global,
                          max_pull=res.max_abs_pull, pulls=res.pulls, params=th, n_starts=len(starts),
                          start_finals=sorted(finals)[:5], secs=time.time() - t0), default=float), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "lower":
        pass
    elif sys.argv[1] == "stage1":
        stage1(int(sys.argv[2]))
    else:
        stage2(sys.argv[2], sys.argv[3], sys.argv[4])


def lower_bound(c_e_s, c_nu_s, seed):
    """chi2 of the N = 12 finite subset (CKM 4 + 7 mass ratios + dm_ratio; no PMNS tables) at 2e16 GeV.
    Every term is also in the full N = 16 chi2, so its minimum is a lower bound on the full minimum
    (up to search quality). DE popsize 10, maxiter 300, L-BFGS-B polish; the 3 stage-1 quark starts
    are injected into the initial population."""
    use = [u for u in M.USE if u not in ("s12sq", "s13sq", "s23sq", "dcp_deg")]
    model = M.model_factory(M.CLEBSCH[c_e_s], M.CLEBSCH[c_nu_s], "NO")
    sc = E.Scorer(use, "NO", scale=M.SCALE)

    def f(x):
        try:
            c = sc.chi2(model(dict(zip(NAMES, x))))
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e12
        return c if np.isfinite(c) else 1e12

    t0 = time.time()
    rng = np.random.default_rng(seed)
    init = LO + rng.random((10 * len(NAMES), len(NAMES))) * (HI - LO)
    st = np.concatenate([np.load(p) for p in sorted(ART.glob("quark_starts_*.npy"))])
    init[: len(st)] = np.clip(st, LO, HI)
    r = differential_evolution(f, list(zip(LO, HI)), seed=seed, init=init, maxiter=300, tol=1e-10, polish=True)
    th = dict(zip(NAMES, map(float, r.x)))
    p = sc(model(th), k=0)
    print(json.dumps(dict(c_e=c_e_s, c_nu=c_nu_s, seed=seed, chi2_lower=float(r.fun), N_sub=len(use),
                          pulls=p.pulls, secs=time.time() - t0, params=th), default=float), flush=True)


if __name__ == "__main__" and sys.argv[1] == "lower":
    lower_bound(sys.argv[2], sys.argv[3], int(sys.argv[4]))

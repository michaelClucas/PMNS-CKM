"""Exp 039 exact fits with the official scorer (src.evaluate), parts (a) and (b).

Each job: scipy differential_evolution on Scorer.chi2_soft(include_finite=True) (the landscape
fit_and_evaluate's "de" method uses), L-BFGS-B polish, then the official src.evaluate.evaluate at the
optimum with the honest k (fitted continuous parameters + bits of every data-selected choice).
Usage: python fit.py [jobs-filter] -> appends JSON lines to fits.jsonl (3 worker processes).
"""
from __future__ import annotations

import functools
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import model as M  # noqa: E402
from src import evaluate as E  # noqa: E402


def make(cls, opt):
    """-> (model_fn(theta), bounds, k_discrete, note). opt: dict of discrete options."""
    o = opt.get("ordering", "NO")
    if cls == "FZ4U":
        f = functools.partial(M.fz4_model, eta=opt["eta"], nu=opt["nu"], flip_phase=opt.get("flip", False), ordering=o)
        return f, M.fz4_bounds(), 2, "bits: eta (1), neutrino option (1)"
    if cls == "FZ4S":
        f = functools.partial(M.fz4_model, eta=opt["eta"], nu=opt["nu"], lshift=tuple(opt["shift"]), ordering=o)
        return f, M.fz4_bounds(), 4, "bits: eta (1), neutrino option (1), lepton sign shifts (2)"
    if cls == "FZ4E":
        f = functools.partial(M.fz4_model, eta=opt["eta"], nu=opt["nu"], a_fixed={"e": 1.0 - 1e-4}, ordering=o)
        return f, M.fz4_bounds(), 4, "a shared by u, d, nu; a_e fixed at its endpoint (choice of sector, 2 bits); eta, nu"
    if cls == "FZ4F":
        f = functools.partial(M.fz4_model, eta=opt["eta"], nu=opt["nu"], per_sector_a=True, ordering=o)
        return f, M.fz4_bounds(True), 2, "diagnostic, a per sector; bits: eta, neutrino option"
    if cls == "DEMU":
        f = functools.partial(M.demu_model, nu=opt["nu"], ordering=o)
        return f, M.demu_bounds(), 1, "bits: neutrino option (1)"
    if cls == "DEMU2":
        f = functools.partial(M.demu_model, nu=opt["nu"], ordering=o)
        return f, M.demu_bounds(True), 1, "phi_u = phi_nu, phi_d = phi_e; bits: neutrino option (1)"
    if cls == "DEMFX":
        f = functools.partial(M.demfx_model, ordering=o)
        return f, M.demfx_bounds(), 0, "literature structure, no data-selected choice except ordering"
    if cls == "DEMFX2":
        f = functools.partial(M.demfx_model, ordering=o)
        return f, M.demfx_bounds(True), 0, "literature structure with separate phi_u, phi_d"
    if cls == "GEO":
        f = functools.partial(M.geo_model, p=opt["p"], nu=opt["nu"], ordering=o)
        return f, M.geo_bounds(opt["p"]), 7, "bits: 64 patterns x 2 neutrino options = 7"
    raise KeyError(cls)


def fit(job, use=None, popsize=10, maxiter=300):
    from scipy.optimize import differential_evolution, minimize
    use = list(use or job.get("use") or M.USE)
    t0 = time.time()
    cls, opt, seed, scale_name = job["cls"], job["opt"], job["seed"], job["scale"]
    o = opt.get("ordering", "NO")
    model, bounds, kd, note = make(cls, opt)
    for k, v in job.get("fix", {}).items():  # held-out stage: shared parameters fixed
        bounds.pop(k)
    fixed = dict(job.get("fix", {}))
    names = list(bounds)
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    scale = M.SCALES[scale_name]
    sc = E.Scorer(use, o, mass_floor=M.FLOOR, scale=scale)

    def th_of(x):
        return {**fixed, **dict(zip(names, map(float, x)))}

    def f(x):
        try:
            c = sc.chi2_soft(model(th_of(x)), include_finite=True)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e9
        return c if np.isfinite(c) else 1e9

    init = "sobol"
    r = differential_evolution(f, list(zip(lo, hi)), seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-9,
                               init=init, polish=False)
    p = minimize(f, r.x, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxfun=6000))
    x = p.x if p.fun < r.fun else r.x
    th = th_of(x)
    k_cont = len(names)
    k = k_cont + kd
    res = E.evaluate(model(th), o, use=use, k=k, k_discrete=kd, mass_floor=M.FLOOR, scale=scale)
    pred = model(th)
    return dict(cls=cls, opt=opt, seed=seed, scale=scale_name, use=use, chi2=res.chi2, chi2_global=res.chi2_global,
                N=res.N, k=res.k, k_cont=res.k_continuous, k_discrete=res.k_discrete, dof=res.dof, p=res.p,
                p_global=res.p_global, max_pull=res.max_abs_pull, compression_ok=bool(res.compression_ok),
                fit_ok=bool(res.fit_ok), pulls=res.pulls, params=th, note=note,
                pred={k2: pred[k2] for k2 in M.USE}, de_chi2=float(r.fun), secs=time.time() - t0)


def fit_from(job, start, maxfun=6000):
    """L-BFGS-B + Nelder-Mead from a given start (e.g. the emulator optimum), official evaluate."""
    from scipy.optimize import minimize
    cls, opt, scale_name = job["cls"], job["opt"], job["scale"]
    o = opt.get("ordering", "NO")
    model, bounds, kd, note = make(cls, opt)
    names = list(bounds)
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    x0 = np.clip([start[n] for n in names], lo, hi)
    scale = M.SCALES[scale_name]
    sc = E.Scorer(M.USE, o, mass_floor=M.FLOOR, scale=scale)

    def f(x):
        try:
            c = sc.chi2_soft(model(dict(zip(names, map(float, x)))), include_finite=True)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e9
        return c if np.isfinite(c) else 1e9

    p = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxfun=maxfun))
    q = minimize(f, p.x, method="Nelder-Mead", bounds=list(zip(lo, hi)), options=dict(maxfev=maxfun))
    x = q.x if q.fun < p.fun else p.x
    th = dict(zip(names, map(float, x)))
    res = E.evaluate(model(th), o, use=M.USE, k=len(names) + kd, k_discrete=kd, mass_floor=M.FLOOR, scale=scale)
    return dict(cls=cls, opt=opt, seed="emulator-start", scale=scale_name, use=M.USE, chi2=res.chi2,
                chi2_global=res.chi2_global, N=res.N, k=res.k, k_cont=res.k_continuous, k_discrete=res.k_discrete,
                dof=res.dof, p=res.p, p_global=res.p_global, max_pull=res.max_abs_pull,
                compression_ok=bool(res.compression_ok), fit_ok=bool(res.fit_ok), pulls=res.pulls, params=th,
                note=note, start_chi2=float(f(x0)))


def jobs():
    J = []
    for sc in ("MZ", "GUT"):
        for eta in (1, -1):
            for nu in ("dirac", "seesaw"):
                for s in (1, 2):
                    J.append(dict(cls="FZ4U", opt=dict(eta=eta, nu=nu), seed=s, scale=sc))
        for nu in ("dirac", "seesaw"):
            J.append(dict(cls="DEMU", opt=dict(nu=nu), seed=1, scale=sc))
        for o in ("NO", "IO"):
            J.append(dict(cls="DEMFX", opt=dict(ordering=o), seed=1, scale=sc))
    for eta in (1, -1):
        for nu in ("dirac", "seesaw"):
            J.append(dict(cls="FZ4F", opt=dict(eta=eta, nu=nu), seed=1, scale="MZ"))
    # second batch (added after the first: DEMU's single phi forces delta_CKM ~ 0)
    for sc in ("MZ", "GUT"):
        for nu in ("dirac", "seesaw"):
            J.append(dict(cls="DEMU2", opt=dict(nu=nu), seed=1, scale=sc, batch=2))
    for o in ("NO", "IO"):
        J.append(dict(cls="DEMFX2", opt=dict(ordering=o), seed=1, scale="MZ", batch=2))
    # third batch (added after seeing FZ4U fail on s23sq): fixed -1 factors on lepton entries
    for sh in ((1, 0), (0, 1), (1, 1)):
        for eta in (1, -1):
            for nu in ("dirac", "seesaw"):
                J.append(dict(cls="FZ4S", opt=dict(eta=eta, nu=nu, shift=list(sh)), seed=1, scale="MZ", batch=3))
    # fourth batch (post hoc, from FZ4F: a_u ~ a_d ~ 0.17, a_e ~ 0.9-1.0): a_e fixed at 1
    for eta in (1, -1):
        for nu in ("dirac", "seesaw"):
            J.append(dict(cls="FZ4E", opt=dict(eta=eta, nu=nu), seed=1, scale="MZ", batch=4))
    return J


def run_one(job):
    try:
        out = fit(job)
    except Exception as e:  # noqa: BLE001  (logged, never silently dropped)
        out = dict(job, error=repr(e))
    return out


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    J = [j for j in jobs() if flt in json.dumps(j)]
    print(f"{len(J)} jobs", flush=True)
    with mp.Pool(2) as pool, open(HERE / "fits.jsonl", "a") as fh:
        for out in pool.imap_unordered(run_one, J):
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            print(out.get("cls"), out.get("opt"), out.get("scale"), out.get("seed"), "chi2",
                  round(out.get("chi2", np.nan), 2), "k", out.get("k"), "N", out.get("N"),
                  "p_glob", out.get("p_global"), "maxpull", round(out.get("max_pull", np.nan), 2),
                  out.get("error", ""), f"{out.get('secs', 0):.0f}s", flush=True)

"""Step 1: lepton-only tau region for the exp 017 S4 tau-only neutrino lead.

Model: S4, L ~ 3', type-I seesaw with constant M_D and M_N from the single weight-2 form (lead_scan.py),
charged-lepton rows e-set 19 = (4 1', 6 1 [2 couplings], 6 1') [the lead] and, as a variant, e-set 20 =
(4 1', 6 1', 8 1 [2 couplings]) [second best in exp 017]. NO. chi2 exactly as exp 017 lead_scan
(search.exact_chi2_fn: e log mass ratios with 1% floor, dm21/|dm3l|, PMNS 4 via src.evaluate NuFIT tables).
At each tau the 3 charged-lepton couplings are profiled (random batch + Nelder-Mead).

    python lepton_region.py        -> artifacts/fixedtau/lepton_grid_*.npz, lepton_region.json
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, "..", ".."))]

import numpy as np  # noqa: E402

WORKERS = int(os.environ.get("FT_WORKERS", "5"))
ESETS = tuple(int(x) for x in os.environ.get("FT_ESETS", "19").split(","))  # run used 19 only (20 aborted for time)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tau_to_u(tau):
    import models as md
    x = tau.real
    ylo = np.sqrt(np.clip(1 - x ** 2, 0, None))
    return np.stack([x + 0.5, (tau.imag - ylo) / (md.YMAX - ylo)], -1)


_CACHE = {}


def _setup(ie):
    if ie in _CACHE:
        return _CACHE[ie]
    import lead_scan  # noqa: F401  (patches models.nu_matrix for the seesaw)
    import models as md
    import search as se
    from src.nulls import Universe
    u = Universe.real("NO")
    T = se.targets(u, "NO")
    e = md.sector_sets(4, "3p")[ie]
    a = md.LeptonAssign(4, "3p", e, ("SS", 1))
    sc = u.scorer(se.USE8[4:], "NO")
    _CACHE[ie] = (a, T, sc, se, md)
    return _CACHE[ie]


def profile_point(ie, tau, warm=None, rng=None, n_samp=1500):
    from scipy.optimize import minimize
    a, T, sc, se, md = _setup(ie)
    rng = rng or np.random.default_rng(0)
    lo, hi = se.bounds(a)
    uu = tau_to_u(np.array([tau]))[0]
    f = se.exact_chi2_fn(a, T, sc, "NO")
    Z = rng.uniform(lo, hi, size=(n_samp, len(lo)))
    Z[:, :2] = uu
    if warm is not None:
        Z[0, 2:] = warm
    o = se.sector_obs(a, Z, "NO")
    cg = se.chi2_lepton_gauss(o, T)
    starts = [Z[j, 2:] for j in np.argsort(cg)[:2]]
    if warm is not None:
        starts.append(np.asarray(warm))
    best = (np.inf, None)
    for s in starts:
        fp = lambda p: f(np.concatenate([uu, np.clip(p, lo[2:], hi[2:])])) + 10 * float(((p - np.clip(p, lo[2:], hi[2:])) ** 2).sum())
        r = minimize(fp, s, method="Nelder-Mead", options={"maxfev": 600, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True})
        if r.fun < best[0]:
            best = (float(r.fun), np.clip(r.x, lo[2:], hi[2:]))
    return best


def _task(args):
    ie, taus, warms, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for t, w in zip(taus, warms):
        c, p = profile_point(ie, t, w, rng)
        out.append((c, p))
    return ie, out


def run_grid(pool, ie, taus, warms=None, tag=""):
    warms = warms if warms is not None else [None] * len(taus)
    n = len(taus)
    chunk = 20
    tasks = [(ie, taus[i:i + chunk], warms[i:i + chunk], [ie, i, len(tag)]) for i in range(0, n, chunk)]
    res = []
    for _, out in pool.imap(_task, tasks):
        res += out
    c = np.array([r[0] for r in res])
    P = np.array([r[1] for r in res])
    return c, P


def domain_mask(tau):
    return (np.abs(tau.real) <= 0.5) & (np.abs(tau) >= 1)


def _global_task(args):
    """Free-tau multi-start fit (5 params) from one start; returns (chi2, z)."""
    from scipy.optimize import minimize
    ie, z0 = args
    a, T, sc, se, md = _setup(ie)
    lo, hi = se.bounds(a)
    f = se.exact_chi2_fn(a, T, sc, "NO")
    fb = lambda x: f(np.clip(x, lo, hi)) + 10.0 * float(((x - np.clip(x, lo, hi)) ** 2).sum())
    r = minimize(fb, z0, method="Nelder-Mead", options={"maxfev": 3000, "xatol": 1e-8, "fatol": 1e-7, "adaptive": True})
    r = minimize(fb, r.x, method="Nelder-Mead", options={"maxfev": 1500, "xatol": 1e-9, "fatol": 1e-8, "adaptive": True})
    z = np.clip(r.x, lo, hi)
    tau = md.tau_from_unit(z[None, :2])[0]
    return float(r.fun), complex(tau), z[2:].tolist()


def global_search(pool, ie, n_pre=200_000, n_starts=80, seed=0):
    a, T, sc, se, md = _setup(ie)
    rng = np.random.default_rng([seed, ie])
    lo, hi = se.bounds(a)
    Z = rng.uniform(lo, hi, size=(n_pre, len(lo)))
    o = se.sector_obs(a, Z, "NO")
    cg = se.chi2_lepton_gauss(o, T)
    order = np.argsort(cg)
    starts, taus = [], []
    for j in order:  # distinct tau starts
        t = md.tau_from_unit(Z[j:j + 1, :2])[0]
        if all(abs(t - s) > 0.02 for s in taus):
            starts.append(Z[j]); taus.append(t)
        if len(starts) >= n_starts:
            break
    lead = json.load(open(os.path.join(HERE, "..", "20261118-017-modular-search", "lead_tauonly_nu.json")))
    for r in lead[:2]:  # exp 017 optima for e-sets 19, 20 (z = u_tau + couplings) and their gCP images
        z = np.array(r["z"])
        if len(z) == len(lo):
            starts.append(z)
            zi = z.copy(); zi[0] = 1 - zi[0]
            starts.append(zi)
    res = pool.map(_global_task, [(ie, s) for s in starts])
    return sorted(res, key=lambda r: r[0])


def flood(pool, ie, tau0, warm0, thr, h):
    """Grid flood fill from tau0 (step h) over points with chi2 <= thr; returns dict (i,j)->(tau, chi2, P)."""
    key0 = (round(tau0.real / h), round(tau0.imag / h))
    done = {}
    frontier = {key0: warm0}
    while frontier:
        keys = list(frontier)
        taus = [complex(i * h, j * h) for i, j in keys]
        c, P = run_grid(pool, ie, taus, [frontier[k] for k in keys], tag="f")
        new = {}
        for k, t, ci, pi in zip(keys, taus, c, P):
            done[k] = (t, float(ci), pi)
        for k, t, ci, pi in zip(keys, taus, c, P):
            if ci <= thr:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        kk = (k[0] + di, k[1] + dj)
                        tt = complex(kk[0] * h, kk[1] * h)
                        if kk not in done and kk not in new and abs(tt.real) <= 0.5 and abs(tt) >= 1:
                            new[kk] = pi
        frontier = new
        if len(done) > 20000:
            log("flood capped")
            break
    # re-warm pass: neighbours better by > 0.05 restart the point
    keys = list(done)
    for _ in range(2):
        redo, warms = [], []
        for k in keys:
            best = min(((done[(k[0] + a_, k[1] + b_)][1], done[(k[0] + a_, k[1] + b_)][2]) for a_ in (-1, 0, 1)
                        for b_ in (-1, 0, 1) if (k[0] + a_, k[1] + b_) in done), key=lambda x: x[0])
            if best[0] < done[k][1] - 0.05 and done[k][1] < thr + 20:
                redo.append(k); warms.append(best[1])
        if not redo:
            break
        c, P = run_grid(pool, ie, [done[k][0] for k in redo], warms, tag="r")
        for k, ci, pi in zip(redo, c, P):
            if ci < done[k][1]:
                done[k] = (done[k][0], float(ci), pi)
    return done


if __name__ == "__main__":
    import multiprocessing as mp
    from scipy.stats import chi2 as chi2d
    H = float(os.environ.get("FT_H", "0.0005"))
    out = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for ie in ESETS:
            g = global_search(pool, ie)
            gmin = g[0][0]
            log("e-set", ie, "global min", round(gmin, 3), g[0][1], "| top minima:",
                [(round(c, 2), complex(round(t.real, 4), round(t.imag, 4))) for c, t, _ in g[:12]])
            # distinct basins within min + 10
            basins = []
            for c, t, P in g:
                if c <= gmin + 10 and all(abs(t - b[1]) > 0.01 for b in basins):
                    basins.append((c, t, P))
            log("e-set", ie, "basins within +10:", [(round(c, 2), t) for c, t, _ in basins])
            allpts = {}
            for c, t, P in basins:
                d = flood(pool, ie, t, np.array(P), gmin + 9.0, H)
                allpts.update({k: v for k, v in d.items() if k not in allpts or v[1] < allpts[k][1]})
                log("e-set", ie, "flood from", t, "points", len(d))
            ks = list(allpts)
            tf = np.array([allpts[k][0] for k in ks])
            cf = np.array([allpts[k][1] for k in ks])
            Pf = np.array([allpts[k][2] for k in ks])
            gmin = float(min(gmin, cf.min()))
            np.savez(os.path.join(HERE, "..", "..", "artifacts", "fixedtau", f"lepton_fine_{ie}.npz"),
                     tau=tf, chi2=cf, P=Pf, gmin=gmin, h=H)
            j = int(np.argmin(cf))
            rec = {"eset": ie, "chi2_min": gmin, "tau_best": [tf[j].real, tf[j].imag], "P_best": Pf[j].tolist(),
                   "p_lepton": float(chi2d.sf(gmin, 7 - 5)), "grid_step": H,
                   "global_minima": [(c, [t.real, t.imag]) for c, t, _ in g[:20]]}
            for lvl in (2.30, 6.18):
                m = cf <= gmin + lvl
                pts = tf[m]
                cm = cf[m]
                rec[f"n_pts_{lvl}"] = int(m.sum())
                rec[f"area_{lvl}"] = float(m.sum() * H * H)
                isl, left = [], set(range(len(pts)))
                while left:
                    st = [left.pop()]
                    comp = list(st)
                    while st:
                        q = st.pop()
                        nbrs = [k for k in list(left) if abs(pts[k] - pts[q]) < 1.5 * H]
                        for k in nbrs:
                            left.discard(k)
                        st += nbrs
                        comp += nbrs
                    cp = pts[comp]
                    jm = int(np.argmin(cm[comp]))
                    isl.append({"n": len(comp), "re": [float(cp.real.min()), float(cp.real.max())],
                                "im": [float(cp.imag.min()), float(cp.imag.max())],
                                "chi2_min": float(cm[comp][jm]), "tau_min": [float(cp[jm].real), float(cp[jm].imag)]})
                rec[f"islands_{lvl}"] = sorted(isl, key=lambda r: r["chi2_min"])
            out[str(ie)] = rec
            log("e-set", ie, {k: v for k, v in rec.items() if k.startswith(("chi2", "tau", "n_", "area", "islands"))})
    json.dump(out, open(os.path.join(HERE, "lepton_region.json"), "w"), indent=1)
    log("done")

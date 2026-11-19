"""Steps 2-3: quark sector at the lepton-fixed tau, and the random-tau comparison.

Pipeline (frozen before the first run; identical for the lepton region and every random tau):
  stage A  every assignment of the group, couplings only, at each tau point of the set:
           1500 uniform prior samples -> 12 best -> batched projected Levenberg-Marquardt (40 iters).
  stage B  the TOP_B best assignments from A: 8000 samples -> 64 starts -> LM 120 iters, same tau points.
Observables (N = 8): CKM s12, s13, s23, delta (PDG, asymmetric Gaussian, same as src.evaluate) and
ln(m_u/m_c), ln(m_c/m_t), ln(m_d/m_s), ln(m_s/m_b) at M_Z (src.rg), errors with 2% floor on the three
light-quark ratios and 1% on m_c/m_t.

    python quark_search.py real     # S4 (968) + A4 (12,100) at the lepton tau region -> real_*.csv
    python quark_search.py null     # S4 at 100 random tau (reduced pipeline) + lepton tau -> null_*.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, "..", ".."))]

import numpy as np  # noqa: E402

import ftcore as fc  # noqa: E402
import models as md  # noqa: E402
import modforms as mf  # noqa: E402

WORKERS = int(os.environ.get("FT_WORKERS", "5"))
CHEAP = dict(n_samp=1500, n_starts=12, iters=40)
HEAVY = dict(n_samp=8000, n_starts=64, iters=120)
NULL_CHEAP = dict(n_samp=800, n_starts=6, iters=30)
TOP_B = 40
NULL_TOP_B = 10
N_OBS = 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


_VALS = {}


def _vals(N, qrep, taus):
    key = (N, qrep, tuple(np.round(taus, 12)))
    if key not in _VALS:
        if len(_VALS) > 8:
            _VALS.clear()
        _VALS[key] = fc.row_values(N, qrep, md.row_types(N, qrep), np.array(taus))
    return _VALS[key]


def _fit_task(args):
    N, idxs, taus, opts, seed = args
    cat = md.quark_catalog(N)
    out = []
    for i in idxs:
        a = cat[i]
        vals = _vals(N, a.qrep, taus)
        rng = np.random.default_rng([seed, N, i])
        c, P = fc.fit_fixed_tau(a, vals, len(taus), rng, **opts)
        j = int(np.argmin(c))
        out.append((i, float(c[j]), int(j), P[j].tolist(), c.tolist()))
    return out


def run_fits(pool, N, idxs, taus, opts, seed, chunk=8):
    tasks = [(N, idxs[k:k + chunk], list(taus), opts, seed) for k in range(0, len(idxs), chunk)]
    res = []
    for r in pool.imap_unordered(_fit_task, tasks):
        res += r
    return sorted(res, key=lambda r: r[1])


def pulls_at(N, i, tau, P):
    a = md.quark_catalog(N)[i]
    o = fc.quark_obs_tau(a, np.array([tau]), np.array([P]))
    R = fc.residuals(o, fc.quark_targets())[0]
    return {k: float(o[k][0]) for k in o}, dict(zip(fc.QKEYS + fc.CKMKEYS, R.tolist()))


def eval_ckm_chi2(obs):
    """CKM part through the official src.evaluate Scorer (cross-check of the residual code)."""
    from src.nulls import Universe
    sc = Universe.real("NO").scorer(fc.CKMKEYS, "NO")
    return sc.chi2({k: obs[k] for k in fc.CKMKEYS})


def search(pool, N, taus, cheap, heavy, top_b, seed, idxs=None):
    cat = md.quark_catalog(N)
    idxs = list(range(len(cat))) if idxs is None else list(idxs)
    A = run_fits(pool, N, idxs, taus, cheap, seed)
    top = [r[0] for r in A[:top_b]]
    B = run_fits(pool, N, top, taus, heavy, seed + 1, chunk=1)
    best = {r[0]: r for r in A}
    for r in B:
        if r[1] < best[r[0]][1]:
            best[r[0]] = r
    return A, B, sorted(best.values(), key=lambda r: r[1])


# ---------------------------------------------------------------- region


def lepton_region(eset="19", lvl=6.18):
    rec = json.load(open(os.path.join(HERE, "lepton_region.json")))[eset]
    d = np.load(os.path.join(HERE, "..", "..", "artifacts", "fixedtau", f"lepton_fine_{eset}.npz"))
    gmin = float(d["gmin"])
    m = d["chi2"] <= gmin + lvl
    return d["tau"][m], d["chi2"][m] - gmin, rec


def rep_points(taus, dchi, k=5):
    """Best point + farthest-point sample of the region (k points)."""
    pts = [int(np.argmin(dchi))]
    while len(pts) < min(k, len(taus)):
        d = np.min(np.abs(taus[:, None] - taus[pts][None, :]), 1)
        pts.append(int(np.argmax(d)))
    return taus[pts]


def area_uniform_tau(rng, n, ymax=4.0):
    out = []
    while len(out) < n:
        x = rng.uniform(-0.5, 0.5)
        y = rng.uniform(np.sqrt(3) / 2, ymax)
        if x * x + y * y >= 1:
            out.append(complex(x, y))
    return np.array(out)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def cmd_real():
    import multiprocessing as mp
    from src.stats import p_value
    taus6, d6, rec = lepton_region("19", 6.18)
    taus2, d2, _ = lepton_region("19", 2.30)
    K = rep_points(taus6, d6, 5)
    log("lepton region (6.18):", len(taus6), "grid points; rep points", K.tolist())
    rows = []
    summary = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for N, taus_A in ((4, K), (3, K[:1])):
            g = mf.GROUP[N]
            ncat = len(md.quark_catalog(N))
            log(g, "stage A on", ncat, "assignments at", len(taus_A), "tau points")
            A = run_fits(pool, N, list(range(ncat)), taus_A, CHEAP, 20)
            log(g, "stage A best", [(r[0], round(r[1], 1)) for r in A[:10]])
            top = [r[0] for r in A[:TOP_B]]
            B = run_fits(pool, N, top, K, HEAVY, 21, chunk=1)
            log(g, "stage B best", [(r[0], round(r[1], 1)) for r in B[:10]])
            # stage C: top 5 heavy over a 30-point subsample of the 6.18 region, and the 2.30 region best
            sub = rep_points(taus6, d6, min(30, len(taus6)))
            C = run_fits(pool, N, [r[0] for r in B[:5]], sub, HEAVY, 22, chunk=1)
            log(g, "stage C best", [(r[0], round(r[1], 1)) for r in C])
            best = {r[0]: (r, taus_A, "A") for r in A}
            for r in B:
                if r[1] < best[r[0]][0][1]:
                    best[r[0]] = (r, K, "B")
            for r in C:
                if r[1] < best[r[0]][0][1]:
                    best[r[0]] = (r, sub, "C")
            ranked = sorted(best.values(), key=lambda x: x[0][1])
            bits = int(np.ceil(np.log2(ncat)))
            for rank, (r, ts, stage) in enumerate(ranked):
                i, c, j, P = r[0], r[1], r[2], r[3]
                a = md.quark_catalog(N)[i]
                tau = complex(ts[j])
                row = {"group": g, "rank": rank, "idx": i, "label": a.label(), "chi2": c, "N": N_OBS,
                       "k_cont": a.k_cont, "bits": bits, "k": a.k_cont + bits,
                       "p": p_value(c, N_OBS, a.k_cont), "tau_re": tau.real, "tau_im": tau.imag,
                       "in_2.30": bool(np.min(np.abs(taus2 - tau)) < 1e-9) if len(taus2) else False,
                       "stage": stage}
                if rank < 10:
                    obs, pl = pulls_at(N, i, tau, P)
                    row.update({f"pull_{k}": v for k, v in pl.items()})
                    row["chi2_ckm_src_evaluate"] = eval_ckm_chi2(obs)
                    row["P"] = json.dumps([round(x, 6) for x in P])
                rows.append(row)
            summary[g] = {"n_assign": ncat, "n_exact_stageA": ncat, "n_stageA_tau_points": len(taus_A),
                          "n_stageB": len(top), "n_stageC": len(C), "best_chi2": ranked[0][0][1],
                          "best_label": md.quark_catalog(N)[ranked[0][0][0]].label()}
            log(g, "summary", summary[g])
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
    rows = [{k: r.get(k, "") for k in keys} for r in rows]
    write_csv(os.path.join(HERE, "results_fixed_tau.csv"), rows)
    json.dump({"summary": summary, "rep_points": [[t.real, t.imag] for t in K],
               "n_region_6.18": len(taus6), "n_region_2.30": len(taus2)},
              open(os.path.join(HERE, "real_summary.json"), "w"), indent=1)
    log("real done")


def cmd_null(n_rand=int(os.environ.get("FT_NRAND", "100"))):
    """Reduced pipeline (single tau point, NULL_CHEAP on all 968 S4 + NULL_TOP_B heavy) at the lepton
    best tau and at n_rand random tau, uniform in area over the fundamental domain (Im tau <= 4),
    at least 0.05 from the lepton region and its gCP image."""
    import multiprocessing as mp
    taus6, d6, rec = lepton_region("19", 6.18)
    t0 = complex(*rec["tau_best_grid"])
    excl = np.concatenate([taus6, -np.conj(taus6)])
    rng = np.random.default_rng(2020)
    cand = area_uniform_tau(rng, 4 * n_rand)
    cand = np.array([t for t in cand if np.min(np.abs(excl - t)) > 0.05])[:n_rand]
    ncat = len(md.quark_catalog(4))
    out = []
    path = os.path.join(HERE, "null_random_tau.csv")
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for k, t in enumerate([t0] + list(cand)):
            A = run_fits(pool, 4, list(range(ncat)), [t], NULL_CHEAP, 30 + k)
            B = run_fits(pool, 4, [r[0] for r in A[:NULL_TOP_B]], [t], HEAVY, 31 + k, chunk=1)
            best = min(A[0][1], B[0][1])
            bi = B[0][0] if B[0][1] <= A[0][1] else A[0][0]
            out.append({"kind": "lepton" if k == 0 else "random", "j": k, "tau_re": t.real, "tau_im": t.imag,
                        "best_chi2": best, "best_idx": bi, "label": md.quark_catalog(4)[bi].label(),
                        "median_chi2_stageA": float(np.median([r[1] for r in A]))})
            log("null", k, t, "best", round(best, 1), out[-1]["label"])
            write_csv(path, out)
    log("null done")


if __name__ == "__main__":
    {"real": cmd_real, "null": cmd_null}[sys.argv[1]]()

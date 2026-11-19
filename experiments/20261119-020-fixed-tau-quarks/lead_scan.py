"""Exp 016 lead: tau-only neutrino sector. Type-I seesaw with N^c in the same triplet as L,
constant Dirac matrix (weight-0 invariant L N^c), M_N from the single weight-2 invariant
N^c N^c Y^(2). m_nu = M_D^T M_N^-1 M_D has no free parameter besides tau (and a scale).
Scanned with every charged-lepton sector of the catalog (e couplings fitted to m_e/m_mu,
m_mu/m_tau), both orderings, exact fits (search.fit_exact). Trials = e-sets x L reps per group.

    python lead_scan.py
"""
from __future__ import annotations

import functools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, "..", ".."))]

import numpy as np  # noqa: E402

import models as md  # noqa: E402
import modforms as mf  # noqa: E402
import search as se  # noqa: E402


@functools.lru_cache(maxsize=None)
def dirac0(N, rep):
    """Weight-0 invariant C with L^T C N^c invariant: rho^T C rho = C for S, T."""
    S, T = mf.IRREPS[N][rep]
    A = np.concatenate([np.kron(G.T, G.T) - np.eye(9) for G in (S, T)])
    ns = mf._nullspace(A)
    assert ns.shape[0] == 1, ns.shape
    C = ns[0].reshape(3, 3)
    C = C / C.flat[np.argmax(np.abs(C))]
    return C


_orig_nu = md.nu_matrix


def nu_matrix(N, lrep, nu, tau, g):
    if nu[0] != "SS":
        return _orig_nu(N, lrep, nu, tau, g)
    W = mf.weinberg_structures(N, 2, lrep)
    assert W.shape[0] == 1
    MN = mf.eval_series(W, tau, N)[:, 0]
    C = dirac0(N, lrep)
    return np.einsum("ji,bjk,kl->bil", C, np.linalg.inv(MN), C)


md.nu_matrix = nu_matrix


def _task(args):
    N, lrep, ie, o = args
    from src.nulls import Universe
    u = Universe.real(o)
    T = se.targets(u, o)
    e = md.sector_sets(N, lrep)[ie]
    a = md.LeptonAssign(N, lrep, e, ("SS", 1))
    sc = u.scorer(se.USE8[4:], o)
    c, z = se.fit_exact(a, T, sc, o, np.random.default_rng([N, ie, len(o)]), n_samp=6000, n_starts=3)
    ob = se.sector_obs(a, z, o)
    return {"group": mf.GROUP[N], "ordering": o, "label": a.label().replace("W('SS', 1)", "SS"),
            "k_cont": a.k_cont, "chi2_lepton": c, "z": z.tolist(),
            **{k: float(v[0]) for k, v in ob.items()}}


if __name__ == "__main__":
    import multiprocessing as mp
    tasks = [(N, l, i, o) for N in mf.LEVELS for l in mf.TRIPLETS[N] for i in range(len(md.sector_sets(N, l)))
             for o in se.ORDERINGS]
    print("lead tasks", len(tasks), flush=True)
    out = []
    with mp.get_context("spawn").Pool(int(os.environ.get("MODSEARCH_WORKERS", "5"))) as pool:
        for r in pool.imap_unordered(_task, tasks, chunksize=2):
            out.append(r)
    out.sort(key=lambda r: r["chi2_lepton"])
    json.dump(out, open(os.path.join(HERE, "lead_tauonly_nu.json"), "w"), indent=0)
    for r in out[:10]:
        print(f"{r['group']} {r['ordering']} chi2_l {r['chi2_lepton']:.2f} k_cont {r['k_cont']} "
              f"s12sq {r['s12sq']:.3f} s13sq {r['s13sq']:.4f} s23sq {r['s23sq']:.3f} d {r['dcp_deg']:.0f} "
              f"r {r['r_dm']:.4f} | {r['label']}", flush=True)

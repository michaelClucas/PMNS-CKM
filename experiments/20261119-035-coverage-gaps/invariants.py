"""Exp 035 (a): rephasing-invariant CKM <-> PMNS relations (moduli and Jarlskog).

Pre-declared grammar (fixed before any scoring; see README):
  G1 moduli    |U_ij|^2 = c * |V_kl|^p
               (ij) in all 9 PMNS elements, (kl) in the 6 off-diagonal CKM elements,
               p in {1/2, 1, 2}, c in C = {1, 2, 3, 1/2, 1/3, 2/3, 3/2}      9*6*3*7 = 1134
  G2 Jarlskog  J_l = s * c * J_q^p / lambda^n
               s in {+1, -1}, p in {1, 1/2, 1/3}, n in {0..6}, c in C      2*3*7*7 = 294
  total 1428 relations -> ceil(log2 1428) = 11 bits.

Each relation fixes one PMNS combination. Scoring: the relation is imposed by eliminating one
PMNS parameter; the other three PMNS parameters and the CKM quantities used are fitted
(the CKM quantities as nuisances, scored as observables), so N = 4 + n_ckm, k_cont = 3 + n_ckm,
dof = 1. Relations whose Q depends on delta through cos(delta) (|U_mu1|^2 ...) or sin(delta)
(J_l) fix delta up to a two-fold branch: both branches are fitted, the best kept, +1 bit.

Stage 1 (screen): the NuFIT joint chi2 used by src.evaluate is separable,
  chi2 = T12(s12sq) + T13(s13sq) + T23DCP(s23sq, delta)   (src.data.pmns_delta_chi2, method auto),
so it is tabulated on a 4D grid (checked against src.data at random points), Q is computed on
the grid, and the profile chi2(Q) is the binned minimum. A relation's screen chi2 adds the
CKM-side Gaussian. Screen keeps chi2_screen < 9.
Stage 2 (official): survivors are refitted with src.evaluate.fit_and_evaluate (lbfgsb), explicit
use=, alias rules enforced by the Scorer. Kept: max |pull| < 2.
Usage: python invariants.py  (from the repo root or this folder; ~4 processes)
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src.mixing import standard_matrix  # noqa: E402
from src.stats import discrete_bits  # noqa: E402

C_SET = [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (2, 3), (3, 2)]  # (num, den)
CKM_OFF = ["Vus", "Vub", "Vcd", "Vcb", "Vtd", "Vts"]
PMNS_EL = [(i, j) for i in range(3) for j in range(3)]
EL_NAME = {0: "e", 1: "mu", 2: "tau"}
P_MOD = [0.5, 1.0, 2.0]
P_J = [1.0, 0.5, 1.0 / 3.0]
N_J = list(range(7))
EL_ANGLE = {(0, 0): "s12", (0, 1): "s12", (0, 2): "s13", (1, 2): "s23", (2, 2): "s23"}  # delta-free


def grammar():
    rel = []
    for (i, j), v, p, c in itertools.product(PMNS_EL, CKM_OFF, P_MOD, C_SET):
        rel.append(dict(kind="mod", el=(i, j), v=v, p=p, c=c))
    for s, p, n, c in itertools.product((1, -1), P_J, N_J, C_SET):
        rel.append(dict(kind="J", sign=s, p=p, n=n, c=c))
    return rel


def rel_name(r):
    c = f"{r['c'][0]}/{r['c'][1]}" if r["c"][1] != 1 else f"{r['c'][0]}"
    if r["kind"] == "mod":
        i, j = r["el"]
        return f"|U_{EL_NAME[i]}{j + 1}|^2 = {c} |{r['v']}|^{r['p']:g}"
    sg = "+" if r["sign"] > 0 else "-"
    return f"J_l = {sg}{c} J_q^{r['p']:.3g} / lambda^{r['n']}"


def ckm_val(name):
    rec = D.load_ckm()[name]
    return rec["value"], 0.5 * (rec["err_plus"] + rec["err_minus"])


def target(r, ckm=None):
    """(T, sigma_T) of the relation's right-hand side from the CKM loader (linear propagation)."""
    cf = r["c"][0] / r["c"][1]
    if r["kind"] == "mod":
        x, sx = ckm_val(r["v"]) if ckm is None else ckm[r["v"]]
        T = cf * x ** r["p"]
        return T, abs(T * r["p"] * sx / x)
    J, sJ = ckm_val("J") if ckm is None else ckm["J"]
    lam, sl = ckm_val("lambda") if ckm is None else ckm["lambda"]
    T = r["sign"] * cf * J ** r["p"] / lam ** r["n"]
    return T, abs(T) * np.hypot(r["p"] * sJ / J, r["n"] * sl / lam)


def lepton_q(s12sq, s13sq, s23sq, dcp_deg):
    """dict: 9 |U_ij|^2 and J_l (broadcast)."""
    U = standard_matrix(np.arcsin(np.sqrt(s12sq)), np.arcsin(np.sqrt(s13sq)), np.arcsin(np.sqrt(s23sq)),
                        np.radians(dcp_deg))
    out = {(i, j): np.abs(U[..., i, j]) ** 2 for i, j in PMNS_EL}
    out["J"] = np.imag(U[..., 0, 0] * U[..., 1, 1] * np.conj(U[..., 0, 1]) * np.conj(U[..., 1, 0]))
    return out


# ------------------------------------------------------------------ stage 1: grid screen

G12 = np.linspace(0.255, 0.365, 56)
G13 = np.linspace(0.0190, 0.0262, 37)
G23 = np.linspace(0.38, 0.66, 71)
GD = np.arange(0.0, 360.0, 3.0)


def chi2_grid(ordering):
    f = lambda **kw: D.pmns_delta_chi2(ordering, **kw)  # noqa: E731
    t12 = np.array([f(s12sq=x) for x in G12])
    t13 = np.array([f(s13sq=x) for x in G13])
    t23d = np.array([[f(s23sq=a, dcp_deg=b) for b in GD] for a in G23])
    return t12, t13, t23d


def check_separable(ordering, grids, rng, n=50):
    t12, t13, t23d = grids
    worst = 0.0
    for _ in range(n):
        a, b, c, d = rng.integers(len(G12)), rng.integers(len(G13)), rng.integers(len(G23)), rng.integers(len(GD))
        ref = D.pmns_delta_chi2(ordering, s12sq=G12[a], s13sq=G13[b], s23sq=G23[c], dcp_deg=GD[d])
        if np.isfinite(ref):
            worst = max(worst, abs(ref - (t12[a] + t13[b] + t23d[c, d])))
    return worst


def profiles(ordering, grids, nbins=400):
    """profile chi2 of each lepton quantity: dict key -> (bin centres, min chi2)."""
    t12, t13, t23d = grids
    S12, S13, S23, DD = np.meshgrid(G12, G13, G23, GD, indexing="ij")
    chi = t12[:, None, None, None] + t13[None, :, None, None] + t23d[None, None, :, :]
    ok = np.isfinite(chi) & (chi < 30)
    q = lepton_q(S12[ok], S13[ok], S23[ok], DD[ok])
    c = chi[ok]
    prof = {}
    for key, v in q.items():
        lo, hi = v.min(), v.max()
        edges = np.linspace(lo, hi, nbins + 1)
        idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
        m = np.full(nbins, np.inf)
        np.minimum.at(m, idx, c)
        cen = 0.5 * (edges[1:] + edges[:-1])
        prof[key] = (cen[np.isfinite(m)], m[np.isfinite(m)])
    return prof


def screen_chi2(prof, r):
    T, sT = target(r)
    key = r["el"] if r["kind"] == "mod" else "J"
    cen, m = prof[key]
    return float(np.min(m + ((cen - T) / max(sT, 1e-12)) ** 2)), T, sT


# ------------------------------------------------------------------ stage 2: official refit

def ckm_keys(r):
    return ["ckm_" + r["v"]] if r["kind"] == "mod" else (["ckm_J"] + (["ckm_lambda"] if r["n"] else []))


def make_model(r, branch):
    """model(theta) -> pred, eliminating one PMNS parameter; None if the relation has no solution."""
    cf = r["c"][0] / r["c"][1]

    def rhs(th):
        if r["kind"] == "mod":
            return cf * th["ckm_" + r["v"]] ** r["p"]
        lam = th.get("ckm_lambda", 1.0)
        return r["sign"] * cf * th["ckm_J"] ** r["p"] / lam ** r["n"]

    def model(th):
        T = rhs(th)
        p = {k: th[k] for k in th}
        s12, s13, s23 = th.get("s12sq"), th.get("s13sq"), th.get("s23sq")
        if r["kind"] == "mod" and r["el"] in EL_ANGLE:
            el = r["el"]
            if el == (0, 2):
                s13 = T
            elif el in ((0, 1), (0, 0)):
                x = T / (1 - s13)
                s12 = x if el == (0, 1) else 1 - x
            else:
                x = T / (1 - s13)
                s23 = x if el == (1, 2) else 1 - x
            d = th["dcp_deg"]
        else:
            key = r["el"] if r["kind"] == "mod" else "J"
            if key == "J":
                jm = lepton_q(s12, s13, s23, 90.0)["J"]
                sd = T / jm
                if abs(sd) > 1:
                    raise ValueError("no solution")
                d = np.degrees(np.arcsin(sd)) if branch == 0 else 180.0 - np.degrees(np.arcsin(sd))
            else:
                f0 = lepton_q(s12, s13, s23, 0.0)[key]
                f1 = lepton_q(s12, s13, s23, 180.0)[key]
                a, b = 0.5 * (f0 + f1), 0.5 * (f0 - f1)
                cd = (T - a) / b
                if abs(cd) > 1:
                    raise ValueError("no solution")
                d = np.degrees(np.arccos(cd)) * (1 if branch == 0 else -1)
        for nm, val in (("s12sq", s12), ("s13sq", s13), ("s23sq", s23)):
            if not (0 < val < 1):
                raise ValueError("unphysical")
            p[nm] = float(val)
        p["dcp_deg"] = float(np.mod(d, 360.0))
        return p

    return model


def free_pmns(r):
    if r["kind"] == "mod" and r["el"] in EL_ANGLE:
        drop = {"s12": "s12sq", "s13": "s13sq", "s23": "s23sq"}[EL_ANGLE[r["el"]]]
        return [k for k in ("s12sq", "s13sq", "s23sq", "dcp_deg") if k != drop], 1
    return ["s12sq", "s13sq", "s23sq"], 2


PB = {"s12sq": (0.24, 0.38), "s13sq": (0.017, 0.028), "s23sq": (0.36, 0.68), "dcp_deg": (0.0, 360.0)}


def refit(args):
    idx, r, ordering, n_bits = args
    free, nbr = free_pmns(r)
    ck = ckm_keys(r)
    bounds = {k: PB[k] for k in free}
    recs = D.load_ckm()
    for k in ck:
        rec = recs[k[4:]]
        s = 0.5 * (rec["err_plus"] + rec["err_minus"])
        bounds[k] = (rec["value"] - 6 * s, rec["value"] + 6 * s)
    use = list(E.PMNS_KEYS) + ck
    k_disc = n_bits + (1 if nbr == 2 else 0)
    best = None
    for br in range(nbr):
        try:
            fr = E.fit_and_evaluate(make_model(r, br), bounds, ordering, use, k_discrete=k_disc,
                                    periodic=[k for k in free if k == "dcp_deg"], n_starts=6, seed=idx,
                                    method="lbfgsb")
        except Exception as e:  # noqa: BLE001
            continue
        if best is None or fr.result.chi2 < best.result.chi2:
            best = fr
    if best is None:
        return dict(idx=idx, ordering=ordering, name=rel_name(r), status="no fit")
    res = best.result
    return dict(idx=idx, ordering=ordering, name=rel_name(r), chi2=res.chi2, chi2_global=res.chi2_global,
                N=res.N, k=res.k, k_cont=res.k_continuous, k_discrete=res.k_discrete, dof=res.dof,
                p=res.p, p_global=res.p_global, max_pull=res.max_abs_pull,
                pulls={k: float(v) for k, v in res.pulls.items()}, params=best.params,
                compression_ok=bool(res.compression_ok), fit_ok=bool(res.fit_ok), use=use)


def main():
    t0 = time.time()
    rels = grammar()
    M = len(rels)
    bits = discrete_bits(M)
    print(f"grammar: {M} relations (G1 {sum(r['kind'] == 'mod' for r in rels)}, "
          f"G2 {sum(r['kind'] == 'J' for r in rels)}), {bits} bits", flush=True)
    rng = np.random.default_rng(35)
    out = {"M": M, "bits": bits, "screen": {}, "refit": {}}
    jobs = []
    for o in D.ORDERINGS:
        grids = chi2_grid(o)
        print(f"[{o}] separability check, max |grid - src.data| = {check_separable(o, grids, rng):.2e}", flush=True)
        prof = profiles(o, grids)
        sc = []
        for i, r in enumerate(rels):
            c, T, sT = screen_chi2(prof, r)
            sc.append((c, i, T, sT))
        sc.sort()
        out["screen"][o] = [dict(idx=i, name=rel_name(rels[i]), chi2_screen=c, T=T, sigma_T=sT) for c, i, T, sT in sc]
        surv = [i for c, i, _, _ in sc if c < 9]
        print(f"[{o}] screen: {len(surv)} of {M} with chi2_screen < 9; best 5:", flush=True)
        for c, i, T, sT in sc[:5]:
            print(f"    {c:7.3f}  {rel_name(rels[i])}   (T = {T:.4g} +- {sT:.2g})", flush=True)
        jobs += [(i, rels[i], o, bits) for i in surv]
    # sanity: the known theta13 = theta_C / sqrt 2 relation, always refitted
    known = next(i for i, r in enumerate(rels) if r["kind"] == "mod" and r["el"] == (0, 2) and r["v"] == "Vus"
                 and r["p"] == 2.0 and r["c"] == (1, 2))
    for o in D.ORDERINGS:
        if not any(j[0] == known and j[2] == o for j in jobs):
            jobs.append((known, rels[known], o, bits))
    print(f"official refits: {len(jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "4"))) as ex:
        res = list(ex.map(refit, jobs, chunksize=4))
    for o in D.ORDERINGS:
        rows = sorted([x for x in res if x["ordering"] == o and "chi2" in x], key=lambda x: x["chi2_global"])
        out["refit"][o] = rows
        kept = [x for x in rows if x["max_pull"] < 2]
        print(f"[{o}] refitted {len(rows)}; all pulls < 2 sigma: {len(kept)}; fit_ok (p_global > 0.05 too): "
              f"{sum(x['fit_ok'] for x in rows)}; compression_ok: {sum(x['compression_ok'] for x in rows)}", flush=True)
        for x in rows[:15]:
            print(f"    chi2 {x['chi2']:.3f} chi2_g {x['chi2_global']:.3f} p_g {x['p_global']:.3f} "
                  f"maxpull {x['max_pull']:.2f} N {x['N']} k {x['k']}  {x['name']}", flush=True)
        kn = [x for x in rows if x["idx"] == known]
        if kn:
            x = kn[0]
            print(f"[{o}] sanity theta13 = theta_C/sqrt2 ({x['name']}): chi2 {x['chi2']:.2f}, "
                  f"pull s13sq {x['pulls'].get('s13sq', float('nan')):+.2f}", flush=True)
    out["known_idx"] = known
    out["elapsed_s"] = time.time() - t0
    (HERE / "invariants.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

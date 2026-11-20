"""Exp 044 step 5: lepton extension of quark survivors (grammar.md "Lepton extension").

Y_e = C o Y_d with Georgi-Jarlskog C (C_22 = -3, else 1); Y_nu = Y_u; type-I seesaw with
(i) M_R = 1 or (ii) M_R = diag(1, x, y). Full fit N 15 (Y2 of exp 043) and held-out PMNS test
(fit Q8 + mr_e_mu + mr_mu_tau, predict s12sq, s13sq, s23sq, dcp_deg, dm_ratio).
  python lepton.py <survivors.json index list, comma separated>
Output: lepton.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import least_squares

from main import ART, HERE, L, log
from src import evaluate as ev
from src.yukawa import observables_batch

Q8 = L.QUARK_KEYS
CL = ("mr_e_mu", "mr_mu_tau")
PM = ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")
N15 = Q8 + CL + PM
GJ = np.ones((3, 3))
GJ[1, 1] = -3.0


def mats(spec, z, mr):
    k = L.k_cont(spec)
    Yu, Yd = L.build(spec, z[:, :k])
    Ye = GJ[None] * Yd
    B = len(z)
    if mr == "I1":
        MRi = np.tile(np.eye(3), (B, 1, 1)).astype(complex)
    else:
        MRi = np.zeros((B, 3, 3), complex)
        MRi[:, 0, 0] = 1.0
        MRi[:, 1, 1] = 10.0 ** -z[:, k]
        MRi[:, 2, 2] = 10.0 ** -z[:, k + 1]
    Yn = Yu
    m = -np.conj(Yn) @ MRi @ np.conj(np.swapaxes(Yn, 1, 2))
    m = 0.5 * (m + np.swapaxes(m, 1, 2))
    return Yu, Yd, Ye, m


def obs(spec, z, mr):
    with np.errstate(all="ignore"):
        o = observables_batch(*mats(spec, z, mr), ordering="NO", sym_tol=np.inf)
    return o


def make_resid(sc):
    def resid(o):
        R = []
        for i in range(len(o["ckm_s12"])):
            p = {k: float(o[k][i]) for k in sc.use}
            try:
                c = sc.chi2_soft(p, include_finite=True)
                r = sc(p)
                v = np.array([r.pulls.get(k, 0.0) for k in sc.use])
                if not np.isfinite(c) or c >= ev.CHI2_CAP:
                    v = np.full(len(sc.use), np.sqrt(c / len(sc.use)))
            except Exception:
                v = np.full(len(sc.use), 30.0)
            R.append(np.where(np.isfinite(v), v, 30.0))
        return np.array(R)
    return resid


def fit(spec, x0, mr, keys, scale, rng, n_start=6):
    sc = ev.Scorer(keys, "NO", mass_floor=0.02, scale=L.SCALES[scale])
    resid = make_resid(sc)
    nl = 0 if mr == "I1" else 2
    lo, hi = L._box(spec)
    lo = np.concatenate([lo, [-6.0] * nl])
    hi = np.concatenate([hi, [6.0] * nl])
    best = None
    for s in range(n_start):
        z0 = np.concatenate([np.asarray(x0) + (0 if s == 0 else rng.normal(0, 0.05, len(x0))),
                             rng.uniform(-4, 4, nl)])
        z0 = np.clip(z0, lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(lambda z: resid(obs(spec, z[None], mr))[0], z0, bounds=(lo, hi), method="trf",
                              diff_step=1e-6, max_nfev=300)
            z = r.x
        except Exception:
            z = z0
        o = obs(spec, z[None], mr)
        p = {k: float(o[k][0]) for k in keys}
        try:
            res = sc(p, k=len(z))
            c = res.chi2
        except Exception:
            res, c = None, np.inf
        if best is None or c < best[0]:
            best = (c, z, res, p)
    return best


def main(idxs):
    surv = json.load(open(HERE / "survivors.json"))
    rng = np.random.default_rng(4406)
    out = []
    for i in idxs:
        s = surv[i]
        for mr in ("I1", "MRdiag"):
            c, z, res, p = fit(s["spec"], s["x"], mr, N15, s["scale"], rng)
            o = obs(s["spec"], z[None], mr)
            row = {"id": s["id"], "scale": s["scale"], "mr": mr, "k_cont": len(z), "N": 15, "chi2": float(c),
                   "max_pull": None if res is None else float(res.max_abs_pull),
                   "p": None if res is None else float(res.p_global),
                   "pulls": None if res is None else {k: float(v) for k, v in res.pulls.items()},
                   "pred": {k: float(o[k][0]) for k in N15}}
            # held-out: fit Q8 + charged leptons, predict PMNS side
            c10, z10, r10, _ = fit(s["spec"], s["x"], mr, Q8 + CL, s["scale"], rng)
            o10 = obs(s["spec"], z10[None], mr)
            scp = ev.Scorer(PM, "NO", scale=L.SCALES[s["scale"]])
            try:
                rp = scp({k: float(o10[k][0]) for k in PM}, k=0)
                ho = {"chi2_fit10": float(c10), "pmns_chi2": float(rp.chi2), "pmns_p": float(rp.p),
                      "pulls": {k: float(v) for k, v in rp.pulls.items()}}
            except Exception as e:
                ho = {"chi2_fit10": float(c10), "error": str(e)[:200],
                      "pred": {k: float(o10[k][0]) for k in PM}}
            row["heldout"] = ho
            out.append(row)
            log(f"lepton {s['id']} {s['scale']} {mr}: N15 chi2 {c:.4g} maxpull {row['max_pull']}; "
                f"held-out PMNS {ho.get('pmns_chi2', ho.get('error'))}")
    json.dump(out, open(ART / f"lepton_{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    TAG = sys.argv[2]
    main([int(a) for a in sys.argv[1].split(",")])

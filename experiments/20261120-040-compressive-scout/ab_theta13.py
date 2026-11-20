"""Filter (c) for Albright-Barr (hep-ph/0502161 Eqs. 2-5): reachable sin^2 theta13 over the M_R
parameters a, b, c (complex, |.| log-uniform in [0.05, 20], phases uniform), with the charged sector
at the paper's benchmark (Eq. 3), with and without delta_N, delta'_N (|.| log-uniform in [1e-6, 1e-2]).
Keep samples with s12sq in [0.27, 0.35], s23sq in [0.40, 0.62], dm_ratio in [0.02, 0.045] (wide
NuFIT-like windows) and report the s13sq distribution of the kept samples.

    python ab_theta13.py [n] -> ab_theta13.json, stdout (log)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import yukawa as yk  # noqa: E402
import lib40 as L  # noqa: E402

HERE = Path(__file__).resolve().parent


def sample(n, rng, with_dN):
    p = dict(L.AB_BENCH)
    lg = lambda lo, hi, k: np.exp(rng.uniform(np.log(lo), np.log(hi), k))  # noqa: E731
    ph = lambda k: np.exp(1j * rng.uniform(0, 2 * np.pi, k))  # noqa: E731
    a, b, c = (lg(0.05, 20, n) * ph(n) for _ in range(3))
    if with_dN:
        dN, dNp = lg(1e-6, 1e-2, n) * ph(n), lg(1e-6, 1e-2, n) * ph(n)
    else:
        dN = dNp = np.zeros(n)
    Yu = np.broadcast_to(L.ab_U(**p), (n, 3, 3))
    Yd = np.broadcast_to(L.ab_D(**p), (n, 3, 3))
    Ye = np.broadcast_to(L.ab_L(**p), (n, 3, 3))
    mnu = np.empty((n, 3, 3), complex)
    for i in range(n):
        N = L.ab_N(p["eps"], p["eta"], dN[i], dNp[i])
        MR = L.ab_MR(p["eps"], p["eta"], a[i], b[i], c[i])
        mnu[i] = (-N @ np.linalg.solve(MR, N.T)).conj()  # paper Eq. (5), src convention (lib40.ab_obs)
    return yk.observables_batch(Yu, Yd, Ye, mnu, ordering="NO")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    rng = np.random.default_rng(40)
    out = {}
    for with_dN in (False, True):
        o = sample(n, rng, with_dN)
        s13 = o["s13sq"]
        keep = ((o["s12sq"] > 0.27) & (o["s12sq"] < 0.35) & (o["s23sq"] > 0.40) & (o["s23sq"] < 0.62)
                & (o["dm_ratio"] > 0.02) & (o["dm_ratio"] < 0.045) & np.isfinite(s13))
        tag = "with_dN" if with_dN else "dN0"
        res = {"n": n, "kept": int(keep.sum()), "s13sq_all_max": float(np.nanmax(s13)),
               "s13sq_all_q99": float(np.nanquantile(s13, 0.99))}
        if keep.any():
            res.update(s13sq_kept_max=float(s13[keep].max()),
                       s13sq_kept_q=[float(np.quantile(s13[keep], q)) for q in (0.05, 0.5, 0.95)])
        res["frac_all_s13sq_gt_0.02"] = float(np.mean(s13 > 0.02))
        out[tag] = res
        print(tag, json.dumps(res))
    (HERE / "ab_theta13.json").write_text(json.dumps(out, indent=1))

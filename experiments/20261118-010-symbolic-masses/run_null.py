"""Look-elsewhere nulls through the identical brute-force pipeline (all distinct expressions, C <= 5).

  --kind a: real inputs, fake PMNS targets = Haar angles (src.haar.sample_haar_angles, fake i uses
            default_rng([1001, i])), scored with Gaussian NuFIT 1-sigma widths of each ordering
            centred on the fake value; prediction error in quadrature. The real comparison value
            is the same Gaussian score centred on the NuFIT central values (row universe = -1).
  --kind b: real PMNS targets with the NuFIT tables (ProfileScorer, same as run_real.py), fake
            inputs: Wolfenstein lambda~U[.10,.35], A~U[.5,1.2], rhobar, etabar~U[-.5,.5]; mass
            ratios log-uniform over one decade centred on the real value at that scale; fake i uses
            default_rng([2002, i]) (MZ) / [2003, i] (GUT). Input errors: same relative errors as
            the real inputs (absolute for the CKM phase).
Output: null_<kind>_<scale>.csv, one row per (universe, target, ordering): cumulative best
|pull| and number of |pull| < 1 expressions at complexity <= 1..6.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

CMAX = 5  # nulls run the full grammar to C<=5 (C=6 too slow for 300 fakes on the shared machine)
_E = None


def _init():
    global _E
    import symlib as S
    _E = S.Loaded(S.ART / "enum_C6.npz", cmax=CMAX)


def _row(u, tgt, o, best, n1):
    return [u, tgt, o] + [float(x) for x in best] + [int(x) for x in n1]


def job_b(args):
    i, scale = args
    import pipeline as P
    import symlib as S
    real = S.real_inputs(scale)
    rng = np.random.default_rng([2002 if scale == "MZ" else 2003, i])
    base = S.fake_base_inputs(rng, real["base"][4:], 1)[0]
    dirs = S.fake_dirs(real, base)
    v, T = _E.evaluate(base, dirs)
    res = P.run_profile(v, T, _E.comp)
    return [_row(i, t, o, r[0], r[1]) for (t, o), r in res.items()]


def run_a(scale, n, out):
    import pipeline as P
    import symlib as S
    from src.haar import sample_haar_angles
    from src import data
    _init()
    real = S.real_inputs(scale)
    v, T = _E.evaluate(real["base"], real["dirs"])
    centers = [{(t, o): data.load_pmns(o)[t]["value"] for t in S.TARGETS for o in P.ORDERINGS}]
    for i in range(n):
        t12, t13, t23, d = sample_haar_angles(np.random.default_rng([1001, i]))
        fake = {"s12sq": np.sin(t12) ** 2, "s13sq": np.sin(t13) ** 2, "s23sq": np.sin(t23) ** 2,
                "dcp_deg": np.degrees(d)}
        centers.append({(t, o): float(fake[t]) for t in S.TARGETS for o in P.ORDERINGS})
    res = P.run_gauss_many(v, T, _E.comp, centers)
    rows = []
    for u, r in enumerate(res):
        rows += [_row(u - 1, t, o, x[0], x[1]) for (t, o), x in r.items()]
    write(out, rows)


def write(out, rows):
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["universe", "target", "ordering"] + [f"best_C{c}" for c in range(1, CMAX + 1)]
                   + [f"n1_C{c}" for c in range(1, CMAX + 1)])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["a", "b"], required=True)
    ap.add_argument("--scale", choices=["MZ", "GUT"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, f"null_{a.kind}_{a.scale}.csv")
    t0 = time.time()
    if a.kind == "a":
        run_a(a.scale, a.n, out)
    else:
        rows = []
        with Pool(a.workers, initializer=_init) as pool:
            for k, r in enumerate(pool.imap_unordered(job_b, [(i, a.scale) for i in range(a.n)])):
                rows += r
                if k % 25 == 0:
                    print(f"[null b {a.scale}] {k}/{a.n} {time.time() - t0:.0f}s", flush=True)
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
        write(out, rows)
    print(f"[null {a.kind} {a.scale}] done n={a.n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

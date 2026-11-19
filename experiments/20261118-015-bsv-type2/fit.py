"""Step 3a: multi-start GUT-level least-squares fits of the BSV type-II model.

usage: python fit.py <variant>[,<variant>...] <out.jsonl> [n_starts]
Free parameters per fit: model params (VARIANTS) + 10 low-scale nuisances (quark masses, CKM) in pull space.
Two stages: (1) multistart on model params with nuisances at data centres, (2) polish of the 2 best
with the nuisances free. Objective (fit stage): 10 nuisance pulls + 3 charged-lepton GUT residuals (theory floor SIG_TH)
+ NuFIT s12, s13 (1D), (s23, delta) (2D) + dm21/|dm3l| Gaussian. PMNS compared at M_GUT here;
finalize.py adds the GUT->M_Z running shift and scores with src.evaluate.
"""
import json, sys, time, zlib
import numpy as np
from scipy.optimize import least_squares
from common import gut_reference, residuals, VARIANTS, SIGNS, TANB, scenario_name

SCEN = [("SM", None, 3000.0)] + [("MSSM", t, 3000.0) for t in TANB] + [("MSSM", t, 10000.0) for t in (10, 50)]
if len(sys.argv) > 4:  # optional scenario subset, e.g. "SM,5,10" (tanb values at 3 TeV; "10k10" = tanb 10 at 10 TeV)
    want = sys.argv[4].split(",")
    SCEN = [s for s in SCEN if (s[0] == "SM" and "SM" in want) or (s[0] == "MSSM" and ((s[2] == 3000.0 and str(s[1]) in want) or (s[2] == 10000.0 and f"10k{s[1]}" in want)))]
variants = sys.argv[1].split(",")
out = sys.argv[2]
n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 10
BOX = {"x": (-8, 8), "yhat": (-3, 3), "eta": (-0.3, 0.8), "xr": (-8, 8), "xi": (-8, 8),
       "a2": (0, 2 * np.pi), "a3": (0, 2 * np.pi), "p1": (0, 2 * np.pi), "p2": (0, 2 * np.pi), "p3": (0, 2 * np.pi)}
t0 = time.time()
STAGE0 = True  # R0eta run (fits_R0eta.jsonl) was made before stage 0 was added (README)
with open(out, "a") as fh:
    for model, tb, ms in SCEN:
        ref = gut_reference(model, tb, ms)
        for v in variants:
            names, kd = VARIANTS[v]
            lo = np.array([BOX[n][0] for n in names]); hi = np.array([BOX[n][1] for n in names])
            for ordering in ("NO", "IO"):
                signs = SIGNS if v != "C" else [(0.0, 0.0)]
                ns = n_starts if v != "C" else 3 * n_starts
                for sg in signs:
                    rng = np.random.default_rng(zlib.crc32(repr((model, tb, ms, v, ordering, sg)).encode()))
                    n = len(names)
                    # stage 1: model params only, nuisances at data centres (fast multistart)
                    st1 = []
                    for p0 in lo + rng.random((ns, n)) * (hi - lo):
                        if STAGE0:  # stage 0: charged-lepton GUT relations only, to land in the ql-viable region
                            f0 = lambda q: residuals(ref, v, q, np.zeros(10), ordering, sg)[10:13]
                            p0 = least_squares(f0, p0, method="trf", max_nfev=100 * n).x
                        f1 = lambda q: residuals(ref, v, q, np.zeros(10), ordering, sg)
                        s1 = least_squares(f1, p0, method="trf", max_nfev=200 * n)
                        st1.append((2 * s1.cost, s1.x))
                    st1.sort(key=lambda t: t[0])
                    finals = [c for c, _ in st1]
                    # stage 2: polish the 2 best with the 10 nuisances free
                    best = None
                    for _, x1 in st1[:2]:
                        f = lambda q: residuals(ref, v, q[:n], q[n:], ordering, sg)
                        s = least_squares(f, np.r_[x1, np.zeros(10)], method="trf", max_nfev=1500)
                        if best is None or s.cost < best.cost:
                            best = s
                    rec = {"scenario": scenario_name(model, tb, ms), "model": model, "tanb": tb, "msusy": ms,
                           "variant": v, "ordering": ordering, "sign": list(sg), "chi2_gut": 2 * best.cost,
                           "p": best.x[:len(names)].tolist(), "z": best.x[len(names):].tolist(),
                           "n_starts": ns, "chi2_stage1_best": finals[0], "n_stage1_within_1": int(sum(abs(c - finals[0]) < 1 for c in finals))}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"{rec['scenario']:24s} {v:6s} {ordering} sign={np.round(sg,2).tolist()} chi2_gut={rec['chi2_gut']:.2f} "
                          f"stage1={finals[0]:.1f} conv1={rec['n_stage1_within_1']}/{ns} ({time.time()-t0:.0f}s)", flush=True)

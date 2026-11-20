"""Held-out test: fit quark + charged-lepton observables (CKM 4 + 7 mass ratios), freeze those
parameters, then fit only the neutrino-only parameters to PMNS 4 + dm_ratio.

    python heldout.py BPR <seed> [tan_beta]   neutrino-only params: x, y, z of M_R (3) -> N 5, dof 2
    python heldout.py AB  <seed> [tan_beta]   neutrino-only params: a, b, c complex (6) -> N 5 (dof < 0:
                                              reported as a reachability check only)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import evaluate as ev  # noqa: E402
import fit_bpr as FB  # noqa: E402
import fit_more as FM  # noqa: E402

HERE = Path(__file__).resolve().parent
QCL = FB.CKM + FB.MR
PM = FB.PMNS

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    if name == "BPR":
        fn, B, per, kd, tb, nu = FB.model_fn, FB.BOUNDS, FB.PERIODIC, FB.K_BITS, 10.0, FB.NU_KEYS
    else:
        fn, B, per, _, kd, tb = FM.CFG["AB"]
        nu = ["la", "pa", "lb", "pb", "lc", "pc"]
    tb = float(sys.argv[3]) if len(sys.argv) > 3 else tb
    sc = {"mu": 2e16, "model": "MSSM", "tan_beta": tb}
    opts = {"popsize": 20, "maxiter": 600, "tol": 1e-10}
    nu_default = {k: (B[k][0] + B[k][1]) / 2 + 0.37 * (B[k][1] - B[k][0]) / 2 for k in nu}  # nonsingular M_R
    b1 = {k: v for k, v in B.items() if k not in nu}
    f1 = ev.fit_and_evaluate(lambda th: fn({**nu_default, **th}), b1, "NO", QCL, k_discrete=kd,
                             periodic=[p for p in per if p in b1], seed=seed, mass_floor=FB.FLOOR, scale=sc,
                             method="de", optimizer_options=opts)
    fixed = dict(f1.params)
    b2 = {k: B[k] for k in nu}
    f2 = ev.fit_and_evaluate(lambda th: fn({**fixed, **th}), b2, "NO", PM, k_discrete=0,
                             periodic=[p for p in per if p in b2], seed=seed, scale=sc, method="de",
                             optimizer_options=opts)
    full = ev.evaluate(fn({**fixed, **f2.params}), "NO", use=QCL + PM, k=len(B) + kd, k_discrete=kd,
                       mass_floor=FB.FLOOR, scale=sc)
    out = {"model": name, "seed": seed, "tan_beta": tb,
           "stage1": {"chi2": f1.result.chi2, "N": f1.result.N, "k_cont": f1.result.k_continuous,
                      "p": f1.result.p, "pulls": f1.result.pulls, "params": f1.params},
           "stage2": {"chi2": f2.result.chi2, "N": f2.result.N, "k_cont": f2.result.k_continuous,
                      "pulls": f2.result.pulls, "pred": {k: f2.result.pred[k] for k in PM}, "params": f2.params},
           "combined": {"chi2": full.chi2, "N": full.N, "p_global": full.p_global, "pulls": full.pulls}}
    (HERE / f"heldout_{name}_tb{tb:g}_s{seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"stage1_chi2": out["stage1"]["chi2"], "stage2_chi2": out["stage2"]["chi2"],
                      "combined_chi2": full.chi2, "p_global": full.p_global}, default=float))
    print("stage1 pulls", {k: round(v, 2) for k, v in out["stage1"]["pulls"].items()})
    print("stage2 pulls", {k: round(v, 2) for k, v in out["stage2"]["pulls"].items()})
    print("stage2 pred", out["stage2"]["pred"])

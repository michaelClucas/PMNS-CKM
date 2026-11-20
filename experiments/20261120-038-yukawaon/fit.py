"""Step 3: fit the Koide-Nishiura model (arXiv:1510.05370) to our data with the official scorer.

    python fit.py full  <seed>    N = 13 (CKM 4, quark mass ratios 4, PMNS 4, dm_ratio), k_cont = 6
    python fit.py quark <seed>    N = 8  (CKM 4, quark mass ratios 4), k_cont = 5 (no xi_R)

Scale: the paper's matrices give ratios at mu = M_Z (Sec. 4); src.evaluate mr_* keys are M_Z
running ratios; CKM/PMNS compared without running (no `scale`), as in the paper.
Inputs (not scored): mr_e_mu, mr_mu_tau (charged-lepton masses build Phi0), read through
src.evaluate.mass_ratio_targets at every call so the leak probe can see them.
Light-quark floor: mass_floor = 0.02 on every mass key (exps 014, 022).
Parametrisation: eps_u = 1 + b_u (b_u sits near the singular point -1), angles in degrees.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import evaluate as ev  # noqa: E402
import model as M  # noqa: E402

HERE = Path(__file__).resolve().parent
CKM = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad"]
MQ = ["mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b"]
PMNS = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
INPUTS = ["mr_e_mu", "mr_mu_tau"]
import os
FLOOR = float(os.environ.get("YK_FLOOR", "0.02"))  # sensitivity runs: YK_FLOOR=0.3
TAG = "" if FLOOR == 0.02 else f"_floor{FLOOR:g}"
ORDERING = "NO"  # the model yields NO at and around the benchmark; IO checked separately in README
K_BITS = 4  # honest lineage bits (README); the paper's own count is 0
BOUNDS_Q = {"eps_u": (-0.3, 0.3), "b_d": (-20.0, 20.0), "beta_d": (-180.0, 180.0),
            "phi1": (-180.0, 180.0), "phi2": (-180.0, 180.0)}
BOUNDS_FULL = {**BOUNDS_Q, "xi_R": (-2.0e4, 2.0e4)}
PERIODIC = ["beta_d", "phi1", "phi2"]
BENCH = {"eps_u": 1 - 1.011, "b_d": -3.3522, "beta_d": 17.7, "phi1": -176.05, "phi2": -167.91, "xi_R": 2039.6}


def phi0():
    v, _, _ = ev.mass_ratio_targets()
    return M.phi0_from_ratios(v["mr_e_mu"], v["mr_mu_tau"])


def theta_full(t):
    return {"b_u": t["eps_u"] - 1.0, "b_d": t["b_d"], "beta_d": t["beta_d"], "phi1": t["phi1"],
            "phi2": t["phi2"], "xi_R": t.get("xi_R", 2039.6)}


def model_fn(t):
    return M.predict(theta_full(t), phi0())


def run(mode, seed, maxiter=600, popsize=20):
    use = CKM + MQ + (PMNS if mode == "full" else [])
    bounds = BOUNDS_FULL if mode == "full" else BOUNDS_Q
    extra = [{k: BENCH[k] for k in bounds}]
    t0 = time.time()
    fr = ev.fit_and_evaluate(model_fn, bounds, ORDERING, use, inputs=INPUTS, k_discrete=K_BITS,
                             periodic=PERIODIC, seed=seed, extra_starts=extra, mass_floor=FLOOR,
                             method="de", optimizer_options={"popsize": popsize, "maxiter": maxiter, "tol": 1e-9})
    r = fr.result
    out = {"mode": mode, "seed": seed, "mass_floor": FLOOR, "params": fr.params, "theta": theta_full(fr.params),
           "chi2": r.chi2, "chi2_global": r.chi2_global, "N": r.N, "k": r.k, "k_continuous": r.k_continuous,
           "k_discrete": r.k_discrete, "dof": r.dof, "p": r.p, "p_global": r.p_global,
           "max_abs_pull": r.max_abs_pull, "pulls": r.pulls, "terms": r.terms,
           "compression_ok": r.compression_ok, "fit_ok": r.fit_ok, "pred": {k: r.pred[k] for k in r.pred},
           "start_chi2": fr.start_chi2, "nfev": fr.nfev, "seconds": time.time() - t0}
    (HERE / f"fit_{mode}{TAG}_s{seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("mode", "seed", "chi2", "chi2_global", "N", "k", "dof", "p_global",
                                          "max_abs_pull", "compression_ok", "fit_ok", "nfev", "seconds")},
                     default=float))
    print("params", fr.params)
    print("pulls", {k: round(v, 2) for k, v in r.pulls.items()})
    print("start_chi2", fr.start_chi2)
    return out


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), *(int(x) for x in sys.argv[3:]))

"""Official-scorer fit of the Babu-Pati-Rastogi SO(10)/G224 model (hep-ph/0502152 Eqs. 1, 4), the only
model left after the cheap filters with k <= N - 2 (README).

    python fit_bpr.py full  <seed> [tan_beta]   N 16: CKM 4, 7 mass ratios, PMNS 4, dm_ratio
    python fit_bpr.py qcl   <seed> [tan_beta]   N 11: CKM 4, 7 mass ratios (held-out stage 1)

Scale: matrices at mu = 2e16 GeV, MSSM, tan_beta 10 (fixed before fitting; sensitivity runs at 5, 30),
M_SUSY default (3 TeV), no thresholds; NO declared (hierarchical M_R, no ordering bit).
Parameters (13): sigma, eta complex; eps, eta' real; eps', zeta22^d complex (modulus, phase);
M_R = [[x,0,z],[0,0,y],[z,y,1]] real, x, y, z via sinh maps (hierarchy 1 >> y >> z >> x).
zeta22^u = 0 as in the paper's fit A. Bits: 1 (model version: zeta22^u = 0 and real M_R chosen among
the published variants).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src import yukawa as yk  # noqa: E402
import lib40 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
CKM = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad"]
MR = ["mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "mr_b_tau"]
PMNS = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
FLOOR = 0.02
K_BITS = 1
BOUNDS = {"s_re": (-0.5, 0.5), "s_im": (-0.5, 0.5), "e_re": (-0.5, 0.5), "e_im": (-0.5, 0.5),
          "eps": (-0.5, 0.5), "etap": (-0.02, 0.02), "epsp": (0.0, 0.003), "epsp_ph": (-np.pi, np.pi),
          "z22": (0.0, 0.1), "z22_ph": (-np.pi, np.pi), "ux": (-12.0, 12.0), "uy": (-8.0, 8.0), "uz": (-9.0, 9.0)}
PERIODIC = ["epsp_ph", "z22_ph"]
NU_KEYS = ["ux", "uy", "uz"]


def scale(tb):
    return {"mu": 2e16, "model": "MSSM", "tan_beta": float(tb)}


def mats(t):
    p = dict(sigma=t["s_re"] + 1j * t["s_im"], eta=t["e_re"] + 1j * t["e_im"], eps=t["eps"], etap=t["etap"],
             epsp=t["epsp"] * np.exp(1j * t["epsp_ph"]), z22d=t["z22"] * np.exp(1j * t["z22_ph"]), z22u=0.0)
    x, y, z = 1e-7 * np.sinh(t.get("ux", 1.0)), 1e-4 * np.sinh(t.get("uy", 1.0)), 1e-5 * np.sinh(t.get("uz", 1.0))
    return L.bpr_U(**p), L.bpr_D(**p), L.bpr_L(**p), L.bpr_N(**p), L.bpr_MR(x, y, z)


def model_fn(t):
    U, D, Le, N, M = mats(t)
    try:
        return yk.observables(U, D, Le, Y_nu=N, M_R=M, ordering="NO")
    except (np.linalg.LinAlgError, ValueError):
        return {k: float("nan") for k in CKM + MR + PMNS}


def run(mode, seed, tb=10.0, popsize=20, maxiter=800, fix=None):
    use = CKM + MR + (PMNS if mode == "full" else [])
    bounds = dict(BOUNDS) if mode == "full" else {k: v for k, v in BOUNDS.items() if k not in NU_KEYS}
    fixed = dict(fix or {})
    for k in fixed:
        bounds.pop(k, None)
    t0 = time.time()
    fr = ev.fit_and_evaluate(lambda th: model_fn({**fixed, **th}), bounds, "NO", use, k_discrete=K_BITS,
                             periodic=PERIODIC, seed=seed, mass_floor=FLOOR, scale=scale(tb), method="de",
                             optimizer_options={"popsize": popsize, "maxiter": maxiter, "tol": 1e-10})
    r = fr.result
    out = {"mode": mode, "seed": seed, "tan_beta": tb, "params": {**fixed, **fr.params}, "chi2": r.chi2,
           "chi2_global": r.chi2_global, "N": r.N, "k": r.k, "k_continuous": r.k_continuous,
           "k_discrete": r.k_discrete, "dof": r.dof, "p": r.p, "p_global": r.p_global,
           "max_abs_pull": r.max_abs_pull, "pulls": r.pulls, "pred": dict(r.pred),
           "compression_ok": r.compression_ok, "fit_ok": r.fit_ok, "nfev": fr.nfev, "seconds": time.time() - t0}
    return out


if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    tb = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    out = run(mode, seed, tb)
    tag = f"fit_bpr_{mode}_tb{tb:g}_s{seed}"
    (HERE / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("mode", "seed", "tan_beta", "chi2", "chi2_global", "N", "k", "dof",
                                          "p_global", "max_abs_pull", "compression_ok", "fit_ok", "nfev",
                                          "seconds")}, default=float))
    print("pulls", {k: round(v, 2) for k, v in out["pulls"].items()})
    print("params", out["params"])

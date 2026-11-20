"""Official-scorer fits of AKS (arXiv:1301.6764) and CM (arXiv:1307.7711) (README for counts).

    python fit42.py aks <seed>                    N 16, MSSM tan beta 1.49, M_SUSY 750 GeV (paper's fit), NO (m1 = 0)
    python fit42.py cm  <seed> [NO|IO]            N 16, MSSM tan beta 10 (paper's value), M_SUSY 3 TeV
    options: --drop k1,k2   score without these keys (threshold proxy, README)
             --eta eb,eq,el tan-beta-enhanced thresholds in scale (counted as 3 continuous parameters)
             --tag name     output name suffix

Matrices at mu = 2e16 GeV (src.evaluate scale=, data run up with src.rg). DE: popsize 20, maxiter 800.
The DE fit counts only the parameters it varies; the result is rescored with the honest k
(K_EXTRA continuous + bits, README) via Scorer(...)(pred, k=..., k_discrete=...).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import evaluate as ev  # noqa: E402
import lib42 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
CKM = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad"]
MR = ["mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "mr_b_tau"]
PMNS = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
ALL = CKM + MR + PMNS
NAN = {k: float("nan") for k in ALL}
FLOOR = 0.02
TP = 2 * np.pi

# ---------------------------------------------------------------- AKS
# Y_u / e_u, Y_d / eps_3 entries via signed sinh maps; eps (= |m_b|/|m_a|) of Eq. (2.1); alpha = pi/3.
AKS_SC = {"au": 1e-7, "bu": 1e-6, "cu": 1e-5, "du": 1e-4, "e2": 1e-5, "e102": 1e-5, "e23": 1e-5}
AKS_B = {**{k: (-10.0, 10.0) for k in AKS_SC}, "eps": (0.01, 1.0)}
AKS_PER = []
AKS_SCALE = {"mu": 2e16, "model": "MSSM", "tan_beta": 1.49, "m_susy": 750.0}
AKS_K_EXTRA = 1      # tan beta (fitted in the paper, Table 4)
AKS_BITS = 5         # generous count (README); strict count 11


def aks_params(t):
    p = {k: AKS_SC[k] * np.sinh(t[k]) for k in AKS_SC}
    p.update(eu=1.0, e3=1.0, eps=t["eps"])
    return p


def aks_model(t):
    try:
        return L.aks_obs(aks_params(t))
    except (np.linalg.LinAlgError, ValueError):
        return NAN


# ---------------------------------------------------------------- CM
# b, c, k, h real (Eq. 14, 13), g via signed sinh; (xi0, eta0, eta0'') = (cos th, sin th cos ph, sin th sin ph)
# (the overall scale is absorbed by the neutrino mass scale).
CM_B = {"b": (-0.03, 0.03), "c": (-0.1, 0.1), "k": (-0.2, 0.2), "h": (-0.05, 0.05), "ug": (-10.0, 10.0),
        "th": (0.0, np.pi), "ph": (0.0, TP)}
CM_PER = ["ph"]
CM_SCALE = {"mu": 2e16, "model": "MSSM", "tan_beta": 10.0}
CM_K_EXTRA = 0       # tan beta 10 taken by hand in the paper (Eq. 18), not counted
CM_BITS = 1          # model version (2013 eta'' variant); the ordering bit is added separately (both allowed)


def cm_params(t):
    return dict(b=t["b"], c=t["c"], k=t["k"], h=t["h"], g=1e-7 * np.sinh(t["ug"]), xi=np.cos(t["th"]),
                eta=np.sin(t["th"]) * np.cos(t["ph"]), etapp=np.sin(t["th"]) * np.sin(t["ph"]))


def cm_model_factory(ordering):
    def f(t):
        try:
            return L.cm_obs(cm_params(t), rl_transpose=True, conj=True, ordering=ordering)
        except (np.linalg.LinAlgError, ValueError):
            return NAN
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["aks", "cm"])
    ap.add_argument("seed", type=int)
    ap.add_argument("ordering", nargs="?", default="NO")
    ap.add_argument("--drop", default="")
    ap.add_argument("--eta", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--maxiter", type=int, default=800)
    a = ap.parse_args()
    drop = [k for k in a.drop.split(",") if k]
    use = [k for k in ALL if k not in drop]
    if a.model == "aks":
        fn, bounds, per, scale, kx, bits = aks_model, AKS_B, AKS_PER, dict(AKS_SCALE), AKS_K_EXTRA, AKS_BITS
        ordering_bit = 0
    else:
        fn, bounds, per, scale, kx, bits = (cm_model_factory(a.ordering), CM_B, CM_PER, dict(CM_SCALE), CM_K_EXTRA,
                                            CM_BITS)
        ordering_bit = 1
    if a.eta:
        eb, eq, el = (float(x) for x in a.eta.split(","))
        scale.update(eta_b=eb, eta_q=eq, eta_l=el)
        kx += 3
    t0 = time.time()
    fr = ev.fit_and_evaluate(fn, bounds, a.ordering, use, k_discrete=0, periodic=per, seed=a.seed,
                             mass_floor=FLOOR, scale=scale, method="de",
                             optimizer_options={"popsize": 20, "maxiter": a.maxiter, "tol": 1e-10})
    sc = ev.Scorer(use, a.ordering, mass_floor=FLOOR, scale=scale)
    pred = fn(fr.params)
    k_cont = len(bounds) + kx
    r = sc(pred, k=k_cont + bits + ordering_bit, k_discrete=bits + ordering_bit)
    out = {"model": a.model, "seed": a.seed, "ordering": a.ordering, "scale": scale, "drop": drop,
           "params_fit": fr.params, "chi2": r.chi2, "chi2_global": r.chi2_global, "N": r.N, "k": r.k,
           "k_continuous": r.k_continuous, "k_discrete": r.k_discrete, "dof": r.dof, "p": r.p,
           "p_global": r.p_global, "max_abs_pull": r.max_abs_pull, "pulls": r.pulls, "pred": dict(r.pred),
           "pred_gut": {k: pred[k] for k in ALL + ["R_clebsch"]}, "compression_ok": r.compression_ok,
           "fit_ok": r.fit_ok, "chi2_de_scorer": fr.result.chi2, "nfev": fr.nfev, "seconds": time.time() - t0}
    tag = f"fit_{a.model}_{a.ordering}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    (HERE / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("model", "seed", "ordering", "chi2", "chi2_global", "N", "k", "k_continuous",
                                          "dof", "p_global", "max_abs_pull", "compression_ok", "fit_ok", "nfev",
                                          "seconds")}, default=float))
    print("pulls", {k: round(v, 2) for k, v in sorted(out["pulls"].items(), key=lambda kv: -abs(kv[1]))})
    print("pred_gut", {k: round(v, 6) for k, v in out["pred_gut"].items()})
    print("params", {k: round(float(v), 6) for k, v in fr.params.items()})


if __name__ == "__main__":
    main()

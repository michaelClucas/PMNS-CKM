"""Quick fits of the models that are not covered by F1/F2 (and the uncovered Qin-Ma ansatz) with the official
scorer src.evaluate (import only). Single process. Run: ../../.venv/Scripts/python quickfit.py > quickfit.log

k counting: k_cont = fitted continuous parameters; k_d (bits) = ceil(log2 #options) for data-selected discrete
choices, given twice: 'as published' (only choices the paper itself lists as alternatives) and 'honest' (also the
pattern grammar the authors picked from after seeing data; see README for the option counts).
"""
import json
import math

import numpy as np
from vlib import ckm_central, V_of, obs, PMNS_KEYS, r12, r23
from verify_bilarge import cdsv, rmsv
from verify_csd2_gut import csd2_gut, dm_ratio, lep_ratio
from src.evaluate import fit_and_evaluate, evaluate_model

TWO_PI = 2 * np.pi
WOLF_IN = ("ckm_lambda", "ckm_A")
WOLF_ALL = ("ckm_lambda", "ckm_A", "ckm_rhobar", "ckm_etabar")
U_TB = np.array([[2, np.sqrt(2), 0], [-1, np.sqrt(2), np.sqrt(3)], [1, -np.sqrt(2), np.sqrt(3)]]) / np.sqrt(6)
rows = []


def row(name, fr, n, k_cont, kd_pub, kd_honest, params=None):
    r = fr.result if hasattr(fr, "result") else fr
    d = dict(model=name, ordering=r.ordering, N=n, k_cont=k_cont, k_pub=k_cont + kd_pub, k_honest=k_cont + kd_honest,
             chi2=round(r.chi2, 3), chi2_global=round(r.chi2_global, 3), p_global=r.p_global,
             max_pull=round(max(abs(v) for v in r.pulls.values()), 2),
             pulls={k: round(v, 2) for k, v in r.pulls.items()},
             pred={k: round(float(v), 5) for k, v in (r.pred or {}).items()} if hasattr(r, "pred") else None,
             params=params or getattr(fr, "params", None))
    rows.append(d)
    print(f"{name:34s} {r.ordering}  chi2 {r.chi2:8.3f}  chi2_glob {r.chi2_global:8.3f}  p_glob {r.p_global:.3g}  "
          f"N {n} k_cont {k_cont} k_pub {k_cont + kd_pub} k_honest {k_cont + kd_honest}  pulls "
          f"{ {k: round(v, 2) for k, v in r.pulls.items()} }  params {d['params']}")


def pmns_pred(U):
    o = obs(U)
    return {k: o[k] for k in PMNS_KEYS}


if __name__ == "__main__":
    use4 = PMNS_KEYS
    # --- bi-large CDSV patterns: 1 phase. honest bits: U_nu grammar ~ 8 options per angle (n lambda, 1 - n lambda,
    # n = 1..4) ^3 = 512 x delta_nu {0, pi} x U_l {SO(10), SU(5)} -> 2048 -> 11 bits; published: 2 patterns -> 1 bit
    for pat in ("I", "II"):
        for o in ("NO", "IO"):
            fr = fit_and_evaluate(lambda t, pat=pat: pmns_pred(cdsv(ckm_central(), t["phi"], pat)), {"phi": (0, TWO_PI)},
                                  o, use4, inputs=WOLF_IN, periodic=("phi",), n_starts=16, leak_check=True)
            row(f"bi-large CDSV-{pat}", fr, 4, 1, 1, 11)
    # --- RMSV: psi, phi; types 1-4 -> 2 bits published; honest + grammar for 'theta12 = theta23 = psi lambda,
    # theta13 = lambda' (~ 8 options) -> 5 bits
    for ut in (1, 2, 3, 4):
        fr = fit_and_evaluate(lambda t, ut=ut: pmns_pred(rmsv(ckm_central(), t["psi"], t["phi"], ut)),
                              {"psi": (1.0, 4.4), "phi": (0, TWO_PI)}, "NO", use4, inputs=WOLF_IN, periodic=("phi",),
                              n_starts=16, leak_check=True)
        row(f"bi-large RMSV type {ut}", fr, 4, 2, 2, 5)
    # --- Qin-Ma: U = V U_TB V^dag (orientation fixed by reproducing the paper, verify_qin_ma.py), no parameters
    for o in ("NO", "IO"):
        def qm():
            V = V_of(ckm_central())
            return pmns_pred(V @ U_TB @ V.conj().T)
        r = evaluate_model(qm, o, use4, inputs=WOLF_ALL, k=0)
        print("   Qin-Ma prediction:", {k: round(float(v), 5) for k, v in qm().items()})
        row("Qin-Ma V U_TB V^dag", r, 4, 0, 0, 3, params={})
    # --- CSD2-GUT: eps, eta, omega; use PMNS + dm_ratio + mr_d_s. The model consumes m_e/m_mu (inputs m_e, m_mu),
    # so R_clebsch cannot be scored (alias); m_d/m_s = (m_e/m_mu) c_x^2 / (c_y c_z) is scored instead (R is RG
    # invariant; the pole-vs-MSbar(M_Z) difference of m_e/m_mu, ~1%, sits inside the 2% floor).
    use6 = PMNS_KEYS + ("dm_ratio", "mr_d_s")
    for cl in ((3.0, 1.5, 0.5), (4.5, 1.5, 1.5), (3.0, 1.0, 1.0)):
        def m(t, cl=cl):
            p = pmns_pred(csd2_gut(ckm_central(), t["eps"], t["eta"], t["omega"], cl))
            p["dm_ratio"] = dm_ratio(t["eps"], t["eta"])
            p["mr_d_s"] = lep_ratio() * cl[0] ** 2 / (cl[1] * cl[2])
            return p
        fr = fit_and_evaluate(m, {"eps": (0.01, 0.5), "eta": (0, TWO_PI), "omega": (0, TWO_PI)}, "NO", use6,
                              inputs=WOLF_ALL + ("m_e", "m_mu"), periodic=("eta", "omega"), n_starts=24,
                              leak_check=True, mass_floor=0.02)
        row(f"CSD2-GUT {cl}", fr, 6, 3, 1, 7)   # published: 102 vs 120 variant; honest: 74 tuples x variants
    with open("quickfit.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)

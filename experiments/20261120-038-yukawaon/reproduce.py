"""Step 2: reproduce the published benchmark of arXiv:1510.05370 (Table 1, Eqs. (20)-(23)).

Their inputs, our model code, no measured data from data/ (the charged-lepton masses at M_Z are
the paper-era values the paper says it uses, Xing-Zhang-Zhou arXiv:0712.1419, quoted here as the
paper's inputs, not as our data). Scans the convention variants and reports which reproduces.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

import model as M

HERE = Path(__file__).resolve().parent
# m_e, m_mu, m_tau at M_Z (MeV), Xing-Zhang-Zhou 0712.1419 Table (the paper's ref [10] era inputs)
ME, MMU, MTAU = 0.48657, 102.718, 1746.24
PHI0 = M.phi0_from_ratios(ME / MMU, MMU / MTAU)
BENCH = dict(b_u=-1.011, b_d=-3.3522, beta_d=17.7, phi1=-176.05, phi2=-167.91, xi_R=2039.6)
PUB = {  # paper Eqs. (20)-(23), Table 1
    "sqrt_mu_mc": 0.061, "sqrt_mc_mt": 0.060, "md_ms": 0.049, "ms_mb": 0.027,
    "Vus": 0.2257, "Vcb": 0.03996, "Vub": 0.003701, "Vtd": 0.009173, "dq_deg": 80.99,
    "sin2_2t12": 0.8254, "sin2_2t23": 0.9967, "sin2_2t13": 0.1007, "dl_deg": -68.1, "R_nu": 0.03118,
}
TOL = {"sqrt_mu_mc": 0.0015, "sqrt_mc_mt": 0.0015, "md_ms": 0.0015, "ms_mb": 0.0015}  # printed to 2 digits


def outputs(theta, **kw):
    p, d = M.predict(theta, PHI0, details=True, **kw)
    V, U, m = d["V_ckm"], d["U_pmns"], d["m_nu"]
    w = lambda x: (x + 180) % 360 - 180
    s = lambda x: 4 * x * (1 - x)
    return {
        "sqrt_mu_mc": math.sqrt(p["mr_u_c"]), "sqrt_mc_mt": math.sqrt(p["mr_c_t"]),
        "md_ms": p["mr_d_s"], "ms_mb": p["mr_s_b"],
        "Vus": abs(V[0, 1]), "Vcb": abs(V[1, 2]), "Vub": abs(V[0, 2]), "Vtd": abs(V[2, 0]),
        "dq_deg": w(math.degrees(p["ckm_delta_rad"])),
        "sin2_2t12": s(p["s12sq"]), "sin2_2t23": s(p["s23sq"]), "sin2_2t13": s(p["s13sq"]),
        "dl_deg": w(p["dcp_deg"]), "R_nu": (m[1] ** 2 - m[0] ** 2) / (m[2] ** 2 - m[1] ** 2),
        "ordering": d["ordering"], "m_nu_ratio": (m / m[2]).tolist(),
    }


def rel_dev(o):
    out = {}
    for k, v in PUB.items():
        if k.endswith("_deg"):
            out[k] = abs(o[k] - v)  # degrees
        else:
            out[k] = abs(o[k] - v) / abs(v)
    return out


if __name__ == "__main__":
    rows = {}
    for usym, conj, nuconj, ps in itertools.product([True, False], [False, True], [False, True], [1, -1]):
        o = outputs(BENCH, usym=usym, conj=conj, nuconj=nuconj, phisign=ps)
        dev = rel_dev(o)
        tag = f"usym={usym} conj={conj} nuconj={nuconj} phisign={ps}"
        rows[tag] = {"out": o, "dev": dev}
        print(tag)
        for k in PUB:
            print(f"   {k:11s} ours {o[k]:+.5g}  paper {PUB[k]:+.5g}  dev {dev[k]:.3g}{' deg' if k.endswith('deg') else ''}")
        print("   ordering", o["ordering"], "m_i/m_3", np.round(o["m_nu_ratio"], 4))
    (HERE / "reproduce.json").write_text(json.dumps(rows, indent=1))
    ok = [t for t, v in rows.items() if all(d < (2.0 if k.endswith("deg") else 0.015) for k, d in v["dev"].items())]
    print()
    print("variants reproducing ALL published numbers (rel dev < 1.5%, phases < 2 deg):", ok)

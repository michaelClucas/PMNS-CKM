"""Step 2: pipeline check against arXiv:1503.03306 Table 5 (GUT-scale outputs, tan beta = 5 and 10).

Their inputs, our model code. Fixes the conjugation convention (CONJ, NUCONJ) as the one that
reproduces both published CP-phase signs. No measured data are used here.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

import model as M  # noqa: E402

BENCH = {
    5: dict(u=(0.9566, 0.7346, 0.7198, 0.5961, 0.3224, 0.5435), d=(2.133, 0.8363, 1.108, 1.021)),
    10: dict(u=(0.9182, 0.7087, 0.6910, 0.5768, 0.3095, 0.5218), d=(4.236, 1.661, 2.200, 2.034)),
}
MA, MB = 26.57e-3, 2.684e-3  # eV
PUB = {  # paper Table 5 outputs (tan beta = 5; angles in degrees)
    "th12q": 13.027, "th13q": 0.1802, "th23q": 2.054, "dq": 69.18,
    "th12l": 34.3, "th13l": 8.67, "th23l": 45.8, "dl": -86.7, "dm21": 7.38e-5, "dm31": 2.48e-3,
    "yu/yc": 2.92e-6 / 1.43e-3, "yc/yt": 1.43e-3 / 0.534, "yd/ys": 4.30e-6 / 9.51e-5, "ys/yb": 9.51e-5 / 7.05e-3,
    "ye/ymu": 1.97e-6 / 4.16e-4, "ymu/ytau": 4.16e-4 / 7.05e-3,
}


def outputs(tb, conj, nuconj):
    b = BENCH[tb]
    p = M.predict(b["u"], b["d"], MB / MA, conj=conj, nuconj=nuconj, ma=MA)
    w = lambda x: (x + 180) % 360 - 180
    return {
        "th12q": math.degrees(math.asin(p["ckm_s12"])), "th13q": math.degrees(math.asin(p["ckm_s13"])),
        "th23q": math.degrees(math.asin(p["ckm_s23"])), "dq": w(math.degrees(p["ckm_delta_rad"])),
        "th12l": math.degrees(math.asin(math.sqrt(p["s12sq"]))), "th13l": math.degrees(math.asin(math.sqrt(p["s13sq"]))),
        "th23l": math.degrees(math.asin(math.sqrt(p["s23sq"]))), "dl": w(p["dcp_deg"]),
        "dm21": p["_dm21"], "dm31": p["_dm31"],
        "yu/yc": p["mr_u_c"], "yc/yt": p["mr_c_t"], "yd/ys": p["mr_d_s"], "ys/yb": p["mr_s_b"],
        "ye/ymu": p["mr_e_mu"], "ymu/ytau": p["mr_mu_tau"],
        "theta12e_rad": p["_theta12e"], "theta23e_rad": p["_theta23e"],
    }


def main():
    out = {}
    for conj in (False, True):
        for nuconj in (False, True):
            o = outputs(5, conj, nuconj)
            ok = (np.sign(o["dq"]) == np.sign(PUB["dq"])) and (np.sign(o["dl"]) == np.sign(PUB["dl"]))
            out[f"conj={conj},nuconj={nuconj}"] = dict(o, signs_match=bool(ok))
            print(f"conj={conj} nuconj={nuconj}: dq={o['dq']:.2f} dl={o['dl']:.2f} signs_match={ok}")
    o5 = outputs(5, M.CONJ, M.NUCONJ)
    print("\nquantity      ours(tb5)      published   rel.diff")
    for k, v in PUB.items():
        print(f"{k:10s} {o5[k]:14.5g} {v:12.5g} {(o5[k] - v) / abs(v):+9.3%}")
    print(f"theta12e = {o5['theta12e_rad']:.3e} rad, theta23e = {o5['theta23e_rad']:.3e} rad")
    o10 = outputs(10, M.CONJ, M.NUCONJ)
    out["tb5_selected"], out["tb10_selected"], out["published"] = o5, o10, PUB
    print("tb10:", {k: round(v, 5) for k, v in o10.items() if k in ("th12q", "th13q", "th23q", "dq")})
    Path("reproduce.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

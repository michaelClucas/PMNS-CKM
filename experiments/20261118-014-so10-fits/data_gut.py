"""(Copied from exp 011.) Real-data observable vector z at M_GUT = 2e16 GeV (SM running with src/rg.py) and the noise model.

Writes obs_gut.json: z (NO), z_IO, sigma (NO), sigma_IO, names. All measured inputs come from data/ via src.
Noise model (symmetrised 1 sigma, used both as simulator noise and as the measurement error):
  masses: each mass gets max(PDG relative error, 2%) (2% theory floor: one-loop Yukawa running ~1%,
          thresholds), a log10 ratio gets sqrt(sum of squares)/ln10.
  CKM:    s12 max(PDG, 1%), s13, s23 max(PDG, 3%) relative (in log10), delta_q max(PDG, 0.05 rad).
  PMNS:   NuFIT 6.1 symmetrised errors on s12^2, s13^2, s23^2, delta; Delta m^2 ratio 3% (log10).
GUT-scale value of each quantity = RG-run central value; errors are carried over as relative errors (mass
ratios run multiplicatively), which ignores the small error on the running itself.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import data, rg  # noqa: E402

MU_GUT = 2e16
OUT = pathlib.Path(__file__).with_name("obs_gut.json")


def rel(rec):
    return 0.5 * (rec["err_plus"] + rec["err_minus"]) / rec["value"]


def main():
    c = rg.ckm_at(MU_GUT)
    y = c["yukawas"]
    m = data.load_masses()
    ck = data.load_ckm()
    floor = 0.02
    em = {k: max(rel(m[k]), floor) for k in ("m_u", "m_d", "m_s", "m_c", "m_b", "m_e", "m_mu", "m_tau")}
    em["m_t"] = max(rel(m["m_t_direct"]), floor)
    lr = lambda a, b: np.hypot(em[a], em[b]) / np.log(10)
    out = {"names": None, "mu_gut": MU_GUT, "model": "SM, src/rg.py, Majorana phases 0, m_lightest 0"}
    for o in ("NO", "IO"):
        p = rg.pmns_at(MU_GUT, o)
        pd = data.load_pmns(o)
        sym = lambda k: 0.5 * (pd[k]["err_plus"] + pd[k]["err_minus"])
        z = [np.log10(y["y_u"] / y["y_c"]), np.log10(y["y_c"] / y["y_t"]), np.log10(y["y_d"] / y["y_s"]),
             np.log10(y["y_s"] / y["y_b"]), np.log10(y["y_b"] / y["y_t"]), np.log10(y["y_e"] / y["y_mu"]),
             np.log10(y["y_mu"] / y["y_tau"]), np.log10(y["y_tau"] / y["y_b"]),
             np.sin(c["theta12"]), np.log10(np.sin(c["theta13"])), np.log10(np.sin(c["theta23"])), c["delta"],
             np.sin(p["theta12"]) ** 2, np.sin(p["theta13"]) ** 2, np.sin(p["theta23"]) ** 2, p["delta"],
             np.log10(p["dm21_over_dm3l"])]
        sig = [lr("m_u", "m_c"), lr("m_c", "m_t"), lr("m_d", "m_s"), lr("m_s", "m_b"), lr("m_b", "m_t"),
               lr("m_e", "m_mu"), lr("m_mu", "m_tau"), lr("m_tau", "m_b"),
               max(rel(ck["s12"]), 0.01) * z[8], max(rel(ck["s13"]), 0.03) / np.log(10),
               max(rel(ck["s23"]), 0.03) / np.log(10), max(0.5 * (ck["delta_rad"]["err_plus"] + ck["delta_rad"]["err_minus"]), 0.05),
               sym("s12sq"), sym("s13sq"), sym("s23sq"), np.radians(sym("dcp_deg")), 0.03 / np.log(10)]
        key = "" if o == "NO" else "_IO"
        out["z" + key] = [float(v) for v in z]
        out["sigma" + key] = [float(v) for v in sig]
        # M_Z values of the PMNS block, to quantify SM running of PMNS
        out["pmns_mz" + key] = {k: pd[k]["value"] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg")}
    from so10fit import Z_NAMES
    out["names"] = Z_NAMES
    OUT.write_text(json.dumps(out, indent=1))
    for n, v, s in zip(Z_NAMES, out["z"], out["sigma"]):
        print(f"{n:14s} {v: .5f} +- {s:.5f}")


if __name__ == "__main__":
    main()

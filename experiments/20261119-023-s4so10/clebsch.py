"""Step 3: effective d/e Clebsch factors of 1705.01555 and the R double-ratio pull.

In this model Y^e and Y^d share the matrices Y12, Y22, Y33, YP, with independent coefficients except
y_P (shared). The effective Clebsch factor of each term is the complex ratio of e to d coefficients:
  C12 = y12e / y12d,  C22 = y2e e^{i a_e} / (y2d e^{i a_d}),  C33 = y3e e^{i b_e} / (y3d e^{i b_d}),  CP = 1.
They come from four 45_H VEVs with arbitrary (SM-preserving) directions, so they are continuous
fitted numbers, not group-theory constants.
R = (m_e/m_mu)/(m_d/m_s): exp 021 target 0.0938 +- 0.0027 (2% floor on m_d, m_s) and the official
src.evaluate key R_clebsch (scale translation from M_GUT, mass_floor 0.02).
Usage: python clebsch.py  (reads reproduce.json and fit_*.json present) -> clebsch.json
"""
import json

import numpy as np

from fitlib import HERE, SCALE, FLOOR
from model import PUBLISHED, observables, pred_dimless
from src.evaluate import Scorer

R_EXP021 = (0.0938, 0.0027)


def clebsch(p):
    c12 = p["y12e"] / p["y12d"]
    c22 = p["y2e"] * np.exp(1j * p["ae"]) / (p["y2d"] * np.exp(1j * p["ad"]))
    c33 = p["y3e"] * np.exp(1j * p["be"]) / (p["y3d"] * np.exp(1j * p["bd"]))
    f = lambda z: {"abs": float(abs(z)), "arg_over_pi": float(np.angle(z) / np.pi)}
    return {"C12": f(c12), "C22": f(c22), "C33": f(c33), "CP": f(1.0 + 0j)}


def main():
    sc = Scorer(("R_clebsch",), "NO", mass_floor=FLOOR, scale=SCALE)
    sc_q = Scorer(("R_clebsch",), "NO", mass_floor={"R_clebsch": 0.02 * np.sqrt(2)}, scale=SCALE)
    fac = sc.scale_shift["R_clebsch"]
    pts = {"published_point": PUBLISHED}
    rep = json.loads((HERE / "reproduce.json").read_text())
    pts["refit_paper_inputs"] = rep["refit_paper_inputs"]["params"]
    for f in sorted(HERE.glob("fit_*.json")):
        d = json.loads(f.read_text())
        pts[f.stem] = d["params"]
    out = {"R_target_exp021": R_EXP021, "gut_to_mz_factor": fac}
    for name, p in pts.items():
        pr = pred_dimless(observables(p))
        r_gut = pr["mr_e_mu"] / pr["mr_d_s"]
        r_mz = r_gut * fac
        res = sc({"R_clebsch": r_gut})
        res_q = sc_q({"R_clebsch": r_gut})
        out[name] = {"clebsch": clebsch(p), "R_gut": float(r_gut), "R_mz": float(r_mz),
                     "pull_exp021": float((r_mz - R_EXP021[0]) / R_EXP021[1]),
                     "pull_src_floor002": float(res.pulls["R_clebsch"]),
                     "pull_src_floor_sqrt2x2pct": float(res_q.pulls["R_clebsch"]),
                     "md_over_ms_gut": float(pr["mr_d_s"]), "me_over_mmu_gut": float(pr["mr_e_mu"])}
        c = out[name]["clebsch"]
        print(f"{name:22s} |C12| {c['C12']['abs']:.3f} ({c['C12']['arg_over_pi']:+.2f}pi) "
              f"|C22| {c['C22']['abs']:.3f} ({c['C22']['arg_over_pi']:+.2f}pi) |C33| {c['C33']['abs']:.3f} "
              f"R_MZ {r_mz:.4f} pull {out[name]['pull_exp021']:+.2f} (src {res.pulls['R_clebsch']:+.2f})")
    (HERE / "clebsch.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

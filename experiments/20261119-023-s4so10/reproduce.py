"""Step 1: reproduce the 1705.01555 best fit with the paper's own inputs and errors.

chi^2 = sum ((P - P_obs) / sigma)^2 with the paper's asymmetric 1 sigma ranges (sigma = upper or
lower half-range depending on the side), 18 observables, delta_l not fitted (paper Eq. 25).
Outputs reproduce.json. Usage: python reproduce.py
"""
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

from model import PUBLISHED, REAL, PHASES, observables

HERE = pathlib.Path(__file__).parent
PUB = json.loads((HERE / "pub_inputs.json").read_text())
OBS = PUB["observables"]


def paper_obs(o):
    l12, l13, l23, ld = np.degrees(o["pmns"])
    q12, q13, q23, qd = np.degrees(o["ckm"])
    return {"th12l_deg": l12, "th13l_deg": l13, "th23l_deg": l23, "dl_deg": ld % 360,
            "ye_1e-5": o["ye"][0] / 1e-5, "ymu_1e-3": o["ye"][1] / 1e-3, "ytau_1e-2": o["ye"][2] / 1e-2,
            "dm21_1e-5eV2": o["dm21"] / 1e-5, "dm31_1e-3eV2": o["dm31"] / 1e-3,
            "m1_meV": o["mnu_meV"][0], "m2_meV": o["mnu_meV"][1], "m3_meV": o["mnu_meV"][2],
            "alpha21_deg": o["alpha21"], "alpha31_deg": o["alpha31"],
            "th12q_deg": q12, "th13q_deg": q13, "th23q_deg": q23, "dq_deg": qd % 360,
            "yu_1e-6": o["yu"][0] / 1e-6, "yc_1e-3": o["yu"][1] / 1e-3, "yt": o["yu"][2],
            "yd_1e-5": o["yd"][0] / 1e-5, "ys_1e-4": o["yd"][1] / 1e-4, "yb": o["yd"][2]}


def pulls(po):
    out = {}
    for k, (c, lo, hi) in OBS.items():
        d = po[k] - c
        out[k] = d / ((hi - c) if d > 0 else (c - lo))
    return out


def chi2_parts(po):
    pl = pulls(po)
    lep = sum(v ** 2 for k, v in pl.items() if k.endswith("l_deg") or k.startswith(("ye_", "ymu", "ytau", "dm")))
    tot = sum(v ** 2 for v in pl.values())
    return tot, lep, tot - lep, pl


# parameter vector: log|real| with fixed signs, phases free
SIGNS = {k: np.sign(PUBLISHED[k]) for k in REAL}


def unpack(x):
    p = {k: SIGNS[k] * np.exp(x[i]) for i, k in enumerate(REAL)}
    p.update({k: x[len(REAL) + i] for i, k in enumerate(PHASES)})
    return p


def pack(p):
    return np.array([np.log(abs(p[k])) for k in REAL] + [p[k] for k in PHASES])


def resid(x, conj):
    pl = pulls(paper_obs(observables(unpack(x), conj)))
    return np.array(list(pl.values()))


def main():
    res = {}
    for conj in (False, True):
        po = paper_obs(observables(PUBLISHED, conj))
        tot, lep, q, pl = chi2_parts(po)
        res[f"conj={conj}"] = {"at_published_point": {"chi2": tot, "chi2_lepton": lep, "chi2_quark": q,
                                                     "pred": po, "pulls": pl}}
    conj = min((False, True), key=lambda c: abs(res[f"conj={c}"]["at_published_point"]["pred"]["dq_deg"] - 69.21))
    res["convention"] = {"conj": conj, "reason": "reproduces printed delta_q = 69.21 deg at the published point"}
    best = None
    x0 = pack(PUBLISHED)
    rng = np.random.default_rng(23)
    starts = [x0] + [x0 + rng.normal(0, 0.05, x0.size) for _ in range(15)]
    for s in starts:
        r = least_squares(resid, s, args=(conj,), method="trf", max_nfev=20000, xtol=1e-14, ftol=1e-14)
        c = float(2 * r.cost)
        if best is None or c < best[0]:
            best = (c, r.x)
    p = unpack(best[1])
    po = paper_obs(observables(p, conj))
    tot, lep, q, pl = chi2_parts(po)
    res["refit_paper_inputs"] = {"chi2": tot, "chi2_lepton": lep, "chi2_quark": q, "pred": po, "pulls": pl,
                                 "params": {k: float(v) for k, v in p.items()},
                                 "phases_over_pi": {k: float((p[k] / np.pi) % 2) for k in PHASES}}
    (HERE / "reproduce.json").write_text(json.dumps(res, indent=1, default=float))
    for k in ("conj=False", "conj=True"):
        a = res[k]["at_published_point"]
        print(k, "chi2 at published point", round(a["chi2"], 3), "lep", round(a["chi2_lepton"], 3),
              "dq", round(a["pred"]["dq_deg"], 2), "dl", round(a["pred"]["dl_deg"], 2))
    r = res["refit_paper_inputs"]
    print("refit chi2", round(r["chi2"], 4), "lep", round(r["chi2_lepton"], 4), "dl", round(r["pred"]["dl_deg"], 1))
    print(json.dumps({k: round(v, 3) for k, v in res[f"conj={conj}"]["at_published_point"]["pred"].items()}))
    print(json.dumps({k: round(v, 2) for k, v in res[f"conj={conj}"]["at_published_point"]["pulls"].items()}))


if __name__ == "__main__":
    main()

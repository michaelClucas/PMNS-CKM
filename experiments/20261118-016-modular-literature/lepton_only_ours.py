"""Lepton-only refit of the 2O model to our data (tau + gE2/gE1 + gE3/gE1; 7 observables:
PMNS 4 via src.evaluate, me/mmu, mmu/mtau, Dm21/Dm31). Local multi-start around the lepton island
found by the grid in fit_ours.py. Writes lepton_only_ours.json."""
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

import fit_ours as F
from src import stats

HERE = pathlib.Path(__file__).parent


def resid(x, C):
    return F.resid_lep(x[2:], complex(x[0], x[1]), C)


def main():
    C = F.Chi2(F.build_data("ours"))
    rng = np.random.default_rng(1)
    best = None
    for re0, im0 in [(-0.18, 1.10), (-0.192, 1.083), (-0.205, 1.069), (0.18, 1.10), (-0.17, 1.11)]:
        for _ in range(3):
            x0 = np.array([re0, im0, 0.698, 86.4]) * (1 + 0.01 * rng.standard_normal(4))
            r = least_squares(resid, x0, args=(C,), method="trf", max_nfev=3000, x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
    c2 = 2 * best.cost
    res = resid(best.x, C)
    p = dict(F.BASE, re=best.x[0], im=best.x[1], rE2=best.x[2], rE3=best.x[3])
    o = F.observables(p)
    pred = F.to_mz(o, C.D)
    out = {"x": best.x.tolist(), "chi2": c2, "N": 7, "k": 4, "p": stats.p_value(c2, 7, 4),
           "pred_mz": {k: float(pred[k]) for k in F.PMNS_USE}, "resid_lep": res.tolist()}
    print(json.dumps(out))
    (HERE / "lepton_only_ours.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Filter (a) for Babu-Pati-Rastogi (hep-ph/0502152 Eq. 1): same procedure as bound_R.py.
Free d/e parameters: Re/Im eta, eps, eta', |eps'|, arg eps', |zeta22^d|, arg zeta22^d (8).

    python bound_R_bpr.py  -> bound_R_bpr.json, stdout (log)
"""
import json
import math

import numpy as np

import bound_R as B
import lib40 as L


def build(x):
    p = dict(eta=x[0] + 1j * x[1], eps=x[2], etap=x[3], epsp=x[4] * np.exp(1j * x[5]),
             z22d=x[6] * np.exp(1j * x[7]))
    return L.bpr_D(**p), L.bpr_L(**p)


B.MODELS["BPR"] = ([(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.0, 0.02), (0.0, 0.005), (-math.pi, math.pi),
                    (0.0, 0.1), (-math.pi, math.pi)], build)

if __name__ == "__main__":
    Md, Me = L.bpr_D(**L.BPR_FITA), L.bpr_L(**L.BPR_FITA)
    out = {"fitA_as_printed": {"md_ms": L.ratios(Md)[0], "ms_mb": L.ratios(Md)[1], "me_mmu": L.ratios(Me)[0],
                               "mmu_mtau": L.ratios(Me)[1], "R": L.ratios(Me)[0] / L.ratios(Md)[0]}}
    print("fit A as printed:", out["fitA_as_printed"])
    out["BPR"] = B.run("BPR")
    print("BPR", json.dumps(out["BPR"], default=float))
    (B.HERE / "bound_R_bpr.json").write_text(json.dumps(out, indent=1, default=float))

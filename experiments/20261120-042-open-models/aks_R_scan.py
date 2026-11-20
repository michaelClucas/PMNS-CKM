"""AKS: minimum R = (m_e/m_mu)/(m_d/m_s) over (eps2, eps102, eps23)/eps3 with m_e/m_mu at data, as a function of the
allowed m_s/m_b window (bound42.py uses [0.012, 0.030]; here the window is widened to show the band dependence)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

import bound42 as B  # noqa: E402

out = {}
for lo, hi, mlo, mhi in [(0.012, 0.030, 0.050, 0.068), (0.008, 0.030, 0.050, 0.068), (0.004, 0.05, 0.03, 0.1)]:
    def f(x):
        Md, Me = B.aks_de(x)
        d, e = B.sv(Md), B.sv(Me)
        if min(d[0], e[0]) <= 0:
            return B.BIG
        R = (e[0] / e[1]) / (d[0] / d[1])
        return R + 1e3 * (np.log(e[0] / e[1] / B.V["mr_e_mu"]) ** 2 + B.band(e[1] / e[2], mlo, mhi)
                          + B.band(d[1] / d[2], lo, hi))
    r = differential_evolution(f, [(-10, 10)] * 3, seed=1, popsize=30, maxiter=1000, tol=1e-12)
    Md, Me = B.aks_de(r.x)
    d, e = B.sv(Md), B.sv(Me)
    R = (e[0] / e[1]) / (d[0] / d[1])
    key = f"ms_mb[{lo},{hi}] mmu_mtau[{mlo},{mhi}]"
    out[key] = {"R_min": R, "pull": B.SCR({"R_clebsch": R}).pulls["R_clebsch"], "ms_mb": d[1] / d[2],
                "mmu_mtau": e[1] / e[2], "me_mmu": e[0] / e[1]}
    print(key, out[key], flush=True)
(Path(__file__).parent / "aks_R_scan.json").write_text(json.dumps(out, indent=1, default=float))

"""Rebuild lepton_region.json from artifacts/fixedtau/lepton_fine_19.npz (the e-set 20 variant run was
aborted for time before lepton_region.py wrote its json; e-set 19 numbers are identical to run.log).

    python region_json.py
"""
import json
import os
import re

import numpy as np
from scipy.stats import chi2 as chi2d

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "..", "..", "artifacts", "fixedtau", "lepton_fine_19.npz"))
tau, c, P, gmin, h = d["tau"], d["chi2"], d["P"], float(d["gmin"]), float(d["h"])
j = int(np.argmin(c))
rec = {"eset": 19, "label": "S4 L~3p e[41p+61*+61p] nu[SS] NO", "chi2_min": gmin,
       "tau_best_grid": [float(tau[j].real), float(tau[j].imag)], "chi2_best_grid": float(c[j]),
       "tau_best_free": [-0.17968259279068066, 1.0979643168694648], "P_best": P[j].tolist(),
       "p_lepton": float(chi2d.sf(gmin, 2)), "grid_step": h, "n_grid_evaluated": int(len(tau)),
       "gCP_image": {"tau": [0.1797, 1.098], "chi2_min": 15.92, "dchi2": 15.92 - gmin,
                     "note": "tau -> -tau* gives delta_l -> -delta_l; outside both contours"}}
for lvl in (2.30, 6.18):
    m = c <= gmin + lvl
    rec[f"region_{lvl}"] = {"n_pts": int(m.sum()), "area": float(m.sum() * h * h),
                            "re": [float(tau[m].real.min()), float(tau[m].real.max())],
                            "im": [float(tau[m].imag.min()), float(tau[m].imag.max())]}
json.dump({"19": rec}, open(os.path.join(HERE, "lepton_region.json"), "w"), indent=1)
print(json.dumps(rec, indent=1))

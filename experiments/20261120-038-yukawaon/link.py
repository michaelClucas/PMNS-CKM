"""Does the CKM enter PMNS? Shared-parameter sensitivity at the full best fit (fit_full_s2.json).

For each model parameter: d(observable)/d(parameter) for CKM and PMNS keys, in units of the
observable's rough 1-sigma (CKM: PDG global-fit sigma; PMNS: NuFIT 1D sigma from the scorer pulls'
scale, approximated by fixed widths below, used only for display). Also the PMNS spread when
(phi1, phi2) are redrawn uniformly (the parameters the paper fixes from the CKM) with the rest fixed,
and a check of the error-budget dependence (best-fit point rescored with a 30% mass floor).
"""
import json
from pathlib import Path

import numpy as np

import fit as F

HERE = Path(__file__).resolve().parent
best = json.loads((HERE / "fit_full_s2.json").read_text())
p0 = dict(best["params"])
SIG = {"ckm_s12": 0.0006, "ckm_s13": 0.0001, "ckm_s23": 0.0007, "ckm_delta_rad": 0.025,
       "s12sq": 0.012, "s13sq": 0.0006, "s23sq": 0.02, "dcp_deg": 25.0}
H = {"eps_u": 1e-5, "b_d": 1e-4, "beta_d": 1e-3, "phi1": 1e-3, "phi2": 1e-3, "xi_R": 1e-1}
out = {"params": p0, "deriv_sigma_per_unit": {}}
for n, h in H.items():
    a, b = dict(p0), dict(p0)
    a[n] += h
    b[n] -= h
    pa, pb = F.model_fn(a), F.model_fn(b)
    out["deriv_sigma_per_unit"][n] = {k: (pa[k] - pb[k]) / (2 * h) / s for k, s in SIG.items()}
for n, d in out["deriv_sigma_per_unit"].items():
    print(n, {k: f"{v:+.3g}" for k, v in d.items()})

rng = np.random.default_rng(0)
spread = []
for _ in range(500):
    t = dict(p0)
    t["phi1"], t["phi2"] = rng.uniform(-180, 180, 2)
    p = F.model_fn(t)
    spread.append([p[k] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "ckm_s12", "ckm_s23")])
s = np.array(spread)
out["phi_redraw_pmns_percentiles"] = {k: np.percentile(s[:, i], [5, 50, 95]).tolist()
                                      for i, k in enumerate(("s12sq", "s13sq", "s23sq", "dcp_deg", "ckm_s12", "ckm_s23"))}
print("redraw (phi1, phi2):", json.dumps(out["phi_redraw_pmns_percentiles"]))

use = F.CKM + F.MQ + F.PMNS
for floor in (0.02, 0.30):
    r = F.ev.evaluate(F.model_fn(p0), "NO", use=use, inputs=F.INPUTS, k=6, mass_floor=floor)
    out[f"rescore_floor_{floor}"] = {"chi2": r.chi2, "pulls": r.pulls}
    print(f"floor {floor}: chi2 {r.chi2:.1f}", {k: round(v, 2) for k, v in r.pulls.items()})
(HERE / "link.json").write_text(json.dumps(out, indent=1, default=float))

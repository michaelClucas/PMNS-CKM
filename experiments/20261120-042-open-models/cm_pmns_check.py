"""CM: PMNS-only chi^2 with m_e/m_mu pinned at data, started from the s1 best fit (bound42 (4)'s DE landed in a
theta12 > 45 deg branch). b is rescaled so m_e/m_mu = data at the s1 value of c (and at c from a grid);
(th, ph) re-minimised with Nelder-Mead from the s1 values and from a 12x24 grid."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402

from src import evaluate as ev  # noqa: E402
import fit42 as F  # noqa: E402
import lib42 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
V, _, _ = ev.mass_ratio_targets()
s1 = json.loads((HERE / "fit_cm_NO_s1.json").read_text())["params_fit"]
PM = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
sc = ev.Scorer(PM, "NO", mass_floor=0.02, scale=F.CM_SCALE)
sv = lambda M: np.linalg.svd(M, compute_uv=False)[::-1]  # noqa: E731
out = {}
for c in [s1["c"], 0.01, 0.015, 0.0225, 0.03, -0.0187, -0.0225]:
    sb = np.sign(s1["b"]) if c == s1["c"] else 1.0
    best = None
    for sgn in ([sb] if c == s1["c"] else [1.0, -1.0]):
        f = lambda b: (lambda e: e[0] / e[1])(sv(L.cm_Me(b=sgn * b, c=c).T)) - V["mr_e_mu"]  # noqa: E731
        b = sgn * brentq(f, 1e-6, 0.05)
        base = {**F.cm_params(s1), "b": b, "c": c}

        def chi(x):
            p = dict(base, xi=np.cos(x[0]), eta=np.sin(x[0]) * np.cos(x[1]), etapp=np.sin(x[0]) * np.sin(x[1]))
            return sc.chi2_soft(L.cm_obs(p, conj=True), include_finite=True)
        starts = [[s1["th"], s1["ph"]]] + [[t, p] for t in np.linspace(0.1, 3.0, 12) for p in np.linspace(0, 6.2, 24)]
        vals = sorted(starts, key=chi)[:6]
        for x0 in vals:
            r = minimize(chi, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000})
            if best is None or r.fun < best[0]:
                p = dict(base, xi=np.cos(r.x[0]), eta=np.sin(r.x[0]) * np.cos(r.x[1]), etapp=np.sin(r.x[0]) * np.sin(r.x[1]))
                res = sc(L.cm_obs(p, conj=True))
                best = (r.fun, {"c": c, "b": b, "chi2_pmns": res.chi2, "pulls": res.pulls})
    out[str(c)] = best[1]
    print(json.dumps(best[1], default=float), flush=True)
(HERE / "cm_pmns_check.json").write_text(json.dumps(out, indent=1, default=float))

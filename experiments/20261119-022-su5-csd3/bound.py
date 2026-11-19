"""Analytic lower bound on the primary chi^2 (structure-only argument, see README).

1. Lepton block: U_e = 1 up to theta12e ~ 3e-4 rad, so PMNS + dm_ratio depend only on r.
   Its minimum over r is a lower bound on that part of chi^2.
2. Mass block: R = (m_e/m_mu)/(m_d/m_s) = 2 m_s^2/m_mu^2 >= 8/81 at the GUT scale for this texture
   (equality at Y^d_12 = 0, which kills the only CP phase). For a correlated Gaussian block,
   r^T C^-1 r >= (c.r)^2 / (c^T C c), so the mr block chi^2 >= R pull^2 computed with the
   block's variance of log R (both mr keys floored: 2% * sqrt(2) on R).
The two blocks are disjoint, so chi2_min >= sum.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import evaluate as ev  # noqa: E402
import fit as F  # noqa: E402

out = {}
model_fn = F.make_model()
lep = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
for an in ("IC23_woSKatm", "IC24_wSKatm"):
    for tb in (5, 10):
        sc = dict(F.SCALE, tan_beta=tb)
        fr = ev.fit_and_evaluate(lambda t: model_fn({**F.PAPER, "r": t["r"]}), {"r": (0.0, 1.0)}, "NO", lep,
                                 analysis=an, n_starts=8, mass_floor=F.FLOOR, scale=sc)
        rR = ev.evaluate({"R_clebsch": 8 / 81}, "NO", an, use=["R_clebsch"],
                         mass_floor={"R_clebsch": 0.02 * math.sqrt(2)}, scale=sc)
        b = fr.result.chi2 + rR.chi2
        from src.stats import p_value
        out[f"{an}_tb{tb}"] = {"lepton_chi2_min": fr.result.chi2, "r": fr.params["r"], "lepton_pulls": fr.result.pulls,
                               "R_min_pull": rR.pulls["R_clebsch"], "chi2_lower_bound": b,
                               "p_upper_bound_dof5": p_value(b, 14, 9)}
        print(an, tb, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[f"{an}_tb{tb}"].items()
                                  if k != "lepton_pulls"}))
Path("bound.json").write_text(json.dumps(out, indent=1))

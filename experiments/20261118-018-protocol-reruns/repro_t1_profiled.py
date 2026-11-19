"""T1 reproduction check: the pipeline's template with exp 010's input-error-profiled scorer."""
import json

import numpy as np

import pipelines as PL
import pipeline as P010  # exp 010 pipeline.py (ProfileScorer cache)
from src.nulls import Universe

u = Universe.real("NO")
j = 9412
base = PL.t1_base(u, "GUT")
rel = base.copy(); rel[3] = 1.0
dirs = PL._t1_rel_dirs("GUT") * rel[None, :]
import symlib as S
real = S.real_inputs("GUT")
out = {"base_maxreldiff_vs_exp010": float(np.max(np.abs(base / real["base"] - 1))), "expr": PL.t1_expr(j)}
lst = PL._t1_mapped(base, dirs)["up"]
for o in ("NO", "IO"):
    tot_prof = tot_plain = 0.0
    for k in (0, 1):
        x, dx = lst[k]
        v, T = PL._t1_eval(x, dx)
        p, s = S.predictions(v[j:j + 1], T[j:j + 1], PL.T1_ANG[k])
        tot_prof += float(P010.profile_scorer(PL.T1_ANG[k], o).dchi2(p, s)[0])
        tot_plain += float(P010.profile_scorer(PL.T1_ANG[k], o).dchi2(p, np.zeros_like(s))[0])
        out[f"{o}_{PL.T1_ANG[k]}"] = {"p": float(p[0]), "sigma": float(s[0])}
    out[f"{o}_chi2_profiled"] = tot_prof
    out[f"{o}_chi2_sigma0"] = tot_plain
print(json.dumps(out, indent=1))
(PL.HERE / "repro_t1_profiled.json").write_text(json.dumps(out, indent=1))

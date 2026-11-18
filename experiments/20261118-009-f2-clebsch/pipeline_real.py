"""Real-data value of the null pipeline (f2lib.search with its defaults, the exact function
run_null.py applies to every fake). Writes pipeline_real.json."""
import json

import numpy as np

import f2lib as L
from src import data

PM = ("s12sq", "s13sq", "s23sq", "dcp_deg")
cfgs = L.configs()
out = {}
for scale in ("MZ", "GUT"):
    P, Q, rc, _ = L.build_PQ(cfgs, L.ckm_at_scale(scale))
    tier = np.array([cfgs[i]["tier"] for i in rc])
    sc = {"tab_NO": L.TableChi2("NO"), "tab_IO": L.TableChi2("IO")}
    for o in ("NO", "IO"):
        p = data.load_pmns(o)
        sc[f"gauss_{o}"] = L.GaussChi2([p[k]["value"] for k in PM], o)
    out[scale] = L.search(P, Q, rc, tier, sc)
    # cached variant must agree with search() (used for null a)
    sc_a = {k: v for k, v in sc.items() if k.startswith("gauss")}
    out[scale + "_cached_check"] = L.search_cached(L.obs_cache(P, Q), P, Q, tier, sc_a)
with open(L.HERE / "pipeline_real.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))

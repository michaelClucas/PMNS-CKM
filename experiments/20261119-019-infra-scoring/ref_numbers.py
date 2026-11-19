"""Reference numbers for the exp 019 README: mr_* data at M_Z, splitting observables, running factors.

Run from the repo root: .venv/Scripts/python experiments/20261119-019-infra-scoring/ref_numbers.py
Writes ref_numbers.json next to this file.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from src import data, evaluate as ev, rg  # noqa: E402

out = {}
r = rg.mass_ratios()
out["mr_MZ"] = {k: {"value": r["value"][k], "rel_err": r["err"][k] / r["value"][k]} for k in r["names"]}
sd = np.sqrt(np.diag(r["cov_log"]))
out["mr_MZ_corr"] = np.round(r["cov_log"] / np.outer(sd, sd), 3).tolist()
y = rg.derived_yukawas_mz_unc()
out["yukawa_MZ_rel_err"] = {k: y["err"][k] / y["value"][k] for k in y["names"]}
for o in data.ORDERINGS:
    b = ev._dm_best(o, data.DEFAULT_ANALYSIS)
    out[f"dm_{o}"] = {"best": b, "dm_ratio_chi2_at_record": ev.dm_ratio_delta_chi2(o, b["dm_ratio"]),
                      "dm_ratio_chi2_at_+-3pct": [ev.dm_ratio_delta_chi2(o, b["dm_ratio"] * f) for f in (0.97, 1.03)]}
use = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg",
       "dm_ratio") + ev.MASS_RATIO_KEYS
for name, spec in (("SM_2e16", {"mu": 2e16}), ("MSSM_tb10_2e16", {"mu": 2e16, "model": "MSSM", "tan_beta": 10.0})):
    sc = ev.Scorer(use, "NO", scale=spec)
    out["running_" + name] = {"shift_or_factor": sc.scale_shift, "systematic": sc.scale_systematic}
with open(HERE / "ref_numbers.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(json.dumps({k: out[k] for k in ("mr_MZ", "dm_NO", "dm_IO")}, indent=1, default=float))
for k in ("running_SM_2e16", "running_MSSM_tb10_2e16"):
    print(k, {u: round(v, 5) for u, v in out[k]["shift_or_factor"].items()})
    print("  sys", {u: float(f"{v:.2g}") for u, v in out[k]["systematic"].items()})

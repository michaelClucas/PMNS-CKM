"""Random-tau comparison: fraction of random tau whose best S4 quark chi2 (reduced pipeline) is <= the
lepton tau's, with a two-sided 95% Clopper-Pearson interval. Writes null_summary.json.

    python analyze_null.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, "..", ".."))]

import numpy as np  # noqa: E402

from src.stats import clopper_pearson  # noqa: E402

rows = list(csv.DictReader(open(os.path.join(HERE, "null_random_tau.csv"))))
lep = [r for r in rows if r["kind"] == "lepton"][0]
rnd = [r for r in rows if r["kind"] == "random"]
c0 = float(lep["best_chi2"])
c = np.array([float(r["best_chi2"]) for r in rnd])
im = np.array([float(r["tau_im"]) for r in rnd])
out = {"lepton_tau": [float(lep["tau_re"]), float(lep["tau_im"])], "lepton_best_chi2": c0,
       "n_random": len(c), "random_best_chi2_quantiles": dict(zip(["min", "q10", "q50", "q90", "max"],
                                                                np.quantile(c, [0, .1, .5, .9, 1]).tolist()))}
for name, sel in (("all", np.ones(len(c), bool)), ("im_le_2", im <= 2)):
    k = int((c[sel] <= c0).sum())
    n = int(sel.sum())
    lo, hi = clopper_pearson(k, n)
    out[f"rate_{name}"] = {"k": k, "n": n, "rate": k / n if n else float("nan"), "cp95": [float(lo), float(hi)]}
out["rank_of_lepton_tau_among_all"] = int((c < c0).sum()) + 1
out["n_random_with_p_ge_0.05_possible"] = "none: chi2 thresholds for dof 2-4 are < 10"
out["random_best_overall"] = min(rnd, key=lambda r: float(r["best_chi2"]))
json.dump(out, open(os.path.join(HERE, "null_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

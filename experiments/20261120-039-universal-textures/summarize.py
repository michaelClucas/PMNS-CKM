"""Exp 039: best fit per (class, options, scale) from fits.jsonl -> summary.json and a printed table.
chi2 is the official src.evaluate value (inf when a PMNS prediction lies outside the NuFIT tables);
chi2_soft_gauss = DE objective - 1e4 is the scorer's Gaussian fallback (chi2_soft) at the optimum,
reported only to say how far outside the tables the best point is."""
from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / "fits.jsonl")]
best = {}
for r in rows:
    if "error" in r:
        print("ERROR", r)
        continue
    key = (r["cls"], json.dumps(r["opt"], sort_keys=True), r["scale"])
    soft = r["de_chi2"] - 1e4 if r["de_chi2"] >= 1e4 else r["de_chi2"]
    r["chi2_soft_gauss"] = soft if not np.isfinite(r["chi2"]) else r["chi2"]
    if key not in best or (r["chi2"], r["chi2_soft_gauss"]) < (best[key]["chi2"], best[key]["chi2_soft_gauss"]):
        best[key] = r
out = []
for (c, o, s), r in sorted(best.items()):
    worst = sorted(r["pulls"].items(), key=lambda kv: -abs(kv[1]))[:3]
    row = dict(cls=c, opt=json.loads(o), scale=s, N=r["N"], k=r["k"], k_cont=r["k_cont"], k_discrete=r["k_discrete"],
               chi2=r["chi2"], chi2_soft_gauss=r["chi2_soft_gauss"], p_global=r["p_global"], max_pull=r["max_pull"],
               compression_ok=r["compression_ok"], fit_ok=r["fit_ok"], worst_pulls=worst, pred=r["pred"],
               params=r["params"])
    out.append(row)
    print(f"{c:6s} {o:32s} {s:3s} N {r['N']} k {r['k']:2d} ({r['k_cont']}+{r['k_discrete']}) chi2 {r['chi2']:.4g} "
          f"(gauss {r['chi2_soft_gauss']:.4g}) p_glob {r['p_global']:.2g} comp {r['compression_ok']} fit {r['fit_ok']} "
          + " ".join(f"{k}:{v:.3g}" for k, v in worst))
json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=float)

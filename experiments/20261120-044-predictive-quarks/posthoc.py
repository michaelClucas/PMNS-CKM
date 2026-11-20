"""Exp 044 post-processing: GUT-cut check, miss-rate detail, survivor table, lepton results -> posthoc.json"""
import json

import numpy as np

from main import ART, HERE, L
from src import evaluate as ev

out = {}
# GUT cut check: did any of the random check sample (MZ chi2 >= 200) pass at a GUT scale?
chk = set(json.load(open(ART / "gut_check_ids_A.json")))
for sc in ("GUT10", "GUT30"):
    rows = [json.loads(l) for l in open(ART / f"fits_A_{sc}.jsonl")]
    c = [r for r in rows if r["id"] in chk]
    out[f"gut_check_{sc}"] = {"n": len(c), "pass": sum(bool(r.get("fit_ok")) for r in c),
                              "min_chi2": float(min(r["chi2"] for r in c))}
# miss-rate detail
tm = [json.loads(l) for l in open(ART / "thorough_miss_A_MZ.jsonl")]
out["miss"] = {"n": len(tm), "pass": sum(bool(r.get("fit_ok")) for r in tm),
               "min_thorough_chi2": float(min(r["chi2"] for r in tm)),
               "median_ratio_thorough_over_fast": float(np.median([r["chi2"] / r["fast_chi2"] for r in tm]))}
# proxy vs official on near refits
rel = []
for sc in ("MZ", "GUT10", "GUT30"):
    for l in open(ART / f"thorough_near_A_{sc}.jsonl"):
        r = json.loads(l)
        if np.isfinite(r["chi2"]) and r["chi2"] > 1e-6:
            rel.append(abs(r["proxy_at_x"] - r["chi2"]) / r["chi2"])
out["proxy_vs_official"] = {"n": len(rel), "max_rel": float(max(rel)), "median_rel": float(np.median(rel))}
# survivors: best scale per entry
surv = json.load(open(HERE / "survivors.json"))
best = {}
for s in surv:
    if s["id"] not in best or s["chi2"] < best[s["id"]]["chi2"]:
        best[s["id"]] = s
table = []
for s in sorted(best.values(), key=lambda s: s["chi2"]):
    table.append({"id": s["id"], "texture": L.spec_id(s["spec"]), "scales_pass": sorted({t["scale"] for t in surv
                                                                                           if t["id"] == s["id"]}),
                  "best_scale": s["scale"], "k_cont": s["k"], "predicts": 8 - s["k"], "chi2": round(s["chi2"], 3),
                  "p": round(s["p"], 3), "max_pull": round(s["max_pull"], 2),
                  "pulls": {k: round(v, 2) for k, v in s["pulls"].items()},
                  "x": [round(v, 4) for v in s["x"]], "fingerprint_class": s["fingerprint_class"]})
out["survivors"] = table
# leptons
lep = []
for tag in "abc":
    f = ART / f"lepton_{tag}.json"
    if f.exists():
        lep += json.load(open(f))
sc = ev.Scorer(("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"), "NO")
lrows = []
for r in lep:
    p = r["pred"]
    pm = {k: p[k] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")}
    lrows.append({"id": r["id"], "scale": r["scale"], "mr": r["mr"], "k_cont": r["k_cont"], "chi2_N15": r["chi2"],
                  "pmns_gauss_soft": round(sc.chi2_soft(pm, include_finite=True) - ev.CHI2_CAP, 1)
                  if not np.isfinite(sc.chi2(pm)) else round(sc.chi2(pm), 2),
                  "pred": {k: round(v, 5) for k, v in p.items() if k in pm or k in ("mr_e_mu", "mr_mu_tau")},
                  "heldout_pmns_chi2": r["heldout"].get("pmns_chi2")})
out["lepton"] = lrows
out["lepton_n_finite"] = int(sum(np.isfinite(r["chi2_N15"]) for r in lrows))
json.dump(out, open(HERE / "posthoc.json", "w"), indent=1)
for k in ("gut_check_GUT10", "gut_check_GUT30", "miss", "proxy_vs_official", "lepton_n_finite"):
    print(k, out[k])
for t in table:
    print(t["id"], t["texture"], t["scales_pass"], t["chi2"], t["p"], t["max_pull"], t["pulls"])
for r in lrows:
    print(r["id"], r["scale"], r["mr"], r["chi2_N15"], r["pmns_gauss_soft"], r["pred"])

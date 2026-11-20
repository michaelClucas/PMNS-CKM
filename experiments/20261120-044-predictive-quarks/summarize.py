"""Exp 044 summary: filter kill fractions, fit pass counts per scale and k, survivors table.
Writes summary.json and survivors.json (small) in this folder."""
from __future__ import annotations

import collections
import json
import math

import numpy as np
from scipy.stats import beta

from main import ART, HERE, L, load

BITS_A = math.log2(64438)
BITS_G44 = math.log2(20402378)
SCALE_BITS = 2


def cp95(x, n):
    lo = 0.0 if x == 0 else beta.ppf(0.025, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(0.975, x + 1, n - x)
    return [float(lo), float(hi)]


def main():
    out = {"bits": {"A": BITS_A, "G44": BITS_G44, "scale": SCALE_BITS}}
    surv = []
    for g in ("A", "B"):
        fn = ART / f"filter_{g}.jsonl"
        if not fn.exists():
            continue
        specs = {sp["id"]: sp for sp in load(g)}
        fl = [json.loads(l) for l in open(fn)]
        n = len(fl)
        cp = sum(r["cp_dead"] for r in fl)
        per = {sc: sum(r["alive"][sc] for r in fl) for sc in ("MZ", "GUT10", "GUT30")}
        anyal = sum(any(r["alive"].values()) for r in fl)
        bytk = collections.Counter((specs[r["id"]]["t"], r["k"]) for r in fl)
        aliv_tk = collections.Counter((specs[r["id"]]["t"], r["k"]) for r in fl if any(r["alive"].values()))
        G = {"n": n, "cp_killed": cp, "sector_alive": per, "alive_any_scale": anyal,
             "by_type_k": {f"{t}{k}": [bytk[(t, k)], aliv_tk[(t, k)]] for t, k in sorted(bytk)}}
        fits = {}
        for sc in ("MZ", "GUT10", "GUT30"):
            ff = ART / f"fits_{g}_{sc}.jsonl"
            if not ff.exists():
                continue
            rows = [json.loads(l) for l in open(ff)]
            ok = [r for r in rows if r.get("fit_ok")]
            c2 = np.array([r["chi2"] for r in rows])
            fits[sc] = {"fitted": len(rows), "of_alive": per[sc], "pass": len(ok), "cp95": cp95(len(ok), len(rows)),
                        "chi2_quantiles": [float(np.nanquantile(np.where(np.isfinite(c2), c2, 1e9), q))
                                           for q in (0.0, 0.01, 0.1, 0.5)],
                        "n_chi2_lt_50": int(np.sum(c2 < 50)), "n_chi2_lt_100": int(np.sum(c2 < 100))}
            for r in ok:
                surv.append({"grammar": g, "scale": sc, "spec": specs[r["id"]], **{k: r[k] for k in (
                    "id", "k", "chi2", "p", "max_pull", "pulls", "x", "pred")}})
            th = ART / f"thorough_near_{g}_{sc}.jsonl"
            if th.exists():
                tr = [json.loads(l) for l in open(th)]
                fits[sc]["thorough_near"] = {"n": len(tr), "pass": sum(bool(r.get("fit_ok")) for r in tr),
                                             "proxy_vs_official_max_rel": float(np.nanmax(
                                                 [abs(r["proxy_at_x"] - r["chi2"]) / max(r["chi2"], 1e-9)
                                                  for r in tr if np.isfinite(r["chi2"])] or [0]))}
                for r in tr:
                    if r.get("fit_ok") and not any(s["id"] == r["id"] and s["scale"] == sc for s in surv):
                        surv.append({"grammar": g, "scale": sc, "spec": specs[r["id"]], "via": "thorough",
                                     **{k: r[k] for k in ("id", "k", "chi2", "p", "max_pull", "pulls", "x", "pred")}})
            tm = ART / f"thorough_miss_{g}_{sc}.jsonl"
            if tm.exists():
                tr = [json.loads(l) for l in open(tm)]
                fits[sc]["thorough_miss"] = {"n": len(tr), "now_pass": sum(bool(r.get("fit_ok")) for r in tr),
                                             "chi2_improved_by_gt_10pct": sum(r["chi2"] < 0.9 * r["fast_chi2"] for r in tr)}
        G["fits"] = fits
        out[g] = G
    for s in surv:
        s["predicts"] = 8 - s["k"]
        s["k_honest_A"] = s["k"] + BITS_A + SCALE_BITS
        s["k_honest_G44"] = s["k"] + BITS_G44 + SCALE_BITS
    # numeric fingerprint dedupe of survivors
    fp = {}
    for s in surv:
        s["fingerprint_class"] = fp.setdefault(L.fingerprint(s["spec"]), len(fp))
    out["n_survivors"] = len(surv)
    out["n_survivor_classes"] = len(fp)
    json.dump(out, open(HERE / "summary.json", "w"), indent=1)
    json.dump(surv, open(HERE / "survivors.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])
    for s in sorted(surv, key=lambda s: s["chi2"])[:40]:
        print(s["grammar"], s["scale"], s["id"], L.spec_id(s["spec"]), "k", s["k"], "chi2", round(s["chi2"], 2),
              "p", round(s["p"], 3), "maxpull", round(s["max_pull"], 2))


if __name__ == "__main__":
    main()

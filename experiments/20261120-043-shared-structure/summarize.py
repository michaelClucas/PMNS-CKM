"""fits.jsonl (+ screen.json, bank.json) -> summary.json and printed tables."""
import json
from collections import defaultdict

import numpy as np

import lib43 as L
from src.stats import clopper_pearson

import sys
R = [json.loads(x) for x in open(sys.argv[1] if len(sys.argv) > 1 else "../../artifacts/exp043/fits_full.jsonl")]
G = defaultdict(list)
for r in R:
    G[r["group"]].append(r)
    G[f"{r['group']}/|Z|={len(r['spec']['Z'])}"].append(r)
    if r["group"] == "C1-S":
        G[f"C1-S/{r.get('sub')}"].append(r)
    if r["group"] == "C2-S":
        G[f"C2-S/{r.get('sub')}"].append(r)
        G[f"C2-S/{r['spec']['nu']}"].append(r)


def worst(r, n=3):
    p = sorted(r["pulls"].items(), key=lambda kv: -abs(kv[1]))[:n]
    return ", ".join(f"{k} {v:+.1f}" for k, v in p)


out = {"groups": {}}
print(f"{'group':14s} {'n':>4s} {'pass':>4s} {'frac':>6s} {'CP95':>16s} {'l':>7s} {'k':>7s} "
      f"{'med chi2':>9s} {'min chi2':>9s} best (worst pulls)")
for g in sorted(G):
    rs = G[g]
    n, k = len(rs), sum(r["pass"] for r in rs)
    lo, hi = clopper_pearson(k, n)
    chi = np.array([r["chi2"] for r in rs])
    gauss = np.array([r["proxy_chi2"] for r in rs])
    fin = chi[np.isfinite(chi)]
    b = min(rs, key=lambda r: (not r["pass"], r["max_pull"], r["proxy_chi2"]))
    ls = sorted({r["l_e"] + r["l_nu"] for r in rs})
    ks = sorted({r["k_cont"] for r in rs})
    rec = {"n": n, "pass": k, "frac": k / n, "cp95": [lo, hi], "l": ls, "k_cont": ks,
           "n_finite_chi2": int(len(fin)), "median_proxy_chi2": float(np.median(gauss)),
           "min_official_chi2": float(fin.min()) if len(fin) else None,
           "best": {"spec": b["spec"], "chi2": b["chi2"], "max_pull": b["max_pull"], "proxy_chi2": b["proxy_chi2"],
                    "pulls": b["pulls"], "pred": b["pred"]},
           "n_fit_ok": sum(r["fit_ok"] for r in rs)}
    out["groups"][g] = rec
    print(f"{g:14s} {n:4d} {k:4d} {k / n:6.3f} [{lo:.3f}, {hi:.3f}] {str(ls[0]) + '-' + str(ls[-1]):>7s} "
          f"{str(ks[0]) + '-' + str(ks[-1]):>7s} {np.median(gauss):9.3g} "
          f"{(fin.min() if len(fin) else np.inf):9.3g} {b['chi2']:.3g} maxpull {b['max_pull']:.2f} ({worst(b)})")

# pass rate vs number of lepton-only parameters (Y2 groups)
byl = defaultdict(list)
for r in R:
    if r["keys"] == "Y2":
        byl[r["l_e"] + r["l_nu"]].append(r["pass"])
# pass rate vs k_cont - N (excess of continuous parameters over scored observables)
byx = defaultdict(list)
for r in R:
    byx[(r["keys"], r["k_cont"] - (10 if r["keys"] == "Y1" else 15))].append(r["pass"])
out["by_excess"] = {}
print("\npass rate by (set, k_cont - N)")
for key in sorted(byx):
    n, k = len(byx[key]), sum(byx[key])
    lo, hi = clopper_pearson(k, n)
    out["by_excess"][f"{key[0]} {key[1]:+d}"] = {"n": n, "pass": k, "cp95": [lo, hi]}
    print(f"{key[0]} k-N {key[1]:+3d}: {k}/{n}  CP95 [{lo:.3f}, {hi:.3f}]")
out["Y2_by_l"] = {}
print("\nY2 pass rate by lepton-only parameter count l")
for l in sorted(byl):
    n, k = len(byl[l]), sum(byl[l])
    lo, hi = clopper_pearson(k, n)
    out["Y2_by_l"][l] = {"n": n, "pass": k, "cp95": [lo, hi]}
    print(f"l {l:2d}: {k}/{n}  CP95 [{lo:.3f}, {hi:.3f}]")

# C1 per pattern
print("\nC1-S per pattern (profiled): best official max pull")
out["C1_by_Z"] = {}
byz = defaultdict(list)
for r in G["C1-S"]:
    byz["".join(r["spec"]["Z"])].append(r)
for z, rs in sorted(byz.items()):
    b = min(rs, key=lambda r: (r["max_pull"], r["chi2"]))
    out["C1_by_Z"][z] = {"n": len(rs), "pass": sum(r["pass"] for r in rs), "best_C": b["spec"]["ce"],
                         "chi2": b["chi2"], "max_pull": b["max_pull"], "pulls": b["pulls"]}
    print(z, len(rs), "pass", sum(r["pass"] for r in rs), "best C", b["spec"]["ce"], f"chi2 {b['chi2']:.3g}",
          f"maxpull {b['max_pull']:.2f}", worst(b, 4))

# proxy validation: proxy chi2 vs official chi2 at the optima
pairs = [(r["proxy_chi2"], r["chi2"], r["proxy_maxres"] < 2, r["pass"], r["polished"]) for r in R]
fin = [(a, b) for a, b, *_ in pairs if np.isfinite(b)]
a, b = np.array(fin).T if fin else (np.array([]), np.array([]))
rel = np.abs(a - b) / np.maximum(b, 1)
agree = sum((p == q) for _, _, p, q, pol in pairs if not pol)
npol = sum(1 for *_, pol in pairs if not pol)
out["proxy_validation"] = {"n_finite": int(len(a)), "median_rel_diff": float(np.median(rel)) if len(a) else None,
                           "q90_rel_diff": float(np.quantile(rel, 0.9)) if len(a) else None,
                           "spearman": float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
                           if len(a) > 2 else None,
                           "pass_agreement_unpolished": f"{agree}/{npol}",
                           "n_polished": sum(1 for *_, pol in pairs if pol),
                           "n_infinite_official": int(sum(1 for _, b_, *__ in pairs if not np.isfinite(b_)))}
print("\nproxy validation", out["proxy_validation"])
json.dump(out, open("summary.json", "w"), indent=1, default=float)

"""Exp 045 summary: best fit per (structure, scale) over seeds, eta and scale profiles, ranked table.

Writes summary.json and prints the tables used in README.md (summarize.log).
p-values: chi^2 survival with dof = 8 - k_cont (k_cont counts eta_b or the fitted scale where used).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import chi2 as C2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import structures as S  # noqa: E402

N = 8


def load(fn):
    p = HERE / fn
    return [json.loads(l) for l in open(p)] if p.exists() else []


def p_of(c, k):
    return float(C2.sf(c, N - k)) if N - k > 0 else float("nan")


fits = [d for d in load("fits.jsonl") if "error" not in d and not d.get("tag")]
errs = [d for d in load("fits.jsonl") if "error" in d]
best, seeds = {}, {}
for d in fits:
    key = (d["name"], d["scale"])
    seeds.setdefault(key, []).append(d["chi2"])
    if key not in best or d["chi2"] < best[key]["chi2"]:
        best[key] = d

eta = {}
for d in load("eta.jsonl"):
    if d["name"] not in eta or d["chi2"] < eta[d["name"]]["chi2"]:
        eta[d["name"]] = d
miss = {(d["name"], d["scale"]): d for d in load("miss.jsonl")}


def worst(pulls, n=3):
    ks = sorted(pulls, key=lambda k: -abs(pulls[k]))[:n]
    return ", ".join(f"{k} {pulls[k]:+.1f}" for k in ks)


rows = []
names = list(S.TEXTURES) + list(S.RELATIONS) + ["HS"]
print("Per-scale best fits (chi2 over seeds; spread = max - min over seeds)")
print("| structure | scale | k_cont + bits | chi2 | p | seeds spread | worst pulls |")
print("|---|---|---|---|---|---|---|")
for n in names:
    for sc in ("MZ", "G10", "G30", "SM3", "SM5", "SM7", "SM9", "SM12", "SM16"):
        if (n, sc) not in best:
            continue
        d = best[(n, sc)]
        kc = d["k"] - d["k_discrete"]
        sp = max(seeds[(n, sc)]) - min(seeds[(n, sc)])
        print(f"| {n} | {sc} | {kc} + {d['k_discrete']} | {d['chi2']:.2f} | {p_of(d['chi2'], kc):.3g} | "
              f"{sp:.2g} ({len(seeds[(n, sc)])}) | {worst(d['pulls'])} |")
    if n in eta:
        e = eta[n]
        kc = e["k_fit"] + 1
        print(f"| {n} | G30 + eta_b {e['eta_b']:+.2f} | {kc} + 0 | {e['chi2']:.2f} | {p_of(e['chi2'], kc):.3g} | local | "
              f"{worst(e['pulls'])} |")

# primary verdict row per structure: best p among the options its paper allows, with honest k
STATED = {n: "GUT" for n in ("DHR", "RRR1", "RRR2", "RRR3", "RRR4", "RRR5")}
STATED.update({n: "MZ" for n in ("GR1", "LG_S8", "LG_S9", "LG_S10", "LG_S12", "LG_S13", "LG_S15", "F4")})
STATED.update({n: "MZ" for n in S.RELATIONS})
STATED.update({"F6": "ANY", "BB": "SCAN", "HS": "SCAN"})
summary = {}
for n in names:
    opts = []  # (label, chi2, k_cont, bits, pulls)
    st = STATED[n]
    if st in ("MZ", "ANY") and (n, "MZ") in best:
        d = best[(n, "MZ")]
        opts.append(("M_Z", d["chi2"], d["k"] - d["k_discrete"], d["k_discrete"], d["pulls"]))
    if st in ("GUT", "ANY"):
        for sc in ("G10", "G30"):
            if (n, sc) in best:
                d = best[(n, sc)]
                opts.append((f"GUT tanb {sc[1:]}", d["chi2"], d["k"] - d["k_discrete"], d["k_discrete"] + 1, d["pulls"]))
        if n in eta:
            e = eta[n]
            opts.append((f"GUT tanb 30, eta_b {e['eta_b']:+.2f}", e["chi2"], e["k_fit"] + 1, 1, e["pulls"]))
    if n == "BB":  # paper's benchmark scales 1e3, 1e6, 1e16 GeV (Sec. 4.2.3): a 3-way choice, 2 bits, not fitted
        for sc in ("SM3", "SM5", "SM16"):
            if (n, sc) in best:
                d = best[(n, sc)]
                opts.append((f"SM mu 1e{sc[2:]} (benchmark)", d["chi2"], d["k"] - d["k_discrete"], 2, d["pulls"]))
    elif st == "SCAN":
        sm = [best[(n, f"SM{e}")] for e in (3, 5, 7, 9, 12, 16) if (n, f"SM{e}") in best]
        if sm:
            d = min(sm, key=lambda d: d["chi2"])
            opts.append((f"SM mu {d['scale'][2:]} (scale fitted)", d["chi2"], d["k"] - d["k_discrete"] + 1, d["k_discrete"],
                         d["pulls"]))
    if not opts:
        continue
    lab, c, kc, bits, pulls = max(opts, key=lambda o: (p_of(o[1], o[2]) if N - o[2] > 0 else -1, -o[1]))
    summary[n] = {"option": lab, "chi2": c, "k_cont": kc, "bits": bits, "p": p_of(c, kc), "pulls": pulls,
                  "max_pull": max(abs(v) for v in pulls.values()), "options": [(o[0], o[1], o[2]) for o in opts]}

print("\nPrimary rows (best allowed option per structure)")
for n, s in sorted(summary.items(), key=lambda kv: (-(kv[1]["p"] if kv[1]["p"] == kv[1]["p"] else -1), kv[1]["chi2"])):
    m = miss.get((n, "MZ")) or miss.get((n, "G30")) or miss.get((n, "G10"))
    ms = f"; drop {m['dropped']}: chi2_7 {m['chi2_7']:.1f}, its pull at 7-key fit {m['pull_dropped_at_7fit']:+.1f}" if m else ""
    print(f"{n:8s} {s['option']:30s} k {s['k_cont']}+{s['bits']} chi2 {s['chi2']:.2f} p {s['p']:.3g} "
          f"max|pull| {s['max_pull']:.1f} [{worst(s['pulls'])}]{ms}")
json.dump({"summary": summary, "miss": {f"{k[0]}@{k[1]}": v for k, v in miss.items()}, "errors": errs},
          open(HERE / "summary.json", "w"), indent=1)
print("\nerrors:", len(errs))

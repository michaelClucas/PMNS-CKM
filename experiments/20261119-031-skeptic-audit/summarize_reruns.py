"""Exp 031: collect the DE rerun JSON (artifacts/031/) into reruns_summary.json and a markdown table.

usage (repo root): .venv/Scripts/python experiments/20261119-031-skeptic-audit/summarize_reruns.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "031"
rows = []
for f in sorted(ART.glob("bsv_*.json")):
    d = json.loads(f.read_text())
    rows.append({"exp": "015 BSV", "fit": f"{d['variant']} {d['scenario']} NO", "seed": d["seed"],
                 "committed_chi2": d["committed_best_chi2_gut"], "de_chi2": d["best_chi2_gut_de"],
                 "note": f"fit-stage chi2_gut, N 18; max nuisance pull {d['best_max_abs_nuisance_pull']:.1f}; "
                         f"DE popsize {d.get('popsize', 20)} maxiter {d['maxiter']}",
                 "pmns_gut": d["best_pmns_gut"]})
for f in sorted(ART.glob("twoO_de_s*.json")):
    d = json.loads(f.read_text())
    if d["maxiter"] < 100:  # smoke test
        continue
    b = d["best"]
    worst = max(b["pulls"].items(), key=lambda kv: abs(kv[1]))
    rows.append({"exp": "016 2O", "fit": f"joint, N 15{' (committed point in DE init)' if d['use_x0'] else ''}",
                 "seed": d["seed"], "committed_chi2": d["committed_json_chi2"], "de_chi2": b["chi2"],
                 "note": f"tau {b['tau'][0]:+.3f}+{b['tau'][1]:.3f}i; worst pull {worst[0]} {worst[1]:+.1f}",
                 "pmns_gut": None})
(pathlib.Path(__file__).with_name("reruns_summary.json")).write_text(json.dumps(rows, indent=1))
print("| exp | fit | seed | committed χ² | DE χ² | note |")
print("|---|---|---|---|---|---|")
for r in rows:
    c = "-" if r["committed_chi2"] is None else f"{r['committed_chi2']:.1f}"
    print(f"| {r['exp']} | {r['fit']} | {r['seed']} | {c} | {r['de_chi2']:.1f} | {r['note']} |")

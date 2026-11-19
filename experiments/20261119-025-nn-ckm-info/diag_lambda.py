"""Diagnostic: N1 fake statistic T (main flows, point) vs the fake's Wolfenstein parameters.
Reads results.json; regenerates the N1 draws with analyze.n1_ckms (same seeds)."""
import json
import sys
import numpy as np
from analyze import n1_ckms

r = json.load(open("results.json"))
T = np.array(r["main"]["fake_T_point_bits"])
Treal = r["main"]["T_point_bits"]
_, w = n1_ckms(len(T))
out = {}
for name, j, edges in (("lambda", 0, [0.05, 0.15, 0.2, 0.25, 0.3, 0.5]), ("A", 1, [0.3, 0.6, 0.9, 1.2, 2.0])):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (w[:, j] >= a) & (w[:, j] < b)
        rows.append(dict(bin=[a, b], n=int(m.sum()), median_T=float(np.median(T[m])), frac_ge_real=float(np.mean(T[m] >= Treal))))
    out[name] = rows
near = (np.abs(w[:, 0] - 0.225) < 0.03) & (np.abs(w[:, 1] - 0.83) < 0.2)
out["near_real_lambda_A"] = dict(n=int(near.sum()), frac_ge_real=float(np.mean(T[near] >= Treal)), median_T=float(np.median(T[near])))
print(json.dumps(out, indent=1))
json.dump(out, open("diag_lambda.json", "w"), indent=1)

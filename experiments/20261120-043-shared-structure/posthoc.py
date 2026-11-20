"""Post hoc (not pre-registered): C1-S pass rates by subset and |Z|; quark-Jacobian rank at the
stage-0 solutions (is there a quark-flat direction that the charged leptons can use?)."""
import json
from collections import defaultdict
import numpy as np
import lib43 as L
from src.stats import clopper_pearson

import sys
R = [json.loads(x) for x in open(sys.argv[1] if len(sys.argv) > 1 else "../../artifacts/exp043/fits_full.jsonl")]
out = {}
c = defaultdict(lambda: [0, 0])
for r in R:
    if r["group"] == "C1-S":
        key = f"{r['sub']} |Z|={len(r['spec']['Z'])}"
        c[key][0] += r["pass"]; c[key][1] += 1
for k, (p, n) in sorted(c.items()):
    lo, hi = clopper_pearson(p, n)
    out[f"C1-S {k}"] = {"pass": p, "n": n, "cp95": [lo, hi]}
    print("C1-S", k, f"{p}/{n} [{lo:.3f}, {hi:.3f}]")
# passes with screen chi2 among random subset
bank = json.load(open("bank.json"))
P = L.Proxy(L.QUARK_KEYS)
for b in bank:
    if not b["viable"]:
        continue
    Z = b["Z"]; spec = {"Z": Z, "e": "S", "ce": [1.0] * (len(Z) + 1), "nu": None}
    svs = []
    for x in b["bank"][:5]:
        J = L._fd_jac(spec, P, np.array(x), h=1e-6)
        s = np.linalg.svd(J, compute_uv=False)
        svs.append(int((s > 1e-6 * s[0]).sum()))
    out["rank " + "".join(Z)] = {"k_quark": 3 * len(Z), "jacobian_rank": svs}
    print("".join(Z), "quark params", 3 * len(Z), "Jacobian rank (8 obs) at bank points", svs)
# Y2 passes with k_cont < N
lowk = [r for r in R if r["keys"] == "Y2" and r["pass"] and r["k_cont"] < 15]
for r in lowk:
    print("Y2 pass k<N:", r["group"], r["spec"], r["k_cont"], r["l_e"] + r["l_nu"], f"chi2 {r['chi2']:.2f} maxpull {r['max_pull']:.2f}")
out["Y2_pass_k_lt_N"] = [{"group": r["group"], "spec": r["spec"], "k_cont": r["k_cont"], "chi2": r["chi2"],
                          "max_pull": r["max_pull"], "pulls": r["pulls"]} for r in lowk]
json.dump(out, open("posthoc.json", "w"), indent=1)

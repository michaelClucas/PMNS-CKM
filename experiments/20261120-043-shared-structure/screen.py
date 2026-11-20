"""C1 profiling screen: every charged-lepton Clebsch matrix C in CSET^(|Z|+1) (c33 > 0: the overall
sign of Y_e is unphysical) at the stage-0 quark solutions. Screen chi2 = whitened (mr_e_mu, mr_mu_tau)
with quark parameters fixed, minimised over the bank. Writes screen.json (per Z: best C list, a random
screened-out sample, the bank indices that gave each minimum, and the screen chi2 distribution)."""
import itertools
import json
import sys

import numpy as np

import lib43 as L

N_TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_RAND = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_BANK = 20
rng = np.random.default_rng(4301)
Winv, mrlog = L._mr_whitener(("mr_e_mu", "mr_mu_tau"))
bank = json.load(open("bank.json"))
out = []
for b in bank:
    if not b["viable"]:
        continue
    Z = b["Z"]
    nz = len(Z)
    ZZ = Z + ["33"]
    pos = [L.CSET[L.CSET > 0] if i == nz else L.CSET for i in range(nz + 1)]
    combos = np.array(list(itertools.product(*pos)))
    sols = b["bank"][:N_BANK]
    spec = {"Z": Z, "e": "S", "ce": [1.0] * (nz + 1), "nu": None}
    best = np.full(len(combos), np.inf)
    arg = np.zeros(len(combos), int)
    for j, x in enumerate(sols):
        Yd = L.build(spec, np.array([x]))[1][0]
        for s in range(0, len(combos), 100000):
            C = combos[s:s + 100000]
            Ce = L._place(len(C), C.astype(complex), ZZ)
            Ye = Ce * Yd[None]
            w = np.sqrt(np.clip(np.linalg.eigvalsh(Ye @ L._dag(Ye)), 0, None))
            with np.errstate(all="ignore"):
                z = np.log(np.stack([w[:, 0] / w[:, 1], w[:, 1] / w[:, 2]], 1)) - mrlog
            y = z @ Winv.T
            c2 = np.where(np.isfinite(y).all(1), (y ** 2).sum(1), np.inf)
            m = c2 < best[s:s + 100000]
            best[s:s + 100000][m] = c2[m]
            arg[s:s + 100000][m] = j
    order = np.argsort(best)
    top = order[:N_TOP]
    rest = order[N_TOP:]
    rnd = rng.choice(rest, min(N_RAND, len(rest)), replace=False)
    q = np.quantile(best[np.isfinite(best)], [0, 0.001, 0.01, 0.1, 0.5])
    rec = {"Z": Z, "n_combos": int(len(combos)), "n_bank": len(sols),
           "screen_quantiles": {"min": q[0], "q0.001": q[1], "q0.01": q[2], "q0.1": q[3], "median": q[4]},
           "n_below_4": int((best < 4).sum()), "n_below_25": int((best < 25).sum()),
           "top": [{"C": combos[i].tolist(), "screen": float(best[i]), "bank_idx": int(arg[i])} for i in top],
           "random": [{"C": combos[i].tolist(), "screen": float(best[i]), "bank_idx": int(arg[i])} for i in rnd]}
    out.append(rec)
    print("".join(Z), len(combos), "screen min %.3g" % q[0], "n<4", rec["n_below_4"], "n<25", rec["n_below_25"],
          "top C", rec["top"][0]["C"], flush=True)
json.dump(out, open("screen.json", "w"))
print("screen done")

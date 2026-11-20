"""Stage 0: quark-only fits per pattern Z (claims.md 'Quark-viability conditioning').
Writes bank.json: per Z, official best quark fit and a bank of quark solutions (proxy chi2 < 6).
Usage: python stage0.py [n_starts] [n_workers]"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import lib43 as L

N_STARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def patterns():
    return [list(c) for n in (3, 4) for c in itertools.combinations(L.P_ALL, n)]


# hierarchical log10 start ranges per position (theory-space prior, not fitted numbers)
RU = {"11": (-7, -5), "12": (-6, -3), "13": (-6, -2.5), "22": (-4, -1.5), "23": (-3, -0.5)}
RD = {"11": (-6, -3), "12": (-5, -1.5), "13": (-5, -1.5), "22": (-3.5, -0.5), "23": (-2, -0.2)}


def start(Z, rng):
    nz = len(Z)
    u = rng.choice([-1, 1], nz) * (np.array([rng.uniform(*RU[z]) for z in Z]) + 7.0)
    d = np.array([rng.uniform(*RD[z]) for z in Z])
    return np.concatenate([u, d, rng.uniform(-np.pi, np.pi, nz)])


def run(Z):
    t = time.time()
    rng = np.random.default_rng(int("".join(Z)))
    spec = {"Z": Z, "e": "S", "ce": [1.0] * (len(Z) + 1), "nu": None}
    nz = len(Z)
    sols = []
    for s in range(N_STARTS):
        x0 = start(Z, rng)
        x, c, mr = L.fit_one(spec, L.QUARK_KEYS, x0)
        sols.append((c, mr, x))
    sols.sort(key=lambda r: r[0])
    off = L.official(spec, L.QUARK_KEYS, sols[0][2])
    bank = [list(map(float, x)) for c, mr, x in sols if c < 6]
    return {"Z": Z, "best_proxy": sols[0][0], "best_maxres": sols[0][1], "official_chi2": off["chi2"],
            "official_max_pull": off["max_pull"], "viable": bool(off["max_pull"] < 2), "pulls": off["pulls"],
            "n_bank": len(bank), "bank": bank, "start_chi2": [r[0] for r in sols], "seconds": time.time() - t}


if __name__ == "__main__":
    t0 = time.time()
    with Pool(WORKERS) as p:
        out = p.map(run, patterns(), chunksize=1)
    for r in out:
        print("".join(f"{z} " for z in r["Z"]), f"proxy {r['best_proxy']:.3g} official {r['official_chi2']:.3g}"
              f" maxpull {r['official_max_pull']:.2f} viable {r['viable']} bank {r['n_bank']} {r['seconds']:.0f}s",
              flush=True)
    json.dump(out, open("bank.json", "w"))
    print("stage0 done", time.time() - t0)

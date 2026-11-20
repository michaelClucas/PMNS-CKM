"""Step 3 of claims.md: fit the structure groups. Jobs are interleaved across groups (round robin)
so a run stopped at the wall budget still samples every group. Appends to fits.jsonl.
Usage: python main.py [workers] [scale]"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import lib43 as L

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
OPTS = ("D", "I1", "IR", "II")
BANK = {"".join(b["Z"]): b for b in json.load(open("bank.json"))}
VIABLE = [b["Z"] for b in BANK.values() if b["viable"] and b["n_bank"] > 0]


def jobs():
    rng = np.random.default_rng(4302)
    pos_c = L.CSET[L.CSET > 0]
    G = {}
    G["C1-S"] = []
    for s in json.load(open("screen.json")):
        Z = s["Z"]
        for kind in ("top", "random"):
            for e in s[kind]:
                G["C1-S"].append({"group": "C1-S", "sub": kind, "screen": e["screen"], "keys": "Y1",
                                  "spec": {"Z": Z, "e": "S", "ce": e["C"], "nu": None}, "bank_idx": e["bank_idx"]})

    def rZ():
        return list(VIABLE[rng.integers(len(VIABLE))])

    def rC(n, pos33=True):
        c = [float(rng.choice(L.CSET)) for _ in range(n - 1)]
        return c + [float(rng.choice(pos_c))] if pos33 else c + [float(rng.choice(L.CSET))]

    for m in (1, 2):
        g = f"C1-P{m}"
        G[g] = []
        for _ in range(int(round(150 * SCALE))):
            Z = rZ()
            c = rC(len(Z) + 1)
            for i in rng.choice(len(c), m, replace=False):
                c[i] = None
            G[g].append({"group": g, "keys": "Y1", "spec": {"Z": Z, "e": "P", "ce": c, "nu": None}})
    G["C1-F"] = [{"group": "C1-F", "keys": "Y1", "seed": s,
                  "spec": {"Z": list(Z), "e": "F", "ce": [None] * len(Z) + [1.0], "nu": None}}
                 for Z in VIABLE for s in range(2)]

    def nu_spec(Z, opt, ones=False):
        n = len(Z) + 1
        return {"Z": Z, "e": "F", "ce": [None] * len(Z) + [1.0], "nu": opt,
                "cnu": [1.0] * n if ones else rC(n, False), "cR": [1.0] * n if ones else rC(n, False)}

    G["C2-S"] = [{"group": "C2-S", "sub": "ones", "keys": "Y2", "spec": nu_spec(list(Z), o, True)}
                 for Z in VIABLE for o in OPTS]
    G["C2-S"] += [{"group": "C2-S", "sub": "random", "keys": "Y2", "spec": nu_spec(rZ(), OPTS[rng.integers(4)])}
                  for _ in range(int(round(300 * SCALE)))]
    for J in (1, 2):
        g = f"C3-P{J}"
        G[g] = []
        for _ in range(int(round(150 * SCALE))):
            sp = nu_spec(rZ(), OPTS[rng.integers(4)])
            sp["extra"] = [str(x) for x in rng.choice(L.NU_POS, J, replace=False)]
            G[g].append({"group": g, "keys": "Y2", "spec": sp})
    G["C2-F"] = [{"group": "C2-F", "keys": "Y2", "seed": s,
                  "spec": {"Z": list(Z), "e": "F", "ce": [None] * len(Z) + [1.0], "nu": "F"}}
                 for Z in VIABLE for s in range(3)]
    G["C3-SS"] = []
    for _ in range(int(round(100 * SCALE))):
        Z = rZ()
        sp = nu_spec(Z, OPTS[rng.integers(4)])
        sp["e"], sp["ce"] = "S", rC(len(Z) + 1)
        G["C3-SS"].append({"group": "C3-SS", "keys": "Y2", "spec": sp})
    for g in G.values():
        rng.shuffle(g)
    order = ["C1-S", "C2-F", "C1-F", "C2-S", "C1-P1", "C3-P1", "C1-P2", "C3-P2", "C3-SS"]
    out, i = [], 0
    while any(G[g] for g in order):
        for g in order:
            # C1-S has ~4x more jobs but each is cheap (Y1, 2 starts): take 2 per round
            for _ in range(2 if g == "C1-S" else 1):
                if G[g]:
                    out.append(G[g].pop())
    for n, j in enumerate(out):
        j["id"] = n
    return out


def work(job):
    t = time.time()
    rng = np.random.default_rng(43000 + job["id"])
    spec = job["spec"]
    keys = L.Y1 if job["keys"] == "Y1" else L.Y2
    bank = BANK["".join(spec["Z"])]["bank"]
    if "bank_idx" in job:
        qs = [bank[job["bank_idx"]], bank[rng.integers(len(bank))]]
        qs = qs + [bank[rng.integers(len(bank))] for _ in range(2)]
        r = L.fit_structure(spec, keys, bank, 4, rng, quark_starts=qs)
    else:
        r = L.fit_structure(spec, keys, bank, 4 if job["keys"] == "Y1" else 6, rng)
    r.update({k: job[k] for k in ("group", "id") if k in job})
    for k in ("sub", "screen"):
        if k in job:
            r[k] = job[k]
    r["seconds"] = time.time() - t
    return r


if __name__ == "__main__":
    J = jobs()
    from collections import Counter
    print("jobs", len(J), dict(Counter(j["group"] for j in J)), flush=True)
    done = set()
    try:
        for line in open("fits.jsonl"):
            done.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass
    J = [j for j in J if j["id"] not in done]
    t0 = time.time()
    with Pool(WORKERS) as p, open("fits.jsonl", "a") as f:
        for n, r in enumerate(p.imap_unordered(work, J, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n + 1}/{len(J)} {time.time() - t0:.0f}s last {r['group']} pass {r['pass']} "
                      f"chi2 {r['chi2']:.3g}", flush=True)
    print("main done", time.time() - t0)

"""Exp 044 driver (grammar.md).

  python main.py enum              -> artifacts/044/A.jsonl (sub-grammar A), B_sample.jsonl (3000 of B)
  python main.py filter A|B W      -> filters 1-2 at the three scales, sector solutions (W workers)
  python main.py fit A|B SCALE W   -> full fits of filter survivors at SCALE, random order, appended to
                                      artifacts/044/fits_<grammar>_<SCALE>.jsonl (resumable)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib44 as L  # noqa: E402

ART = HERE.parents[1] / "artifacts" / "044"
ART.mkdir(parents=True, exist_ok=True)
SCALES = ("MZ", "GUT10", "GUT30")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(HERE / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def enum():
    t0 = time.time()
    out = open(ART / "A.jsonl", "w")
    nA = 0
    tie_index = []  # (t, Z, ties) with counts for B sampling
    for t in "HS":
        pairs, nseen = L.pattern_pairs(t)
        for Z in pairs:
            for ties in L.tie_sets(t, Z):
                if not ties:
                    for ph in L.iter_phase_assignments(t, Z, ()):
                        sp = L.make_spec(t, Z, (), ph)
                        sp["id"] = f"A{nA}"
                        out.write(json.dumps(sp) + "\n")
                        nA += 1
                else:
                    c = L.count_entries(t, Z, ties)
                    if c:
                        tie_index.append((t, Z, ties, c))
    out.close()
    nB = sum(c for *_, c in tie_index)
    rng = np.random.default_rng(4401)
    w = np.array([c for *_, c in tie_index], float)
    picks = rng.choice(len(tie_index), size=3000, p=w / w.sum())
    with open(ART / "B_sample.jsonl", "w") as f:
        for n, i in enumerate(picks):
            t, Z, ties, c = tie_index[i]
            j = rng.integers(c)
            for m, ph in enumerate(L.iter_phase_assignments(t, Z, ties)):
                if m == j:
                    break
            sp = L.make_spec(t, Z, ties, ph)
            sp["id"] = f"B{n}"
            f.write(json.dumps(sp) + "\n")
    log(f"enum: A {nA} entries, B {nB} entries ({len(tie_index)} pattern+tie sets), B sample 3000 (seed 4401); "
        f"G44 total {nA + nB} = {np.log2(nA + nB):.2f} bits; {time.time() - t0:.0f}s")


def load(g):
    fn = ART / ("A.jsonl" if g == "A" else "B_sample.jsonl")
    return [json.loads(l) for l in open(fn)]


# ---------------------------------------------------------------- filter stage

def _sector_job(args):
    key, spec, seed = args
    rng = np.random.default_rng(seed)
    s = key[1]
    out = {}
    for sc in SCALES:
        m, sols = L.sector_solutions(spec, s, sc, rng)
        out[sc] = (m, sols)
    return repr(key), out


def filt(g, W):
    specs = load(g)
    t0 = time.time()
    rng = np.random.default_rng(4402)
    cp_dead = [L.cp_dead(sp, rng) for sp in specs]
    log(f"filter {g}: CP filter killed {sum(cp_dead)}/{len(specs)} ({time.time() - t0:.0f}s)")
    keys = {}
    for sp, dead in zip(specs, cp_dead):
        if dead:
            continue
        for s in L.SECT:
            k = L.sector_key(sp, s)
            if k not in keys:
                keys[k] = sp
    jobs = [(k, sp, 4403 + i) for i, (k, sp) in enumerate(keys.items())]
    log(f"filter {g}: {len(jobs)} distinct sector families")
    res = {}
    with mp.Pool(W) as pool:
        for n, (k, out) in enumerate(pool.imap_unordered(_sector_job, jobs, chunksize=8)):
            res[k] = out
            if n % 2000 == 0:
                log(f"  sector {n}/{len(jobs)} {time.time() - t0:.0f}s")
    json.dump(res, open(ART / f"sectors_{g}.json", "w"))
    surv = {sc: 0 for sc in SCALES}
    rows = []
    for sp, dead in zip(specs, cp_dead):
        kc = L.k_cont(sp)
        crit = L.chi2_crit(kc)
        alive = {}
        for sc in SCALES:
            ok = not dead and all(res[repr(L.sector_key(sp, s))][sc][0] <= crit for s in L.SECT)
            alive[sc] = ok
            surv[sc] += ok
        rows.append({"id": sp["id"], "k": kc, "cp_dead": dead, "alive": alive,
                     "sector_min": {sc: [res[repr(L.sector_key(sp, s))][sc][0] for s in L.SECT] for sc in SCALES}
                     if not dead else None})
    with open(ART / f"filter_{g}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    anyalive = sum(any(r["alive"].values()) for r in rows)
    log(f"filter {g}: survivors per scale {surv}; alive at any scale {anyalive}/{len(rows)} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- fit stage

_SEC = None


def _init(g):
    global _SEC
    _SEC = json.load(open(ART / f"sectors_{g}.json"))
    for sc in SCALES:
        L.scorer(sc)


def _fit_job(args):
    sp, sc, seed = args
    rng = np.random.default_rng(seed)
    sols = {s: _SEC[repr(L.sector_key(sp, s))][sc][1] for s in L.SECT}
    t = time.time()
    try:
        r = L.fit_from_sectors(sp, sc, sols, rng, n_start=8, iters=50)
    except Exception as e:
        r = {"chi2": float("inf"), "max_pull": float("inf"), "fit_ok": False, "error": str(e)[:200]}
    r.update({"id": sp["id"], "scale": sc, "k": L.k_cont(sp), "sec": round(time.time() - t, 3)})
    return r


def fit(g, sc, W, limit=None):
    specs = {sp["id"]: sp for sp in load(g)}
    fl = [json.loads(l) for l in open(ART / f"filter_{g}.jsonl")]
    ids = [r["id"] for r in fl if r["alive"][sc]]
    rng = np.random.default_rng(4404)
    rng.shuffle(ids)
    outfn = ART / f"fits_{g}_{sc}.jsonl"
    done = set()
    if outfn.exists():
        done = {json.loads(l)["id"] for l in open(outfn)}
    todo = [i for i in ids if i not in done]
    if limit and str(limit).endswith(".json"):  # GUT subset (deviation logged in run.log)
        keep = set(json.load(open(ART / limit)))
        todo = [i for i in todo if i in keep]
    elif limit:
        todo = todo[:int(limit)]
    log(f"fit {g} {sc}: {len(ids)} survivors, {len(done)} done, running {len(todo)} with {W} workers")
    t0 = time.time()
    jobs = [(specs[i], sc, 10_000 + int(i[1:])) for i in todo]
    npass = 0
    with mp.Pool(W, initializer=_init, initargs=(g,)) as pool, open(outfn, "a") as f:
        for n, r in enumerate(pool.imap_unordered(_fit_job, jobs, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            npass += bool(r.get("fit_ok"))
            if n % 1000 == 0:
                f.flush()
                log(f"  fit {g} {sc} {n}/{len(jobs)} pass so far {npass} {time.time() - t0:.0f}s")
    log(f"fit {g} {sc}: finished {len(jobs)} in {time.time() - t0:.0f}s, pass {npass}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "enum":
        enum()
    elif cmd == "filter":
        filt(sys.argv[2], int(sys.argv[3]))
    elif cmd == "fit":
        fit(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None)

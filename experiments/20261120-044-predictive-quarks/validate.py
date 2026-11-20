"""Exp 044 validation and near-pass refits.

  python validate.py near A SCALE CUT W    thorough refit of every fast fit with chi2 < CUT at SCALE
  python validate.py miss A SCALE N W      thorough refit of N random fast-fit failures (miss-rate estimate)
Output: artifacts/044/thorough_<mode>_<g>_<scale>.jsonl
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time

import numpy as np

from main import ART, HERE, L, load, log


def _job(args):
    sp, sc, seed, fast = args
    for s in ("MZ", "GUT10", "GUT30"):
        L.scorer(s)
    rng = np.random.default_rng(seed)
    t = time.time()
    try:
        r = L.fit_thorough(sp, sc, rng)
    except Exception as e:
        r = {"chi2": float("inf"), "max_pull": float("inf"), "fit_ok": False, "error": str(e)[:200]}
    # proxy vs official at the same point
    try:
        pc = float((L.proxy(sc)(L.observe(sp, np.array(r["x"])[None])) ** 2).sum())
    except Exception:
        pc = float("nan")
    r.update({"id": sp["id"], "scale": sc, "k": L.k_cont(sp), "fast_chi2": fast, "proxy_at_x": pc,
              "sec": round(time.time() - t, 2)})
    return r


def run(mode, g, sc, arg, W):
    specs = {sp["id"]: sp for sp in load(g)}
    fits = [json.loads(l) for l in open(ART / f"fits_{g}_{sc}.jsonl")]
    if mode == "near":
        sel = [f for f in fits if f["chi2"] < float(arg)]
    else:
        fails = [f for f in fits if not f.get("fit_ok")]
        rng = np.random.default_rng(4405)
        sel = [fails[i] for i in rng.choice(len(fails), size=min(int(arg), len(fails)), replace=False)]
    log(f"validate {mode} {g} {sc}: {len(sel)} refits")
    jobs = [(specs[f["id"]], sc, 20_000 + int(f["id"][1:]), f["chi2"]) for f in sel]
    out = ART / f"thorough_{mode}_{g}_{sc}.jsonl"
    t0 = time.time()
    n_ok = 0
    with mp.Pool(W) as pool, open(out, "w") as fo:
        for n, r in enumerate(pool.imap_unordered(_job, jobs)):
            fo.write(json.dumps(r) + "\n")
            n_ok += bool(r.get("fit_ok"))
    log(f"validate {mode} {g} {sc}: done in {time.time() - t0:.0f}s, fit_ok {n_ok}/{len(jobs)}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))

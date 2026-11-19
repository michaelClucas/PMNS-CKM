"""Inverted-ordering synthetic data for exp 028 (secondary analysis).

Uses the exp 025 generators and filter unchanged (imported from exp 025 gen.py), except that the
ordering cut keeps IO instead of NO. Families: F1, F2 (PMNS independent of the ordering label),
GEN, CTRL, MN (kept only if >= MIN_KEEP samples within the wall cap).
Usage: python gen_io.py [n_target] [workers]   -> artifacts/028/data_io.npz, gen_io_stats.json
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "20261119-025-nn-ckm-info"), str(ROOT)]
import gen  # noqa: E402  (exp 025 generators, filter, features)

ART = ROOT / "artifacts" / "028"
FAMS = ("F1", "F2", "GEN", "CTRL", "MN")
WALL = 900.0
MIN_KEEP = 2000


def _task(args):
    fam, n_target, seed = args
    rng = np.random.default_rng(seed)
    t0 = time.time()
    parts, n_raw, n_ok = [], 0, 0
    chunk = 4000 if fam in ("F1", "F2") else 50000
    while n_ok < n_target and time.time() - t0 < WALL:
        o = gen.GENS[fam](rng, chunk)
        io = o["io"] > 0.5
        o = dict(o, io=np.zeros_like(o["io"]))  # bypass exp 025's NO cut, then select IO
        c, m, y, _, ok = gen.features(o)
        ok &= io
        parts.append((c[ok], m[ok], y[ok]))
        n_raw += len(ok)
        n_ok += int(ok.sum())
    c, m, y = (np.concatenate([p[i] for p in parts])[:n_target] for i in range(3))
    return fam, c, m, y, n_raw, n_ok, time.time() - t0


def main(n_target=6000, workers=2):
    ART.mkdir(parents=True, exist_ok=True)
    res, stats = {}, {}
    with mp.Pool(workers) as pool:
        for fam, c, m, y, n_raw, n_ok, dt in pool.imap_unordered(
                _task, [(f, n_target, [28, 9, i]) for i, f in enumerate(FAMS)]):
            stats[fam] = dict(n_raw=n_raw, n_pass_io=n_ok, kept=len(c), acceptance_io=n_ok / max(n_raw, 1), seconds=dt,
                              included=len(c) >= MIN_KEEP)
            print(f"{fam:5s} IO kept {len(c):6d} / raw {n_raw:9d} ({dt:.0f} s)", flush=True)
            if len(c) >= MIN_KEEP:
                res[fam] = (c, m, y)
    fams = [f for f in FAMS if f in res]
    # all kept samples; unequal family counts are undone in the mixture by w_f p(f|c) / p_train(f)
    fam_id = np.concatenate([np.full(len(res[f][0]), i) for i, f in enumerate(fams)])
    arr = {k: np.concatenate([res[f][j] for f in fams]) for j, k in enumerate(("ctx_ckm", "ctx_m", "y"))}
    np.savez_compressed(ART / "data_io.npz", fam=fam_id, families=np.array(fams), **arr)
    json.dump(dict(n_target=n_target, n_per_family={f: len(res[f][0]) for f in fams}, families=fams, wall_cap_s=WALL, stats=stats),
              open(HERE / "gen_io_stats.json", "w"), indent=1)
    print("saved", len(fam_id), fams, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 6000, int(a[1]) if len(a) > 1 else 2)

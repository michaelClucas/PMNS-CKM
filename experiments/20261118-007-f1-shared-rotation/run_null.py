"""Look-elsewhere null for the F1 search: the identical `f1lib.search` pipeline on fake universes.

kind a: fake PMNS target = Haar-random U(3) (angles extracted), real central CKM kept.
        Scored with GaussChi2 (NuFIT 1 sigma widths of NO and of IO, centred on the fake
        values) because NuFIT chi^2 tables exist only around the real data.
kind b: real PMNS data with the NuFIT tables (NO and IO), V_CKM replaced by a
        hierarchy-preserving fake: Wolfenstein lambda ~ U[0.10, 0.35], A ~ U[0.5, 1.2],
        rhobar, etabar ~ U[-0.5, 0.5] (f1lib.ckm_like_prior).

Fake i uses rng = default_rng([seed, i]). Output rows are appended to null_<kind>.csv
(one per fake); already-finished indices are skipped, so the command can be rerun to
resume or extend.

    python run_null.py --kind a --n 300 --workers 3
"""
import argparse
import csv
import multiprocessing as mp
import time

import common  # noqa: F401  (sets threads and paths)
import numpy as np

from common import HERE, configs
from f1lib import GaussChi2, TableChi2, build_T, ckm_central, ckm_like_prior, observables
from src.haar import haar_unitary

SEEDS = {"a": 1001, "b": 2002}
_G = {}


def _init(kind):
    raw, distinct, _ = configs()
    _G.update(kind=kind, raw=raw, distinct=distinct)
    if kind == "a":
        V = ckm_central()
        _G["T"], _G["Ts"] = build_T(V, distinct), build_T(V, raw)
    else:
        _G["tab"] = {"NO": TableChi2("NO"), "IO": TableChi2("IO")}


def _one(i):
    from f1lib import search
    kind = _G["kind"]
    rng = np.random.default_rng([SEEDS[kind], i])
    info = {}
    if kind == "a":
        tgt = [float(v) for v in observables(haar_unitary(rng))]
        info = dict(zip(("t_s12sq", "t_s13sq", "t_s23sq", "t_dcp"), tgt))
        scorers = {"NO": GaussChi2(tgt, "NO"), "IO": GaussChi2(tgt, "IO")}
        T, Ts = _G["T"], _G["Ts"]
    else:
        V, p = ckm_like_prior(rng)
        info = dict(zip(("lam", "A", "rhobar", "etabar"), p))
        scorers = _G["tab"]
        T, Ts = build_T(V, _G["distinct"]), build_T(V, _G["raw"])
    t0 = time.time()
    r = search(T, scorers, Ts)
    row = {"i": i, **info}
    for o in ("NO", "IO"):
        row[f"best_cont_{o}"] = r[o]["best_cont"]
        row[f"cfg_cont_{o}"] = r[o]["cfg_cont"]
        row[f"best_struct_{o}"] = r[o]["best_struct"]
        row[f"cfg_struct_{o}"] = r[o]["cfg_struct"]
    row["sec"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("a", "b"), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = HERE / f"null_{a.kind}.csv"
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {int(r["i"]) for r in csv.DictReader(fh)}
    todo = [i for i in range(a.n) if i not in done]
    print(f"[null {a.kind}] seed {SEEDS[a.kind]}, {len(done)} done, {len(todo)} to run, "
          f"{a.workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(a.kind,)) as pool:
        fh, w = None, None
        for n, row in enumerate(pool.imap_unordered(_one, todo), 1):
            if w is None:
                new = not out.exists()
                fh = open(out, "a", newline="")
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row)
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"[null {a.kind}] {n}/{len(todo)} fakes, {time.time() - t0:.0f} s", flush=True)
        if fh:
            fh.close()


if __name__ == "__main__":
    main()

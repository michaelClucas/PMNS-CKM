"""Look-elsewhere null for the F2 pipeline (f2lib.search / search_cached, identical to the
real-data `real_pipeline.json` values), at M_Z and at 2e16 GeV.

kind a: fake PMNS target = Haar U(3) (angles via src.mixing.extract_angles), real CKM kept.
        Scored with GaussChi2 (NuFIT 1 sigma widths of NO and of IO, centred on the fake
        values), as exp 007 null (a). Real comparison value: same Gaussian chi^2 centred on
        the NuFIT central values (real_pipeline.json gauss_NO / gauss_IO).
kind b: real PMNS data with the NuFIT tables, CKM replaced by a hierarchy-preserving fake
        (Wolfenstein lambda ~ U[0.10, 0.35], A ~ U[0.5, 1.2], rhobar, etabar ~ U[-0.5, 0.5],
        as exp 007). GUT-scale fake = the same fake run with the real-CKM SM running ratios.

Fake i uses default_rng([seed, i]). Rows append to null_<kind>.csv; finished indices are
skipped (resumable).   python run_null.py --kind b --n 300 --workers 2
"""
import argparse
import csv
import multiprocessing as mp
import time

import numpy as np

import f2lib as L
from src.haar import haar_unitary
from src.mixing import extract_angles

SEEDS = {"a": 1009, "b": 2009}
_G = {}


def _init(kind):
    cfgs = L.configs()
    _G.update(kind=kind, cfgs=cfgs, run=L.gut_running())
    if kind == "a":
        for scale in ("MZ", "GUT"):
            P, Q, rc, _ = L.build_PQ(cfgs, L.ckm_at_scale(scale))
            tier = np.array([cfgs[i]["tier"] for i in rc])
            _G[scale] = (P, Q, tier, L.obs_cache(P, Q))
    else:
        _G["tab"] = {"tab_NO": L.TableChi2("NO"), "tab_IO": L.TableChi2("IO")}


def _flat(scale, res):
    row = {}
    for sk, r in res.items():
        for var, d in r.items():
            for t, v in d.items():
                row[f"{scale}_{sk}_{var}_{t}"] = v
    return row


def one(i):
    t0 = time.time()
    rng = np.random.default_rng([SEEDS[_G["kind"]], i])
    row = {"i": i}
    if _G["kind"] == "a":
        t12, t13, t23, d = (float(x) for x in extract_angles(haar_unitary(rng)))
        tgt = [np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2, np.degrees(d)]
        row.update(s12sq=tgt[0], s13sq=tgt[1], s23sq=tgt[2], dcp_deg=tgt[3])
        sc = {f"gauss_{o}": L.GaussChi2(tgt, o) for o in ("NO", "IO")}
        for scale in ("MZ", "GUT"):
            P, Q, tier, cache = _G[scale]
            row.update(_flat(scale, L.search_cached(cache, P, Q, tier, sc)))
    else:
        ckm, w = L.ckm_like_prior(rng)
        row.update(lam=w[0], A=w[1], rhobar=w[2], etabar=w[3])
        for scale, c in (("MZ", ckm), ("GUT", L.to_gut(ckm, _G["run"]))):
            P, Q, rc, _ = L.build_PQ(_G["cfgs"], c)
            tier = np.array([_G["cfgs"][j]["tier"] for j in rc])
            row.update(_flat(scale, L.search(P, Q, rc, tier, _G["tab"])))
    row["sec"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["a", "b"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--start", type=int, default=0, help="first fake index (to split a run across processes)")
    a = ap.parse_args()
    path = L.HERE / f"null_{a.kind}.csv"
    done = set()
    if path.exists():
        with open(path) as fh:
            done = {int(r["i"]) for r in csv.DictReader(fh)}
    todo = [i for i in range(a.start, a.n) if i not in done]
    print(f"[null {a.kind}] {len(todo)} fakes to run, {len(done)} done", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(a.kind,)) as pool:
        fh, w = None, None
        for n, row in enumerate(pool.imap_unordered(one, todo)):
            if w is None:
                new = not path.exists()
                fh = open(path, "a", newline="")
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row)
            fh.flush()
            if n % 20 == 0:
                print(f"[null {a.kind}] {n + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        if fh:
            fh.close()
    print(f"[null {a.kind}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

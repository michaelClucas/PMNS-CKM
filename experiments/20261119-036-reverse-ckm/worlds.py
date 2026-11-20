"""Exp 036 injection worlds: python worlds.py <family> [n_worlds]
Fresh draws from src.theoryspace (rng default_rng([36, 9, b]) per batch of 2000 accepted NO draws), first
n_worlds whose true s12sq, s13sq, s23sq lie in the N2 box; then measurement noise at today's precision
(prereg.md "Injection"). Writes worlds_<family>.json."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from common import HERE, ROOT  # noqa: F401  (sets sys.path)
from src import data as sd, evaluate as ev, nulls, theoryspace as ts
from oracle import HI, LO

FCODE = {"GEN": 9, "F2": 10}


def main(fam, n_worlds=20, batch=2000, max_batches=60):
    t0 = time.time()
    recs = sd.load_ckm()
    _, names, cov = ev.mass_ratio_targets()
    idx = [names.index(k) for k in ts.MR_KEYS]
    cov6 = np.asarray(cov)[np.ix_(idx, idx)]
    worlds, n_raw, n_acc = [], 0, 0
    for b in range(max_batches):
        rng = np.random.default_rng([36, FCODE[fam], b])
        o, st = ts.sample_accepted(fam, batch, rng)
        n_raw += st["n_raw"]
        n_acc += st["kept"]
        s = np.stack([o["s12sq"], o["s13sq"], o["s23sq"]], 1)
        inbox = np.all((s > LO[:3]) & (s < HI[:3]), 1)
        for r in np.flatnonzero(inbox):
            if len(worlds) >= n_worlds:
                break
            w = len(worlds)
            nr = np.random.default_rng([36, 4, FCODE[fam], w])
            truth = {k: float(o[k][r]) for k in ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq",
                                                   "s23sq", "dcp_deg", "dm_ratio") + ts.MR_KEYS}
            meas = {}
            for k, rk in (("ckm_s12", "s12"), ("ckm_s13", "s13"), ("ckm_s23", "s23")):
                z = nr.standard_normal()
                rel = (recs[rk]["err_plus"] if z > 0 else recs[rk]["err_minus"]) / recs[rk]["value"]
                meas[k] = truth[k] * (1 + z * rel)
            z = nr.standard_normal()
            meas["ckm_delta_rad"] = truth["ckm_delta_rad"] + z * (recs["delta_rad"]["err_plus"] if z > 0 else recs["delta_rad"]["err_minus"])
            ln = np.log([truth[k] for k in ts.MR_KEYS]) + nr.multivariate_normal(np.zeros(6), cov6)
            meas.update({k: float(np.exp(v)) for k, v in zip(ts.MR_KEYS, ln)})
            meas["dm_ratio"] = truth["dm_ratio"]
            off = nulls.pmns_noise("NO", nr)
            for k in ("s12sq", "s13sq", "s23sq"):
                meas[k] = float(np.clip(truth[k] + off[k], 1e-6, 1 - 1e-6))
            meas["dcp_deg"] = float((truth["dcp_deg"] + off["dcp_deg"]) % 360.0)
            worlds.append({"index": w, "batch": b, "row": int(r), "truth": truth, "meas": meas})
        print(f"batch {b}: accepted {n_acc}, raw {n_raw}, worlds {len(worlds)} ({time.time() - t0:.0f}s)", flush=True)
        if len(worlds) >= n_worlds:
            break
    json.dump({"family": fam, "n_raw": n_raw, "n_accepted": n_acc, "worlds": worlds}, open(HERE / f"worlds_{fam}.json", "w"), indent=1)
    with open(HERE / "run.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} worlds {fam}: {len(worlds)} in-box worlds from {n_acc} accepted / {n_raw} raw draws\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)

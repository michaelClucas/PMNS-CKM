"""Exp 041 stage-1 diagnostic: fit quarks + charged leptons only (N 10: CKM 4, four quark and two
charged-lepton mass ratios; k_cont 8) for each of the 8 representation patterns (neutrino option
irrelevant, the Dirac entry is used). Shows whether the charged-lepton Clebsches alone already fail,
before any neutrino observable. -> fits_qe.jsonl"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit41 as F  # noqa: E402
import grammar as g  # noqa: E402

if __name__ == "__main__":
    with open(HERE / "fits_qe.jsonl", "w") as fh:
        for e in g.ENTRIES:
            if e["nu"] != "dirac":
                continue
            r = F.fit(e["id"], use=g.QEUSE, seed=1, popsize=12, maxiter=200)
            r["reps"] = e["reps"]
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            worst = sorted(r["pulls"].items(), key=lambda kv: -abs(kv[1]))[:3]
            print(g.name(e)[:3], "chi2", round(r["chi2"], 1), "p", r["p"], {a: round(b, 1) for a, b in worst},
                  f"{r['secs']:.0f}s", flush=True)

"""Exp 031: what B_ckm does a TRUE relation produce through the program's null pipelines?

Exp 030's power checks inject universes in which the bi-large (CDSV) relation holds exactly
(best-fit prediction + measurement noise) and run the full N1/N2 nulls on each. Each injected
null stores src.infogain null_info, including B_ckm. Comparing those values with the
"CKM carries <~ 1-2 bits" statement (exp 021, quoted in STATE and in exps 025/028) shows whether
that number is an upper bound on information or the ceiling of the instrument.

usage (repo root): .venv/Scripts/python experiments/20261119-031-skeptic-audit/bckm_injected.py
Output: bckm_injected.json next to this file.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
E = ROOT / "experiments" / "20261119-030-engine-bilarge"
out = {}
for nul in ("N1", "N2"):
    for o in ("NO", "IO"):
        d = json.loads((E / f"power_{nul}_{o}.json").read_text())
        b = np.array([n["info"]["B_ckm"] for n in d["nulls"] if n.get("info") and n["info"].get("B_ckm") is not None])
        nrun = [n["n_run"] for n in d["nulls"]]
        real = json.loads((E / f"null_{nul}_{o}.json").read_text()).get("info", {}).get("B_ckm")
        out[f"{nul}_{o}"] = dict(n_injections=len(b), B_ckm_min=float(b.min()), B_ckm_median=float(np.median(b)),
                               B_ckm_max=float(b.max()), frac_le_2bits=float(np.mean(b <= 2.0)),
                               injected_null_n_run_median=float(np.median(nrun)), real_data_B_ckm=real,
                               power=d["fraction"])
        print(f"{nul} {o}: injected-true B_ckm min/median/max {b.min():+.2f}/{np.median(b):+.2f}/{b.max():+.2f} bits, "
              f"{np.mean(b <= 2.0):.0%} <= 2 bits; real-data B_ckm {real}; power {d['fraction']}")
pathlib.Path(__file__).with_name("bckm_injected.json").write_text(json.dumps(out, indent=1))

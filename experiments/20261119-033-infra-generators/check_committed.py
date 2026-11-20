"""Rescore committed fit points with the current src (default arguments) and compare to the files.

Run from the repo root: .venv/Scripts/python experiments/20261119-033-infra-generators/check_committed.py
Writes committed_check.json next to this file.
- exp 022 fit_primary / fit_tb10 / fit_alt / fit_eta_minus / fit_v1: model at the stored params.
- exp 023 fit_eb0 / fit_eb08: stored GUT predictions (eb08: eta_b applied linearly as in exp 023).
"""
import json
import os
import pathlib
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E22 = ROOT / "experiments/20261119-022-su5-csd3"
E23 = ROOT / "experiments/20261119-023-s4so10"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E22))

from src import evaluate as ev  # noqa: E402
import fit as F  # noqa: E402
import model as M  # noqa: E402

out = {}
for tag in ("primary", "tb10", "alt", "eta_minus", "v1"):
    d = json.loads((E22 / f"fit_{tag}.json").read_text())
    scale = dict(d["scale"])
    scale["majorana"] = tuple(scale.get("majorana", (0.0, 0.0)))
    fn = F.make_model(d["eta"])
    sc = ev.Scorer(d["use"], "NO", d["analysis"], mass_floor=d["mass_floor"], scale=scale)
    c = sc.chi2(fn(d["params"]))
    out[f"022_{tag}"] = {"committed": d["chi2"], "now": c, "abs_diff": abs(c - d["chi2"])}

for tag in ("eb0", "eb08"):
    d = json.loads((E23 / f"fit_{tag}.json").read_text())
    sc = ev.Scorer(d["use"], "NO", mass_floor=0.02,
                   scale={"mu": 2e16, "model": "MSSM", "tan_beta": 5.0, "m_susy": 1000.0})
    c = sc.chi2(d["pred_gut"])
    out[f"023_{tag}"] = {"committed": d["chi2"], "now": c, "abs_diff": abs(c - d["chi2"])}

out["max_abs_diff"] = max(v["abs_diff"] for v in out.values())
(HERE / "committed_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

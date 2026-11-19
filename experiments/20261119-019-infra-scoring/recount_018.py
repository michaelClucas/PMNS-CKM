"""Recount exp 018's stored per-fake summaries with src.nulls.is_hit_fit_only (p tolerance P_RTOL).

Run from the repo root: .venv/Scripts/python experiments/20261119-019-infra-scoring/recount_018.py
"exact" is exp 018's own rule (max|pull| < 2 and p_global >= ref, no tolerance).
"""
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.nulls import is_hit_fit_only  # noqa: E402


def null_runs(x):
    if isinstance(x, dict):
        if "fakes" in x and "ref" in x:
            yield x
        for v in x.values():
            yield from null_runs(v)
    elif isinstance(x, list):
        for v in x:
            yield from null_runs(v)


for f in sorted(glob.glob(str(ROOT / "experiments/20261118-018-protocol-reruns/*_*.json"))):
    name = pathlib.Path(f).name
    if name.startswith(("leak", "repro")):
        continue
    with open(f) as fh:
        d = json.load(fh)
    n = fakes = exact = tol = 0
    for run in null_runs(d):
        ref = run["ref"]
        if not ref:
            continue
        n += 1
        for x in run["fakes"]:
            r = x.get("result")
            fakes += 1
            if not r:
                continue
            exact += r["max_abs_pull"] < 2 and r["p_global"] >= ref["p_global"]
            tol += is_hit_fit_only(r, ref)
    if n:
        print(f"{name:28s} runs {n:2d} fakes {fakes:5d} hits exact {exact:4d} with tolerance {tol:4d}")

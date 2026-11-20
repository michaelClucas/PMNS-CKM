"""Recompute exp 019's ref_numbers.json with the current src and compare to the committed file.

Copied from exp 026 (unchanged logic). Run from the repo root:
.venv/Scripts/python experiments/20261119-033-infra-generators/check_ref_numbers.py
Writes ref_check.json next to this file (max abs / rel difference per section). Does not touch exp 019.
"""
import json
import pathlib
import runpy
import shutil
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "experiments/20261119-019-infra-scoring"
ref = json.loads((SRC / "ref_numbers.json").read_text())

# run exp 019's script from a temporary copy so its output lands outside the repo
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td) / "a" / "b"
    tmp.mkdir(parents=True)
    shutil.copy(SRC / "ref_numbers.py", tmp / "ref_numbers.py")
    sys.path.insert(0, str(ROOT))
    src = (tmp / "ref_numbers.py").read_text().replace("sys.path.insert(0, str(HERE.parents[1]))",
                                                        f"sys.path.insert(0, {str(ROOT)!r})")
    (tmp / "ref_numbers.py").write_text(src)
    runpy.run_path(str(tmp / "ref_numbers.py"), run_name="__main__")
    new = json.loads((tmp / "ref_numbers.json").read_text())


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


a, b = dict(leaves(ref)), dict(leaves(new))
assert set(a) == set(b), sorted(set(a) ^ set(b))
out = {"n_numbers": len(a), "sections": {}}
for p in a:
    sec = p.split("/")[1].split("[")[0]
    d = abs(float(a[p]) - float(b[p]))
    r = d / max(abs(float(a[p])), 1e-300)
    s = out["sections"].setdefault(sec, {"max_abs": 0.0, "max_rel": 0.0})
    s["max_abs"], s["max_rel"] = max(s["max_abs"], d), max(s["max_rel"], r)
out["max_abs_all"] = max(s["max_abs"] for s in out["sections"].values())
out["max_rel_all"] = max(s["max_rel"] for s in out["sections"].values())
out["pass_1e-9"] = bool(out["max_abs_all"] <= 1e-9)
(HERE / "ref_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

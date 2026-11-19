"""Resolve the FN prior-predictive min chi2_5 discrepancy (exp 014: 216, exp 021: 79).

Run from this folder: ../../.venv/Scripts/python fn_prior_min.py  -> fn_prior_min.json
Needs exp 014's so10fit.py (no PT samples needed). Single process.
"""
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261118-014-so10-fits"))
import so10fit as S  # noqa: E402

N = 200000


def min_chi2_no(prob, thp):
    zp, op = prob.predict(thp)
    ok = np.all(np.isfinite(zp), 1)
    zp, io = zp[ok], op["io"][ok]
    d = zp[:, 12:17] - prob.val[12:17]
    d[:, 3] = S.wrap(d[:, 3])
    c = ((d / prob.err[12:17]) ** 2).sum(1)
    return float(np.where(io, np.inf, c).min())


def draw(rng, lo, hi):
    return lo + (hi - lo) * rng.random((N, len(lo)))


out = {}
fn = S.Problem("FN", "ours")
mn = S.Problem("MN", "ours")
flo, fhi = S.fn_box("ours")
mlo, mhi = S.mn_box("ours", mn.val)

# exp 014 analyze_heldout.py: fresh default_rng(5) per model run -> FN takes the first draw
out["exp014_stream_first_draw"] = min_chi2_no(fn, draw(np.random.default_rng(5), flo, fhi))
# exp 021 bound.py: one default_rng(5) shared across cases MN_ours, MN_drrel, FN_ours -> FN takes the third draw
rng = np.random.default_rng(5)
draw(rng, mlo, mhi)
draw(rng, mlo, mhi)
out["exp021_stream_third_draw"] = min_chi2_no(fn, draw(rng, flo, fhi))
# spread of the statistic across independent seeds
seeds = list(range(100, 100 + int(os.environ.get("NSEEDS", "10"))))
out["other_seeds"] = {s: min_chi2_no(fn, draw(np.random.default_rng(s), flo, fhi)) for s in seeds}
v = np.array(list(out["other_seeds"].values()))
out["other_seeds_summary"] = dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
(HERE / "fn_prior_min.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

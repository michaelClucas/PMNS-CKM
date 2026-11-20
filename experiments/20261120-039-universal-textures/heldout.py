"""Exp 039 held-out test: fit quarks + charged leptons, predict PMNS.

Stage 1: fit every parameter except the two neutrino mass parameters to QUSE (CKM 4 + six mass
ratios), neutrino parameters fixed at an arbitrary value (they do not enter QUSE).
Stage 2: freeze everything from stage 1 (shared a / phi, charged-lepton masses) and fit only the two
neutrino mass parameters to LUSE (PMNS 4 + dm_ratio, N 5, k_cont 2). The PMNS is "predicted from the
quark side" if stage 2 fits (p >= 0.05, all pulls < 2).
Also stage 2 profile: s12sq, s13sq, s23sq over a 60 x 60 grid of the neutrino parameters, to show the
reachable PMNS set at the quark-side optimum.
Usage: python heldout.py  -> heldout.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fit as F  # noqa: E402
import model as M  # noqa: E402

CASES = [("FZ4U", dict(eta=1, nu="dirac")), ("FZ4U", dict(eta=1, nu="seesaw")),
         ("FZ4U", dict(eta=-1, nu="dirac")), ("FZ4U", dict(eta=-1, nu="seesaw")),
         ("DEMU", dict(nu="dirac")), ("DEMU", dict(nu="seesaw"))]
NUPAR = {"FZ4U": ("lr12_n", "lr23_n"), "DEMU": ("dl_n", "sg_n")}


def run(case):
    cls, opt = case
    nup = NUPAR[cls]
    fix0 = {nup[0]: -0.5, nup[1]: -1.0} if cls == "FZ4U" else {nup[0]: 0.1, nup[1]: 0.05}
    s1 = F.fit(dict(cls=cls, opt=opt, seed=1, scale="MZ", fix=fix0, use=M.QUSE))
    fixed = {k: v for k, v in s1["params"].items() if k not in nup}
    s2 = F.fit(dict(cls=cls, opt=opt, seed=1, scale="MZ", fix=fixed, use=M.LUSE), popsize=20, maxiter=200)
    model, bounds, _, _ = F.make(cls, opt)
    g0 = np.linspace(*bounds[nup[0]], 60)
    g1 = np.linspace(*bounds[nup[1]], 60)
    pts = []
    for x in g0:
        for y in g1:
            try:
                o = model({**fixed, nup[0]: float(x), nup[1]: float(y)})
            except ValueError:
                continue
            pts.append([o["s12sq"], o["s13sq"], o["s23sq"], o["dm_ratio"]])
    pts = np.array(pts)
    near = pts[np.abs(np.log(pts[:, 3] / 0.0299)) < 0.3] if len(pts) else pts
    rng = lambda a: [float(np.nanmin(a)), float(np.nanmax(a))] if len(a) else None  # noqa: E731
    return dict(cls=cls, opt=opt,
                stage1=dict(chi2=s1["chi2"], N=s1["N"], k_cont=s1["k_cont"], p=s1["p"], max_pull=s1["max_pull"],
                            pulls=s1["pulls"], params=s1["params"]),
                stage2=dict(chi2=s2["chi2"], N=s2["N"], k_cont=s2["k_cont"], p=s2["p"], max_pull=s2["max_pull"],
                            pulls=s2["pulls"], params={k: s2["params"][k] for k in nup}, pred=s2["pred"]),
                grid_range_all=dict(s12sq=rng(pts[:, 0]), s13sq=rng(pts[:, 1]), s23sq=rng(pts[:, 2])) if len(pts) else None,
                grid_range_dm_ok=dict(s12sq=rng(near[:, 0]), s13sq=rng(near[:, 1]), s23sq=rng(near[:, 2]),
                                      n=int(len(near))) if len(near) else None)


if __name__ == "__main__":
    with mp.Pool(2) as pool:
        out = pool.map(run, CASES)
    json.dump(out, open(HERE / "heldout.json", "w"), indent=1)
    for r in out:
        print(r["cls"], r["opt"], "stage1 chi2", round(r["stage1"]["chi2"], 2), "N", r["stage1"]["N"],
              "| stage2 chi2", round(r["stage2"]["chi2"], 2), "N", r["stage2"]["N"], "p", r["stage2"]["p"],
              "maxpull", round(r["stage2"]["max_pull"], 2), r["grid_range_dm_ok"], flush=True)

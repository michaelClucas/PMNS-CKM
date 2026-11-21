"""Exp 047 (a): repeatability of exp 044's own production fitter on its survivors.

Reruns lib44's pipeline for one entry exactly as main.py does (sector_solutions per sector, then
fit_from_sectors with n_start 8, iters 50) with 20 different seeds, and counts how often it gets the entry
to fit_ok. A low rate means exp 044's single-seed scan of the other 64k entries is search-limited
(survivor list = lower bound).
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
E44 = HERE.parent / "20261120-044-predictive-quarks"
sys.path.insert(0, str(E44))
import lib44 as L  # noqa: E402


def main():
    surv = json.loads((E44 / "survivors.json").read_text())
    out = {}
    for sid, scale in (("A6842", "MZ"), ("A6624", "GUT10"), ("A47977", "MZ")):
        spec = next(s for s in surv if s["id"] == sid and s["scale"] == scale)["spec"]
        res = []
        for seed in range(20):
            rng = np.random.default_rng(10_000 + seed)
            sols = {s: L.sector_solutions(spec, s, scale, rng)[1] for s in L.SECT}
            r = L.fit_from_sectors(spec, scale, sols, rng, n_start=8, iters=50)
            res.append((float(r["chi2"]), bool(r["fit_ok"])))
        c = [a for a, _ in res]
        out[sid] = {"scale": scale, "chi2": c, "n_fit_ok": sum(b for _, b in res), "n_seeds": len(res)}
        print(sid, scale, f"fit_ok in {out[sid]['n_fit_ok']}/20 seeds; chi2 min {min(c):.3f} median {np.median(c):.2f}",
              flush=True)
    (HERE / "rerun044_fitter.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

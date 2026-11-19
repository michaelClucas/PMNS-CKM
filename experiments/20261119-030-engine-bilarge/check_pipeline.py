"""Checks of the frozen pipeline (bilarge.py) before any fake is run.
1. vector_chi2 == Universe.pmns_delta_chi2 on random points, real and N2 universes, both orderings.
2. Exp 027 quickfit fits reproduced inside the grammar (CDSV-I/II NO, RMSV types 1-4 NO).
3. Leak probe of one configuration's prediction (inputs lambda, A declared).
4. Timing and the real-data selection.
Run: ../../.venv/Scripts/python check_pipeline.py > check_pipeline.log
"""
import json
import time

import numpy as np

import bilarge as B
from src import nulls as N
from src.evaluate import evaluate_model

out = {}
rng = np.random.default_rng(30)
worst = 0.0
for kind in ("real", "N2"):
    for o in ("NO", "IO"):
        u = N.Universe.real(o)
        if kind == "N2":
            u = N.n2_universe(u, np.random.default_rng([30, 1]))
        c = u.pmns_data_centre(o)
        pts = [(c["s12sq"] + rng.normal(0, .02), c["s13sq"] + rng.normal(0, .001), c["s23sq"] + rng.normal(0, .05),
                rng.uniform(0, 360)) for _ in range(300)]
        a = np.array(pts).T
        v = B.vector_chi2(u, o, *a)
        s = np.array([u.pmns_delta_chi2(o, *p) for p in pts])
        both = np.isfinite(v) & np.isfinite(s)
        assert np.array_equal(np.isfinite(v), np.isfinite(s)), (kind, o)
        worst = max(worst, float(np.max(np.abs(v[both] - s[both]))))
out["vector_chi2_max_abs_diff"] = worst
print("vector_chi2 vs Universe.pmns_delta_chi2 max |diff|:", worst)

u = N.Universe.real("NO")
ck = u.load_ckm()
lam, A = ck["lambda"]["value"], ck["A"]["value"]
sc = u.scorer(B.USE, "NO", inputs=B.INPUTS)
ref = {}
for name, cfg, th in (("CDSV-I", (("l", 2), ("l", 1), ("1-l", 1), np.pi, "I"), {"phi": 3.894128571190876}),
                      ("CDSV-II", (("l", 2), ("l", 1), ("l", 3), np.pi, "II"), {"phi": 5.531270126172538}),
                      ("RMSV1", ("RMSV", 1), {"psi": 2.9248490895355843, "phi": 0.7448876485882988}),
                      ("RMSV2", ("RMSV", 2), {"psi": 2.9811135566200284, "phi": 0.7362923494998037}),
                      ("RMSV3", ("RMSV", 3), {"psi": 2.9248490895345487, "phi": 2.3967050050251935}),
                      ("RMSV4", ("RMSV", 4), {"psi": 2.9032347787032267, "phi": 3.8902931118836857})):
    s12, s13, s23, d = B.obs_of(B.model_U(cfg, lam, A, th))
    ref[name] = sc.chi2({"s12sq": s12, "s13sq": s13, "s23sq": s23, "dcp_deg": d})
print("exp 027 quickfit points rescored (027: CDSV-I 8.208, CDSV-II 7.973, RMSV 4.790/20.012/4.790/5.133):",
      {k: round(v, 3) for k, v in ref.items()})
out["quickfit_rescored_NO"] = ref


def model():
    c = N.Universe.real("NO").load_ckm()
    from src import data as sd
    cc = sd.load_ckm()
    s12, s13, s23, d = B.obs_of(B.model_U(("RMSV", 1), cc["lambda"]["value"], cc["A"]["value"],
                                          {"psi": 2.92, "phi": 0.74}))
    return {"s12sq": float(s12), "s13sq": float(s13), "s23sq": float(s23), "dcp_deg": float(d)}


r = evaluate_model(model, "NO", B.USE, inputs=B.INPUTS, leak_check=True, k=2)
out["leak_probe"] = "passed"
print("leak probe passed (inputs ckm_lambda, ckm_A)")

t0 = time.time()
allr = B.search(N.Universe.real("NO"), return_all=True)
dt = time.time() - t0
out["seconds_per_pipeline"] = dt
print(f"pipeline time {dt:.2f} s")
rows = []
for res, cfg, th in allr:
    rows.append(dict(config=B.config_name(cfg), ordering=res.ordering, chi2=res.chi2, chi2_global=res.chi2_global,
                     p_global=res.p_global, k=res.k, k_cont=res.k_continuous, N=res.N, max_pull=res.max_abs_pull,
                     pulls=res.pulls, pred=res.pred, theta=th))
    print(f"  {B.config_name(cfg):55s} {res.ordering} chi2 {res.chi2:7.3f} chi2_g {res.chi2_global:7.3f} "
          f"p_g {res.p_global:.4f} maxpull {res.max_abs_pull:.2f} theta {th}")
out["real_polished"] = rows
out["real_selected"] = rows[0]
json.dump(out, open("check_pipeline.json", "w"), indent=1, default=float)

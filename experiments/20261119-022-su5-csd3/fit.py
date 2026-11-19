"""Step 3: fit the frozen 1503.03306 structure to our data (pre-registration in README.md).

Usage: python fit.py [tag]   tags: primary (default), v1, tb10, alt, eta_minus
"""
import json
import math
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import evaluate as ev  # noqa: E402
import model as M  # noqa: E402

USE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_e_mu", "mr_mu_tau",
       "s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
SCALE = {"mu": 2e16, "model": "MSSM", "tan_beta": 5, "m_susy": 1000.0, "m_lightest": 0.0, "majorana": (0.0, 0.0)}
FLOOR = 0.02
K_DISCRETE = 1  # sign of eta (README pre-registration)
BOUNDS = {**{n: (-10.0, 10.0) for n in ("u11", "u12", "u13", "u22", "u23", "d11", "d12", "d22")}, "r": (0.0, 1.0)}
# paper tb=5 benchmark, normalised to u33 = d33 = 1 (extra start)
PAPER = {"u11": 0.9566 / 0.5435, "u12": 0.7346 / 0.5435, "u13": 0.7198 / 0.5435, "u22": 0.5961 / 0.5435,
         "u23": 0.3224 / 0.5435, "d11": 2.133 / 1.021, "d12": 0.8363 / 1.021, "d22": 1.108 / 1.021, "r": 2.684 / 26.57}


def make_model(eta=M.ETA):
    def model_fn(t):
        u = (t["u11"], t["u12"], t["u13"], t["u22"], t["u23"], 1.0)
        d = (t["d11"], t["d12"], t["d22"], 1.0)
        return M.scored(M.predict(u, d, t["r"], eta=eta))
    return model_fn


def run(tag="primary", n_starts=48):
    use, scale, analysis, eta = list(USE), dict(SCALE), "IC23_woSKatm", M.ETA
    if tag == "v1":
        use.insert(7, "mr_s_b")
    elif tag == "tb10":
        scale["tan_beta"] = 10
    elif tag == "alt":
        analysis = "IC24_wSKatm"
    elif tag == "eta_minus":
        eta = -M.ETA
    model_fn = make_model(eta)
    t0 = time.time()
    fr = ev.fit_and_evaluate(model_fn, BOUNDS, "NO", use, analysis=analysis, k_discrete=K_DISCRETE,
                             n_starts=n_starts, seed=22, extra_starts=[PAPER], leak_check=True,
                             mass_floor=FLOOR, scale=scale)
    r = fr.result
    th = fr.params
    full = M.predict((th["u11"], th["u12"], th["u13"], th["u22"], th["u23"], 1.0),
                     (th["d11"], th["d12"], th["d22"], 1.0), th["r"], eta=eta)
    pred = M.scored(full)
    pred["R_clebsch"] = pred["mr_e_mu"] / pred["mr_d_s"]
    rr = ev.evaluate(pred, "NO", analysis, k=0, use=["R_clebsch"], mass_floor=FLOOR, scale=scale)
    out = {
        "tag": tag, "use": use, "scale": {k: list(v) if isinstance(v, tuple) else v for k, v in scale.items()},
        "analysis": analysis, "eta": eta, "mass_floor": FLOOR,
        "chi2": r.chi2, "chi2_global": r.chi2_global, "N": r.N, "k": r.k, "k_continuous": r.k_continuous,
        "k_discrete": r.k_discrete, "dof": r.dof, "p": r.p, "p_global": r.p_global, "max_abs_pull": r.max_abs_pull,
        "fit_ok": r.fit_ok, "compression_ok": r.compression_ok, "pulls": r.pulls, "terms": r.terms,
        "pred_gut": pred, "params": th, "scale_shift": r.scale_shift, "scale_systematic": r.scale_systematic,
        "R_pred_gut": pred["R_clebsch"], "R_pull": rr.pulls["R_clebsch"], "R_chi2": rr.chi2,
        "theta12e_rad": full["_theta12e"], "theta23e_rad": full["_theta23e"], "theta13e_rad": full["_theta13e"],
        "start_chi2_sorted": sorted(fr.start_chi2)[:10], "n_starts": fr.n_starts,
        "n_within_1e-3_of_best": fr.n_distinct_minima(1e-3), "nfev": fr.nfev, "seconds": time.time() - t0,
    }
    Path(f"fit_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[{tag}] chi2 {r.chi2:.3f} chi2_global {r.chi2_global:.3f} N {r.N} k {r.k} (cont {r.k_continuous}) "
          f"dof {r.dof} p {r.p:.4g} p_global {r.p_global:.4g} max|pull| {r.max_abs_pull:.2f} fit_ok {r.fit_ok} "
          f"compression_ok {r.compression_ok}")
    for k, v in r.pulls.items():
        print(f"   pull {k:14s} {v:+.2f}   pred(GUT) {pred.get(k, float('nan')):.6g}")
    print(f"   R(GUT pred) {pred['R_clebsch']:.5f}  R pull {rr.pulls['R_clebsch']:+.2f}")
    print(f"   theta12e {full['_theta12e']:.3e} rad  theta23e {full['_theta23e']:.3e} rad")
    print(f"   params {json.dumps({k: round(v, 5) for k, v in th.items()})}")
    print(f"   best starts {[round(c, 3) for c in sorted(fr.start_chi2)[:6]]}  ({time.time() - t0:.0f} s, nfev {fr.nfev})")
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["primary"]
    for t in tags:
        run(t, n_starts=int(os.environ.get("NSTARTS", 48)))

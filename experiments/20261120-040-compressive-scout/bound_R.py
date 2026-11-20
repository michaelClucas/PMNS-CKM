"""Filter (a): reachable Clebsch double ratio R = (m_e/m_mu)/(m_d/m_s) in the models whose d and e
matrices share parameters (Albright-Barr / Ji-Li-Mohapatra lopsided SO(10); UTZ at LO and HO).

R is RG-invariant (exp 021), so it is compared directly with the data value via the official scorer
(key R_clebsch, mass_floor sqrt(2)*2%, the exp 022 convention). Minimise R's pull over the model's
d/e parameters, subject to m_e/m_mu equal to data (penalty) and loose GUT-scale bands
m_mu/m_tau in [0.050, 0.068], m_s/m_b in [0.012, 0.030] (wide on purpose: a conservative bound).

    python bound_R.py   -> bound_R.json, stdout (log)
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

from src import evaluate as ev  # noqa: E402
import lib40 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
V, _, _ = ev.mass_ratio_targets()
R0 = V["mr_e_mu"] / V["mr_d_s"]
FLOOR = {"R_clebsch": 0.02 * math.sqrt(2)}


def pullR(R):
    return ev.evaluate({"R_clebsch": R}, "NO", use=["R_clebsch"], mass_floor=FLOOR).pulls["R_clebsch"]


SIG = (1 / 9 - R0) / pullR(1 / 9)  # 1 sigma of R under this floor


def band(x, lo, hi):
    return 0.0 if lo <= x <= hi else (math.log(x / lo) if x < lo else math.log(x / hi)) ** 2


def objective(Md, Me):
    rd, re = L.ratios(Md), L.ratios(Me)
    if not all(np.isfinite([*rd, *re])) or min(rd[0], re[0]) <= 0:
        return 1e6, None
    R = re[0] / rd[0]
    pen = 1e4 * (math.log(re[0] / V["mr_e_mu"]) ** 2 + band(re[1], 0.050, 0.068) + band(rd[1], 0.012, 0.030))
    return ((R - R0) / SIG) ** 2 + pen, dict(R=R, me_mmu=re[0], mmu_mtau=re[1], md_ms=rd[0], ms_mb=rd[1])


MODELS = {
    # name: (bounds, builder(x) -> (Md, Me))
    "AB_lopsided": ([(0.3, 6.0), (0.01, 0.6), (1e-4, 0.05), (0.0, 0.05), (0.0, 2 * math.pi)],
                    lambda x: (L.ab_D(*x), L.ab_L(*x))),
    "UTZ_LO": ([(-0.03, 0.03), (-0.3, 0.3), (-math.pi, math.pi), (-math.pi, math.pi)],
               lambda x: L.utz_de(*x)),
    "UTZ_HO": ([(-0.03, 0.03), (-0.3, 0.3), (-math.pi, math.pi), (-math.pi, math.pi), (-0.1, 0.1),
                (-math.pi, math.pi)], lambda x: L.utz_de(*x)),
}


def run(name, seeds=(1, 2, 3)):
    bounds, build = MODELS[name]
    f = lambda x: objective(*build(x))[0]  # noqa: E731
    best = None
    for s in seeds:
        r = differential_evolution(f, bounds, seed=s, popsize=30, maxiter=1500, tol=1e-12, polish=False,
                                   workers=1)
        r2 = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 40000})
        x = r2.x if r2.fun < r.fun else r.x
        if best is None or f(x) < f(best):
            best = x
    val, info = objective(*build(best))
    info.update(objective=val, x=[float(v) for v in best], pull_R=pullR(info["R"]))
    return info


if __name__ == "__main__":
    out = {"R_data": R0, "sigma_R": SIG, "pull_R_1_9": pullR(1 / 9)}
    print(f"R data {R0:.5f}, sigma {SIG:.5f}; R = 1/9 pull {pullR(1/9):+.2f}")
    # published/benchmark points
    pts = {"AB_bench (hep-ph/0502161 Eq. 3)": (L.ab_D(**L.AB_BENCH), L.ab_L(**L.AB_BENCH)),
           "UTZ_LO_paper (Table III)": L.utz_de(0.0042, -0.0545, 0.13, 1.83),
           "UTZ_HO_paper (Table III)": L.utz_de(0.00416, -0.0566, 0.0, 2.0, 0.0145, math.pi)}
    out["benchmarks"] = {}
    for k, (Md, Me) in pts.items():
        R = L.ratios(Me)[0] / L.ratios(Md)[0]
        out["benchmarks"][k] = {"R": R, "pull": pullR(R)}
        print(f"{k}: R {R:.4f} pull {pullR(R):+.2f}")
    for name in MODELS:
        info = run(name)
        out[name] = info
        print(name, json.dumps(info, default=float))
    (HERE / "bound_R.json").write_text(json.dumps(out, indent=1, default=float))

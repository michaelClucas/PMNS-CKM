"""Parameter-independent bounds on the observables that dominate the AKS and CM misfits (exp 038 bound.py style).

(1) R = (m_e/m_mu)/(m_d/m_s), RG-invariant (exp 021), scored alone (key R_clebsch, floor sqrt2 * 2%, exp 040
    convention). Minimum R pull over each model's d/e parameters with m_e/m_mu at data (penalty) and loose GUT
    bands m_mu/m_tau in [0.050, 0.068], m_s/m_b in [0.012, 0.030] (exp 040 bound_R.py). Also the maximum
    reachable mr_d_s pull bound: m_d/m_s range at the R minimum.
    AKS: e2, e102, e23 (Y_d / eps3), Eq. (3.8).  CM: b, c, Eq. (14).
(2) AKS PMNS: minimum of the official PMNS chi^2 (s12sq, s13sq, s23sq, dcp_deg, dm_ratio; NO; AKS scale) over eps
    and the d/e parameters, with m_e/m_mu within 10% of data and m_mu/m_tau in the band above. The up sector and
    every quark observable are left free, so this bounds the lepton-sector chi^2 for any quark fit.
(3) CM delta_CKM: range of delta_q over (b, c, k, h, g) with ckm_s12, mr_u_c, mr_c_t, mr_d_s within 3 sigma
    (official scorer, CM scale) and s13/s23 within 20% of the data ratio (both rescale by the same threshold factor,
    so the bound holds with any eta). The pulls of the extremes are reported (official scorer, ckm_delta_rad).
(4) CM PMNS: as (2), over (b, c) and the neutrino angles (th, ph), CM scale, NO.

    python bound42.py -> bound42.json, stdout
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

from src import evaluate as ev  # noqa: E402
import fit42 as F  # noqa: E402
import lib42 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
V, _, _ = ev.mass_ratio_targets()
R0 = V["mr_e_mu"] / V["mr_d_s"]
RFLOOR = {"R_clebsch": 0.02 * math.sqrt(2)}
SCR = ev.Scorer(["R_clebsch"], "NO", mass_floor=RFLOOR)
SIG = (1 / 9 - R0) / SCR({"R_clebsch": 1 / 9}).pulls["R_clebsch"]
BIG = 1e6


def sv(M):
    return np.linalg.svd(np.asarray(M, complex), compute_uv=False)[::-1]


def band(x, lo, hi):
    return 0.0 if lo <= x <= hi else (math.log(x / lo) if x < lo else math.log(x / hi)) ** 2


def de(f, bounds, seeds=(1, 2), popsize=30, maxiter=1000):
    best = None
    for s in seeds:
        r = differential_evolution(f, bounds, seed=s, popsize=popsize, maxiter=maxiter, tol=1e-12, polish=False)
        r2 = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000})
        x = r2.x if r2.fun < r.fun else r.x
        if best is None or f(x) < f(best):
            best = x
    return best


# ---------------------------------------------------------------- (1) R
def r_obj(Md, Me, sign=1.0):
    d, e = sv(Md), sv(Me)
    if min(d[0], e[0]) <= 0 or not np.all(np.isfinite([*d, *e])):
        return BIG, None
    rd, re = (d[0] / d[1], d[1] / d[2]), (e[0] / e[1], e[1] / e[2])
    R = re[0] / rd[0]
    pen = 1e4 * (math.log(re[0] / V["mr_e_mu"]) ** 2 + band(re[1], 0.050, 0.068) + band(rd[1], 0.012, 0.030))
    val = ((R - R0) / SIG) ** 2 if sign > 0 else -R
    return val + pen, dict(R=R, me_mmu=re[0], mmu_mtau=re[1], md_ms=rd[0], ms_mb=rd[1])


def aks_de(x):
    p = dict(e2=1e-5 * np.sinh(x[0]), e102=1e-5 * np.sinh(x[1]), e23=1e-5 * np.sinh(x[2]), e3=1.0)
    return L.aks_Yd(**p), L.aks_Ye(**p)


def cm_de(x):
    p = dict(b=x[0], c=x[1])
    return L.cm_Md(**p).T, L.cm_Me(**p).T


R_MODELS = {"AKS": ([(-10, 10)] * 3, aks_de), "CM": ([(-0.03, 0.03), (-0.1, 0.1)], cm_de)}


def bound_R(name):
    bounds, build = R_MODELS[name]
    res = {}
    for lab, sgn in (("min_pull", 1.0), ("max_R", -1.0)):
        x = de(lambda x: r_obj(*build(x), sgn)[0], bounds)
        val, info = r_obj(*build(x), sgn)
        info.update(x=[float(v) for v in x], penalty_ok=bool(abs(val - (((info["R"] - R0) / SIG) ** 2 if sgn > 0
                                                                        else -info["R"])) < 1e-3),
                    pull_R=SCR({"R_clebsch": info["R"]}).pulls["R_clebsch"])
        res[lab] = info
    return res


# ---------------------------------------------------------------- (2), (4) PMNS-only lower bounds
PM = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]


def pmns_bound(model):
    if model == "AKS":
        sc = ev.Scorer(PM, "NO", mass_floor=0.02, scale=F.AKS_SCALE)
        bounds = [(-10, 10)] * 3 + [(0.01, 1.0)]

        def build(x):
            p = dict(L.AKS_T4, e2=1e-5 * np.sinh(x[0]), e102=1e-5 * np.sinh(x[1]), e23=1e-5 * np.sinh(x[2]),
                     e3=1.0, eps=x[3])
            return L.aks_obs(p)
    else:
        sc = ev.Scorer(PM, "NO", mass_floor=0.02, scale=F.CM_SCALE)
        bounds = [(-0.03, 0.03), (-0.1, 0.1), (0.0, np.pi), (0.0, 2 * np.pi)]

        def build(x):
            p = dict(L.CM_EQ16, b=x[0], c=x[1], xi=np.cos(x[2]), eta=np.sin(x[2]) * np.cos(x[3]),
                     etapp=np.sin(x[2]) * np.sin(x[3]))
            return L.cm_obs(p, conj=True)

    def f(x):
        try:
            o = build(x)
        except (np.linalg.LinAlgError, ValueError):
            return BIG
        pen = 1e4 * (band(o["mr_e_mu"], 0.9 * V["mr_e_mu"], 1.1 * V["mr_e_mu"]) + band(o["mr_mu_tau"], 0.050, 0.068))
        return sc.chi2_soft(o, include_finite=True) + pen

    x = de(f, bounds, popsize=20, maxiter=400)
    o = build(x)
    r = sc(o)
    return {"chi2_pmns_min": r.chi2, "pulls": r.pulls, "pred_gut": {k: o[k] for k in PM + ["mr_e_mu", "mr_mu_tau"]},
            "x": [float(v) for v in x]}


# ---------------------------------------------------------------- (3) CM delta_CKM range
def cm_delta_range():
    keys = ["ckm_s12", "mr_u_c", "mr_c_t", "mr_d_s"]
    sc = ev.Scorer(keys, "NO", mass_floor=0.02, scale=F.CM_SCALE)
    sd = ev.Scorer(["ckm_delta_rad"], "NO", scale=F.CM_SCALE)
    bounds = [(-0.03, 0.03), (-0.1, 0.1), (-0.2, 0.2), (-0.05, 0.05), (-10, 10)]

    def build(x):
        return L.cm_obs(dict(L.CM_EQ58, b=x[0], c=x[1], k=x[2], h=x[3], g=1e-7 * np.sinh(x[4])), conj=True)

    # target ratio from the scorer: the pred values that give zero pull
    zero = {}
    for key in ("ckm_s13", "ckm_s23"):
        s1 = ev.Scorer([key], "NO", scale=F.CM_SCALE)
        lo, hi = 1e-4, 0.2
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if s1({key: mid}).pulls[key] < 0:
                lo = mid
            else:
                hi = mid
        zero[key] = 0.5 * (lo + hi)
    rat0 = zero["ckm_s13"] / zero["ckm_s23"]

    def pen(o):
        r = sc(o)
        p = sum(max(0.0, abs(v) - 3.0) ** 2 for v in r.pulls.values())
        return p + band(o["ckm_s13"] / o["ckm_s23"], 0.8 * rat0, 1.2 * rat0)

    out = {"s13_s23_data_ratio_gut": rat0, "zero_pull_gut": zero}
    for lab, sgn in (("min", 1.0), ("max", -1.0)):
        def f(x):
            try:
                o = build(x)
            except (np.linalg.LinAlgError, ValueError):
                return BIG
            d = o["ckm_delta_rad"]
            d = d - 2 * np.pi if d > np.pi else d
            return sgn * d + 1e3 * pen(o)
        x = de(f, bounds, popsize=20, maxiter=600)
        o = build(x)
        d = o["ckm_delta_rad"]
        out[lab] = {"delta_rad": d, "delta_deg": math.degrees(d), "penalty": pen(o),
                    "pull": sd(o).pulls["ckm_delta_rad"], "x": [float(v) for v in x],
                    "others": sc(o).pulls, "s13_s23": o["ckm_s13"] / o["ckm_s23"]}
    return out


if __name__ == "__main__":
    out = {"R_data": R0, "sigma_R": SIG}
    print(f"R data {R0:.5f} sigma {SIG:.5f}", flush=True)
    for pt, (Md, Me) in {"AKS_T4": (L.aks_Yd(**L.AKS_T4), L.aks_Ye(**L.AKS_T4)),
                         "CM_Eq16": (L.cm_Md(**L.CM_EQ16).T, L.cm_Me(**L.CM_EQ16).T)}.items():
        d, e = sv(Md), sv(Me)
        R = (e[0] / e[1]) / (d[0] / d[1])
        out[pt] = {"R": R, "pull": SCR({"R_clebsch": R}).pulls["R_clebsch"]}
        print(pt, out[pt], flush=True)
    for m in ("AKS", "CM"):
        out["R_" + m] = bound_R(m)
        print("R", m, json.dumps(out["R_" + m], default=float), flush=True)
    out["CM_delta"] = cm_delta_range()
    print("CM delta", json.dumps(out["CM_delta"], default=float), flush=True)
    for m in ("AKS", "CM"):
        out["PMNS_" + m] = pmns_bound(m)
        print("PMNS", m, json.dumps(out["PMNS_" + m], default=float), flush=True)
    (HERE / "bound42.json").write_text(json.dumps(out, indent=1, default=float))

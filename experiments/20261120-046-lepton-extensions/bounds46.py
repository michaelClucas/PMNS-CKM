"""Exp 046 parameter-independent bounds (extensions.md, Tests 1). Output: bounds.json, printed log.

(a) RRR5 + Georgi-Jarlskog: the 2x2 light blocks [[0,F],[F*,E]] (down) and [[0,F],[F*,-3E]] (charged
    leptons) give m_s m_d = m_e m_mu = |F|^2, m_s - m_d = |E|, m_mu - m_e = 3|E| exactly (Y_d block
    diagonal), so m_e/m_mu is a function of m_d/m_s alone. Minimise the (mr_d_s, mr_e_mu) marginal of the
    official mass-ratio block over m_d/m_s, at GUT tan beta 30 for eta_b in a grid (R is threshold-
    invariant; the grid shows it).
(b) HS minimal SU(5) / Pati-Salam: m_e/m_mu = m_d/m_s and m_mu/m_tau = m_s/m_b at the model scale; minimise
    the (mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau) marginal over the two free ratios.
(c) M_R = 1: dm_ratio = (y2^4 - y1^4)/(y3^4 - y1^4) of Y_nu's singular values; evaluate with the measured
    ratios (u or d sector) and with each ratio at +5 sigma (2% floor).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib46 as L  # noqa: E402
from src import evaluate as E  # noqa: E402


def gj_lepton_ratio(x):
    """m_e/m_mu from m_d/m_s = x under GJ (m_s = 1)."""
    E_ = 1.0 - x
    F2 = x
    s = 3 * E_
    m_mu = 0.5 * (s + np.sqrt(s * s + 4 * F2))
    m_e = F2 / m_mu
    return m_e / m_mu


def bound_gj(eta_b):
    sc = E.Scorer(["mr_d_s", "mr_e_mu"], "NO", mass_floor=0.02, scale=dict(L.SCALES["G30"], eta_b=eta_b))
    f = lambda lx: sc.chi2({"mr_d_s": np.exp(lx), "mr_e_mu": gj_lepton_ratio(np.exp(lx))})
    r = minimize_scalar(f, bounds=(np.log(0.01), np.log(0.2)), method="bounded", options={"xatol": 1e-10})
    x = float(np.exp(r.x))
    res = sc({"mr_d_s": x, "mr_e_mu": gj_lepton_ratio(x)})
    return {"eta_b": eta_b, "chi2": res.chi2, "md_ms_gut": x, "me_mmu_gut": gj_lepton_ratio(x),
            "R": gj_lepton_ratio(x) / x, "pulls": res.pulls}


def bound_su5(sc_tag):
    keys = ["mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau"]
    sc = E.Scorer(keys, "NO", mass_floor=0.02, scale=L.SCALES[sc_tag])
    f = lambda z: sc.chi2({"mr_d_s": np.exp(z[0]), "mr_e_mu": np.exp(z[0]),
                           "mr_s_b": np.exp(z[1]), "mr_mu_tau": np.exp(z[1])})
    best = min((minimize(f, z0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
                for z0 in ([np.log(0.05), np.log(0.02)], [np.log(0.005), np.log(0.06)], [np.log(0.015), np.log(0.035)])),
               key=lambda r: r.fun)
    a, b = np.exp(best.x)
    res = sc({"mr_d_s": a, "mr_e_mu": a, "mr_s_b": b, "mr_mu_tau": b})
    return {"scale": sc_tag, "chi2": res.chi2, "common_12": float(a), "common_23": float(b), "pulls": res.pulls}


def bound_mr1():
    out = {}
    tg = E.mass_ratio_targets()[0]
    for sec, (r12, r23) in {"u": ("mr_u_c", "mr_c_t"), "d": ("mr_d_s", "mr_s_b")}.items():
        for tag, f in (("central", 1.0), ("+5sigma(2%)", 1.10)):
            a, b = tg[r12] * f, tg[r23] * f
            y = np.array([a * b, b, 1.0])
            dr = (y[1] ** 4 - y[0] ** 4) / (y[2] ** 4 - y[0] ** 4)
            out[f"{sec} {tag}"] = {"dm_ratio": dr, "delta_chi2": E.dm_ratio_delta_chi2("NO", dr)}
    return out


if __name__ == "__main__":
    res = {"gj": [bound_gj(e) for e in (0.0, -0.15, -0.25, -0.35)],
           "su5": [bound_su5(s) for s in ("SM5", "SM7", "SM9")],
           "mr1": bound_mr1(), "dm_ratio_data_NO": E._dm_best("NO", E._data.DEFAULT_ANALYSIS)}
    for r in res["gj"]:
        print(f"GJ eta_b {r['eta_b']:+.2f}: chi2 >= {r['chi2']:.2f}  md/ms {r['md_ms_gut']:.4f} me/mmu "
              f"{r['me_mmu_gut']:.5f} R {r['R']:.4f} pulls {r['pulls']}")
    for r in res["su5"]:
        print(f"HS R=1 {r['scale']}: chi2 >= {r['chi2']:.1f} pulls {r['pulls']}")
    for k, v in res["mr1"].items():
        print(f"M_R=1 {k}: dm_ratio {v['dm_ratio']:.3e} delta chi2 {v['delta_chi2']}")
    print("dm_ratio data NO", res["dm_ratio_data_NO"])
    json.dump(res, open(HERE / "bounds.json", "w"), indent=1, default=float)

"""Refit the 2O model to the published inputs of arXiv:2409.15823 (Tables 1-2), starting from its
printed best fit. Checks that our implementation finds their chi2_comb = 20.3 (and the separate
lepton 4.74 / quark ~0 minima). The inputs below are the paper's numbers (GUT scale, MSSM
tan beta = 5, Bari 2021 lepton fit with sin^2 theta23 set to 0.5), not our data.
Writes fit_pub_inputs.json.
"""
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

from model_2o import observables
from reproduce_2o import P_COMB, P_LEP, P_QRK

HERE = pathlib.Path(__file__).parent

# Table 1: best fit and 1 sigma = (3 sigma range)/6 as fractional accuracy
LEP = {"dm21": (7.36e-5, 0.023), "s12sq": (0.303, 0.045), "DM2": (2.485e-3, 0.011),
       "s13sq": (0.0223, 0.030), "s23sq": (0.5, 0.067),
       "mtau": (1.293, 0.007 / 1.293), "r_emu": (0.00474, 0.00004 / 0.00474), "r_mutau": (0.0588, 0.0005 / 0.0588)}
QRK = {"mt": (89.213, 2.219 / 89.213), "r_uc": (0.00193, 0.00060 / 0.00193), "r_ct": (0.00280, 0.00012 / 0.00280),
       "mb": (0.965, 0.011 / 0.965), "r_ds": (0.0505, 0.0062 / 0.0505), "r_sb": (0.0182, 0.0010 / 0.0182),
       "th12q": (0.2274, 0.0007 / 0.2274), "th13q": (0.00349, 0.00013 / 0.00349), "th23q": (0.0400, 0.0006 / 0.0400),
       "dq": (np.radians(69.21), np.radians(3.11) / np.radians(69.21))}
# paper footnote: sigma(s12sq) = 0.013 used for the combined chi2
LEP["s12sq"] = (0.303, 0.013 / 0.303)

NAMES = ["re", "im", "gE1", "rE2", "rE3", "nu_scale", "gu1", "ru2", "ru3", "gd1", "rd2", "rd3", "rd4", "rd5"]
LEP_P = ["re", "im", "gE1", "rE2", "rE3", "nu_scale"]
QRK_P = ["re", "im", "gu1", "ru2", "ru3", "gd1", "rd2", "rd3", "rd4", "rd5"]


def pred(p):
    o = observables(p)
    o["mtau"] = o["mtau"]  # gE1 in GeV (the paper's "MeV" label is a typo: MeV gives m_tau = 1.3 MeV)
    o["dq"] = (o["dq"] + np.pi) % (2 * np.pi) - np.pi
    return o


def resid(x, names, base, targets):
    p = dict(base)
    p.update(zip(names, x))
    if p["im"] <= 0.5:
        return np.full(len(targets), 1e3)
    o = pred(p)
    return np.array([(o[k] - v) / (s * v) for k, (v, s) in targets.items()])


def fit(names, base, targets, n_jitter=30, seed=0):
    rng = np.random.default_rng(seed)
    x0 = np.array([base[n] for n in names])
    best = None
    for j in range(n_jitter):
        x = x0 if j == 0 else x0 * (1 + 0.02 * rng.standard_normal(len(x0)))
        r = least_squares(resid, x, args=(names, base, targets), method="lm", max_nfev=20000, x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    p = dict(base)
    p.update(zip(names, best.x))
    res = resid(best.x, names, base, targets)
    return p, float(np.sum(res ** 2)), {k: float(v) for k, v in zip(targets, res)}


def main():
    out = {}
    at_pub = {}
    for tag, P, T in (("comb", P_COMB, {**LEP, **QRK}), ("lep", P_LEP, LEP), ("qrk", P_QRK, QRK)):
        r = resid(np.array([P[n] for n in NAMES]), NAMES, P, T)
        at_pub[tag] = float(np.sum(r ** 2))
    print("chi2 at printed points (published: comb 20.3, lep 4.74, qrk 0.0025):", at_pub)
    out["chi2_at_printed_point"] = at_pub
    for tag, names, P, T in (("lep", LEP_P, P_LEP, LEP), ("qrk", QRK_P, P_QRK, QRK), ("comb", NAMES, P_COMB, {**LEP, **QRK})):
        p, c2, pulls = fit(names, P, T)
        out[tag] = {"chi2": c2, "params": {n: float(p[n]) for n in names}, "pulls": pulls}
        print(f"refit {tag}: chi2 {c2:.3f}; tau = {p['re']:+.4f} + {p['im']:.4f} i")
        print("   pulls:", {k: round(v, 2) for k, v in pulls.items()})
    (HERE / "fit_pub_inputs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

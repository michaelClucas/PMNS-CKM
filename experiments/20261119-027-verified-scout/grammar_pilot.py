"""Pilot look-elsewhere check for the bi-large (Cabibbo-seed) family. NOT the criterion-4 protocol (no power check,
no translated tables for N2); informs whether a pre-registered test is worth running.

Grammar (frozen here, before looking at the outcome): U = U_l(lambda, A, phi)^dag U_nu with
  sin th12nu, sin th13nu, sin th23nu each in {n lambda, 1 - n lambda : n = 1..4}   (8^3 = 512)
  delta_nu in {0, pi}                                                              (x2)
  U_l in {CDSV-I form (SO(10)), CDSV-II form (SU(5))}                              (x2)
  phi fitted on a 120-point grid (3 deg)                                           -> 2048 configs, 11 bits + 1 phase
Score: vectorized NuFIT 6.1 table chi^2 (exp 009 f2lib.TableChi2 = src.data.pmns_delta_chi2, checked there), NO.
Real data: rank of CDSV-I/II inside the grammar. Null (N1-style, STATE ranges): 100 fake hierarchical CKMs
(lambda U[0.05, 0.5], A U[0.3, 2]); hit = fake best chi^2 <= real best chi^2. Options with |sin| > 1 are invalid.
Run: ../../.venv/Scripts/python grammar_pilot.py > grammar_pilot.log   (~2 min, 1 core)
"""
import itertools
import json
import sys

import numpy as np
from vlib import ROOT, ckm_central
sys.path.insert(0, str(ROOT / "experiments" / "20261118-009-f2-clebsch"))
from f2lib import TableChi2  # noqa: E402
from src.mixing import extract_angles  # noqa: E402

PHI = np.linspace(0, 2 * np.pi, 120, endpoint=False)
OPTS = [("l", n) for n in range(1, 5)] + [("1-l", n) for n in range(1, 5)]


def val(o, lam):
    return o[1] * lam if o[0] == "l" else 1 - o[1] * lam


def R12(t):
    c, s = np.cos(t), np.sin(t)
    M = np.zeros(np.shape(t) + (3, 3), complex); M[..., 0, 0] = c; M[..., 0, 1] = s; M[..., 1, 0] = -s; M[..., 1, 1] = c; M[..., 2, 2] = 1
    return M


def R23(t):
    c, s = np.cos(t), np.sin(t)
    M = np.zeros(np.shape(t) + (3, 3), complex); M[..., 1, 1] = c; M[..., 1, 2] = s; M[..., 2, 1] = -s; M[..., 2, 2] = c; M[..., 0, 0] = 1
    return M


def Pd(a, b):
    M = np.zeros(np.shape(a) + (3, 3), complex); M[..., 0, 0] = np.exp(1j * a); M[..., 1, 1] = np.exp(1j * b); M[..., 2, 2] = 1
    return M


def Ul_batch(lam, A, form):
    t12, t23 = np.arcsin(lam), np.arcsin(A * lam ** 2)
    if form == 0:
        return R23(t23) @ Pd(-PHI / 2, PHI / 2) @ R12(t12) @ Pd(PHI / 2, -PHI / 2)
    F = Pd(-PHI / 2, PHI / 2)
    return F @ np.swapaxes(R12(t12), -1, -2) @ F.conj() @ np.swapaxes(R23(t23), -1, -2)


def Unu(s12, s13, s23, d):
    t12, t13, t23 = np.arcsin(s12), np.arcsin(s13), np.arcsin(s23)
    c13 = np.cos(t13)
    U13 = np.array([[c13, 0, s13 * np.exp(-1j * d)], [0, 1, 0], [-s13 * np.exp(1j * d), 0, c13]])
    return R23(t23) @ U13 @ R12(t12)


def search(lam, A, tab):
    """best chi2 over the grammar; returns (best, config) and per-config minima."""
    Uls = [Ul_batch(lam, A, f).conj().swapaxes(-1, -2) for f in (0, 1)]
    best, arg, allmin = np.inf, None, {}
    for o12, o13, o23 in itertools.product(OPTS, repeat=3):
        s = [val(o, lam) for o in (o12, o13, o23)]
        if any(not (0 < x < 1) for x in s):
            continue
        for d in (0.0, np.pi):
            un = Unu(*s, d)
            for f in (0, 1):
                U = Uls[f] @ un
                t12, t13, t23, dl = extract_angles(U)
                c = tab(np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2, np.degrees(dl))
                m = float(np.min(c))
                allmin[(o12, o13, o23, d, f)] = m
                if m < best:
                    best, arg = m, (o12, o13, o23, d, f, float(PHI[int(np.argmin(c))]))
    return best, arg, allmin


if __name__ == "__main__":
    tab = TableChi2("NO")
    c = ckm_central()
    best, arg, allmin = search(c["lambda"], c["A"], tab)
    vals = np.array(list(allmin.values()))
    key_I = (("l", 2), ("l", 1), ("1-l", 1), np.pi, 0)
    key_II = (("l", 2), ("l", 1), ("l", 3), np.pi, 1)
    out = {"n_configs": len(allmin), "real_best": best, "real_best_config": str(arg),
           "CDSV_I_grid": allmin[key_I], "CDSV_II_grid": allmin[key_II],
           "n_le_CDSV_I": int(np.sum(vals <= allmin[key_I])), "n_lt_4": int(np.sum(vals < 4)),
           "n_lt_8": int(np.sum(vals < 8))}
    print(json.dumps(out, indent=1))
    rng = np.random.default_rng(2027)
    fakes = []
    for i in range(100):
        lam, A = rng.uniform(0.05, 0.5), rng.uniform(0.3, 2.0)
        b, a, _ = search(lam, A, tab)
        fakes.append(b)
    fakes = np.array(fakes)
    out["fake_best_quantiles"] = np.quantile(fakes, [0.05, 0.25, 0.5, 0.75]).tolist()
    out["fake_hits_le_real_best"] = int(np.sum(fakes <= best))
    out["fake_hits_le_CDSV_I"] = int(np.sum(fakes <= allmin[key_I]))
    print(json.dumps({k: out[k] for k in ("fake_best_quantiles", "fake_hits_le_real_best", "fake_hits_le_CDSV_I")}))
    json.dump(out, open("grammar_pilot.json", "w"), indent=1)

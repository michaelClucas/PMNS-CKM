"""Exp 035 (b): Dirac neutrinos with Y_nu = Y_u up to GUT Clebsch factors.

Structure read from arXiv:2511.16196 (Chen, Fang, Zhou), model M1, appendix eq. (12)
(renormalizable SO(10), complex 10_H + 126bar_H; basis Y_126 real diagonal):
    Y_u  = S + D
    Y_d  = r2 S + e^{i w} r1 D
    Y_e  = r2 S - 3 e^{i w} r1 D
    Y_nu = S - 3 D                  (paper: Majorana, M_R = v_R D, type-I seesaw)
S complex symmetric (10_H), D real diagonal (126bar_H). The -3 is the 126 Clebsch.
This experiment replaces the seesaw by DIRAC neutrinos, m_nu = Y_nu v_nu (separate tiny vev,
absolute scale not scored), and generalizes the two Clebsch factors:
    Y_e  = r2 S + c_e e^{i w} r1 D,     Y_nu = S + c_nu D,     c_e, c_nu in {1, -3, 1/3, 9/2, 4/9, 6}
36 pairs = 6 bits. (c_e, c_nu) = (-3, -3) is the paper's structure; c = 1 sets Y_e = Y_d / Y_nu = Y_u.
Exact limit (S or D alone): Y_nu = Y_u (or Y_u^T, same singular values), see exact_limit().

Parameters (continuous, k_cont = 17): D = diag(d1, d2, 1) (overall scale cancels in every scored
ratio; d_i >= 0 wlog, signs absorbed by rephasing into the complex S), S: 6 log10-moduli + 6 phases,
r1, r2 (log10), w. r1 and r2 do not cancel: they set the relative weight of S and D in Y_d / Y_e.
N = 16: CKM 4 (ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad), PMNS 4, mass ratios 7 (mr_u_c, mr_c_t,
mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau), dm_ratio. Scored at GUT scale: predictions are
given at mu = 2e16 GeV, scale={"mu": 2e16, "model": "SM"} (src.evaluate translation with src.rg;
PMNS running there is the Majorana kappa running at m_lightest = 0, negligible in the SM for
hierarchical spectra, and a Dirac spectrum runs with y_tau only, also negligible).
Dirac U_nu: left rotation of Y_nu (src.yukawa.left_rotation), passed to the engine as the symmetric
matrix m = U_nu^* diag(y) U_nu^dag, whose Takagi factor is U_nu (same PMNS, same singular values).
Usage: python dirac.py <c_e> <c_nu> <ordering> <seed> [maxiter]   -> JSON line on stdout
       python dirac.py exact                                       -> exact-limit numbers
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from src import evaluate as E  # noqa: E402
from src import rg  # noqa: E402
from src.yukawa import left_rotation, observables  # noqa: E402
from src.stats import discrete_bits  # noqa: E402

CLEBSCH = {"1": 1.0, "-3": -3.0, "1/3": 1 / 3, "9/2": 4.5, "4/9": 4 / 9, "6": 6.0}
USE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg",
       "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "mr_b_tau", "dm_ratio"]
SCALE = {"mu": 2e16, "model": "SM"}
IJ = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
BOUNDS = {"ld1": (-7.0, 0.5), "ld2": (-5.0, 0.5)}
for a, b in IJ:
    BOUNDS[f"lS{a}{b}"] = (-8.0, 0.5)
    BOUNDS[f"pS{a}{b}"] = (0.0, 2 * np.pi)
BOUNDS.update({"lr1": (-3.0, 2.0), "lr2": (-3.0, 2.0), "w": (0.0, 2 * np.pi)})
PERIODIC = [k for k in BOUNDS if k.startswith("pS")] + ["w"]


def dirac_mnu(Y_nu):
    U, y = left_rotation(Y_nu)
    return U.conj() @ np.diag(y) @ U.conj().T


def matrices(th, c_e, c_nu):
    D = np.diag([10 ** th["ld1"], 10 ** th["ld2"], 1.0]).astype(complex)
    S = np.zeros((3, 3), complex)
    for a, b in IJ:
        S[a, b] = S[b, a] = 10 ** th[f"lS{a}{b}"] * np.exp(1j * th[f"pS{a}{b}"])
    r1, r2, ph = 10 ** th["lr1"], 10 ** th["lr2"], np.exp(1j * th["w"])
    Yu = S + D
    Yd = r2 * S + ph * r1 * D
    Ye = r2 * S + c_e * ph * r1 * D
    Yn = S + c_nu * D
    return Yu, Yd, Ye, Yn


def model_factory(c_e, c_nu, ordering):
    def model(th):
        Yu, Yd, Ye, Yn = matrices(th, c_e, c_nu)
        return observables(Yu, Yd, Ye, m_nu=dirac_mnu(Yn), ordering=ordering)
    return model


def fit(c_e_s, c_nu_s, ordering, seed, maxiter=100, popsize=8):
    """Budget-capped global fit: scipy DE on the official Scorer.chi2_soft(include_finite=True) landscape
    (the objective fit_and_evaluate's "de" method uses), L-BFGS-B polish, then the official
    src.evaluate.evaluate at the optimum. fit_and_evaluate(method="de") itself was too slow here
    (its unconditional Nelder-Mead polish of up to 68k evaluations at 2.3 ms each)."""
    from scipy.optimize import differential_evolution, minimize
    c_e, c_nu = CLEBSCH[c_e_s], CLEBSCH[c_nu_s]
    t0 = time.time()
    model = model_factory(c_e, c_nu, ordering)
    names = list(BOUNDS)
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])
    sc = E.Scorer(USE, ordering, scale=SCALE)
    nfev = [0]

    def f(x):
        nfev[0] += 1
        try:
            c = sc.chi2_soft(model(dict(zip(names, x))), include_finite=True)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e9
        return c if np.isfinite(c) else 1e9

    r = differential_evolution(f, list(zip(lo, hi)), seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-8,
                               init="sobol", polish=False)
    p = minimize(f, r.x, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxfun=4000))
    x = p.x if p.fun < r.fun else r.x
    th = dict(zip(names, map(float, x)))
    kd = discrete_bits(len(CLEBSCH) ** 2)
    res = E.evaluate(model(th), ordering, use=USE, k=len(names) + kd, k_discrete=kd, scale=SCALE)
    return dict(c_e=c_e_s, c_nu=c_nu_s, ordering=ordering, seed=seed, chi2=res.chi2, chi2_global=res.chi2_global,
                N=res.N, k=res.k, k_cont=res.k_continuous, dof=res.dof, p_global=res.p_global,
                max_pull=res.max_abs_pull, pulls=res.pulls, params=th, nfev=nfev[0], secs=time.time() - t0,
                de_chi2=float(r.fun), maxiter=maxiter, popsize=popsize)


def exact_limit():
    """Y_nu = Y_u (or Y_u^T) and Y_e = Y_d (or Y_d^T) exactly (any common overall factor).

    Singular values of Y^T equal those of Y, so every exact variant predicts
    mr_e_mu = mr_d_s, mr_mu_tau = mr_s_b, mr_b_tau = 1, dm_ratio = (y_c^2 - y_u^2)/(y_t^2 - y_u^2)
    (NO labelling), whatever the right-handed rotations. Fitted parameters: the four quark ratios
    at 2e16 GeV (k_cont = 4); scored: all 7 mass ratios + dm_ratio (N = 8) at GUT scale."""
    def model(th):
        uc, ct, ds, sb = (10 ** th[k] for k in ("luc", "lct", "lds", "lsb"))
        yu, yc = uc * ct, ct
        return {"mr_u_c": uc, "mr_c_t": ct, "mr_d_s": ds, "mr_s_b": sb, "mr_e_mu": ds, "mr_mu_tau": sb,
                "mr_b_tau": 1.0, "dm_ratio": (yc ** 2 - yu ** 2) / (1 - yu ** 2)}
    use = ["mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "mr_b_tau", "dm_ratio"]
    b = {"luc": (-5, -0.5), "lct": (-5, -0.5), "lds": (-4, -0.3), "lsb": (-4, -0.3)}
    out = {}
    for o in ("NO", "IO"):
        fr = E.fit_and_evaluate(model, b, o, use, method="lbfgsb", n_starts=16, scale=SCALE)
        r = fr.result
        out[o] = dict(chi2=r.chi2, chi2_global=r.chi2_global, N=r.N, k=r.k, dof=r.dof, p_global=r.p_global,
                      max_pull=r.max_abs_pull, pulls=r.pulls, params=fr.params)
    out["note"] = ("PMNS: Y_e = Y_d and Y_nu = Y_u give U_PMNS = V_CKM^dag (s13^2 ~ 1e-5); transposed variants "
                   "leave PMNS to unconstrained right-handed rotations (not predicted). The model has no IO "
                   "solution (hierarchical y), IO row shows the forced-IO score of the same spectrum.")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "exact":
        print(json.dumps(exact_limit(), default=float))
    else:
        mi = int(sys.argv[5]) if len(sys.argv) > 5 else 100
        print(json.dumps(fit(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), mi), default=float), flush=True)

"""Official-scorer fits of the two other filter survivors (README): Albright-Barr lopsided SO(10)
(hep-ph/0502161 Eqs. 2-5, delta_N = delta'_N = 0) and the UTZ model (1710.01741 Eq. V.1, LO).

    python fit_more.py AB  <seed> [tan_beta]   N 16 (CKM 4, 7 mass ratios, PMNS 4, dm_ratio), k_cont 12
    python fit_more.py UTZ <seed> [tan_beta]   N 15 (no mr_b_tau: c_d, c_e are messenger-dependent), k_cont 10

Scale: 2e16 GeV, MSSM (both models are SUSY GUTs), tan_beta fixed before fitting (AB 5, the paper's
value; UTZ 30, "larger tan beta" per the paper); NO declared (hierarchical spectra, no ordering bit).
AB: a, b, c complex (log-modulus, phase); sigma, eps, delta, delta', phi, eta. 1 bit (model version).
UTZ: (a, b)_e, gamma_e, delta_e shared with d (b_d = -b_e/3), (a, b)_u with (gamma, delta)_u = 0,
(a, b)_nu with (gamma, delta)_nu = (2 pi/5, 0), Majorana (x, y) with (rho, phi) = (0, -2 pi/5), all as in
Table III; the six discrete phases are fixed, not fitted. Bits: README (4 generous, 10 strict).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src import yukawa as yk  # noqa: E402
import lib40 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
CKM = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad"]
MR6 = ["mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau"]
PMNS = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
FLOOR = 0.02
NAN = {k: float("nan") for k in CKM + MR6 + PMNS + ["mr_b_tau"]}
TP = 2 * np.pi


def ab_model(t):
    p = dict(sigma=t["sigma"], eps=t["eps"], delta=t["delta"], deltap=t["deltap"], phi=t["phi"],
             eta=10 ** t["leta"])
    for k in "abc":
        p[k] = 10 ** t["l" + k] * np.exp(1j * t["p" + k])
    try:
        return L.ab_obs(p)
    except (np.linalg.LinAlgError, ValueError):
        return NAN


AB_B = {"sigma": (0.3, 4.0), "eps": (0.02, 0.5), "delta": (1e-4, 0.05), "deltap": (0.0, 0.05), "phi": (0, TP),
        "leta": (-7.0, -3.0), "la": (-2.0, 2.0), "pa": (0, TP), "lb": (-2.0, 2.0), "pb": (0, TP),
        "lc": (-2.0, 2.0), "pc": (0, TP)}
AB_PER = ["phi", "pa", "pb", "pc"]


def slog(u, s):
    """signed log map: s * sinh(u)."""
    return s * np.sinh(u)


def utz_model(t):
    Md, Me = L.utz_de(t["a_e"], t["b_e"], t["g_e"], t["d_e"], t.get("dp_d", 0.0), np.pi)
    Mu = L.utz_M(t["a_u"], t["b_u"], 0.0, 0.0)
    Mn = L.utz_M(slog(t["ua_n"], 1e-6), slog(t["ub_n"], 1e-6), TP / 5, 0.0)
    # Sec. V relabelling for the Majorana matrix: a -> y (A term, phase rho), b -> x (B term, phase phi)
    Mm = L.utz_M(slog(t["uy"], 1e-14), slog(t["ux"], 1e-14), 0.0, -TP / 5)
    try:
        # Sec. V step 6: M -> M M_R^-1 M^T; src convention needs its complex conjugate (reproduce_utz.py)
        mn = (Mn @ np.linalg.solve(Mm, Mn.T)).conj()
        return yk.observables(Mu, Md, Me, m_nu=0.5 * (mn + mn.T), ordering="NO")
    except (np.linalg.LinAlgError, ValueError):
        return NAN


UTZ_B = {"a_e": (-0.03, 0.03), "b_e": (-0.3, 0.3), "g_e": (-np.pi, np.pi), "d_e": (-np.pi, np.pi),
         "a_u": (-0.003, 0.003), "b_u": (-0.03, 0.03), "ua_n": (-10.0, 10.0), "ub_n": (-10.0, 10.0),
         "ux": (-12.0, 12.0), "uy": (-12.0, 12.0)}
UTZ_PER = ["g_e", "d_e"]
UTZHO_B = {**UTZ_B, "dp_d": (-0.1, 0.1)}  # H.O. term (Eq. II.8), psi_d = pi as in Table III

CFG = {"AB": (ab_model, AB_B, AB_PER, CKM + MR6 + ["mr_b_tau"] + PMNS, 1, 5.0),
       "UTZ": (utz_model, UTZ_B, UTZ_PER, CKM + MR6 + PMNS, 4, 30.0),
       "UTZHO": (utz_model, UTZHO_B, UTZ_PER, CKM + MR6 + PMNS, 5, 30.0)}

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    fn, B, per, use, kd, tb = CFG[name]
    tb = float(sys.argv[3]) if len(sys.argv) > 3 else tb
    sc = {"mu": 2e16, "model": "MSSM", "tan_beta": tb}
    t0 = time.time()
    fr = ev.fit_and_evaluate(fn, B, "NO", use, k_discrete=kd, periodic=per, seed=seed, mass_floor=FLOOR,
                             scale=sc, method="de", optimizer_options={"popsize": 20, "maxiter": 800, "tol": 1e-10})
    r = fr.result
    out = {"model": name, "seed": seed, "tan_beta": tb, "params": fr.params, "chi2": r.chi2,
           "chi2_global": r.chi2_global, "N": r.N, "k": r.k, "k_continuous": r.k_continuous, "k_discrete": r.k_discrete,
           "dof": r.dof, "p_global": r.p_global, "max_abs_pull": r.max_abs_pull, "pulls": r.pulls,
           "pred": dict(r.pred), "compression_ok": r.compression_ok, "fit_ok": r.fit_ok, "nfev": fr.nfev,
           "seconds": time.time() - t0}
    tag = f"fit_{name}_tb{tb:g}_s{seed}"
    (HERE / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("model", "seed", "tan_beta", "chi2", "N", "k", "dof", "p_global",
                                          "max_abs_pull", "compression_ok", "fit_ok", "nfev", "seconds")},
                     default=float))
    print("pulls", {k: round(v, 2) for k, v in out["pulls"].items()})
    print("params", out["params"])

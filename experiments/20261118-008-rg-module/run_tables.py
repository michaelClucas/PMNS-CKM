"""Validation table and running table for src/rg.py.

Run from repo root:
    .venv/Scripts/python experiments/20261118-008-rg-module/run_tables.py > experiments/20261118-008-rg-module/run.log

Published reference values below are literature numbers (not measured data) from
arXiv:2510.01312v2 (Antusch, Hinze, Saad), Table 2 (SM, 2024 PDG inputs) and Table 6
(MSSM, M_SUSY = 3 TeV, M_GUT = 2e16 GeV, 2024 PDG inputs, zero SUSY threshold corrections).
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src import rg  # noqa: E402

D = 180 / np.pi
AHS_SM = {  # mu: {quantity: value}, arXiv:2510.01312v2 Table 2
    1e12: {"theta12": 0.22707, "theta23": 4.650e-2, "theta13": 4.11e-3, "delta": 1.139, "y_u": 3.37e-6,
           "y_c": 1.71e-3, "y_t": 0.5138, "y_d": 0.75e-5, "y_s": 1.50e-4, "y_b": 0.719e-2,
           "y_e": 2.7978e-6, "y_mu": 5.8939e-4, "y_tau": 1.0012e-2, "g1": 0.534498, "g2": 0.55157, "g3": 0.6052},
    1e16: {"theta12": 0.22708, "theta23": 4.743e-2, "theta13": 4.19e-3, "delta": 1.139, "y_u": 2.87e-6,
           "y_c": 1.45e-3, "y_t": 0.4454, "y_d": 0.65e-5, "y_s": 1.29e-4, "y_b": 0.606e-2,
           "y_e": 2.6935e-6, "y_mu": 5.6745e-4, "y_tau": 0.9639e-2, "g1": 0.575240, "g2": 0.52296, "g3": 0.5310},
}
AHS_MSSM = {  # tan beta: values at 2e16 GeV, SM-normalised Yukawas, arXiv:2510.01312v2 Table 6
    10: {"theta12": 0.22702, "theta23": 3.874e-2, "theta13": 3.41e-3, "delta": 1.139, "y_u": 2.76e-6, "y_c": 1.39e-3,
         "y_t": 0.4947, "y_d": 0.49e-5, "y_s": 0.98e-4, "y_b": 0.550e-2, "y_e": 2.0095e-6, "y_mu": 4.2334e-4,
         "y_tau": 0.7223e-2},
    50: {"theta12": 0.22702, "theta23": 3.679e-2, "theta13": 3.24e-3, "delta": 1.139, "y_u": 2.79e-6, "y_c": 1.41e-3,
         "y_t": 0.5312, "y_d": 0.59e-5, "y_s": 1.18e-4, "y_b": 0.766e-2, "y_e": 2.4248e-6, "y_mu": 5.1104e-4,
         "y_tau": 1.0089e-2},
}
MZ_CKM = {"theta12": 0.2270, "theta23": 4.194e-2, "theta13": 3.70e-3}  # Eq. (6)


def flat(c):
    out = {k: c[k] for k in ("theta12", "theta13", "theta23", "delta")}
    out.update(c["yukawas"])
    out.update(dict(zip(("g1", "g2", "g3"), c["g"])))
    return out


def row(name, ours, pub, mz=None):
    s = f"| {name} | {ours:.5g} | {pub:.5g} | {100 * (ours / pub - 1):+.2f}% |"
    s += f" {(ours - mz) / (pub - mz):.3f} |" if mz is not None else " |"
    return s


def validation():
    print("## Validation 1: M_Z Yukawas derived from data/ vs arXiv:2510.01312 Eq. (4)\n")
    print("| quantity | ours | published | rel. diff | |\n|---|---|---|---|---|")
    ours = rg.derived_yukawas_mz()
    for k, v in rg.AHS_EQ4_YUKAWAS.items():
        print(row(k, ours[k], v))
    print("\n## Validation 2: SM running from the published M_Z inputs (Eq. 4, Eq. 6)\n")
    print("Last column: (ours - M_Z value)/(published - M_Z value) for CKM angles = fraction of the RG shift reproduced.\n")
    print("| quantity @ scale | ours | published | rel. diff | shift ratio |\n|---|---|---|---|---|")
    for mu, ref in AHS_SM.items():
        c = flat(rg.ckm_at(mu, yukawas=rg.AHS_EQ4_YUKAWAS, ckm=rg.AHS_EQ6_CKM))
        for k, v in ref.items():
            print(row(f"{k} @ {mu:.0e}", c[k], v, MZ_CKM.get(k)))
    print("\n## Validation 3: MSSM (M_SUSY = 3 TeV) at 2e16 GeV from the published M_Z inputs\n")
    print("| quantity @ tan beta | ours | published | rel. diff | shift ratio |\n|---|---|---|---|---|")
    for tb, ref in AHS_MSSM.items():
        c = flat(rg.ckm_at(2e16, "MSSM", tb, 3000.0, yukawas=rg.AHS_EQ4_YUKAWAS, ckm=rg.AHS_EQ6_CKM))
        for k, v in ref.items():
            print(row(f"{k} @ tb={tb}", c[k], v, MZ_CKM.get(k)))


def analytic_rates(o, ytau, C, phi):
    """hep-ph/0305273 eqs. (8)-(10): leading order in theta13."""
    t12, t13, t23, d = o["theta12"], o["theta13"], o["theta23"], o["delta"]
    m1, m2, m3 = o["m"]
    dm21, datm = m2 ** 2 - m1 ** 2, m3 ** 2 - m2 ** 2
    z = dm21 / datm
    k = C * ytau ** 2 / (32 * np.pi ** 2)
    e1, e2 = np.exp(1j * phi[0]), np.exp(1j * phi[1])
    d12 = -k * np.sin(2 * t12) * np.sin(t23) ** 2 * abs(m1 * e1 + m2 * e2) ** 2 / dm21
    d13 = k * np.sin(2 * t12) * np.sin(2 * t23) * m3 / (datm * (1 + z)) * (
        m1 * np.cos(phi[0] - d) - (1 + z) * m2 * np.cos(phi[1] - d) - z * m3 * np.cos(d))
    d23 = -k * np.sin(2 * t23) / datm * (np.cos(t12) ** 2 * abs(m2 * e2 + m3) ** 2
                                         + np.sin(t12) ** 2 * abs(m1 * e1 + m3) ** 2 / (1 + z))
    return np.array([d12, d13, d23])


def numeric_rates(ordering, ml, phi, model, tb, t13=1e-4, h=1e-6):
    """Central-difference d(theta)/dln(mu) at M_Z, with theta13 small so the LO formulas apply."""
    pm = (33.76 / D, t13, 43.27 / D, 1.0)
    s = rg.build_mz_state(ordering, ml, phi, pmns=pm)
    mz = s["mu"]
    if model == "MSSM":  # MSSM RGEs applied directly at M_Z for a pointwise check
        s = dict(rg._match(s, tb, True), normalisation="MSSM")
        f = lambda mu: rg.run(mz, mu, s, "MSSM", tb, m_susy=0.5 * mz, rtol=1e-12)  # noqa: E731
    else:
        f = lambda mu: rg.run_sm(mz, mu, s, rtol=1e-12)  # noqa: E731
    a, b = (rg.observables(f(mz * np.exp(x)), tb)["pmns"] for x in (h, -h))
    num = np.array([(a[k] - b[k]) / (2 * h) for k in ("theta12", "theta13", "theta23")])
    o = rg.observables(s, tb)
    ytau = o["y"]["y_tau"] / (np.cos(np.arctan(tb)) if model == "MSSM" else 1.0)
    return num, analytic_rates(o["pmns"], ytau, -1.5 if model == "SM" else 1.0, phi)


PMNS_CASES = [("NO", 0.05, (0.0, 0.0), "SM", None), ("IO", 0.05, (0.0, 0.0), "SM", None),
              ("NO", 0.0, (0.0, 0.0), "MSSM", 50), ("NO", 0.1, (0.5, 1.0), "MSSM", 30),
              ("IO", 0.0, (np.pi, 0.0), "MSSM", 50), ("IO", 0.05, (0.0, 0.0), "MSSM", 10)]


def pmns_validation():
    print("\n## Validation 4: PMNS running rates at M_Z vs analytic one-loop formulas (hep-ph/0305273 eqs. 8-10)\n")
    print("theta13 set to 1e-4 rad (the formulas are leading order in theta13); rates in rad per e-fold.\n")
    print("| case | dth12 ours | dth12 analytic | dth13 ours | dth13 analytic | dth23 ours | dth23 analytic |")
    print("|---|---|---|---|---|---|---|")
    for c in PMNS_CASES:
        n, a = numeric_rates(*c)
        ph = tuple(round(x, 2) for x in c[2])
        lab = f"{c[3]}{'' if c[4] is None else ' tb=' + str(c[4])} {c[0]} m0={c[1]} phi={ph}"
        print(f"| {lab} | " + " | ".join(f"{x:.4e} | {y:.4e}" for x, y in zip(n, a)) + " |")


def running_table():
    print("\n## Running table (inputs from data/: CKM PDG 2026, PMNS NuFIT 6.1 default analysis, Majorana phases 0)\n")
    print("CKM angles in degrees.\n")
    print("| model | mu [GeV] | theta12 | theta13 | theta23 | delta | m_u/m_c | m_c/m_t | m_d/m_s | m_s/m_b | m_b/m_tau |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    models = [("SM", None), ("MSSM", 10), ("MSSM", 50)]
    for model, tb in models:
        for mu in (91.1879, 1e12, 2e16):
            c = rg.ckm_at(mu, model, tb)
            r = c["ratios"]
            lab = model if tb is None else f"MSSM tb={tb}"
            print(f"| {lab} | {mu:.4g} | {c['theta12'] * D:.4f} | {c['theta13'] * D:.4f} | {c['theta23'] * D:.4f} | "
                  f"{c['delta'] * D:.2f} | {r['mu_over_mc']:.4e} | {r['mc_over_mt']:.4e} | {r['md_over_ms']:.4f} | "
                  f"{r['ms_over_mb']:.4e} | {r['mb_over_mtau']:.4f} |")
    print("\nPMNS angles in degrees, masses in eV (m = kappa v(M_Z)^2/4, SM normalisation).\n")
    print("| model | ord | m_light | mu [GeV] | theta12 | theta13 | theta23 | delta | m1 | m2 | m3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, tb in models:
        for ordering in ("NO", "IO"):
            for ml in (0.0, 0.01, 0.05):
                for mu in (91.1879, 1e12, 2e16):
                    p = rg.pmns_at(mu, ordering, ml, model=model, tan_beta=tb)
                    lab = model if tb is None else f"MSSM tb={tb}"
                    m = p["m"]
                    print(f"| {lab} | {ordering} | {ml} | {mu:.4g} | {p['theta12'] * D:.3f} | {p['theta13'] * D:.3f} | "
                          f"{p['theta23'] * D:.3f} | {p['delta'] * D:.1f} | {m[0]:.4g} | {m[1]:.4g} | {m[2]:.4g} |")


if __name__ == "__main__":
    validation()
    pmns_validation()
    running_table()

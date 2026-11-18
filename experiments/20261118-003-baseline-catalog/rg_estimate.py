"""Order-of-magnitude RG shift of PMNS angles between M_Z and M_GUT (Majorana, Weinberg operator).

Uses the leading-order analytic RGEs of Antusch, Kersten, Lindner, Ratz, hep-ph/0305273,
eqs. (8)-(10), with constant right-hand side over t = ln(M_GUT/M_Z) (linear approximation).
C = -3/2 (SM), C = 1 (MSSM); y_tau = sqrt2 m_tau / (v cos beta), v = 246 GeV (their conventions).
The linear approximation overestimates/underestimates when the shift is large; any entry
with |shift| > ~2 deg only says "running is large here, integrate numerically".
In the MSSM the operator runs as SM below M_SUSY; ignoring that overstates MSSM running a bit.
Signs: positive = angle grows toward high scale. Phases phi1 = phi2 = 0 maximise theta12 running;
phi1 - phi2 = pi suppresses it.

Run from repo root: .venv/Scripts/python experiments/20261118-003-baseline-catalog/rg_estimate.py
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
with open(os.path.join(ROOT, "data", "pmns.json")) as f:
    pmns = json.load(f)["analyses"]["IC24_wSKatm"]
with open(os.path.join(ROOT, "data", "masses.json")) as f:
    mtau = json.load(f)["charged_leptons"]["m_tau"]["value"]

D = np.pi / 180
V_EW = 246.0          # GeV, convention of hep-ph/0305273
M_Z, M_GUT = 91.19, 2e16  # GeV; M_Z rounded, only enters through a log
T = np.log(M_GUT / M_Z)


def shifts(order, mlight, C, ytau, phi1=0.0, phi2=0.0):
    P = pmns[order]
    t12, t13, t23 = (P[k]["value"] * D for k in ("theta12_deg", "theta13_deg", "theta23_deg"))
    d = P["dcp_deg"]["value"] * D
    dm21 = P["dm21sq_1e-5eV2"]["value"] * 1e-5
    dm3l = abs(P["dm3l_sq_1e-3eV2"]["value"]) * 1e-3
    if order == "NO":
        m1 = mlight
        m2 = np.sqrt(m1**2 + dm21)
        m3 = np.sqrt(m1**2 + dm3l)          # dm3l = dm31 for NO
    else:
        m3 = mlight
        m2 = np.sqrt(m3**2 + dm3l)          # dm3l = dm32 (<0) for IO
        m1 = np.sqrt(m2**2 - dm21)
    datm = m3**2 - m2**2
    zeta = dm21 / datm
    k = C * ytau**2 / (32 * np.pi**2)
    s12, s23, c12, c23 = np.sin(t12), np.sin(t23), np.cos(t12), np.cos(t23)
    e1, e2 = np.exp(1j * phi1), np.exp(1j * phi2)
    d12 = -k * np.sin(2 * t12) * s23**2 * abs(m1 * e1 + m2 * e2) ** 2 / dm21
    d13 = k * np.sin(2 * t12) * np.sin(2 * t23) * m3 / (datm * (1 + zeta)) * (
        m1 * np.cos(phi1 - d) - (1 + zeta) * m2 * np.cos(phi2 - d) - zeta * m3 * np.cos(d))
    d23 = -k * np.sin(2 * t23) / datm * (c12**2 * abs(m2 * e2 + m3) ** 2
                                         + s12**2 * abs(m1 * e1 + m3) ** 2 / (1 + zeta))
    # dtheta/dt with t increasing toward the UV: shift M_Z -> M_GUT
    return np.array([d12, d13, d23]) * T / D


print(f"t = ln(M_GUT/M_Z) = {T:.2f}; NuFIT 1 sigma (IC24 NO): th12 ~0.42, th13 ~0.11, th23 ~0.9 deg")
print(f"{'model':16s} {'ord':3s} {'m_light':>8s} {'phases':>8s} {'d th12':>9s} {'d th13':>9s} {'d th23':>9s}  [deg]")
ytau_sm = np.sqrt(2) * mtau / V_EW
cases = [("SM", -1.5, ytau_sm)]
for tb in (10, 30, 50):
    cb = 1 / np.sqrt(1 + tb**2)
    cases.append((f"MSSM tb={tb}", 1.0, ytau_sm / cb))
for name, C, yt in cases:
    for order in ("NO", "IO"):
        for ml in (0.0, 0.01, 0.05, 0.1):
            for ph, lab in (((0.0, 0.0), "0,0"), ((np.pi, 0.0), "pi,0")):
                s = shifts(order, ml, C, yt, *ph)
                print(f"{name:16s} {order:3s} {ml:8.3f} {lab:>8s} {s[0]:9.3g} {s[1]:9.3g} {s[2]:9.3g}")

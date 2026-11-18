"""Rough one-observable pulls of each baseline vs data/ (low-energy values, no RG running).

Not a chi^2 evaluation: errors treated as independent, asymmetric 1 sigma, no correlations,
no chi^2 profiles. The proper evaluator supersedes this. Reads data/*.json directly because
src/ has no loader function yet; switch to the src loader when it exists.

Run from repo root:  .venv/Scripts/python experiments/20261118-003-baseline-catalog/quick_status.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import baselines as B  # noqa: E402

D = np.pi / 180


def load(name):
    with open(os.path.join(ROOT, "data", name + ".json")) as f:
        return json.load(f)


ckm, pmns, mass = load("ckm"), load("pmns"), load("masses")
sp = ckm["standard_parametrization"]
thC = np.arcsin(sp["s12"]["value"])
thcb = np.arcsin(sp["s23"]["value"])
thub = np.arcsin(sp["s13"]["value"])
dq = sp["delta_rad"]["value"]
Vckm = B.standard_matrix(thC, thub, thcb, dq)


def pull(pred, e):
    v, ep, em = e["value"], e["err_plus"], e["err_minus"]
    return (pred - v) / (ep if pred > v else em)


def row(label, pred, e, unit=""):
    print(f"  {label:44s} pred {pred:9.4f}{unit}  data {e['value']:9.4f}  pull {pull(pred, e):+7.2f}")


print(f"CKM input: theta_C = {thC / D:.3f} deg, theta_cb = {thcb / D:.3f} deg, "
      f"theta_ub = {thub / D:.4f} deg, delta = {dq:.3f} rad")

for an in ("IC23_woSKatm", "IC24_wSKatm"):
    for mo in ("NO", "IO"):
        P = pmns["analyses"][an][mo]
        t12, t13, t23 = (P[k]["value"] * D for k in ("theta12_deg", "theta13_deg", "theta23_deg"))
        print(f"\n=== {an} {mo}")
        q = B.qlc(thC, thcb)
        row("QLC  th12 = 45 - theta_C [deg]", q["th12"] / D, P["theta12_deg"])
        row("QLC  th23 = 45 - theta_cb [deg]", q["th23"] / D, P["theta23_deg"])
        row("QLC' th23 = 45 + theta_cb [deg]", (np.pi / 4 + thcb) / D, P["theta23_deg"])
        row("SC   th12 = 45 - th13(data) [deg]", B.self_complementarity(t13)["th12"] / D, P["theta12_deg"])
        for n in B.LO_TH12:
            row(f"{n:4s} s12^2 (LO)", B.lo_pattern(n)["s2_12"], P["s12sq"])
        row("TBM/BM/GR/HEX s23^2 = 1/2 (LO)", 0.5, P["s23sq"])
        row("TBC  th13 = asin(s_C/sqrt2) [deg]", B.tbc_king(thC)["th13"] / D, P["theta13_deg"])
        row("lepton-GST th13 = asin(sqrt(me/mmu)/sqrt2)",
            B.lepton_gst(mass["charged_leptons"]["m_e"]["value"],
                         mass["charged_leptons"]["m_mu"]["value"])["th13"] / D, P["theta13_deg"])
        row("TM1  s12^2 from th13(data)", np.sin(B.tm1(t13)["th12"]) ** 2, P["s12sq"])
        row("TM2  s12^2 from th13(data)", np.sin(B.tm2(t13)["th12"]) ** 2, P["s12sq"])
        for name, f in (("TM1", B.tm1), ("TM2", B.tm2)):
            cd = f(t13, t23)["cos_delta"]
            print(f"  {name}  cos(delta) from data angles             {cd:+.3f}  "
                  f"(data delta = {P['dcp_deg']['value']:.0f} deg, cos = {np.cos(P['dcp_deg']['value'] * D):+.3f})")
        for n in B.LO_TH12:
            cd = B.solar_sum_rule_cosdelta(t12, t13, t23, B.LO_TH12[n])
            print(f"  solar sum rule {n:4s} cos(delta)               {cd:+.3f}")
        # V_CKM^dag P X: best over the two free phases (k = 2 + choice of X)
        for order in ("Vdag_X", "V_X_Vdag"):
            for n in B.LO_TH12:
                best = (np.inf, None)
                grid = np.linspace(0, 2 * np.pi, 73)
                for a in grid:
                    for b in grid:
                        o = B.ckm_dagger_x(Vckm, n, (a, b), order=order)
                        c = sum(pull(o[k] / D, P[kk]) ** 2 for k, kk in
                                (("th12", "theta12_deg"), ("th13", "theta13_deg"), ("th23", "theta23_deg")))
                        if c < best[0]:
                            best = (c, o)
                c, o = best
                print(f"  {order:8s} X={n:4s} best over 2 phases: sum pull^2(angles) = {c:7.2f}  "
                      f"th12 {o['th12'] / D:.2f} th13 {o['th13'] / D:.2f} th23 {o['th23'] / D:.2f} "
                      f"delta {o['delta'] / D:.0f}")
        # KM-phase sum rule (2003.09921): delta_KM^q + delta_KM^l
        dkq = B.km_phase(Vckm)
        dl = P["dcp_deg"]["value"] * D
        dkl = B.km_phase(B.standard_matrix(t12, t13, t23, dl))
        s = np.angle(np.exp(1j * (dkq + dkl))) / D
        print(f"  KM phases: delta_KM^q = {dkq / D:.1f}, delta_KM^l = {dkl / D:.1f}, sum mod 360 -> {s:+.1f} deg")

print("\n=== within-sector mass relations")
mq = mass["quarks"]
r = mq["m_d"]["value"] / mq["m_s"]["value"]
print(f"  GST sqrt(m_d/m_s) = {np.sqrt(r):.4f} (m_d, m_s at 2 GeV) vs |V_us| = {sp['s12']['value']:.4f}; "
      f"m_s/m_d error alone gives +-{0.5 * np.sqrt(r) * np.hypot(mq['m_d']['err_plus'] / mq['m_d']['value'], mq['m_s']['err_plus'] / mq['m_s']['value']):.4f}")
cl = mass["charged_leptons"]
mt = B.koide_predict_m3(cl["m_e"]["value"], cl["m_mu"]["value"])
print(f"  Koide m_tau pred = {mt * 1e3:.3f} MeV vs {cl['m_tau']['value'] * 1e3:.3f} "
      f"+{cl['m_tau']['err_plus'] * 1e3:.3f} MeV, pull {pull(mt, cl['m_tau']):+.2f}")
ns = mass["neutrino_splittings"]["IC24_wSKatm"]["NO"]
for signs in ((1, 1, 1), (-1, 1, 1)):
    roots = B.koide_neutrino_m1(ns["dm21sq"]["value"], ns["dm31sq"]["value"], signs)
    print(f"  Koide-type neutrino (NO, signs {signs}): m1 roots [eV] = {[round(x, 5) for x in roots]}")

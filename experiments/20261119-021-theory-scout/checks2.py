"""Checks of the neutrino-alignment and published-point ingredients of proposals P1-P3. Writes checks2.json.

Run from this folder: ../../.venv/Scripts/python checks2.py   (single process, < 1 min)

C4  CSD(n) two-RH-neutrino seesaw (Littlest Seesaw, King arXiv:1512.07531):
    m_nu = m_a (0,1,1)(0,1,1)^T + m_b e^{i eta} (1,n,n-2)(1,n,n-2)^T, U_e = 1, NO, m1 = 0.
    One real parameter r = m_b/m_a (eta fixed); fitted to the Delta m^2 ratio plus the NuFIT PMNS tables.
    Both CP conventions (U and U*) are reported since the sign of delta depends on the Weinberg-operator convention.
C5  Published best-fit / benchmark predictions of the GUT models in proposals.md, scored against NuFIT 6.1 (NO) with
    the 2D (s23, delta) table and 1D profiles (relative to the NO minimum). Numbers are the papers' printed values.
C6  Double ratio R = (m_e/m_mu)/(m_d/m_s) for the diagonal-Clebsch case (m_e/m_d = c1, m_mu/m_s = c2 => R = c1/c2),
    e.g. 1503.03306 (4/9, 9/2) and Georgi-Jarlskog (1/3, 3).
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.data import load_masses, load_pmns, pmns_delta_chi2  # noqa: E402
from src.mixing import extract_angles  # noqa: E402
from src.rg import derived_yukawas_mz  # noqa: E402

OUT = {}


def csd(n, eta, r):
    a = np.array([0, 1, 1.0])
    b = np.array([1, n, n - 2.0])
    m = np.outer(a, a) + r * np.exp(1j * eta) * np.outer(b, b)
    w, v = np.linalg.eigh(m @ m.conj().T)
    return np.sqrt(np.clip(w, 0, None)), v


def c4():
    sp = load_masses()["neutrino_splittings"]["IC23_woSKatm"]["NO"]
    ratio = sp["dm21sq"]["value"] / sp["dm31sq"]["value"]
    rerr = ratio * math.hypot(sp["dm21sq"]["err_plus"] / sp["dm21sq"]["value"], sp["dm31sq"]["err_plus"] / sp["dm31sq"]["value"])
    res = {}
    for n in (3, 4):
        for eta_lab, eta in (("2pi/3", 2 * math.pi / 3), ("-2pi/3", -2 * math.pi / 3), ("pi/3", math.pi / 3), ("-pi/3", -math.pi / 3)):
            best = None
            for r in np.logspace(-3, 0, 800):
                mm, v = csd(n, eta, r)
                rat = (mm[1] ** 2 - mm[0] ** 2) / (mm[2] ** 2 - mm[0] ** 2)
                for conj in (False, True):
                    U = v.conj() if conj else v
                    t12, t13, t23, d = extract_angles(U)
                    obs = dict(s12sq=math.sin(t12) ** 2, s13sq=math.sin(t13) ** 2, s23sq=math.sin(t23) ** 2,
                               dcp_deg=math.degrees(d) % 360)
                    c = pmns_delta_chi2("NO", **obs) + ((rat - ratio) / rerr) ** 2
                    if best is None or c < best["chi2"]:
                        best = dict(chi2=float(c), r=float(r), conj=conj, dm_ratio=float(rat), **obs)
            res[f"CSD{n}_eta={eta_lab}"] = best
    return dict(N=5, k_cont=1, note="N = 4 PMNS + Delta m^2 ratio; m1 = 0 predicted; U_e = 1 (no charged-lepton correction)",
                fits=res)


def score(s12sq, s13sq, s23sq, dcp):
    tot = pmns_delta_chi2("NO", s12sq=s12sq, s13sq=s13sq, s23sq=s23sq, dcp_deg=dcp)
    parts = {k: pmns_delta_chi2("NO", **{k: v}) for k, v in (("s12sq", s12sq), ("s13sq", s13sq), ("s23sq", s23sq), ("dcp_deg", dcp))}
    return dict(chi2_4=float(tot), chi2_1d=parts)


def c5():
    s2 = lambda deg: math.sin(math.radians(deg)) ** 2
    pts = {
        "1305.6612 AGMS best fit (th12 34.29, th13 9.44, th23 38.49, delta 268.79)": (34.29, 9.44, 38.49, 268.79),
        "1406.7005 A-to-Z PS CSD4 abstract (th12 32.5 mid of 31-34, th13 9, th23 45, delta 260)": (32.5, 9.0, 45.0, 260.0),
        "1406.7005 A-to-Z best case in its ranges (th12 31, th13 8.5, th23 44.5, delta 255)": (31.0, 8.5, 44.5, 255.0),
        "1503.03306 A4xSU(5) CSD3 (th12 34, th13 8.7, th23 46, delta -87)": (34.0, 8.7, 46.0, 273.0),
        "1705.01555 S4xSO(10) delta window centres 294.5 / 239 (angles at NuFIT best)": None,
    }
    p = load_pmns("NO")
    out = {}
    for lab, v in pts.items():
        if v is None:
            for dd in (294.5, 239.0, 280.7, 225.0):
                out[f"1705.01555 delta={dd} (angles at NuFIT best)"] = score(p["s12sq"]["value"], p["s13sq"]["value"], p["s23sq"]["value"], dd)
            continue
        out[lab] = score(s2(v[0]), s2(v[1]), s2(v[2]), v[3])
    return out


def c6():
    Y = derived_yukawas_mz()
    R = (Y["y_e"] / Y["y_mu"]) / (Y["y_d"] / Y["y_s"])
    ms = load_masses()
    rel_pdg = math.hypot(ms["m_d"]["err_plus"] / ms["m_d"]["value"], ms["m_s"]["err_plus"] / ms["m_s"]["value"])
    out = dict(R=R, sigma_floor=R * math.sqrt(2) * 0.02, sigma_pdg=R * rel_pdg, cases={})
    for lab, c1, c2 in (("1503.03306 (4/9, 9/2)", 4 / 9, 9 / 2), ("Georgi-Jarlskog (1/3, 3)", 1 / 3, 3), ("minimal SU(5) (1,1)", 1, 1),
                        ("(1/2, 9/2)", 1 / 2, 9 / 2), ("(1/2, 6)", 1 / 2, 6), ("(2/3, 6)", 2 / 3, 6), ("(1/3, 3) x (1, 1) alt", 1 / 3, 3)):
        pr = c1 / c2
        out["cases"][lab] = dict(R_pred=pr, pull_floor=(pr - R) / out["sigma_floor"], pull_pdg=(pr - R) / out["sigma_pdg"])
    ms_md = 1 / (Y["y_d"] / Y["y_s"])
    out["ms_over_md"] = ms_md
    out["AGMS_pred_ms_md_18.95_pull_pdg"] = (18.95 - ms_md) / math.hypot(ms_md * rel_pdg, 0.24)
    out["AGMS_pred_ms_md_18.95_pull_floor"] = (18.95 - ms_md) / math.hypot(ms_md * math.sqrt(2) * 0.02, 0.24)
    return out


if __name__ == "__main__":
    OUT["C4_csd"] = c4()
    OUT["C5_published_points"] = c5()
    OUT["C6_diag_double_ratio"] = c6()
    (HERE / "checks2.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps(OUT, indent=1))

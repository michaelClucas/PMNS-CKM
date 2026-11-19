"""C7: what charged-lepton 1-2 correction does CSD3 (Littlest Seesaw) need to fit NuFIT 6.1? Writes checks3.json.

U_PMNS = U_e^dag U_nu, U_e = diag(1, e^{i phi}, 1) R12(theta12e) (left-handed e rotation), U_nu from CSD3 with
eta = 2pi/3 and r = m_b/m_a profiled. chi2 = NuFIT tables (NO) + Delta m^2 ratio. Grid over theta12e in [0, 0.25],
phi in [0, 2pi). Run from this folder: ../../.venv/Scripts/python checks3.py  (single process, ~2-4 min)
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.data import load_masses, pmns_delta_chi2  # noqa: E402
from src.mixing import extract_angles  # noqa: E402

sys.path.insert(0, str(HERE))
from checks2 import csd  # noqa: E402

sp = load_masses()["neutrino_splittings"]["IC23_woSKatm"]["NO"]
ratio = sp["dm21sq"]["value"] / sp["dm31sq"]["value"]
rerr = ratio * math.hypot(sp["dm21sq"]["err_plus"] / sp["dm21sq"]["value"], sp["dm31sq"]["err_plus"] / sp["dm31sq"]["value"])

eta = 2 * math.pi / 3
rs = np.linspace(0.085, 0.12, 36)
nu = []
for r in rs:
    mm, v = csd(3, eta, r)
    nu.append((((mm[1] ** 2 - mm[0] ** 2) / (mm[2] ** 2 - mm[0] ** 2) - ratio) / rerr) ** 2)
    nu[-1] = (nu[-1], v)


def chi(te, ph, r_i):
    c2, v = nu[r_i]
    c, s = math.cos(te), math.sin(te)
    Ue = np.diag([1, np.exp(1j * ph), 1]) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    t12, t13, t23, d = extract_angles(Ue.conj().T @ v)
    obs = dict(s12sq=math.sin(t12) ** 2, s13sq=math.sin(t13) ** 2, s23sq=math.sin(t23) ** 2, dcp_deg=math.degrees(d) % 360)
    return pmns_delta_chi2("NO", **obs) + c2, obs


tes = np.linspace(0, 0.25, 26)
phs = np.linspace(0, 2 * math.pi, 37)[:-1]
prof = []
best = None
r_sel = [i for i in range(len(rs)) if nu[i][0] < 9]
for te in tes:
    bt = None
    for ph in phs:
        for i in r_sel:
            c, obs = chi(te, ph, i)
            if bt is None or c < bt[0]:
                bt = (c, ph, rs[i], obs)
    prof.append(dict(theta12e=float(te), chi2=float(bt[0]), phi=float(bt[1]), r=float(bt[2]), **bt[3]))
    if best is None or bt[0] < best["chi2"]:
        best = prof[-1]
allowed = [p["theta12e"] for p in prof if p["chi2"] - best["chi2"] < 4]
out = dict(best=best, theta12e_allowed_dchi2_lt4=[min(allowed), max(allowed)] if allowed else None,
           chi2_at_theta12e_0=prof[0]["chi2"], profile=prof, N=5, k_cont=3,
           note="k_cont = r, theta12e, phi when both are free; in P1 theta12e and phi come from Y_d x Clebsch (0 extra)")
(HERE / "checks3.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=1))
for p in prof:
    print(f"theta12e {p['theta12e']:.3f} chi2 {p['chi2']:.2f} phi {math.degrees(p['phi']):.0f}")

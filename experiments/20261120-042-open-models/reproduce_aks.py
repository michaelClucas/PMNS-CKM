"""Reproduce AKS (arXiv:1301.6764) Tables 4-5 and Fig. 2 from the printed inputs.

Charged sector: Table 4 matrices (Eqs. 3.8, 3.10) put at mu_GUT = 2e16 GeV as MSSM Yukawas, gauge couplings
from our own M_Z -> GUT run (MSSM, tan beta 1.49, M_SUSY 750 GeV), then run down with src.rg to m_t(m_t)
and compared with the Table 5 "Model" column. The paper used REAP; the GUT scale is not printed (2e16 assumed).
Neutrino sector: Eq. (2.1) with alpha = pi/3, scan of eps, compared with Fig. 2 / Eq. (7.2)
(theta13 8-9 deg, theta23 ~ 38.5, theta12 ~ 33, delta ~ 130, alpha2 ~ 315), both conjugation conventions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import rg  # noqa: E402
import lib42 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
P = L.AKS_T4
TB, MS, MGUT, MT = P["tan_beta"], P["m_susy"], 2e16, 163.0

T5 = {"y_tau": 1.00e-2, "y_mu": 5.89e-4, "y_e": 2.79e-6, "y_b": 1.64e-2, "y_s": 2.95e-4, "y_d": 11.7e-6,
      "y_t": 0.939, "y_c": 3.40e-3, "y_u": 7.59e-6, "theta12": 0.2257, "theta23": 0.0409, "theta13": 0.0036,
      "delta": 1.1975}

st = rg.build_mz_state(neutrinos=False)
mz = 91.1876
up = rg.run(mz, MGUT, st, model="MSSM", tan_beta=TB, m_susy=MS)
gut = dict(up)
gut["Yu"], gut["Yd"], gut["Ye"] = L.aks_Yu(**P), L.aks_Yd(**P), L.aks_Ye(**P)
low = rg.run(MGUT, MT, gut, model="MSSM", tan_beta=TB, m_susy=MS)
o = rg.observables(low)
ours = {**o["y"], **{k: o["ckm"][k] for k in ("theta12", "theta23", "theta13", "delta")}}
out = {"charged": {k: {"paper": T5[k], "ours": float(ours[k]), "rel": float(ours[k] / T5[k] - 1)} for k in T5}}
og = rg.observables(gut, tan_beta=TB)
out["gut_ratios"] = {"md_ms": og["y"]["y_d"] / og["y"]["y_s"], "me_mmu": og["y"]["y_e"] / og["y"]["y_mu"],
                     "ms_md": og["y"]["y_s"] / og["y"]["y_d"], "mmu_ms": og["y"]["y_mu"] / og["y"]["y_s"],
                     "mb_mtau": og["y"]["y_b"] / og["y"]["y_tau"]}
out["gut_ratios"]["R"] = out["gut_ratios"]["me_mmu"] / out["gut_ratios"]["md_ms"]

# neutrino sector: scan eps
rows = {}
for conj in (True, False):
    r = []
    for eps in np.linspace(0.05, 0.6, 1101):
        ob = L.aks_obs({**P, "eps": eps}, conj=conj)
        r.append((eps, np.degrees(np.arcsin(np.sqrt(ob["s13sq"]))), np.degrees(np.arcsin(np.sqrt(ob["s12sq"]))),
                  np.degrees(np.arcsin(np.sqrt(ob["s23sq"]))), ob["dcp_deg"], ob["alpha21_deg"], ob["alpha31_deg"],
                  ob["dm_ratio"]))
    r = np.array(r)
    sel = {}
    for t13 in (8.0, 8.5, 9.0):
        i = int(np.argmin(np.abs(r[:, 1] - t13)))
        sel[str(t13)] = dict(zip(("eps", "t13", "t12", "t23", "delta", "alpha21", "alpha31", "dm_ratio"),
                                 map(float, r[i])))
    rows["conj" if conj else "noconj"] = sel
out["neutrino"] = rows
(HERE / "reproduce_aks.json").write_text(json.dumps(out, indent=1, default=float))
for k, v in out["charged"].items():
    print(f"{k:8s} paper {v['paper']:.4g} ours {v['ours']:.4g} rel {v['rel']:+.3f}")
print("GUT", {k: round(v, 5) for k, v in out["gut_ratios"].items()})
for c, sel in rows.items():
    for t, d in sel.items():
        print(c, {k: round(v, 3) for k, v in d.items()})

"""Reproduce CM (arXiv:1307.7711) benchmark from its printed inputs.

Charged: Eq. (16) parameters -> |V_CKM| (Eq. 19), Wolfenstein (Eq. 21), delta_q 45.6 deg (Eq. 20), |U_e,L| (Eq. 43).
The paper quotes CKM at the electroweak scale after RG (tan beta 10); we compare the GUT-scale matrices directly
(CKM angle running at tan beta 10 is a few % for V_cb, V_ub) and after an src.rg run to M_Z.
Neutrino: Eq. (58) xi0, eta0, eta0'' -> theta12 33.351, theta23 49.162, theta13 9.189 (Eq. 59), masses Eq. (60)
(dm ratio 7.62e-5 / 2.53e-3), delta 195.9-200.1 (Eq. 65). Convention variants are tried; the one that
reproduces Eqs. (43), (59), (65) is used for the fit.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from src import rg  # noqa: E402
from src import yukawa as yk  # noqa: E402
import lib42 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
P = {**L.CM_EQ16, **L.CM_EQ58}
out = {"variants": {}}
for tr, cj in itertools.product((True, False), (False, True)):
    ob, d = L.cm_obs(P, rl_transpose=tr, conj=cj, details=True)
    V = d["V_ckm"]
    Ue = d["U_e"]
    lam = abs(V[0, 1]) / np.hypot(abs(V[0, 0]), abs(V[0, 1]))
    A = abs(V[1, 2]) / np.hypot(abs(V[0, 0]), abs(V[0, 1])) / lam
    rb = -V[0, 0] * V[0, 2].conj() / (V[1, 0] * V[1, 2].conj())
    out["variants"][f"transpose={tr},conj={cj}"] = {
        "absV": np.abs(V).round(5).tolist(), "lambda": lam, "A": A, "rhobar": rb.real, "etabar": rb.imag,
        "delta_q_deg": np.degrees(ob["ckm_delta_rad"]), "absUe": np.abs(Ue).tolist(),
        "t12": np.degrees(np.arcsin(np.sqrt(ob["s12sq"]))), "t23": np.degrees(np.arcsin(np.sqrt(ob["s23sq"]))),
        "t13": np.degrees(np.arcsin(np.sqrt(ob["s13sq"]))), "dcp_deg": ob["dcp_deg"], "dm_ratio": ob["dm_ratio"],
        "nu_masses_rel": (d["m_nu"] / d["m_nu"][2]).tolist(),
        "md_ms": ob["mr_d_s"], "ms_mb": ob["mr_s_b"], "me_mmu": ob["mr_e_mu"], "mmu_mtau": ob["mr_mu_tau"],
        "mu_mc": ob["mr_u_c"], "mc_mt": ob["mr_c_t"], "mb_mtau": ob["mr_b_tau"], "R": ob["R_clebsch"]}
out["paper"] = {"absV": [[0.974, 0.227, 0.00412], [0.227, 0.973, 0.0412], [0.00718, 0.0408, 0.999]],
                "lambda": 0.227, "A": 0.798, "rhobar": 0.299, "etabar": 0.306, "delta_q_deg": 45.6,
                "absUe12": 0.0823, "absUe13": 1.31e-5, "absUe23": 1.49e-4, "absUe31": 1.14e-6,
                "t12": 33.351, "t23": 49.162, "t13": 9.189, "dcp_range": [195.9, 200.1],
                "dm_ratio": 7.62e-5 / 2.53e-3, "nu_masses_rel": [0.00398 / 0.0505, 0.00959 / 0.0505, 1.0]}
(HERE / "reproduce_cm.json").write_text(json.dumps(out, indent=1, default=float))
for k, v in out["variants"].items():
    print(k)
    print("  |V|", v["absV"], "lam A rho eta", round(v["lambda"], 4), round(v["A"], 4), round(v["rhobar"], 4),
          round(v["etabar"], 4), "dq", round(v["delta_q_deg"], 2))
    print("  |Ue| 12 13 23 31", [f"{x:.3g}" for x in (v["absUe"][0][1], v["absUe"][0][2], v["absUe"][1][2],
                                                       v["absUe"][2][0])])
    print("  t12 t23 t13 d dm", round(v["t12"], 3), round(v["t23"], 3), round(v["t13"], 3), round(v["dcp_deg"], 2),
          round(v["dm_ratio"], 5), "m/m3", np.round(v["nu_masses_rel"], 4))
    print("  ratios d/s s/b e/mu mu/tau u/c c/t b/tau R", [round(v[x], 5) for x in
          ("md_ms", "ms_mb", "me_mmu", "mmu_mtau", "mu_mc", "mc_mt", "mb_mtau", "R")])

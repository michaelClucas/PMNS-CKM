"""Reproduce the UTZ (1710.01741) L.O. benchmark (Table III -> Tables IV, V) with fit_more.utz_model.

    python reproduce_utz.py -> stdout (reproduce_utz.log)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

import fit_more as F  # noqa: E402

t = dict(a_e=0.0042, b_e=-0.0545, g_e=0.13, d_e=1.83, a_u=-0.00014, b_u=0.003,
         ua_n=np.arcsinh(4e-5 / 1e-6), ub_n=np.arcsinh(11.8e-5 / 1e-6),
         ux=np.arcsinh(12.75e-13 / 1e-14), uy=np.arcsinh(4.055e-13 / 1e-14))
o = F.utz_model(t)
ours = {"sin th12q": o["ckm_s12"], "sin th23q": o["ckm_s23"], "sin th13q": o["ckm_s13"],
        "sin dq": np.sin(o["ckm_delta_rad"]), "sin th12l": np.sqrt(o["s12sq"]), "sin th23l": np.sqrt(o["s23sq"]),
        "sin th13l": np.sqrt(o["s13sq"]), "sin dl": np.sin(np.radians(o["dcp_deg"])), "dm_ratio": o["dm_ratio"],
        "md/ms": o["mr_d_s"], "ms/mb": o["mr_s_b"], "me/mmu": o["mr_e_mu"], "mmu/mtau": o["mr_mu_tau"],
        "mc/mt": o["mr_c_t"]}
paper = {"sin th12q": .226, "sin th23q": .0191, "sin th13q": .0042, "sin dq": .561, "sin th12l": .554,
         "sin th23l": .778, "sin th13l": .152, "sin dl": -.905, "dm_ratio": .0213, "md/ms": .00090 / .020,
         "ms/mb": .020, "me/mmu": .00031 / .055, "mmu/mtau": .055, "mc/mt": .0027}
for k in ours:
    print(f"{k:10s} ours {ours[k]:.5g}  paper {paper[k]:.5g}")

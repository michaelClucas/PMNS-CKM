"""Step 2: data run to M_GUT = 2e16 GeV with src.rg (SM; MSSM tanb x M_SUSY). Writes rg_scan.csv."""
import csv, time
import numpy as np
from common import gut_reference, TANB, scenario_name

rows = []
scen = [("SM", None, 3000.0)] + [("MSSM", t, ms) for ms in (3000.0, 10000.0, 1000.0) for t in TANB]
t0 = time.time()
for model, tb, ms in scen:
    r = gut_reference(model, tb, ms)
    yu, yd, ye = r["yu"], r["yd"], r["ye"]
    g = r["ckm_gut"]
    rows.append({"scenario": scenario_name(model, tb, ms), "model": model, "tanb": tb or "", "msusy_GeV": ms if model == "MSSM" else "",
                 "mb_over_mtau": yd[2] / ye[2], "ms_over_mmu": yd[1] / ye[1], "md_over_me": yd[0] / ye[0],
                 "eps_btau": (yd[2] - ye[2]) / yd[2],
                 "mu_over_mc": yu[0] / yu[1], "mc_over_mt": yu[1] / yu[2], "md_over_ms": yd[0] / yd[1], "ms_over_mb": yd[1] / yd[2],
                 "me_over_mmu": ye[0] / ye[1], "mmu_over_mtau": ye[1] / ye[2],
                 "y_t_SMnorm": r["state"]["normalisation"] and yu[2] * (np.sin(np.arctan(tb)) if tb else 1),
                 "y_b_SMnorm": yd[2] * (np.cos(np.arctan(tb)) if tb else 1),
                 "theta12_q": g[0], "theta13_q": g[1], "theta23_q": g[2], "delta_q": g[3],
                 "J_tt": r["J"][0, 0], "J_tb": r["J"][0, 1], "J_bt": r["J"][1, 0], "J_bb": r["J"][1, 1],
                 "J_taut": r["J"][2, 0], "J_taub": r["J"][2, 1]})
    print(f"{rows[-1]['scenario']:28s} mb/mtau={yd[2]/ye[2]:.4f} th23q={g[2]:.5f} th13q={g[1]:.6f} ({time.time()-t0:.0f}s)", flush=True)
with open("rg_scan.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

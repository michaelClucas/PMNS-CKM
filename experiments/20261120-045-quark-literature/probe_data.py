"""Print the 8 quark targets and sigmas the official scorer uses (M_Z, PDG 2026, 2% floor)."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src import evaluate as E, data as D
Q8 = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b"]
for k, v in D.load_ckm().items():
    if k in ("s12", "s13", "s23", "delta_rad", "lambda", "A", "rhobar", "etabar", "Vub", "Vcb", "Vtd", "Vts", "Vus"):
        print("ckm", k, v)
print("mr targets", E.mass_ratio_targets())
t = time.time(); sc = E.Scorer(Q8, "NO", mass_floor=0.02); print("scorer M_Z build s", time.time() - t)
t = time.time(); sc2 = E.Scorer(Q8, "NO", mass_floor=0.02, scale={"mu": 2e16, "model": "MSSM", "tan_beta": 10.0}); print("scorer GUT build s", time.time() - t)
print("GUT shifts", sc2.scale_shift)
t = time.time(); sc3 = E.Scorer(Q8, "NO", mass_floor=0.02, scale={"mu": 2e16, "model": "MSSM", "tan_beta": 10.0, "eta_b": 0.2}); print("eta scorer build s", time.time() - t)
print("GUT eta 0.2 shifts", sc3.scale_shift)

"""Parameter counting: Jacobian rank of the 8 quark observables for each texture (writes ranks.json),
and the Harrison-Scott benchmark reproduction (Table 1 of 2501.18508)."""
import json, sys
import numpy as np
import structures as S
from lib45 import reduce_params

out = {}
for name, (fac, scale, note) in S.TEXTURES.items():
    tex = fac()
    n_all = len(tex.all_params())
    rank, pinned, n_active = reduce_params(tex)
    out[name] = {"n_symbols": n_all, "rank": rank, "pinned_phases": pinned, "n_fit_params": n_active,
                 "fit_params": list(tex.bounds())}
    print(f"{name:8s} symbols {n_all:2d} rank {rank} pinned {pinned} -> fit params {n_active}", flush=True)
json.dump(out, open("ranks.json", "w"), indent=1)

print("\nHarrison-Scott benchmark (published params), paper Table 1 fitted column at 1e4 TeV:")
print("paper: mu/mc 2.00e-3, md/ms 4.97e-2, mc/mt 3.46e-3, ms/mb 1.968e-2, lambda 0.22499, A 0.876, rhobar 0.1519, etabar 0.3477")
from src.mixing import standard_matrix
for reading in ("mod", "cplx"):
    p = S.hs_model(S.HS_PUB, reading)
    s12, s13, s23, d = p["ckm_s12"], p["ckm_s13"], p["ckm_s23"], p["ckm_delta_rad"]
    lam = s12; A = s23 / s12 ** 2
    V = standard_matrix(np.arcsin(s12), np.arcsin(s13), np.arcsin(s23), d)
    z = -(V[0, 0] * np.conj(V[0, 2])) / (V[1, 0] * np.conj(V[1, 2]))
    al = np.degrees(S.alpha_of(V))
    print(f"{reading}: mu/mc {p['mr_u_c']:.4g} md/ms {p['mr_d_s']:.4g} mc/mt {p['mr_c_t']:.4g} ms/mb {p['mr_s_b']:.4g} "
          f"lambda {lam:.5f} A {A:.4f} rhobar {z.real:.4f} etabar {z.imag:.4f} alpha {al:.2f} delta {np.degrees(d):.2f}")

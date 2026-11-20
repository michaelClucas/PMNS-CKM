"""Parameter-independent mass-ratio bounds of the model (our charged-lepton inputs, no fit to quarks).

Down sector: Y_d = Phi0 (1 + B X3)^-1 Phi0, B = b_d e^{i beta_d} complex. Grid over B; report the
range of m_s/m_b among points with m_d/m_s within +-5% of the data, and vice versa.
Up sector: Y_u = Phi0 Pbar (1 + b_u X3)^-1 P Phi0. Grid over (eps_u = 1 + b_u, phi1, phi2); range of
m_u/m_c among points with m_c/m_t within +-5% of the data. Then a local minimisation of
m_u/m_c subject to m_c/m_t = data (penalty) from the best grid points, as a check.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import fit as F
import model as M

HERE = Path(__file__).resolve().parent
P = F.phi0()
V, _, _ = F.ev.mass_ratio_targets()
out = {"targets": {k: V[k] for k in ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")}}


def sv(Y):
    return np.linalg.svd(Y, compute_uv=False)[::-1]


rows = []
for b in np.concatenate([-np.logspace(-2, 2.5, 500), np.logspace(-2, 2.5, 500)]):
    for be in np.radians(np.arange(0, 180, 0.5)):
        s = sv(P @ M.inv_democratic(b * np.exp(1j * be)) @ P)
        rows.append((s[0] / s[1], s[1] / s[2]))
a = np.array(rows)
m = np.abs(np.log(a[:, 0] / V["mr_d_s"])) < 0.05
m2 = np.abs(np.log(a[:, 1] / V["mr_s_b"])) < 0.05
out["down"] = {"ms_mb_range_given_md_ms": [a[m, 1].min(), a[m, 1].max()],
               "md_ms_range_given_ms_mb": [a[m2, 0].min(), a[m2, 0].max()], "n_grid": len(a)}


def up(x):
    e, p1, p2 = x
    D = np.diag([np.exp(-1j * np.radians(p1)), np.exp(-1j * np.radians(p2)), 1.0])
    s = sv(P @ D @ M.inv_democratic(e - 1.0) @ D.conj() @ P)
    return s[0] / s[1], s[1] / s[2]


rng = np.random.default_rng(0)
X = np.column_stack([np.concatenate([-np.logspace(-4, -0.5, 1), np.zeros(0)])] * 0) if False else None
pts = []
for e in np.concatenate([-np.logspace(-4, -0.3, 60), np.logspace(-4, -0.3, 60)]):
    for p1 in np.arange(-180, 180, 6.0):
        for p2 in np.arange(-180, 180, 6.0):
            r = up((e, p1, p2))
            pts.append((e, p1, p2, r[0], r[1]))
u = np.array(pts)
mu = np.abs(np.log(u[:, 4] / V["mr_c_t"])) < 0.05
i = np.argmin(np.where(mu, u[:, 3], np.inf))
out["up_grid"] = {"mu_mc_range_given_mc_mt": [u[mu, 3].min(), u[mu, 3].max()], "argmin": u[i].tolist(),
                  "n_grid": len(u)}


def pen(x):
    r = up(x)
    return np.log(r[0]) + 1e3 * np.log(r[1] / V["mr_c_t"]) ** 2


best = None
for j in np.argsort(np.where(mu, u[:, 3], np.inf))[:20]:
    r = minimize(pen, u[j, :3], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    if best is None or r.fun < best.fun:
        best = r
out["up_min"] = {"x": best.x.tolist(), "mu_mc": up(best.x)[0], "mc_mt": up(best.x)[1]}

# chi^2 lower bound from the two parameter-independent relations (official scorer, mass block only)
bpred = {"mr_u_c": out["up_min"]["mu_mc"], "mr_c_t": V["mr_c_t"], "mr_d_s": V["mr_d_s"],
         "mr_s_b": out["down"]["ms_mb_range_given_md_ms"][0]}
rb = F.ev.evaluate(bpred, "NO", use=F.MQ, inputs=F.INPUTS, k=0, mass_floor=F.FLOOR)
out["chi2_bound_mass_block"] = {"chi2": rb.chi2, "pulls": rb.pulls, "pred": bpred}
print(json.dumps(out, indent=1, default=float))
print("mass-block chi2 lower bound (approx.; +-5% grid tolerance on the matched ratios):", rb.chi2)
(HERE / "bound.json").write_text(json.dumps(out, indent=1, default=float))

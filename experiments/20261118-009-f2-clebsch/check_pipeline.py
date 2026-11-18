"""Pipeline checks and known-result sanity tests. Writes sanity.json.

1. fast_obs == src.mixing.extract_angles; TableChi2 == src.data.pmns_delta_chi2;
   v_e("std") == src.mixing.standard_matrix.
2. TBC (Antusch-Gross-Maurer-Sluka 1305.6612, King 1205.0506): TBM with th12e = theta_C
   gives s13 = sin(theta_C)/sqrt2 for every phi.
3. Solar sum rule (src.baselines.solar_sum_rule_cosdelta): every std-order config with
   th13e = 0 must satisfy it exactly with th12nu of its pattern; rev order breaks it.
4. TM1 / TM2 sum rules (src.baselines.tm1/tm2) for the TM tier with th12e = 0, and the
   fitted chi^2 against the exp 006 baselines B2-TM1, B1-TBM.
"""
import json

import numpy as np
from scipy.optimize import minimize

import f2lib as L
from src import data
from src.baselines import LO_TH12, solar_sum_rule_cosdelta, tm1, tm2
from src.mixing import standard_matrix

rng = np.random.default_rng(9)
out = {}
cfgs = L.configs()
out["tier_counts"] = L.tier_counts(cfgs)
ckm = L.ckm_mz()

# 1 ------------------------------------------------------------------
idx = rng.choice(len(cfgs), 3000)
mx = 0.0
for i in idx:
    c = cfgs[i]
    th = rng.uniform(0, 0.5) if c["tier"] == "TM" else None
    phi = rng.uniform(0, 2 * np.pi)
    U = L.full_matrix(c, ckm, phi, th)
    a = np.array(L.obs_exact(U))
    E = np.array([U[i_, j_] for i_, j_ in L.ENTRIES])
    b = np.array([float(v) for v in L.fast_obs(E)])
    d = np.abs(a - b)
    d[3] = min(d[3], 360 - d[3])
    mx = max(mx, d.max())
out["fast_obs_max_abs_diff"] = mx
for o in ("NO", "IO"):
    T = L.TableChi2(o)
    P = data.load_pmns(o)
    md = 0.0
    for _ in range(3000):
        v = [P[k]["value"] + rng.normal(0, 3) * P[k]["err_plus"] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg")]
        a = data.pmns_delta_chi2(o, *v)
        b = float(T(*v))
        if np.isfinite(a) or np.isfinite(b):
            md = max(md, abs(a - b))
    out[f"table_chi2_max_abs_diff_{o}"] = md
t = rng.uniform(0, 1, 4)
out["v_e_std_vs_standard_matrix"] = float(np.abs(L.v_e(*t, "std") - standard_matrix(*t)).max())

Pf, Qf, rcf, _ = L.build_PQ(cfgs, ckm)
Ps, Qs, rcs, _ = L.build_PQ_slow(cfgs, ckm)
out["build_PQ_vectorized_vs_slow"] = float(max(np.abs(Pf - Ps).max(), np.abs(Qf - Qs).max(), np.abs(rcf - rcs).max()))

# 2 ------------------------------------------------------------------
c = next(c for c in cfgs if c["tier"] == "A" and c["nu"] == "TBM" and c["c12"] == 1)
s13 = [L.obs_exact(L.full_matrix(c, ckm, p))[1] for p in np.linspace(0, 2 * np.pi, 13)]
target = np.sin(ckm[0]) ** 2 / 2
out["TBC_s13sq_pred"] = s13[0]
out["TBC_sin2thetaC_over_2"] = target
out["TBC_max_dev_over_phi"] = float(np.max(np.abs(np.array(s13) - target)))

# 3 ------------------------------------------------------------------
dev_std, dev_rev = 0.0, 0.0
for c in cfgs:
    if c["tier"] == "TM" or c["c13"] != 0 or c["c12"] == 0:
        continue
    for p in rng.uniform(0, 2 * np.pi, 3):
        U = L.full_matrix(c, ckm, p)
        s12, s13_, s23, d = L.obs_exact(U)
        cd = solar_sum_rule_cosdelta(np.arcsin(np.sqrt(s12)), np.arcsin(np.sqrt(s13_)),
                                     np.arcsin(np.sqrt(s23)), LO_TH12[c["nu"]])
        dev = abs(cd - np.cos(np.radians(d)))
        if c["order"] == "std":
            dev_std = max(dev_std, dev)
        elif c["c23"] != 0:
            dev_rev = max(dev_rev, dev)
out["solar_sum_rule_max_dev_std_order"] = float(dev_std)
out["solar_sum_rule_max_dev_rev_order"] = float(dev_rev)

# 4 ------------------------------------------------------------------
dev = {"TM1": 0.0, "TM2": 0.0}
for c in cfgs:
    if c["tier"] != "TM" or c["c12"] != 0:
        continue
    kind = c["nu"].split("@")[0]
    for th in (0.1, 0.2, 0.3):
        s12, s13_, s23, d = L.obs_exact(L.full_matrix(c, ckm, 0.3, th))
        t13, t23 = np.arcsin(np.sqrt(s13_)), np.arcsin(np.sqrt(s23))
        r = (tm1 if kind == "TM1" else tm2)(t13, t23)
        dev[kind] = max(dev[kind], abs(np.sin(r["th12"]) ** 2 - s12), abs(r["cos_delta"] - np.cos(np.radians(d))))
out["TM_sum_rule_max_dev"] = dev


def fit_tm(c, o):
    T = L.TableChi2(o)

    def f(x):
        return float(T(*L.obs_exact(L.full_matrix(c, ckm, 0.0, x[0]))))
    best = min((minimize(f, [x0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-10)) for x0 in (0.1, 0.2, 0.3)),
               key=lambda r: r.fun)
    return float(best.fun), float(best.x[0])


tm = {}
for c in cfgs:
    if c["tier"] == "TM" and c["c12"] == 0 and c["nu"].startswith("TM1"):
        for o in ("NO", "IO"):
            tm[f"{c['nu']}_{o}"] = fit_tm(c, o)
out["TM1_fixed_phase_chi2_thnu"] = tm
out["ref_exp006_B2-TM1_chi2"] = {"NO": 5.1146, "IO": 1.8208, "note": "phi_nu fitted there (k_c=2)"}

with open(L.HERE / "sanity.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps(out, indent=1, default=str))

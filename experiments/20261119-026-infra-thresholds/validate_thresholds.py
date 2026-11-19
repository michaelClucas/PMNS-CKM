"""Validate the tan-beta-enhanced SUSY threshold corrections added to src.rg / src.evaluate.

Run from the repo root: .venv/Scripts/python experiments/20261119-026-infra-thresholds/validate_thresholds.py
Writes thresholds.json next to this file. Single process, ~2-4 min.

Checks
1. eta = 0 is bit-identical to the old tree-level path (ckm_at, pmns_at, mass_ratios, Scorer).
2. Up-then-down running with thresholds returns the M_Z state.
3. Leading-order matching relations (Antusch-Maurer arXiv:1306.6879 Eqs. (17)-(24)) at M_SUSY.
4. Comparison with Antusch-Maurer's GUT-scale results (M_SUSY = 1 TeV):
   a. Fig. 6b, y_tau/y_b(M_GUT) vs eta_b at tan(beta_bar) = 15 (contours nearly flat in tan beta there): the eta_b at which each
      published contour level is crossed (contours read off the figure by eye, +-0.02 in eta_b).
   b. Fig. 5, (1+eta_q)/(1+eta_b) theta_i3(M_GUT) / theta_i3(M_Z) at tan(beta_bar) = 20,
      eta_b = 0 (read off as 3.385e-3 / 3.64e-3 and 3.915e-2 / 4.208e-2 using their Table 1).
   c. Eq. (25): threshold-invariant GUT ratios (1+eta_l) y_mu / ((1+eta_q) y_s) = 4.36 +- 0.23
      and (1+eta_l) y_e / ((1+eta_q) y_d) = 0.41 (+0.02 -0.06). Their inputs differ from ours
      (PDG 2012 vs 2026), so the running factor GUT / M_Z is compared, with their M_Z values
      from Table 1 (y_mu/y_s = 1.8910, y_e/y_d = 0.17688), and the absolute ratio is listed.
      Eq. (26) (y_mu y_d / (y_s y_e) = 10.7 +1.8 -0.8) likewise.
5. Exp 023's linear treatment (model side: s13, s23, m_s/m_b divided by 1+eta_b, m_b/m_tau
   multiplied) vs the full treatment (scale={..., "eta_b": x}): translated low-scale values for
   a prediction equal to the data's own GUT values, and the chi2 of exp 023's eb08 best fit.
"""
import json
import os
import pathlib
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from src import rg  # noqa: E402
from src.evaluate import Scorer  # noqa: E402

GUT, MS = 2e16, 1000.0
out = {}

# 1. eta = 0 bit identity
a = rg.ckm_at(GUT, "MSSM", 20, MS)
b = rg.ckm_at(GUT, "MSSM", 20, MS, thresholds={"eta_b": 0.0, "eta_q": 0.0, "eta_l": 0.0}, cache=False)
pa = rg.pmns_at(GUT, model="MSSM", tan_beta=20, m_susy=MS)
pb = rg.pmns_at(GUT, model="MSSM", tan_beta=20, m_susy=MS, thresholds={"eta_b": 0.0})
same = all(a[k] == b[k] for k in ("theta12", "theta13", "theta23", "delta")) and \
    all(a["ratios"][k] == b["ratios"][k] for k in a["ratios"]) and \
    all(pa[k] == pb[k] for k in ("theta12", "theta13", "theta23", "delta", "dm21_over_dm3l"))
out["eta0_bit_identical"] = bool(same)

# 2. up-then-down identity with thresholds
eta = {"eta_b": -0.4, "eta_q": 0.1, "eta_l": 0.05}
s0 = rg.build_mz_state("NO", 0.01)
up = rg.run(s0["mu"], GUT, s0, "MSSM", 20, MS, thresholds=eta)
dn = rg.run(GUT, s0["mu"], up, "MSSM", 20, MS, thresholds=eta)
out["up_down_max_rel"] = max(float(np.max(np.abs(dn[k] - s0[k])) / np.max(np.abs(s0[k])))
                             for k in ("Yu", "Yd", "Ye", "kappa"))

# 3. leading-order matching at M_SUSY (just above the threshold)
mu1 = MS * (1 + 1e-9)
r0 = rg.ckm_at(mu1, "MSSM", 20, MS)
lo_rel = {}
for e in ({"eta_b": -0.3}, {"eta_q": 0.2}, {"eta_b": -0.3, "eta_q": 0.2, "eta_l": -0.1}):
    eb, eq, el = (e.get(k, 0.0) for k in rg.THRESHOLD_KEYS)
    r = rg.ckm_at(mu1, "MSSM", 20, MS, thresholds=e)
    y0, y1 = r0["yukawas"], r["yukawas"]
    T = (1 + eq) / (1 + eb)
    lo_rel[json.dumps(e)] = {
        "y_b": y1["y_b"] * (1 + eb) / y0["y_b"] - 1, "y_s": y1["y_s"] * (1 + eq) / y0["y_s"] - 1,
        "y_d": y1["y_d"] * (1 + eq) / y0["y_d"] - 1, "y_mu": y1["y_mu"] * (1 + el) / y0["y_mu"] - 1,
        "y_tau": y1["y_tau"] / y0["y_tau"] - 1, "theta13": r["theta13"] * T / r0["theta13"] - 1,
        "theta23": r["theta23"] * T / r0["theta23"] - 1, "theta12": r["theta12"] / r0["theta12"] - 1,
        "delta": r["delta"] - r0["delta"]}
out["leading_order_rel_dev"] = lo_rel


# 4a. Fig. 6b contours
def ytau_over_yb(eb, tb=20.0):
    y = rg.ckm_at(GUT, "MSSM", tb, MS, thresholds={"eta_b": eb})["yukawas"]
    return y["y_tau"] / y["y_b"]


fig6b = {"0.75": -0.41, "1.0": -0.22, "1.25": -0.02, "1.5": 0.14, "1.75": 0.33, "2.0": 0.50}
ours = {}
for lev, eb_pub in fig6b.items():
    eb = brentq(lambda x: ytau_over_yb(x, 15.0) - float(lev), -0.6, 0.9, xtol=1e-5)
    ours[lev] = {"eta_b_published_readoff": eb_pub, "eta_b_ours": round(eb, 4), "diff": round(eb - eb_pub, 4)}
out["fig6b_tb15"] = ours
out["ytau_over_yb_eta0_tb20"] = ytau_over_yb(0.0)
out["ytau_over_yb_eta0_tb10"] = ytau_over_yb(0.0, 10.0)

# 4b. Fig. 5 theta_i3 running factor
mz = rg.ckm_at(rg._mz())
fig5 = {}
for eb in (-0.4, 0.0, 0.4):
    g = rg.ckm_at(GUT, "MSSM", 20, MS, thresholds={"eta_b": eb})
    fig5[str(eb)] = {"theta13_ratio": g["theta13"] / (1 + eb) / mz["theta13"],
                     "theta23_ratio": g["theta23"] / (1 + eb) / mz["theta23"]}
out["fig5_tb20"] = {"ours": fig5, "published_eta0": {"theta13_ratio": 3.385 / 3.64, "theta23_ratio": 3.915 / 4.208}}

# 4c. Eq. (25), (26)
ymz = rg.derived_yukawas_mz()
eq25 = {}
for tb in (10.0, 30.0):
    for e in ({}, {"eta_b": -0.3, "eta_q": 0.2, "eta_l": 0.1}):
        y = rg.ckm_at(GUT, "MSSM", tb, MS, thresholds=e or None)["yukawas"]
        eb, eq, el = (e.get(k, 0.0) for k in rg.THRESHOLD_KEYS)
        mu_s = (1 + el) * y["y_mu"] / ((1 + eq) * y["y_s"])
        e_d = (1 + el) * y["y_e"] / ((1 + eq) * y["y_d"])
        eq25[f"tb{tb:g} {json.dumps(e)}"] = {
            "mu_s_gut": mu_s, "e_d_gut": e_d, "eq26_gut": y["y_mu"] * y["y_d"] / (y["y_s"] * y["y_e"]),
            "mu_s_factor": mu_s / (ymz["y_mu"] / ymz["y_s"]), "e_d_factor": e_d / (ymz["y_e"] / ymz["y_d"])}
out["eq25"] = {"ours": eq25, "published": {"mu_s_gut": 4.36, "mu_s_err": 0.23, "e_d_gut": 0.41,
                                           "eq26_gut": 10.7, "mu_s_factor": 4.36 / 1.8910,
                                           "e_d_factor": 0.41 / 0.17688, "eq26_factor": 10.7 / (1.8910 / 0.17688)},
               "ours_mz": {"mu_s": ymz["y_mu"] / ymz["y_s"], "e_d": ymz["y_e"] / ymz["y_d"]}}

# 5. exp 023 linear treatment vs full
USE = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b",
       "mr_e_mu", "mr_mu_tau", "mr_b_tau", "s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio")
SC = {"mu": GUT, "model": "MSSM", "tan_beta": 5.0, "m_susy": MS}
base = Scorer(USE, "NO", mass_floor=0.02, scale=SC)


def lin(pred, eb):  # exp 023 fitlib.apply_eta_b
    q = dict(pred)
    q["ckm_s13"], q["ckm_s23"] = pred["ckm_s13"] / (1 + eb), pred["ckm_s23"] / (1 + eb)
    q["mr_s_b"], q["mr_b_tau"] = pred["mr_s_b"] / (1 + eb), pred["mr_b_tau"] * (1 + eb)
    return q


def data_gut(eb):
    """The data's own GUT-scale values with threshold eta_b (scored keys that change with eta_b)."""
    g = rg.ckm_at(GUT, "MSSM", 5.0, MS, thresholds={"eta_b": eb} if eb else None)
    r = rg.mass_ratios(GUT, "MSSM", 5.0, MS, errors=False, thresholds={"eta_b": eb} if eb else None)["value"]
    return {"ckm_s13": np.sin(g["theta13"]), "ckm_s23": np.sin(g["theta23"]), "mr_s_b": r["mr_s_b"],
            "mr_b_tau": r["mr_b_tau"]}


cmp5 = {}
for eb in (-0.01, -0.05, -0.1, -0.2, -0.4, -0.8):
    full = Scorer(USE, "NO", mass_floor=0.02, scale=dict(SC, eta_b=eb))
    dg = data_gut(eb)
    row = {}
    for pert in (1.0, 1.05):  # at the data's GUT value, and 5% above it
        pred = {k: base.best[k] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg")}
        pred.update({k: 0.1 for k in USE if k not in pred})
        pred.update({k: v * pert for k, v in dg.items()})
        pf = full._prep(pred)
        pl = base._prep(lin(pred, eb))
        row[f"x{pert}"] = {k: {"full": pf[k], "linear": pl[k], "rel_diff": pf[k] / pl[k] - 1} for k in dg}
    cmp5[str(eb)] = row
out["exp023_linear_vs_full"] = cmp5

f23 = json.loads((ROOT / "experiments/20261119-023-s4so10/fit_eb08.json").read_text())
eb = f23["eta_b"]
raw = dict(f23["pred_gut"])  # stored after the linear eta_b treatment; undo it
raw["ckm_s13"] *= 1 + eb
raw["ckm_s23"] *= 1 + eb
raw["mr_s_b"] *= 1 + eb
raw["mr_b_tau"] /= 1 + eb
use23 = tuple(f23["use"])
lin_sc = Scorer(use23, "NO", mass_floor=0.02, scale=SC)
full_sc = Scorer(use23, "NO", mass_floor=0.02, scale=dict(SC, eta_b=eb))
rl, rf = lin_sc(lin(raw, eb)), full_sc(raw)
out["exp023_eb08_rescore"] = {
    "eta_b": eb, "chi2_committed": f23["chi2"], "chi2_linear_recomputed": rl.chi2, "chi2_full": rf.chi2,
    "pulls_linear": rl.pulls, "pulls_full": rf.pulls,
    "note": "same parameter point (no refit); full = threshold matching inside src.rg running"}

(HERE / "thresholds.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("exp023_linear_vs_full",)}, indent=1, default=float))
for ebk, row in cmp5.items():
    print(ebk, {p: {k: round(v["rel_diff"], 5) for k, v in d.items()} for p, d in row.items()})

"""Step 3b: GUT->M_Z running of each best fit with src.rg, refit with the running shift, official scoring.

usage: python finalize.py fits_R0.jsonl [fits_C.jsonl ...]  -> results_<variant>.csv, best_fits_<variant>.json
For each (scenario, variant, ordering): best sign pattern from fit.py, then twice:
  run the model's GUT matrices (Y = M^dagger, kappa ∝ M_nu) down to M_Z with src.rg (same model/tanb/M_SUSY),
  shift = PMNS(M_Z) - PMNS(M_GUT), refit (least squares, start at the previous best) with that shift.
Final PMNS values are the src.rg M_Z values (not the shifted GUT values).
Scoring:
  official: src.evaluate.evaluate(pred, ordering, use=OFFICIAL_USE (6 quark masses + 4 CKM + 4 PMNS),
            inputs=(m_e, m_mu, m_tau), k=k_cont+k_d, k_discrete=k_d)
  supplementary (src.evaluate cannot score them): 3 charged-lepton GUT relations with a SIG_TH floor,
            dm21/|dm3l| Gaussian (NuFIT errors).  N_total = 14 + 3 + 1 = 18.
Also: CKM-systematic variation (MSSM theta13^q, theta23^q at M_GUT scaled by 0.98, exp 008) for MSSM fits.
"""
import csv, json, sys
import numpy as np
from scipy.optimize import least_squares
from common import (gut_reference, residuals, predict, VARIANTS, PMNS_KEYS, OFFICIAL_USE, LEPTON_INPUTS, SIG_TH,
                    ratio_data, M_GUT, rg, D)
from src.evaluate import evaluate
from src.stats import p_value

MZ = rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"]


def rundown(ref, mats, ordering):
    Mu, Md, Me, Mnu = mats
    s = dict(ref["state"])
    kref = np.linalg.norm(ref["state"]["kappa"], 2)
    s.update(Yu=Mu.conj().T.copy(), Yd=Md.conj().T.copy(), Ye=Me.conj().T.copy(),
             kappa=Mnu / np.linalg.norm(Mnu, 2) * kref, ordering=ordering)
    og = rg.observables(s, ref["tanb"])
    out = rg.run(M_GUT, MZ, s, model=ref["model"], tan_beta=ref["tanb"], m_susy=ref["msusy"])
    out["ordering"] = ordering
    oz = rg.observables(out, ref["tanb"])

    def lep(o):
        p, m = o["pmns"], o["pmns"]["m"]
        r = ((m[1] ** 2 - m[0] ** 2) / abs(m[2] ** 2 - (m[0] ** 2 if ordering == "NO" else m[1] ** 2)))
        return {"s12sq": np.sin(p["theta12"]) ** 2, "s13sq": np.sin(p["theta13"]) ** 2,
                "s23sq": np.sin(p["theta23"]) ** 2, "dcp_deg": np.degrees(p["delta"]) % 360, "ratio": r}
    return lep(og), lep(oz)


def refit(ref, v, p, z, ordering, sign, shift):
    n = len(p)
    f = lambda q: residuals(ref, v, q[:n], q[n:], ordering, sign, shift)
    s = least_squares(f, np.r_[p, z], method="trf", max_nfev=4000)
    return s.x[:n], s.x[n:], 2 * s.cost


def wrap180(d):
    return (d + 180) % 360 - 180


def finalize(rec, ckm_sys=1.0):
    ref = gut_reference(rec["model"], rec["tanb"], rec["msusy"])
    if ckm_sys != 1.0:
        ref = dict(ref)
        g = ref["ckm_gut"]
        ref["ckm_gut"] = (g[0], g[1] * ckm_sys, g[2] * ckm_sys, g[3])
    v, o, sg = rec["variant"], rec["ordering"], tuple(rec["sign"])
    p, z = np.array(rec["p"]), np.array(rec["z"])
    if ckm_sys != 1.0:
        p, z, _ = refit(ref, v, p, z, o, sg, None)
    shift = None
    for _ in range(3):
        pr = predict(ref, v, p, z, o, sg)
        g, mz = rundown(ref, pr["mats"], o)
        new = {k: (wrap180(mz[k] - g[k]) if k == "dcp_deg" else mz[k] - g[k]) for k in list(PMNS_KEYS) + ["ratio"]}
        if shift is not None and max(abs(new[k] - shift[k]) for k in new) < 1e-6:
            break
        shift = new
        p, z, c2 = refit(ref, v, p, z, o, sg, shift)
    pr = predict(ref, v, p, z, o, sg)
    g, mz = rundown(ref, pr["mats"], o)
    # consistency: our GUT PMNS vs src.rg.observables on the same GUT state
    cons = max(abs(g[k] - pr["pmns"][k]) for k in ("s12sq", "s13sq", "s23sq"))
    names, kd = VARIANTS[v]
    k_cont = len(names) + 10
    pred = dict(pr["nu"])
    pred.update({k: mz[k] for k in PMNS_KEYS})
    off = evaluate(pred, o, use=OFFICIAL_USE, inputs=LEPTON_INPUTS, k=k_cont + kd, k_discrete=kd)
    lep_pull = (np.log(pr["me"]) - np.log(pr["e_target"])) / SIG_TH
    r0, re = ratio_data(o)
    r_pull = (mz["ratio"] - r0) / re
    chi2_sup = float(np.sum(lep_pull ** 2) + r_pull ** 2)
    N = 18
    chi2_tot = off.chi2 + chi2_sup
    dof = N - k_cont
    p_tot = p_value(chi2_tot, N, k_cont) if dof > 0 else float("nan")
    p_tot_g = p_value(chi2_tot + off.ordering_offset, N, k_cont) if dof > 0 else float("nan")
    pulls = dict(off.pulls)
    pulls.update({"lep_e": lep_pull[0], "lep_mu": lep_pull[1], "lep_tau": lep_pull[2], "dm_ratio": r_pull})
    x, yh, ph, eta = __import__("common").unpack(v, p, sg)
    row = {"scenario": rec["scenario"], "model": rec["model"], "tanb": rec["tanb"], "msusy": rec["msusy"],
           "variant": v, "ordering": o, "sign_phi1_phi2": f"{sg[0]:.2f},{sg[1]:.2f}", "ckm_sys": ckm_sys,
           "mb_over_mtau_GUT_data": ref["mb_over_mtau"],
           "mb_over_mtau_GUT_model": float(np.linalg.svd(pr["mats"][1], compute_uv=False)[0] / pr["me"][2]),
           "x": str(np.round(x, 5)), "yhat": yh, "eta": eta,
           "phases": ",".join(f"{a:.3f}" for a in ph),
           "chi2_gut_fitstage": rec["chi2_gut"], "chi2_official": off.chi2, "chi2_lepton": float(np.sum(lep_pull ** 2)),
           "chi2_ratio": r_pull ** 2, "chi2_total": chi2_tot, "N_total": N, "k_cont": k_cont, "k_d": kd,
           "k_total": k_cont + kd, "dof": dof, "compressive_k<=N-2": (k_cont + kd) <= N - 2,
           "p_total": p_tot, "p_total_global": p_tot_g, "N_official": off.N, "dof_official": off.dof,
           "p_official": off.p, "p_global_official": off.p_global, "max_abs_pull_all": max(abs(u) for u in pulls.values()),
           "consistency_gut_pmns": cons,
           **{f"pred_{k}": mz[k] for k in list(PMNS_KEYS) + ["ratio"]},
           **{f"shift_{k}": mz[k] - g[k] for k in ("s12sq", "s13sq", "s23sq", "ratio")},
           **{f"pull_{k}": u for k, u in pulls.items()}}
    return row, {"p": p.tolist(), "z": z.tolist()}


if __name__ == "__main__":
    recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
    best = {}
    for r in recs:
        key = (r["scenario"], r["variant"], r["ordering"])
        if key not in best or r["chi2_gut"] < best[key]["chi2_gut"]:
            best[key] = r
    rows, params = [], {}
    for key, r in best.items():
        variants = [1.0] + ([0.98] if r["model"] == "MSSM" and r["variant"] != "R0" and r["msusy"] == 3000.0 and r["ordering"] == "NO" else [])
        for cs in variants:
            row, pz = finalize(r, cs)
            rows.append(row)
            params["|".join(key) + f"|ckm{cs}"] = pz
            print(f"{row['scenario']:24s} {row['variant']:6s} {row['ordering']} ckm_sys={cs} chi2_tot={row['chi2_total']:.2f} "
                  f"(off {row['chi2_official']:.2f} lep {row['chi2_lepton']:.2f} ratio {row['chi2_ratio']:.2f}) "
                  f"k={row['k_total']} dof={row['dof']} p={row['p_total']:.3g} maxpull={row['max_abs_pull_all']:.1f} "
                  f"s23sq={row['pred_s23sq']:.3f} s13sq={row['pred_s13sq']:.4f} s12sq={row['pred_s12sq']:.3f} "
                  f"d={row['pred_dcp_deg']:.0f} r={row['pred_ratio']:.4f} eta={row['eta']:.3f}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = sys.argv[1].replace("fits_", "results_").replace(".jsonl", ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    json.dump(params, open(out.replace("results_", "best_fits_").replace(".csv", ".json"), "w"), indent=0)

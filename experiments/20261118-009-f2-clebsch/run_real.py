"""Real-data F2 enumeration at M_Z and at 2e16 GeV (SM running of the CKM inputs).

Outputs (per scale S in {MZ, GUT}, ordering O in {NO, IO}):
  fits_S_O.csv           every config: best fitted-phase chi^2 and best structured-phase chi^2
  top30_S_fit_O.csv      top 30 fitted-phase configs, rescored with src.evaluate
  top30_S_struct_O.csv   top 30 structured-phase (config, phi) pairs, rescored with src.evaluate
  ckm_spread.csv         CKM-input error propagation for the top 5 of each table
  real_pipeline.json     the null pipeline (f2lib.search) run on the real data
"""
import csv
import json
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import f2lib as L
from src import data
from src.evaluate import evaluate
from src.haar import split_normal
from src.stats import trials_factor_pvalue

PM = ("s12sq", "s13sq", "s23sq", "dcp_deg")
t0 = time.time()
cfgs = L.configs()
counts = L.tier_counts(cfgs)
K_FIT = counts["ALL"]
K_STRUCT = 4 * counts["ALL"]
NPHI = 720


def log(*a):
    print(f"[run_real {time.time() - t0:7.1f}s]", *a, flush=True)


def inputs_of(c):
    i = ["ckm_s12"] if c["c12"] else []
    if c["c23"]:
        i.append("ckm_s23")
    if c["c13"]:
        i += ["ckm_s13", "ckm_delta_rad"]
    return i


def fit_all(ckm, o):
    """Full refined fit of every config for one ordering. Returns list of dict rows."""
    T = L.TableChi2(o)
    P, Q, rc, rt = L.build_PQ(cfgs, ckm)
    g = L.grid_scan(P, Q, {"c": T}, NPHI)["c"]
    phi, chi = L.golden_refine(P, Q, T, g[1], half=2 * np.pi / NPHI)
    chi = np.minimum(chi, g[0])
    st = L.struct_scan(P, Q, {"c": T})["c"]  # (R, 4)
    rows = []
    for c in cfgs:
        m = np.where(rc == c["id"])[0]
        r = dict(id=c["id"], name=c["name"], tier=c["tier"], nu=c["nu"], c12=str(c["c12"]),
                 c23=str(c["c23"]), c13=str(c["c13"]), order=c["order"])
        j = m[np.argmin(chi[m])]
        r["chi2_fit"], r["phi_fit"], r["thnu_fit"] = float(chi[j]), float(phi[j]), float(rt[j])
        if c["tier"] == "TM":  # polish (th_nu, phi) jointly
            def f(x):
                return float(T(*L.obs_exact(L.full_matrix(c, ckm, x[1], x[0]))))
            res = minimize(f, [rt[j], phi[j]], method="Nelder-Mead",
                           options=dict(xatol=1e-9, fatol=1e-10, maxiter=2000))
            if res.fun < r["chi2_fit"]:
                r["chi2_fit"], r["thnu_fit"], r["phi_fit"] = float(res.fun), float(res.x[0]), float(res.x[1] % (2 * np.pi))
        for k in range(4):
            jj = m[np.argmin(st[m, k])]
            cs, th = float(st[jj, k]), float(rt[jj])
            if c["tier"] == "TM":
                def g1(x, k=k):
                    return float(T(*L.obs_exact(L.full_matrix(c, ckm, L.STRUCT_PHI[k], x))))
                lo, hi = max(th - 0.01, 0.0), min(th + 0.01, 0.5)
                res = minimize_scalar(g1, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-10))
                if res.fun < cs:
                    cs, th = float(res.fun), float(res.x)
            r[f"chi2_s{k}"], r[f"thnu_s{k}"] = cs, th
        rows.append(r)
    return rows


def score(c, ckm, o, phi, thnu, k_cont, K, k_own):
    U = L.full_matrix(c, ckm, phi, None if np.isnan(thnu) else thnu)
    pred = dict(zip(PM, L.obs_exact(U)))
    ev = evaluate(pred, o, k=k_cont, use=PM, inputs=inputs_of(c))
    p_loc = ev.p if np.isfinite(ev.p) else np.nan
    row = dict(name=c["name"], tier=c["tier"], nu=c["nu"], c12=str(c["c12"]), c23=str(c["c23"]),
               c13=str(c["c13"]), order=c["order"], phi_deg=float(np.degrees(phi) % 360),
               thnu=thnu, chi2=ev.chi2, chi2_global=ev.chi2_global, N=ev.N, k_cont=k_cont,
               K_trials=K, k_bits=k_cont + np.log2(K), dof_local=ev.N - k_cont, p_local=p_loc,
               p_trials=float(trials_factor_pvalue(p_loc, K)) if np.isfinite(p_loc) else np.nan,
               K_own_tier=k_own,
               p_trials_own_tier=float(trials_factor_pvalue(p_loc, k_own)) if np.isfinite(p_loc) else np.nan,
               max_abs_pull=ev.max_abs_pull)
    for n in PM:
        row[f"pred_{n}"] = pred[n]
        row[f"pull_{n}"] = ev.pulls[n]
    return row


OWN = {"A": counts["A"], "B": counts["AB"], "C": counts["ABC"], "TM": counts["ALL"]}


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ckm_draws(rng, n, scale, run):
    c = data.load_ckm()
    s = [split_normal(rng, c[k]["value"], c[k]["err_plus"], c[k]["err_minus"], n) for k in ("s12", "s13", "s23")]
    d = split_normal(rng, c["delta_rad"]["value"], c["delta_rad"]["err_plus"], c["delta_rad"]["err_minus"], n)
    out = [(np.arcsin(a), np.arcsin(b), np.arcsin(e), dd) for a, b, e, dd in zip(s[0], s[1], s[2], d)]
    if scale == "GUT":
        out = [L.to_gut(x, run) for x in out]
    return out


def main():
    run = L.gut_running()
    log("GUT running", json.dumps(run))
    summary = {"tier_counts": counts, "K_fit": K_FIT, "K_struct": K_STRUCT,
               "bits_fit": float(np.log2(K_FIT)), "bits_struct": float(np.log2(K_STRUCT))}
    spread_rows = []
    rng = np.random.default_rng(3009)
    for scale in ("MZ", "GUT"):
        ckm = L.ckm_at_scale(scale)
        summary[f"ckm_{scale}"] = ckm
        # null pipeline on real data (comparison values for the nulls)
        P, Q, rc, rt = L.build_PQ(cfgs, ckm)
        tier = np.array([cfgs[i]["tier"] for i in rc])
        best = {o: data.load_pmns(o) for o in ("NO", "IO")}
        sc = {"tab_NO": L.TableChi2("NO"), "tab_IO": L.TableChi2("IO")}
        for o in ("NO", "IO"):
            sc[f"gauss_{o}"] = L.GaussChi2([best[o][k]["value"] for k in PM], o)
        summary[f"pipeline_{scale}"] = L.search(P, Q, rc, tier, sc)
        log(scale, "pipeline", json.dumps(summary[f"pipeline_{scale}"]))
        for o in ("NO", "IO"):
            rows = fit_all(ckm, o)
            write(L.HERE / f"fits_{scale}_{o}.csv", rows)
            log(scale, o, "fitted all configs")
            # fitted phase
            top = sorted(rows, key=lambda r: r["chi2_fit"])[:30]
            tf = []
            for r in top:
                c = cfgs[r["id"]]
                kc = 1 + (c["tier"] == "TM")
                tf.append(score(c, ckm, o, r["phi_fit"], r["thnu_fit"], kc, K_FIT, OWN[c["tier"]]))
            write(L.HERE / f"top30_{scale}_fit_{o}.csv", tf)
            # structured phase
            pairs = sorted(((r[f"chi2_s{k}"], r["id"], k, r[f"thnu_s{k}"]) for r in rows for k in range(4)))[:30]
            ts = []
            for chi, i, k, th in pairs:
                c = cfgs[i]
                kc = int(c["tier"] == "TM")
                ts.append(score(c, ckm, o, L.STRUCT_PHI[k], th, kc, K_STRUCT, 4 * OWN[c["tier"]]))
            write(L.HERE / f"top30_{scale}_struct_{o}.csv", ts)
            for var, tab in (("fit", tf), ("struct", ts)):
                log(scale, o, var, "best:", tab[0]["name"], f"chi2={tab[0]['chi2']:.3f}",
                    f"phi={tab[0]['phi_deg']:.1f}", f"p_loc={tab[0]['p_local']:.3g}")
            # sanity ranks
            for nm in ("TBM|c12=1|c23=0|c13=0|std",):
                rr = sorted(rows, key=lambda r: r["chi2_fit"])
                rank = next(i for i, r in enumerate(rr) if r["name"] == nm) + 1
                summary[f"rank_{nm}_{scale}_{o}"] = [rank, rr[rank - 1]["chi2_fit"]]
            # CKM error propagation for top 5 of each variant (phase / th_nu fixed at the fit)
            draws = ckm_draws(rng, 1000, scale, run)
            T = L.TableChi2(o)
            for var, tab in (("fit", tf), ("struct", ts)):
                for r in tab[:5]:
                    c = next(x for x in cfgs if x["name"] == r["name"])
                    th = None if np.isnan(r["thnu"]) else r["thnu"]
                    ob = np.array([L.obs_exact(L.full_matrix(c, d, np.radians(r["phi_deg"]), th)) for d in draws])
                    ch = T(*ob.T)
                    q = np.percentile(ob, [16, 84], axis=0)
                    spread_rows.append(dict(scale=scale, ordering=o, variant=var, name=r["name"],
                                            phi_deg=r["phi_deg"], chi2_central=r["chi2"],
                                            chi2_p16=float(np.percentile(ch, 16)),
                                            chi2_p84=float(np.percentile(ch, 84)),
                                            **{f"hw68_{n}": float((q[1, i] - q[0, i]) / 2) for i, n in enumerate(PM)}))
    write(L.HERE / "ckm_spread.csv", spread_rows)
    with open(L.HERE / "real_pipeline.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()

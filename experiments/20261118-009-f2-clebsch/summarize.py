"""Tier-best table (tier_best.csv) and null summary (null_summary.json).

Tier sets are nested searches: A (th12e only, 60 configs), AB (+ th23e, 1380), ABC
(+ th13e, 17220), ALL (+ TM-type U_nu with fitted th_nu, 17316). K = configs in the set
(fitted phase) or 4 x configs (structured phase). k_cont = fitted continuous parameters
of the best config (phi and/or th_nu); local p uses dof = 4 - k_cont; p_trials = Sidak(K).
Null hit = fake best chi^2 <= real best chi^2 from the same pipeline (pipeline_real.json),
same scale / ordering / variant / tier set.
"""
import csv
import json

import numpy as np

import f2lib as L
from src.stats import fake_universe_rate, p_value, trials_factor_pvalue

cfgs = L.configs()
counts = L.tier_counts(cfgs)


def tier_best():
    rows = []
    for scale in ("MZ", "GUT"):
        for o in ("NO", "IO"):
            with open(L.HERE / f"fits_{scale}_{o}.csv") as fh:
                fits = list(csv.DictReader(fh))
            for tset, tiers in L.TIER_SETS.items():
                sub = [r for r in fits if r["tier"] in tiers]
                for var in ("fit", "struct"):
                    if var == "fit":
                        best = min(sub, key=lambda r: float(r["chi2_fit"]))
                        chi, phi = float(best["chi2_fit"]), np.degrees(float(best["phi_fit"])) % 360
                        kc = 1 + (best["tier"] == "TM")
                        K = counts[tset]
                    else:
                        chi, best, k = min(((float(r[f"chi2_s{k}"]), r, k) for r in sub for k in range(4)),
                                           key=lambda x: x[0])
                        phi = float(np.degrees(L.STRUCT_PHI[k]))
                        kc = int(best["tier"] == "TM")
                        K = 4 * counts[tset]
                    p = p_value(chi, 4, kc)
                    rows.append(dict(scale=scale, ordering=o, tier_set=tset, variant=var, best=best["name"],
                                     chi2=chi, phi_deg=phi, k_cont=kc, K=K, k_bits=kc + np.log2(K),
                                     p_local=p, p_trials=float(trials_factor_pvalue(p, K))))
    with open(L.HERE / "tier_best.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def nulls():
    real = json.loads((L.HERE / "pipeline_real.json").read_text())
    out = {}
    for kind, sk in (("a", "gauss"), ("b", "tab")):
        path = L.HERE / f"null_{kind}.csv"
        if not path.exists():
            continue
        with open(path) as fh:
            rows = list({int(r["i"]): r for r in csv.DictReader(fh)}.values())  # one row per fake index
        n = len(rows)
        for scale in ("MZ", "GUT"):
            for o in ("NO", "IO"):
                for var in ("fit", "struct"):
                    for tset in L.TIER_SETS:
                        col = f"{scale}_{sk}_{o}_{var}_{tset}"
                        rv = real[scale][f"{sk}_{o}"][var][tset]
                        v = np.array([float(r[col]) for r in rows])
                        hits = int(np.sum(v <= rv))
                        fr = fake_universe_rate(hits, n)
                        out[f"{kind}|{scale}|{o}|{var}|{tset}"] = dict(
                            null=kind, scale=scale, ordering=o, variant=var, tier_set=tset, real_best=rv,
                            hits=hits, n=n, rate=fr["rate"], ci95=[fr["lo"], fr["hi"]], upper95=fr["upper"],
                            fake_median_best=float(np.median(v)), fake_p10_best=float(np.percentile(v, 10)))
    with open(L.HERE / "null_summary.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def lo_ckm_spread(tb, n=1000):
    """CKM-input error propagation for the best LO-pattern (tier set ABC) configs, which the
    top-30 tables (dominated by TM-type) do not cover. phi fixed at the fit. -> ckm_spread_lo.csv"""
    from run_real import ckm_draws
    rng = np.random.default_rng(4009)
    run = L.gut_running()
    by = {c["name"]: c for c in cfgs}
    out = []
    for r in tb:
        if r["tier_set"] != "ABC":
            continue
        c = by[r["best"]]
        T = L.TableChi2(r["ordering"])
        ob = np.array([L.obs_exact(L.full_matrix(c, d, np.radians(r["phi_deg"])))
                       for d in ckm_draws(rng, n, r["scale"], run)])
        ch = T(*ob.T)
        q = np.percentile(ob, [16, 84], axis=0)
        ckm0 = L.ckm_at_scale(r["scale"])
        cen = L.obs_exact(L.full_matrix(c, ckm0, np.radians(r["phi_deg"])))
        out.append(dict(scale=r["scale"], ordering=r["ordering"], variant=r["variant"], name=r["best"],
                        phi_deg=r["phi_deg"],
                        **{f"pred_{k}": cen[i] for i, k in enumerate(("s12sq", "s13sq", "s23sq", "dcp_deg"))},
                        chi2_central=r["chi2"], chi2_p16=float(np.percentile(ch, 16)),
                        chi2_p84=float(np.percentile(ch, 84)),
                        **{f"hw68_{k}": float((q[1, i] - q[0, i]) / 2)
                           for i, k in enumerate(("s12sq", "s13sq", "s23sq", "dcp_deg"))}))
    with open(L.HERE / "ckm_spread_lo.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":
    tb = tier_best()
    for r in lo_ckm_spread(tb):
        print(r)
    for r in tb:
        print(f"{r['scale']:3} {r['ordering']} {r['tier_set']:3} {r['variant']:6} {r['best']:38} chi2={r['chi2']:8.3f} "
              f"k_bits={r['k_bits']:5.2f} p={r['p_local']:.3g} p_tr={r['p_trials']:.3g}")
    for k, v in nulls().items():
        print(f"{k:28} real={v['real_best']:7.3f} hits={v['hits']:3}/{v['n']} rate={v['rate']:.3f} "
              f"CI=[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] median_fake={v['fake_median_best']:.2f}")

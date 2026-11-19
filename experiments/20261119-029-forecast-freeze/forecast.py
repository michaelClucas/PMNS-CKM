"""Exp 029 forecast: projected pulls of the frozen relations if future central values equal the
NuFIT 6.1 best fit, with projected uncertainties (sensitivities.json); exclusion fractions; the
inverse scan (which future values single out a relation); Sidak thresholds for the frozen list.

Run from this folder after freeze.py:
    ../../.venv/Scripts/python forecast.py > forecast.log
Outputs: forecast.csv, windows.csv, forecast_summary.json.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2d, norm

import warnings

import freeze as F

warnings.filterwarnings("ignore", category=RuntimeWarning)
from src import data

PMNS = F.PMNS
ORDS = F.ORDS
Z3, Z5 = 3.0, 5.0
P3, P5 = 2 * norm.sf(Z3), 2 * norm.sf(Z5)


def specs_by_id():
    """Rebuild the model callables of the fitted families (same definitions as freeze.py)."""
    from src.baselines import BASELINES
    out = {}
    for bid, b in BASELINES.items():
        out[f"K-{bid}"] = (F.baseline_model(b), dict(b.params), tuple(b.periodic))
    for pat in ("GR2", "TBM"):
        out[f"F2LO-{pat}"] = (F.f2_model(pat), {"phi": (0.0, F.TWO_PI)}, ("phi",))
    for pat in ("I", "II"):
        out[f"BL-CDSV-{pat}"] = (F.cdsv_model(pat), {"phi": (0.0, F.TWO_PI)}, ("phi",))
    for ut in (1, 2, 4):
        out[f"BL-RMSV-{ut}"] = (F.rmsv_model(ut), {"psi": (1.0, 4.4), "phi": (0.0, F.TWO_PI)}, ("phi",))
    return out


def scenario_sigmas():
    s = json.load(open(F.HERE / "sensitivities.json"))["scenarios"]
    out = {}
    for name, sc in s.items():
        out[name] = {}
        for o in ORDS:
            cur = F.gauss_sigma(o)
            d = {}
            for k in PMNS:
                v = sc[k]["sigma"]
                d[k] = cur[k] if v == "current" else (v[o] if isinstance(v, dict) else float(v))
            out[name][o] = d
    out["current_gauss"] = {o: F.gauss_sigma(o) for o in ORDS}
    return out


ALT_OCTANT_SCAN = np.round(np.arange(0.38, 0.64001, 0.0005), 4)


def centres(o, alt=False):
    """Future central values: NuFIT 6.1 best fit of ordering o; alt=True moves s23sq to the
    minimum of the NuFIT 1D s23sq profile in the opposite octant (NO 0.550, Delta chi^2 1.03;
    IO 0.470, Delta chi^2 2.27), other parameters unchanged."""
    p = data.load_pmns(o)
    c = {k: p[k]["value"] for k in PMNS}
    if alt:
        prof = data.pmns_profile_1d(o, "s23sq", ALT_OCTANT_SCAN)
        side = ALT_OCTANT_SCAN > 0.5 if c["s23sq"] < 0.5 else ALT_OCTANT_SCAN < 0.5
        c["s23sq"] = float(ALT_OCTANT_SCAN[side][np.argmin(prof[side])])
    return c


SCENARIOS = (("current_gauss", "current_gauss", False), ("nominal", "nominal", False),
             ("optimistic", "optimistic", False), ("nominal_altoct", "nominal", True),
             ("optimistic_altoct", "optimistic", True))


def resid(k, pred, c):
    d = pred - c
    return F.wrap_deg(d) if k == "dcp_deg" else d


def fit_future(model, bounds, periodic, th0, ctx, cen, sig_eff):
    names = list(bounds)

    def f(x):
        p = model(dict(zip(names, x)), ctx)
        v = sum((resid(k, p[k], cen[k]) / sig_eff[k]) ** 2 for k in PMNS)
        return v if np.isfinite(v) else 1e12

    starts = [np.array([th0[n] for n in names])]
    rng = np.random.default_rng(7)
    for _ in range(5):
        starts.append(np.array([rng.uniform(*bounds[n]) for n in names]))
    best = None
    for x0 in starts:
        r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    th = dict(zip(names, best.x))
    return float(best.fun), th, model(th, ctx)


def z_of(c2, dof):
    p = float(chi2d.sf(c2, dof))
    return p, float(norm.isf(p / 2)) if p > 0 else np.inf


def main():
    rows = list(csv.DictReader(open(F.HERE / "frozen_relations.csv")))
    specs = specs_by_id()
    sigs = scenario_sigmas()
    cen_ctx = F.central_ctx()
    ctxs_opt = F.sample_ctxs(F.N_CKM_SAMPLES, scale_err={"rhobar": 0.25, "etabar": 0.25})
    out = []
    for r in rows:
        o = r["ordering"]
        for scn, sname, alt in SCENARIOS:
            cen = centres(o, alt)
            sg = sigs[sname][o]
            rec = dict(id=r["id"], family=r["family"], ordering=o, scenario=scn,
                       survives_today=r["survives_today"] == "True", k_cont=int(r["k_cont"]))
            if r["family"] == "S":
                t = r["predicts"]
                pred, sp = float(r[f"pred_{t}"]), float(r[f"sig_ckm_{t}"])
                pull = resid(t, pred, cen[t]) / np.hypot(sg[t], sp)
                rec.update(N=1, dof=1, chi2=pull ** 2, p=float(2 * norm.sf(abs(pull))), Z=abs(pull),
                           max_abs_pull=abs(pull), **{f"pull_{k}": (pull if k == t else np.nan) for k in PMNS},
                           **{f"fpred_{k}": (pred if k == t else np.nan) for k in PMNS})
            else:
                model, bounds, periodic = specs[r["id"]]
                th0 = json.loads(r["params_today"])
                if sname == "optimistic":
                    sck = F.ckm_spread(model, th0, PMNS, ctxs_opt)
                else:
                    sck = {k: float(r[f"sig_ckm_{k}"]) for k in PMNS}
                se = {k: np.hypot(sg[k], sck[k]) for k in PMNS}
                c2, th, pred = fit_future(model, bounds, periodic, th0, cen_ctx, cen, se)
                dof = 4 - int(r["k_cont"])
                p, Z = z_of(c2, dof)
                pulls = {k: resid(k, pred[k], cen[k]) / se[k] for k in PMNS}
                rec.update(N=4, dof=dof, chi2=c2, p=p, Z=Z, max_abs_pull=max(abs(v) for v in pulls.values()),
                           **{f"pull_{k}": v for k, v in pulls.items()},
                           **{f"fpred_{k}": pred[k] for k in PMNS}, params_future=json.dumps(
                               {k: round(float(v), 5) for k, v in th.items()}))
            rec["excl_3sigma"] = bool(rec["p"] < P3)
            rec["excl_5sigma"] = bool(rec["p"] < P5)
            out.append(rec)
        print(r["id"], o, [(q["scenario"], round(float(q["chi2"]), 2), round(float(q["Z"]), 2)) for q in out[-5:]], flush=True)
    keys = sorted({k for q in out for k in q}, key=lambda k: list(out[0]).index(k) if k in out[0] else 99)
    with open(F.HERE / "forecast.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for q in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in q.items()})

    # ---- exclusion fractions over relations surviving today in that ordering
    summary = {"P3": P3, "P5": P5, "exclusion": {}}
    summary["centres"] = {o: {"bestfit": centres(o), "alt_octant": centres(o, True)} for o in ORDS}
    for scn, _, _ in SCENARIOS:
        for o in ORDS:
            sel = [q for q in out if q["scenario"] == scn and q["ordering"] == o and q["survives_today"]]
            fam = {}
            for q in sel:
                f = fam.setdefault(q["family"], [0, 0, 0])
                f[0] += 1
                f[1] += q["excl_3sigma"]
                f[2] += q["excl_5sigma"]
            M = len(sel)
            summary["exclusion"][f"{scn}/{o}"] = dict(
                M=M, n_excl_3sigma=sum(q["excl_3sigma"] for q in sel), n_excl_5sigma=sum(q["excl_5sigma"] for q in sel),
                frac_3sigma=sum(q["excl_3sigma"] for q in sel) / M, frac_5sigma=sum(q["excl_5sigma"] for q in sel) / M,
                by_family={k: dict(n=v[0], excl3=v[1], excl5=v[2]) for k, v in fam.items()},
                excluded_3sigma=[q["id"] for q in sel if q["excl_3sigma"]])
    # ---- Sidak over the frozen list (per ordering: only the true ordering's rows are tested)
    sid = {}
    for o in ORDS:
        M = summary["exclusion"][f"nominal/{o}"]["M"]
        d = {"M": M}
        for name, pg in (("3sigma", P3), ("5sigma", P5), ("p0.05", 0.05)):
            pl = 1 - (1 - pg) ** (1 / M)
            d[name] = dict(p_global=pg, p_local=pl, Z_local=float(norm.isf(pl / 2)))
        sid[o] = d
    summary["sidak"] = sid
    print(json.dumps(summary, indent=1))

    # ---- inverse scan
    grids = {"s12sq": np.round(np.arange(0.295, 0.32501, 0.0005), 6),
             "s23sq": np.round(np.arange(0.42, 0.60001, 0.005), 4),
             "dcp_deg": np.arange(0.0, 360.0, 5.0)}
    win = []
    for scn in ("nominal",):
        for o in ORDS:
            sg = sigs[scn][o]
            base = centres(o)
            for t, grid in grids.items():
                for x in grid:
                    cen = dict(base)
                    cen[t] = float(x)
                    compat = []
                    for r in rows:
                        if r["ordering"] != o or r["survives_today"] != "True":
                            continue
                        if r["family"] == "S":
                            if r["predicts"] != t:
                                continue
                            pull = resid(t, float(r[f"pred_{t}"]), x) / np.hypot(sg[t], float(r[f"sig_ckm_{t}"]))
                            Z = abs(pull)
                        else:
                            model, bounds, periodic = specs[r["id"]]
                            th0 = json.loads(r["params_today"])
                            se = {k: np.hypot(sg[k], float(r[f"sig_ckm_{k}"])) for k in PMNS}
                            c2, _, _ = fit_future(model, bounds, periodic, th0, cen_ctx, cen, se)
                            _, Z = z_of(c2, 4 - int(r["k_cont"]))
                        if Z < 2.0:
                            compat.append(r["id"])
                    win.append(dict(scenario=scn, ordering=o, observable=t, value=float(x), n_compatible=len(compat),
                                    compatible=" ".join(compat)))
                print("scan", o, t, flush=True)
    with open(F.HERE / "windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(win[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(win)
    json.dump(summary, open(F.HERE / "forecast_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Exp 046 summary: fits.jsonl (N 15) + fits_PMNS.jsonl (PMNS-only lower bounds) -> summary.json, printed table.

Per option (model, assign, Y_nu, M_R): minimum over seeds of each grid cell (scale, eta), the seed spread
of that cell, and the best cell. p_global recomputed with dof = N - k_cont (k_cont includes eta_b, eta_l
for RRR5 and the fitted scale for HS). Outside the NuFIT tables chi2 = inf, p = 0; chi2_soft is the
minimiser's landscape value (1e4 + Gaussian fallback), reported only to rank failures.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib46 as L  # noqa: E402,F401  (sets sys.path)
from src.stats import p_value  # noqa: E402


def load(fn):
    p = HERE / fn
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def key(d):
    return (d["model"], d["assign"], d["ynu"], d["mr"])


def cellkey(d):
    return (d["scale"], d["eta_b"], d["eta_l"])


def summarize(rows):
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    cells = defaultdict(list)
    for r in rows:
        cells[(key(r), cellkey(r))].append(r)
    opt = defaultdict(list)
    for (k, c), rs in cells.items():
        rs = sorted(rs, key=lambda r: (r["chi2"], r["chi2_soft"]))
        b = rs[0]
        softs = [r["chi2_soft"] for r in rs]
        opt[k].append({"cell": c, "best": b, "n_seeds": len(rs), "soft_spread": max(softs) - min(softs),
                       "chi2_by_seed": [r["chi2"] for r in rs], "soft_by_seed": softs})
    out = {}
    for k, cs in opt.items():
        cs.sort(key=lambda c: (c["best"]["chi2"], c["best"]["chi2_soft"]))
        b = cs[0]["best"]
        N, kc = b["N"], b["k_cont"]
        p = (float(p_value(b["chi2"], N, kc)) if N > kc else float("nan")) if np.isfinite(b["chi2"]) else 0.0
        out[" ".join(k)] = {"N": N, "k_cont": kc, "bits": b["bits"], "k": b["k"], "chi2": b["chi2"],
                            "chi2_soft": b["chi2_soft"], "p_global": p, "max_pull": b["max_pull"],
                            "compression_ok": b["k"] <= N - 2,
                            "fit_ok": bool(np.isfinite(b["chi2"]) and b["max_pull"] < 2 and p > 0.05),
                            "best_cell": cs[0]["cell"], "seeds": cs[0]["n_seeds"],
                            "soft_spread_best_cell": cs[0]["soft_spread"],
                            "soft_by_seed_best_cell": cs[0]["soft_by_seed"],
                            "pulls": b["pulls"], "pred": b["pred"], "params": b["params"],
                            "cells": [(c["cell"], c["best"]["chi2"], c["best"]["chi2_soft"], c["n_seeds"])
                                      for c in cs]}
    return out, errs


if __name__ == "__main__":
    s15, e15 = summarize(load("fits.jsonl"))
    spm, epm = summarize(load("fits_PMNS.jsonl"))
    print(f"N15 options {len(s15)}, errors {len(e15)}; PMNS-only options {len(spm)}, errors {len(epm)}")
    hdr = "option | N | k_cont+bits=k | chi2 | soft | p_global | max pull | comp | fit | best cell | seeds soft"
    print(hdr)
    for k, v in sorted(s15.items()):
        print(f"{k} | {v['N']} | {v['k_cont']}+{v['bits']}={v['k']} | {v['chi2']:.4g} | {v['chi2_soft']:.5g} | "
              f"{v['p_global']:.2g} | {v['max_pull']:.3g} | {v['compression_ok']} | {v['fit_ok']} | {v['best_cell']} | "
              f"{[round(x, 1) for x in v['soft_by_seed_best_cell']]}")
        top = sorted(((n, x) for n, x in v["pulls"].items() if np.isfinite(x)), key=lambda t: -abs(t[1]))[:4]
        print("    pred PMNS", {n: round(v["pred"][n], 4) for n in L.PM + ("dm_ratio", "mr_e_mu", "mr_mu_tau")},
              "largest finite pulls", [(n, round(x, 1)) for n, x in top])
    print("\nPMNS-only lower bounds (use = 4 PMNS + dm_ratio, all parameters free)")
    for k, v in sorted(spm.items()):
        print(f"{k} | chi2 {v['chi2']:.4g} soft {v['chi2_soft']:.5g} | best cell {v['best_cell']} | "
              f"pred {{{', '.join(f'{n}: {v['pred'][n]:.4g}' for n in L.PM + ('dm_ratio',))}}} | "
              f"seeds soft {[round(x, 1) for x in v['soft_by_seed_best_cell']]}")
    json.dump({"N15": s15, "PMNS_only": spm, "errors": e15 + epm}, open(HERE / "summary.json", "w"), indent=1,
              default=lambda o: None if o is None else float(o))

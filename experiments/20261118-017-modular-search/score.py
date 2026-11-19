"""Joint common-tau refinement of the best quark x lepton combinations, scoring through
src.evaluate (via the Universe), emulator-vs-exact comparison, results.csv.

    python score.py      (after search.py joint)
"""
from __future__ import annotations

import csv
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, ROOT]

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import models as md  # noqa: E402
import modforms as mf  # noqa: E402
import search as se  # noqa: E402
from src.nulls import Universe  # noqa: E402
from src.stats import p_value  # noqa: E402

N_REFINE = 4  # quark x lepton combinations refined jointly per group and ordering
N_EXT = 15    # 8 mixing + 6 charged-fermion log mass ratios + dm21/|dm3l|


def trials():
    per = {N: md.catalog_size(N) for N in mf.LEVELS}
    return per, sum(per.values())


def emulator_comparison(rows):
    out = []
    for N in mf.LEVELS:
        for kind, ords in (("q", ("NO",)), ("l", se.ORDERINGS)):
            for o in ords:
                r = [x for x in rows if x["N"] == N and x["kind"] == kind and x["ordering"] == o]
                if len(r) < 4:
                    continue
                e = np.array([x["emu_chi2"] for x in r])
                x_ = np.array([x["exact_chi2"] for x in r])
                rnd = [x for x in r if x["subset"] == "random"]
                top = [x for x in r if x["subset"] == "top"]
                rho_all = spearmanr(e, x_).statistic
                rho_rnd = spearmanr([x["emu_chi2"] for x in rnd], [x["exact_chi2"] for x in rnd]).statistic \
                    if len(rnd) > 3 else float("nan")
                best10 = sorted(r, key=lambda x: x["exact_chi2"])[:10]
                out.append({"group": mf.GROUP[N], "sector": kind, "ordering": o, "n_top": len(top),
                            "n_random": len(rnd), "spearman_all": rho_all, "spearman_random": rho_rnd,
                            "median_exact_top": float(np.median([x["exact_chi2"] for x in top])),
                            "median_exact_random": float(np.median([x["exact_chi2"] for x in rnd])) if rnd else float("nan"),
                            "min_exact_top": float(min(x["exact_chi2"] for x in top)),
                            "min_exact_random": float(min(x["exact_chi2"] for x in rnd)) if rnd else float("nan"),
                            "frac_exact_best10_from_top": sum(x["subset"] == "top" for x in best10) / len(best10)})
    return out


def joint_fit(qa, la, zq, zl, o, u, T):
    scq = u.scorer(se.USE8[:4], o)
    scl = u.scorer(se.USE8[4:], o)
    fq = se.exact_chi2_fn(qa, T, scq, o)
    fl = se.exact_chi2_fn(la, T, scl, o)
    loq, hiq = se.bounds(qa)
    lol, hil = se.bounds(la)
    nq = len(loq) - 2
    lo = np.concatenate([loq, lol[2:]])
    hi = np.concatenate([hiq, hil[2:]])

    def f(z):
        zc = np.clip(z, lo, hi)
        pen = 10.0 * float(((z - zc) ** 2).sum())
        return fq(zc[: 2 + nq]) + fl(np.concatenate([zc[:2], zc[2 + nq:]])) + pen

    best = (np.inf, None)
    for s in (np.concatenate([zq, zl[2:]]), np.concatenate([zl[:2], zq[2:], zl[2:]])):
        x = s
        for it in range(2):
            r = minimize(f, x, method="Nelder-Mead", options={"maxfev": 3000, "adaptive": True,
                                                              "xatol": 1e-7, "fatol": 1e-7})
            x = r.x
        if r.fun < best[0]:
            best = (float(r.fun), np.clip(r.x, lo, hi))
    z = best[1]
    oq = se.sector_obs(qa, z[: 2 + nq])
    ol = se.sector_obs(la, np.concatenate([z[:2], z[2 + nq:]]), o)
    return best[0], z, oq, ol


def score_point(qa, la, z, oq, ol, o, u, T, bits):
    pred = {k: float(oq[k][0]) for k in se.USE8[:4]}
    pred.update({k: float(ol[k][0]) for k in se.USE8[4:]})
    k_cont_ext = 2 + qa.k_cont + la.k_cont
    ratio_terms = {k: ((float((oq if k in oq else ol)[k][0]) - T["ratio"][k][0]) / T["ratio"][k][1]) for k in
                   ("lr_uc", "lr_ct", "lr_ds", "lr_sb", "lr_emu", "lr_mutau", "r_dm")}
    k_core_cont = max(k_cont_ext - 7, 0)
    res = u.evaluate(pred, use=se.USE8, ordering=o, k=k_core_cont + bits, k_discrete=bits,
                     inputs=se.MASS_INPUTS, ordering_selected=True)
    chi2_ext = res.chi2_global + sum(v ** 2 for v in ratio_terms.values())
    p_ext = p_value(chi2_ext, N_EXT, k_cont_ext)
    k_ext = k_cont_ext + bits + 1
    tau = md.tau_from_unit(np.atleast_2d(z[:2]))[0]
    max_pull_ext = max(res.max_abs_pull, max(abs(v) for v in ratio_terms.values()))
    return {
        "group": mf.GROUP[qa.N], "ordering": o, "quarks": qa.label(), "leptons": la.label(),
        "tau_re": tau.real, "tau_im": tau.imag,
        "chi2_ext": chi2_ext, "N_ext": N_EXT, "k_cont_ext": k_cont_ext, "bits": bits + 1, "k_ext": k_ext,
        "p_ext": p_ext, "compression_ok_ext": k_ext <= N_EXT - 2,
        "fit_ok_ext": bool(max_pull_ext < 2 and p_ext > 0.05), "max_pull_ext": max_pull_ext,
        "eval_chi2_global": res.chi2_global, "eval_N": res.N, "eval_k": res.k, "eval_p_global": res.p_global,
        "eval_compression_ok": res.compression_ok, "eval_fit_ok": res.fit_ok, "eval_max_pull": res.max_abs_pull,
        **{f"pull_{k}": v for k, v in res.pulls.items()},
        **{f"pull_{k}": v for k, v in ratio_terms.items()},
        **{f"pred_{k}": v for k, v in pred.items()},
    }


def _joint_task(args):
    N, o, iq, il, zq, zl, bits, extra = args
    u = Universe.real(o)
    T = se.targets(u, o)
    qa, la = md.quark_catalog(N)[iq], md.lepton_catalog(N)[il]
    c, z, oq, ol = joint_fit(qa, la, zq, zl, o, u, T)
    row = score_point(qa, la, z, oq, ol, o, u, T, bits)
    row.update(extra)
    row["joint_fit_chi2"] = c
    row["z"] = " ".join(f"{v:.6g}" for v in z)
    return row


def main():
    rows = json.load(open(os.path.join(HERE, "exact_sector_fits.json")))
    comp = emulator_comparison(rows)
    with open(os.path.join(HERE, "emulator_vs_exact.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)
    per, total = trials()
    bits = math.ceil(math.log2(total))
    print("trials per group", per, "total", total, "bits", bits, flush=True)
    P = pickle.load(open(os.path.join(se.ART, "profiles.pkl"), "rb"))
    prof, sel = P["prof"], P["sel"]
    out, tasks = [], []
    for N in mf.LEVELS:
        for o in se.ORDERINGS:
            qs, ls = sel[(N, "q", "NO")], sel[(N, "l", o)]
            combos = []
            for q in qs:
                pq, zq = prof[(N, "q", "NO", q["i"])]
                for l_ in ls:
                    pl, zl = prof[(N, "l", o, l_["i"])]
                    s = pq[:-1] + pl[:-1]  # last point is each sector's own best tau, not shared
                    j = int(np.argmin(s))
                    combos.append((float(s[j]), q, l_, np.array(zq[j]), np.array(zl[j])))
            combos.sort(key=lambda c: c[0])
            for gsum, q, l_, zq, zl in combos[:N_REFINE]:
                tasks.append((N, o, q["i"], l_["i"], zq, zl, bits,
                              {"grid_chi2": gsum, "sector_chi2_q": q["exact_chi2"], "sector_chi2_l": l_["exact_chi2"]}))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("MODSEARCH_WORKERS", "3"))) as pool:
        for row in pool.imap_unordered(_joint_task, tasks):
            out.append(row)
            print(f"{row['group']} {row['ordering']} grid {row['grid_chi2']:.1f} joint {row['joint_fit_chi2']:.1f} "
                  f"ext {row['chi2_ext']:.1f} p_ext {row['p_ext']:.2g} | {row['quarks']} | {row['leptons']}", flush=True)
    out.sort(key=lambda r: r["chi2_ext"])
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    json.dump({"trials_per_group": per, "total_trials": total, "bits": bits}, open(os.path.join(HERE, "catalog.json"), "w"))
    print("done", len(out), flush=True)


if __name__ == "__main__":
    main()

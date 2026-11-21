"""Exp 047 (b): L-BFGS-B multistart re-fits of DE-only "excluded" fits, through each experiment's own
model code and the official scorer (src.evaluate.fit_and_evaluate, method="lbfgsb").

    python refit_lbfgsb.py <config> [n_sobol] [n_perturb]

configs: AB5, AB30, UTZ, UTZHO (exp 040); YK (exp 038 full, floor 0.02); AKS, CMNO, CMIO (exp 042).
Starts: n_sobol scrambled-Sobol points over the box (seed 4702) + every stored DE optimum + n_perturb
Gaussian perturbations of the best DE point (sigma = 5% of each box width; 2% for half of them).
Each start runs L-BFGS-B, the best gets the Nelder-Mead polish (fit_and_evaluate). The result is
rescored with the experiment's own honest k exactly as in its fit script.
"""
import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from src import evaluate as ev  # noqa: E402

E40 = ROOT / "experiments/20261120-040-compressive-scout"
E38 = ROOT / "experiments/20261120-038-yukawaon"
E42 = ROOT / "experiments/20261120-042-open-models"


def load(d, mod):
    sys.path.insert(0, str(d))
    return importlib.import_module(mod)


def cfg(name):
    if name in ("AB5", "AB30", "UTZ", "UTZHO"):
        F = load(E40, "fit_more")
        key = {"AB5": "AB", "AB30": "AB"}.get(name, name)
        fn, B, per, use, kd, tb = F.CFG[key]
        tb = {"AB5": 5.0, "AB30": 30.0}.get(name, tb)
        files = {"AB5": ["fit_AB_tb5_s1.json"], "AB30": ["fit_AB_tb30_s2.json"], "UTZ": ["fit_UTZ_tb30_s1.json"],
                 "UTZHO": ["fit_UTZHO_tb30_s1.json"]}[name]
        old = [json.loads((E40 / f).read_text()) for f in files]
        des = [o["params"] for o in old]
        if name == "UTZHO":  # the plain UTZ optimum is a point of UTZHO with dp_d = 0
            u = json.loads((E40 / "fit_UTZ_tb30_s1.json").read_text())["params"]
            des.append({**u, "dp_d": 0.0})
        sc = {"mu": 2e16, "model": "MSSM", "tan_beta": tb}
        kw = dict(k_discrete=kd, periodic=per, mass_floor=F.FLOOR, scale=sc)
        return dict(fn=fn, B=B, use=use, ordering="NO", kw=kw, des=des, old=old, files=files,
                    rescore=None)
    if name == "YK":
        F = load(E38, "fit")
        assert F.FLOOR == 0.02
        use = F.CKM + F.MQ + F.PMNS
        files = ["fit_full_s1.json", "fit_full_s2.json"]
        old = [json.loads((E38 / f).read_text()) for f in files]
        des = [o["params"] for o in old] + [{k: F.BENCH[k] for k in F.BOUNDS_FULL}]
        kw = dict(inputs=F.INPUTS, k_discrete=F.K_BITS, periodic=F.PERIODIC, mass_floor=F.FLOOR)
        return dict(fn=F.model_fn, B=F.BOUNDS_FULL, use=use, ordering=F.ORDERING, kw=kw, des=des, old=old,
                    files=files, rescore=None)
    if name in ("AKS", "CMNO", "CMIO"):
        F = load(E42, "fit42")
        if name == "AKS":
            fn, B, per, scale, kx, bits, ob, o = F.aks_model, F.AKS_B, F.AKS_PER, dict(F.AKS_SCALE), \
                F.AKS_K_EXTRA, F.AKS_BITS, 0, "NO"
            files = ["fit_aks_NO_s1.json", "fit_aks_NO_s2.json"]
        else:
            o = name[2:]
            fn, B, per, scale, kx, bits, ob = F.cm_model_factory(o), F.CM_B, F.CM_PER, dict(F.CM_SCALE), \
                F.CM_K_EXTRA, F.CM_BITS, 1
            files = {"NO": ["fit_cm_NO_s1.json", "fit_cm_NO_s2.json", "fit_cm_NO_s3.json"],
                     "IO": ["fit_cm_IO_s1.json"]}[o]
        old = [json.loads((E42 / f).read_text()) for f in files]
        des = [x["params_fit"] for x in old]
        use = list(F.ALL)
        kw = dict(k_discrete=0, periodic=per, mass_floor=F.FLOOR, scale=scale)

        def rescore(pred, use=use, o=o, scale=scale, B=B, kx=kx, bits=bits, ob=ob):
            sc = ev.Scorer(use, o, mass_floor=F.FLOOR, scale=scale)
            return sc(pred, k=len(B) + kx + bits + ob, k_discrete=bits + ob)
        return dict(fn=fn, B=B, use=use, ordering=o, kw=kw, des=des, old=old, files=files, rescore=rescore)
    raise SystemExit(f"unknown config {name}")


def main():
    name = sys.argv[1]
    n_sobol = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    n_pert = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    c = cfg(name)
    B = c["B"]
    names = list(B)
    lo = np.array([B[n][0] for n in names])
    hi = np.array([B[n][1] for n in names])
    rng = np.random.default_rng(4702)
    old_best = min(c["old"], key=lambda o: o["chi2"])
    best_de = old_best.get("params_fit", old_best.get("params"))
    extra = [dict(d) for d in c["des"]]
    for i in range(n_pert):
        s = 0.05 if i < n_pert // 2 else 0.02
        x = np.array([best_de[n] for n in names]) + rng.normal(0, s * (hi - lo))
        x = np.clip(x, lo, hi)
        extra.append(dict(zip(names, x)))
    # chi2 of the stored DE optima, recomputed with the current code
    sc0 = ev.Scorer(c["use"], c["ordering"], inputs=c["kw"].get("inputs", ()), mass_floor=c["kw"]["mass_floor"],
                    scale=c["kw"].get("scale"))
    de_re = [float(sc0.chi2(c["fn"](d))) for d in c["des"]]
    print(f"{name}: stored DE chi2 {[round(o['chi2'], 3) for o in c['old']]}, recomputed {[round(x, 3) for x in de_re]}",
          flush=True)
    t0 = time.time()
    fr = ev.fit_and_evaluate(c["fn"], B, c["ordering"], c["use"], n_starts=n_sobol, seed=4702, extra_starts=extra,
                             method="lbfgsb",
                             optimizer_options={"maxiter": 3000, "maxfun": 300 * len(names)}, **c["kw"])
    r = fr.result if c["rescore"] is None else c["rescore"](c["fn"](fr.params))
    sc = np.array(fr.start_chi2)
    n_s = n_sobol
    out = {"config": name, "files": c["files"], "old_chi2": [o["chi2"] for o in c["old"]],
           "old_fit_ok": [o["fit_ok"] for o in c["old"]], "old_compression_ok": [o["compression_ok"] for o in c["old"]],
           "old_k": old_best["k"], "de_chi2_recomputed": de_re,
           "new_chi2": r.chi2, "new_chi2_global": r.chi2_global, "N": r.N, "k": r.k, "k_continuous": r.k_continuous,
           "dof": r.dof, "p_global": r.p_global, "max_abs_pull": r.max_abs_pull, "fit_ok": r.fit_ok,
           "compression_ok": r.compression_ok, "pulls": r.pulls, "params": fr.params,
           "n_starts": fr.n_starts, "best_sobol_start": float(sc[:n_s].min()),
           "median_sobol_start": float(np.median(sc[:n_s])),
           "best_warm_start": float(sc[n_s:].min()) if len(sc) > n_s else None,
           "n_starts_within_1": int((sc < r.chi2 + 1).sum()), "start_chi2": sc.tolist(),
           "nfev": fr.nfev, "seconds": time.time() - t0}
    (HERE / f"refit_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("config", "old_chi2", "new_chi2", "N", "k", "dof", "p_global",
                                          "max_abs_pull", "fit_ok", "compression_ok", "best_sobol_start",
                                          "median_sobol_start", "best_warm_start", "n_starts_within_1", "nfev",
                                          "seconds")}, default=float), flush=True)
    print("pulls", {k: round(v, 2) for k, v in sorted(r.pulls.items(), key=lambda kv: -abs(kv[1]))}, flush=True)


if __name__ == "__main__":
    main()

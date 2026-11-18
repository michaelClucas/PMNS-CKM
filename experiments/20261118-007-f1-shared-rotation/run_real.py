"""F1 search on the real data (PDG 2026 CKM central values, NuFIT 6.1 default analysis).

1. Configuration catalog and dedup -> configs.csv, trials counts.
2. The look-elsewhere pipeline (f1lib.search) on real data, identical to run_null.py
   -> real_pipeline.json (table chi^2 NO/IO and Gaussian chi^2 NO/IO for null kind a).
3. Full fit: every distinct config, both orderings, N_STARTS Nelder-Mead starts from the
   best separated grid points, scored by src.data.pmns_delta_chi2 (one joint call per
   evaluation) -> fits_<O>.csv, top30_<O>.csv.
4. Structured-phase variant (alpha, beta in {0, 90, 180, 270, 120, 240} deg, no fit) over
   all raw configs -> struct_top30_<O>.csv.
5. CKM uncertainty: top 10 continuous and top 5 structured per ordering, phases fixed,
   CKM Wolfenstein parameters drawn split-normal (seed CKM_SEED) -> ckm_spread.csv.

    python run_real.py
"""
import csv
import json
import time

import common  # noqa: F401
import numpy as np

from common import HERE, configs
from f1lib import (N_GRID, N_STARTS, GaussChi2, TableChi2, build_T, ckm_central,
                   ckm_samples, grid_eval, grid_starts, phase_grid, phis, polish,
                   predict, search, struct_points, fast_observables)
from src import data, stats
from src.haar import haar_unitary

CKM_SEED = 3003
N_CKM = 4000
OBS = ("s12sq", "s13sq", "s23sq", "dcp_deg")


def log(*a):
    print("[real]", *a, flush=True)


class Official:
    def __init__(self, o):
        self.o = o

    def __call__(self, s12, s13, s23, d):
        return data.pmns_delta_chi2(self.o, s12sq=s12, s13sq=s13, s23sq=s23, dcp_deg=d)


def pulls(o, pred):
    P = data.load_pmns(o)
    out = []
    for k, v in zip(OBS, pred):
        r = v - P[k]["value"]
        if k == "dcp_deg":
            r = (r + 180) % 360 - 180
        out.append(r / (P[k]["err_plus"] if r >= 0 else P[k]["err_minus"]))
    return out


def name(cfg):
    f, x, r, c = cfg
    return f"{f}|{x}|r{r}c{c}"


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    raw, distinct, class_of = configs()
    K = len(distinct)
    rep_of_class = {}
    for i, c in enumerate(class_of):
        rep_of_class.setdefault(c, i)
    write_csv(HERE / "configs.csv", [
        {"raw_id": i, "form": f, "X": x, "row_perm": r, "col_perm": c, "class": class_of[i],
         "representative": int(rep_of_class[class_of[i]] == i)}
        for i, (f, x, r, c) in enumerate(raw)])

    # structured-variant trials: distinct predicted observable tuples (Haar V, seed 0)
    Vh = haar_unitary(np.random.default_rng(0))
    Th = build_T(Vh, raw)
    sp = struct_points()
    U = (Th[:, :2].reshape(len(raw), 6, 3) @ phis(sp).T)
    ob = np.stack(fast_observables(*(U[:, i] for i in range(6))), -1)
    ob[..., 3] = np.mod(np.round(ob[..., 3], 4), 360)
    K_struct = len({tuple(r) for r in np.round(ob.reshape(-1, 4), 6)})
    log(f"raw configs {len(raw)}, distinct (continuous phases) K = {K}, "
        f"log2 K = {np.log2(K):.2f}; structured predictions K_s = {K_struct}, "
        f"log2 = {np.log2(K_struct):.2f}")

    V = ckm_central()
    T, Ts = build_T(V, distinct), build_T(V, raw)

    # 2. identical pipeline as the null
    Pc = {o: data.load_pmns(o) for o in ("NO", "IO")}
    centre = {o: [Pc[o][k]["value"] for k in OBS] for o in Pc}
    scorers = {"table_NO": TableChi2("NO"), "table_IO": TableChi2("IO"),
               "gauss_NO": GaussChi2(centre["NO"], "NO"), "gauss_IO": GaussChi2(centre["IO"], "IO")}
    pipe = search(T, scorers, Ts)
    for k, v in pipe.items():
        v["cfg_cont_name"] = name(distinct[v["cfg_cont"]])
        v["cfg_struct_name"] = name(raw[v["cfg_struct"]])
        log(f"pipeline {k}: best cont chi2 {v['best_cont']:.3f} ({v['cfg_cont_name']}), "
            f"best struct chi2 {v['best_struct']:.3f} ({v['cfg_struct_name']})")
    with open(HERE / "real_pipeline.json", "w") as fh:
        json.dump({"K": K, "K_struct": K_struct, "n_raw": len(raw), "pipeline": pipe}, fh, indent=1)

    # 3. full fit, every distinct config
    ab = phase_grid(N_GRID)
    top_cont = {}
    for o in ("NO", "IO"):
        G = grid_eval(T, {o: TableChi2(o)}, ab)[o]
        off = Official(o)
        rows = []
        for c in range(K):
            chi2, a, b = polish(T[c], off, grid_starts(G[c], ab, N_STARTS))
            pred = predict(T[c], (a, b)) if np.isfinite(chi2) else (np.nan,) * 4
            rows.append((chi2, c, a, b, pred))
        rows.sort(key=lambda r: r[0])
        out = []
        for rank, (chi2, c, a, b, pred) in enumerate(rows, 1):
            p = stats.p_value(chi2, 4, 2) if np.isfinite(chi2) else 0.0
            pl = pulls(o, pred) if np.isfinite(chi2) else [np.nan] * 4
            out.append({"rank": rank, "config": name(distinct[c]), "chi2": round(chi2, 4),
                        "p_dof2": f"{p:.4g}", "p_trials": f"{stats.trials_factor_pvalue(p, K):.4g}",
                        "k_cont": 2, "k_bits": round(np.log2(K), 2),
                        "alpha_deg": round(np.degrees(a), 3), "beta_deg": round(np.degrees(b), 3),
                        **{k: round(v, 5) for k, v in zip(OBS, pred)},
                        **{f"pull_{k}": round(v, 2) for k, v in zip(OBS, pl)}})
        write_csv(HERE / f"fits_{o}.csv", out)
        write_csv(HERE / f"top30_{o}.csv", out[:30])
        top_cont[o] = [(r[1], r[2], r[3], r[0]) for r in rows[:10]]
        log(f"full fit {o}: best chi2 {rows[0][0]:.3f} ({name(distinct[rows[0][1]])}), "
            f"#configs chi2<2: {sum(r[0] < 2 for r in rows)}, chi2<6: {sum(r[0] < 6 for r in rows)}")
        for r in out[:5]:
            log("  ", r)

    # 4. structured phases, all raw configs, exact evaluation (no fit)
    top_struct = {}
    for o in ("NO", "IO"):
        S = grid_eval(Ts, {o: TableChi2(o)}, sp)[o]
        flat = np.argsort(S, axis=None)
        out, seen = [], set()
        for f in flat:
            c, g = np.unravel_index(f, S.shape)
            pred = predict(Ts[c], sp[g])
            key = tuple(np.round(pred, 6))
            if key in seen:  # same prediction from an equivalent raw config
                continue
            seen.add(key)
            chi2 = Official(o)(*pred)
            p = stats.p_value(chi2, 4, 0) if np.isfinite(chi2) else 0.0
            out.append({"rank": len(out) + 1, "config": name(raw[c]), "chi2": round(chi2, 4),
                        "p_dof4": f"{p:.4g}", "p_trials": f"{stats.trials_factor_pvalue(p, K_struct):.4g}",
                        "k_cont": 0, "k_bits": round(np.log2(K_struct), 2),
                        "alpha_deg": round(np.degrees(sp[g][0])), "beta_deg": round(np.degrees(sp[g][1])),
                        **{k: round(v, 5) for k, v in zip(OBS, pred)},
                        **{f"pull_{k}": round(v, 2) for k, v in zip(OBS, pulls(o, pred))}})
            if len(out) == 30:
                break
        write_csv(HERE / f"struct_top30_{o}.csv", out)
        top_struct[o] = [r for r in out[:5]]
        log(f"structured {o}: best chi2 {out[0]['chi2']} ({out[0]['config']}, "
            f"alpha {out[0]['alpha_deg']}, beta {out[0]['beta_deg']})")
        for r in out[:5]:
            log("  ", r)

    # 5. CKM uncertainty propagation, phases fixed
    rng = np.random.default_rng(CKM_SEED)
    Vs = ckm_samples(rng, N_CKM)
    rows = []

    def spread(kind, o, cfg, a, b, chi2_c):
        Tk = np.stack([build_T(v, [cfg])[0] for v in Vs])
        Uk = Tk[:, :2].reshape(N_CKM, 6, 3) @ phis((a, b))[0]
        obs = fast_observables(*(Uk[:, i] for i in range(6)))
        chi = TableChi2(o)(*obs)
        r = {"variant": kind, "ordering": o, "config": name(cfg),
             "alpha_deg": round(np.degrees(a), 3), "beta_deg": round(np.degrees(b), 3),
             "chi2_central": round(chi2_c, 4)}
        for k, v in zip(OBS, obs):
            if k == "dcp_deg":
                v = np.mod(v - np.median(v) + 180, 360) + np.median(v) - 180
            q = np.percentile(v, [16, 50, 84])
            r[f"{k}_med"] = round(q[1], 5)
            r[f"{k}_sd68"] = round((q[2] - q[0]) / 2, 5)
        q = np.percentile(chi, [16, 50, 84])
        r.update(chi2_p16=round(q[0], 3), chi2_med=round(q[1], 3), chi2_p84=round(q[2], 3))
        rows.append(r)

    for o in ("NO", "IO"):
        for c, a, b, chi2 in top_cont[o]:
            spread("continuous", o, distinct[c], a, b, chi2)
        for r in top_struct[o]:
            f, x, rc = r["config"].split("|")
            cfg = (f, x, int(rc[1]), int(rc[3]))
            spread("structured", o, cfg, np.radians(r["alpha_deg"]), np.radians(r["beta_deg"]), r["chi2"])
    write_csv(HERE / "ckm_spread.csv", rows)
    log(f"CKM spread: {N_CKM} samples, seed {CKM_SEED}, written ckm_spread.csv")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

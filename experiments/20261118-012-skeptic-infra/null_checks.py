"""Checks on the exp 007 null design. Imports exp 007's own pipeline (read-only) so that the
only thing changed is the one ingredient under test.

1. CKM removed: run the exp 007 search with V_CKM = identity on the real data (tables and
   Gaussian scorer). If the best chi^2 barely changes, the family does not use the CKM.
2. Null (a) with V = identity, same fake seeds as exp 007 (default_rng([1001, i])), paired
   with exp 007's null_a.csv rows. If the per-fake best chi^2 and the hit rate are the same
   with and without the CKM, null (a) cannot measure a CKM-PMNS link.
3. Gaussian vs NuFIT-table chi^2 at the real-data best points (scorer mismatch in null a).

    .venv/Scripts/python experiments/20261118-012-skeptic-infra/null_checks.py --n 100
Writes null_checks.json and null_a_identity.csv here; log to run.log.
"""
import argparse
import csv
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
EXP7 = HERE.parent / "20261118-007-f1-shared-rotation"
sys.path.insert(0, str(EXP7))
import common  # noqa: E402,F401  (exp 007: threads, sys.path)
import numpy as np  # noqa: E402

from common import configs  # noqa: E402
from f1lib import (GaussChi2, TableChi2, build_T, ckm_central, observables,  # noqa: E402
                   predict, search)
from src import data  # noqa: E402
from src.haar import haar_unitary  # noqa: E402
from src.stats import fake_universe_rate  # noqa: E402

OBS = ("s12sq", "s13sq", "s23sq", "dcp_deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    raw, distinct, _ = configs()
    I3 = np.eye(3, dtype=complex)
    out = {}
    real7 = json.load(open(EXP7 / "real_pipeline.json"))["pipeline"]

    centre = {o: [data.load_pmns(o)[k]["value"] for k in OBS] for o in ("NO", "IO")}
    tab = {o: TableChi2(o) for o in ("NO", "IO")}
    gau = {o: GaussChi2(centre[o], o) for o in ("NO", "IO")}

    # 1. real data, CKM removed
    T_I, Ts_I = build_T(I3, distinct), build_T(I3, raw)
    scorers = {"table_NO": tab["NO"], "table_IO": tab["IO"], "gauss_NO": gau["NO"], "gauss_IO": gau["IO"]}
    rI = search(T_I, scorers, Ts_I)
    out["real_identity"] = {k: {"best_cont": v["best_cont"], "best_struct": v["best_struct"]} for k, v in rI.items()}
    out["real_ckm_exp007"] = {k: {"best_cont": real7[k]["best_cont"], "best_struct": real7[k]["best_struct"]}
                              for k in scorers}
    print("[1] real data, V = I vs V = V_CKM (exp 007):", flush=True)
    for k in scorers:
        print(f"    {k}: cont {rI[k]['best_cont']:.3f} vs {real7[k]['best_cont']:.3f}; "
              f"struct {rI[k]['best_struct']:.3f} vs {real7[k]['best_struct']:.3f}", flush=True)

    # 3. Gaussian vs table at the real best points (V = V_CKM)
    V = ckm_central()
    T = build_T(V, distinct)
    out["gauss_vs_table"] = {}
    for o in ("NO", "IO"):
        for which in ("table", "gauss"):
            r = real7[f"{which}_{o}"]
            pred = predict(T[r["cfg_cont"]], np.array(r["ab_cont"]))
            out["gauss_vs_table"][f"{o}_at_{which}_best"] = {
                "pred": pred, "table_chi2": float(tab[o](*pred)), "gauss_chi2": float(gau[o](*pred))}
    print("[3]", json.dumps(out["gauss_vs_table"], indent=1), flush=True)

    # 2. null (a) with V = I, paired with exp 007 null_a.csv
    rows7 = {int(r["i"]): r for r in csv.DictReader(open(EXP7 / "null_a.csv"))}
    rows = []
    t0 = time.time()
    for i in range(a.n):
        rng = np.random.default_rng([1001, i])
        tgt = [float(v) for v in observables(haar_unitary(rng))]
        r7 = rows7[i]
        assert abs(float(r7["t_s12sq"]) - tgt[0]) < 1e-12, "seed mismatch with exp 007"
        sc = {"NO": GaussChi2(tgt, "NO"), "IO": GaussChi2(tgt, "IO")}
        r = search(T_I, sc, Ts_I)
        row = {"i": i}
        for o in ("NO", "IO"):
            for var in ("cont", "struct"):
                row[f"best_{var}_{o}_I"] = r[o][f"best_{var}"]
                row[f"best_{var}_{o}_ckm"] = float(r7[f"best_{var}_{o}"])
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[2] {i + 1}/{a.n} fakes, {time.time() - t0:.0f} s", flush=True)
    with open(HERE / "null_a_identity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out["null_a_paired"] = {}
    for o in ("NO", "IO"):
        for var in ("cont", "struct"):
            vi = np.array([r[f"best_{var}_{o}_I"] for r in rows])
            vc = np.array([r[f"best_{var}_{o}_ckm"] for r in rows])
            refI = rI[f"gauss_{o}"][f"best_{var}"]
            refC = real7[f"gauss_{o}"][f"best_{var}"]
            hI, hC = int((vi <= refI).sum()), int((vc <= refC).sum())
            out["null_a_paired"][f"{o}_{var}"] = {
                "n": len(rows), "real_ref_I": refI, "real_ref_ckm": refC,
                "hits_I": hI, "hits_ckm": hC,
                "rate_I": fake_universe_rate(hI, len(rows)), "rate_ckm": fake_universe_rate(hC, len(rows)),
                "median_abs_diff_fake_best": float(np.median(np.abs(vi - vc))),
                "corr_log_fake_best": float(np.corrcoef(np.log(vi + 1e-9), np.log(vc + 1e-9))[0, 1])}
            print(f"[2] {o} {var}: hits V=I {hI}/{len(rows)} vs V=CKM {hC}/{len(rows)}; "
                  f"median |diff| fake best {np.median(np.abs(vi - vc)):.3f}", flush=True)
    json.dump(out, open(HERE / "null_checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()

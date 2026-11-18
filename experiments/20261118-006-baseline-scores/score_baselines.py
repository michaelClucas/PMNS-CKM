"""Official chi^2 / p table for every scoreable baseline in src.baselines.BASELINES.

Scores each baseline in both orderings (NO, IO) and both NuFIT 6.1 analyses
(IC23_woSKatm default, IC24_wSKatm), fitting continuous parameters where the relation has
them (src.evaluate.fit_and_evaluate, 32 Sobol multi-starts + Nelder-Mead). No RG running:
predictions are compared to low-scale data as given.

Input sensitivity: for relations that plug measured inputs in at central values
(theta_C, the full V_CKM, m_e, m_mu), the best-fit chi^2 is recomputed with each input
shifted by +-1 sigma (default analysis only). max |Delta chi^2_min| is reported in column
input_dchi2; if it is << 1 the plug-in convention is safe.

Run from repo root:
    .venv/Scripts/python experiments/20261118-006-baseline-scores/score_baselines.py
Writes results.csv and run.log next to this file. Uses 2 worker processes.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.baselines import BASELINES, L1_ID, data_context  # noqa: E402
from src.data import ANALYSES, DEFAULT_ANALYSIS, ORDERINGS  # noqa: E402
from src.evaluate import fit_and_evaluate  # noqa: E402

N_STARTS = 32
SEED = 0


def score(task):
    bid, ordering, analysis = task
    b = BASELINES[bid]
    ctx = data_context()
    t0 = time.time()
    fr = fit_and_evaluate(lambda th: b.model(th, ctx), b.params, ordering, b.predicts,
                          analysis=analysis, inputs=b.inputs, k_discrete=b.k_discrete,
                          periodic=b.periodic, n_starts=N_STARTS, seed=SEED)
    sens = np.nan
    if b.inputs and analysis == DEFAULT_ANALYSIS:
        d = []
        for name in b.inputs:
            for n_sig in (-1.0, 1.0):
                c = data_context({name: n_sig})
                f2 = fit_and_evaluate(lambda th: b.model(th, c), b.params, ordering, b.predicts,
                                      analysis=analysis, inputs=b.inputs, k_discrete=b.k_discrete,
                                      periodic=b.periodic, n_starts=4, seed=SEED,
                                      extra_starts=[fr.params] if fr.params else ())
                d.append(abs(f2.result.chi2 - fr.result.chi2))
        sens = max(d)
    r = fr.result
    row = dict(id=bid, name=b.name, relation=b.relation, catalog=b.catalog, ordering=ordering,
               analysis=analysis, N=r.N, k_c=b.k_continuous, k_d=b.k_discrete, k=r.k, dof=r.dof,
               nuisance=" ".join(b.nuisance), chi2=r.chi2, p=r.p, p_fit=r.p_fit,
               chi2_global=r.chi2_global, p_global=r.p_global, max_abs_pull=r.max_abs_pull,
               input_dchi2=sens, n_starts_at_min=fr.n_distinct_minima(1e-3), n_starts=fr.n_starts,
               params=" ".join(f"{k}={v:.6g}" for k, v in fr.params.items()),
               inputs=" ".join(b.inputs), note=b.note, seconds=round(time.time() - t0, 2))
    for k, v in r.pulls.items():
        row[f"pull_{k}"] = v
    for k, v in r.pred.items():
        row[f"pred_{k}"] = v
    return row


def fmt(x, nd=3):
    if isinstance(x, float):
        if np.isnan(x):
            return "-"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def main():
    tasks = [(bid, o, a) for bid in BASELINES for a in ANALYSES for o in ORDERINGS]
    t0 = time.time()
    with mp.Pool(2) as pool:
        rows = pool.map(score, tasks, chunksize=1)
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    lines = [f"score_baselines.py: {len(tasks)} fits, {time.time() - t0:.0f} s, 2 processes, "
             f"n_starts={N_STARTS}, seed={SEED}, no RG running", ""]
    for a in ANALYSES:
        for o in ORDERINGS:
            lines.append(f"=== {a} {o}")
            lines.append(f"{'id':10s} {'N':>2s} {'k':>2s} {'dof':>3s} {'chi2':>9s} {'p':>9s} {'p_fit':>9s} "
                         f"{'chi2_glob':>9s} {'max|pull|':>9s} {'in_dchi2':>8s} {'starts@min':>10s}  params")
            for r in rows:
                if r["analysis"] != a or r["ordering"] != o:
                    continue
                lines.append(f"{r['id']:10s} {r['N']:2d} {r['k']:2d} {r['dof']:3d} {fmt(r['chi2']):>9s} "
                             f"{fmt(r['p']):>9s} {fmt(r['p_fit']):>9s} {fmt(r['chi2_global']):>9s} "
                             f"{fmt(r['max_abs_pull'], 2):>9s} {fmt(r['input_dchi2']):>8s} "
                             f"{r['n_starts_at_min']:>4d}/{r['n_starts']:<5d}  {r['params']}")
            lines.append("")
    lines.append(f"=== lead L1 = {L1_ID}: {BASELINES[L1_ID].relation}")
    for r in rows:
        if r["id"] == L1_ID:
            preds = " ".join(f"{k[5:]}={r[k]:.4g}" for k in r if k.startswith("pred_"))
            pulls = " ".join(f"{k[5:]}={r[k]:+.2f}" for k in r if k.startswith("pull_"))
            lines.append(f"{r['analysis']} {r['ordering']}: chi2={r['chi2']:.3f} chi2_global={r['chi2_global']:.3f} "
                         f"p_fit(dof {r['N'] - r['k_c']})={r['p_fit']:.4f} | {r['params']} | {preds} | 1D pulls {pulls}")
    text = "\n".join(lines)
    (HERE / "run.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

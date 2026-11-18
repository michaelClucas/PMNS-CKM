"""PySR distillation.

(a) sanity: synthetic TBM + th12e (tier A, continuous th12e, phi) -> s13sq. Known answer
    s13sq = sin(th12e)^2 / 2 (TBC when th12e = theta_C).
(b) emulator-inverted best-fit th12e (TBM std, from nn_emulator.py, NO targets drawn within
    NuFIT errors, CKM inputs drawn within PDG errors) versus PMNS and CKM inputs:
    features s12sq, s13sq, s23sq, dcp_deg, theta_C, theta_cb. A CKM-PMNS link would show
    up as theta_C or theta_cb in the selected formula.
Writes pysr_report.json; PySR files under artifacts/009/pysr.
"""
import os

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "2")
import csv
import json

import numpy as np

import f2lib as L

ART = L.ROOT / "artifacts" / "009" / "pysr"
ART.mkdir(parents=True, exist_ok=True)


def fit(X, y, names, tag, niter=40):
    from pysr import PySRRegressor
    m = PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=["sin", "cos", "sqrt"],
                      niterations=niter, maxsize=15, populations=12, parallelism="multithreading",
                      output_directory=str(ART), run_id=tag, random_state=0, deterministic=False,
                      progress=False, verbosity=0)
    m.fit(X, y, variable_names=names)
    eq = m.equations_
    rows = [dict(complexity=int(r.complexity), loss=float(r.loss), equation=str(r.equation)) for _, r in eq.iterrows()]
    return {"best": str(m.sympy()), "pareto": rows}


def main():
    rng = np.random.default_rng(11)
    out = {}
    # (a)
    c = next(c for c in L.configs() if c["tier"] == "A" and c["nu"] == "TBM" and c["c12"] == 1)
    n = 400
    t12 = rng.uniform(0, 0.5, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    y = []
    for a, p in zip(t12, phi):
        cc = dict(c)
        U = L.full_matrix(cc, (a, 0.0, 0.0, 0.0), p)  # theta_C slot carries th12e (c12 = 1)
        y.append(L.obs_exact(U)[1])
    out["a_s13sq_vs_th12e_phi"] = fit(np.column_stack([t12, phi]), np.array(y), ["th12e", "phi"], "a_s13")
    print("a:", out["a_s13sq_vs_th12e_phi"]["best"], flush=True)
    # (b)
    with open(L.ROOT / "artifacts" / "009" / "inversions.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["ordering"] == "NO"]
    names = ["s12sq", "s13sq", "s23sq", "dcp_deg", "theta_C", "theta_cb"]
    X = np.array([[float(r[k]) for k in names] for r in rows])
    yb = np.array([float(r["th12e"]) for r in rows])
    out["b_th12e_inverted_NO"] = fit(X, yb, names, "b_th12e")
    out["b_th12e_over_thetaC_stats"] = {"mean": float(np.mean(yb / X[:, 4])), "std": float(np.std(yb / X[:, 4]))}
    print("b:", out["b_th12e_inverted_NO"]["best"], flush=True)
    with open(L.HERE / "pysr_report.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

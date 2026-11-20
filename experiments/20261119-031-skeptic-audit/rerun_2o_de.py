"""Exp 031: global (differential evolution) rerun of exp 016's 2O joint fit to our data.

Exp 016 (fit_ours.py, variant "ours") minimized resid_all (6 log mass-ratio pulls + Delta m^2 ratio
+ sqrt(PMNS Scorer.chi2_soft) + sqrt(CKM Scorer.chi2_soft)) with local trf least squares from 16
starts (15 tau-profile points + the published point) and got chi2 68.84 (N 15, k_cont 10).
PMNS and CKM use separate Scorers there, so the exp 026 bug (finite terms dropped outside the
tables) cannot apply: the PMNS-only Scorer has no finite terms and the mass terms are added outside.
This rerun asks whether the local search was the limit: same objective, DE over
  Re tau in [-0.5, 0.5], Im tau in [0.86, 3.0] (exp 016's tau grid range),
  each coupling ratio g = BASE * sign(w) * 10^(3 (|w| - 1)), w in [-2, 2] (either sign, 1e-3..1e3 x BASE),
then a trf polish in the original coordinates, scored with exp 016's own scoring code.

usage: python rerun_2o_de.py <seed> [maxiter] [use_x0 0|1]   (run from anywhere)
Output: artifacts/031/twoO_de_s<seed>.json and a SUMMARY line.
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, least_squares

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "20261118-016-modular-literature"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))
import fit_ours as F  # noqa: E402
from model_2o import observables  # noqa: E402
from src import stats  # noqa: E402
from src.evaluate import evaluate  # noqa: E402

NAMES = F.LEP_N + F.QRK_N
BASE = np.array([F.BASE[n] for n in NAMES])


def to_x(w):
    g = BASE * np.sign(w[2:]) * 10 ** (3 * (np.abs(w[2:]) - 1))
    return np.r_[w[:2], g]


def from_x(x):
    g = x[2:] / BASE
    return np.r_[x[:2], np.sign(g) * (1 + np.log10(np.abs(g)) / 3)]


def score(x, C):
    p = dict(F.BASE, re=x[0], im=x[1], **dict(zip(NAMES, x[2:])))
    o = observables(p)
    pred, mass, _, _ = C.parts(o)
    ev = evaluate(pred, "NO", k=10, use=F.PMNS_USE + F.CKM_USE)
    chi2_mass = float(sum(v ** 2 for v in mass.values()))
    return {"chi2": float(ev.chi2 + chi2_mass), "chi2_eval": float(ev.chi2), "chi2_mass": chi2_mass,
            "p": float(stats.p_value(ev.chi2 + chi2_mass, 15, 10)),
            "pulls": {**{k: float(v) for k, v in ev.pulls.items()}, **{k: float(v) for k, v in mass.items()}},
            "tau": [float(x[0]), float(x[1])], "params": {k: float(v) for k, v in zip(NAMES, x[2:])}}


def main():
    seed = int(sys.argv[1])
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    use_x0 = len(sys.argv) > 3 and sys.argv[3] == "1"
    np.seterr(all="ignore")
    D = F.build_data("ours")
    C = F.Chi2(D)
    committed = json.loads((EXP / "fit_ours_ours.json").read_text())
    xc = np.r_[committed["tau"], [committed["params"][n] for n in NAMES]]

    def f(w):
        try:
            r = F.resid_all(to_x(w), C)
        except Exception:  # noqa: BLE001
            return 1e9
        v = float(np.dot(r, r))
        return v if np.isfinite(v) else 1e9

    lo = np.r_[-0.5, 0.86, -2 * np.ones(8)]
    hi = np.r_[0.5, 3.0, 2 * np.ones(8)]
    t0 = time.time()
    kw = dict(x0=np.clip(from_x(xc), lo, hi)) if use_x0 else {}
    d = differential_evolution(f, list(zip(lo, hi)), seed=seed, popsize=25, maxiter=maxiter, tol=1e-10,
                               init="sobol", polish=False, **kw)
    x = to_x(d.x)
    best = None
    for _ in range(3):
        r = least_squares(F.resid_all, x, args=(C,), method="trf", max_nfev=4000, x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
        x = r.x
    xb = best.x if 2 * best.cost < d.fun else to_x(d.x)
    s = score(xb, C)
    sc = score(xc, C)
    out = {"seed": seed, "maxiter": maxiter, "use_x0": use_x0, "de_fun": float(d.fun), "nfev": int(d.nfev),
           "best": s, "committed_rescored_chi2": sc["chi2"], "committed_json_chi2": committed["chi2"],
           "wall_s": time.time() - t0}
    art = ROOT / "artifacts" / "031"
    art.mkdir(parents=True, exist_ok=True)
    (art / f"twoO_de_s{seed}{'_x0' if use_x0 else ''}.json").write_text(json.dumps(out, indent=1))
    print("SUMMARY", json.dumps({"seed": seed, "use_x0": use_x0, "de_fun": out["de_fun"], "chi2": s["chi2"],
                                 "p": s["p"], "tau": s["tau"], "committed": committed["chi2"],
                                 "wall_s": round(out["wall_s"])}), flush=True)
    print("pulls", {k: round(v, 2) for k, v in s["pulls"].items()}, flush=True)


if __name__ == "__main__":
    main()

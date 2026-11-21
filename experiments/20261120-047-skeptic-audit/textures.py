"""Exp 047 (a): independent re-implementation and refit of exp 044 survivors A6842, A6624, A47977.

Written from the texture definitions (survivors.json spec + grammar.md), without importing lib44.
Observables: own eigen-decomposition and own delta extraction; cross-checked against the official
engine src.yukawa.observables. Scoring: official src.evaluate.Scorer (Q8, NO, 2% floor, scale).
Fit: L-BFGS-B multistart (random log-uniform starts + warm starts at the exp 044 point and
perturbations), Nelder-Mead polish of the best few.

Usage: python textures.py [n_random]
"""
import json
import math
import pathlib
import sys
import time

import warnings

import numpy as np
from scipy.optimize import least_squares, minimize

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate as ev  # noqa: E402
from src.yukawa import observables as official_obs  # noqa: E402

Q8 = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")
SCALES = {"MZ": None,
          "GUT10": {"mu": 2e16, "model": "MSSM", "tan_beta": 10},
          "GUT30": {"mu": 2e16, "model": "MSSM", "tan_beta": 30}}
PI = math.pi

# Texture definitions, transcribed by hand from survivors.json.
# Each entry: (sector, i, j, modulus, phase); modulus = ("p", k) free log10 param k, or a number;
# phase = number or ("p", k) free phase param k. H: Hermitian (diagonal real, sign via phase 0/pi).
# S: complex symmetric. Gauge-fixed entries carry phase 0 (as in the exp 044 canonicalisation).
TEX = {
    # H; u: 11, 13, 23, 33; d: 11, 12, 13, 33. Fixed phases d12 = +pi/2, d13 = 0, signs +.
    "A6842": dict(t="H", scale="MZ", k=6, entries=[
        ("u", 3, 3, 1.0, 0.0), ("u", 2, 3, ("p", 0), 0.0), ("u", 1, 3, ("p", 1), 0.0), ("u", 1, 1, ("p", 2), 0.0),
        ("d", 3, 3, 1.0, 0.0), ("d", 1, 3, ("p", 3), 0.0), ("d", 1, 2, ("p", 4), PI / 2), ("d", 1, 1, ("p", 5), 0.0)]),
    # H; u: 11, 13, 23, 33; d: 12, 13, 33. d13 phase free, d12 = -pi/2, u11 sign +.
    "A6624": dict(t="H", scale="GUT10", k=6, entries=[
        ("u", 3, 3, 1.0, 0.0), ("u", 2, 3, ("p", 0), 0.0), ("u", 1, 3, ("p", 1), 0.0), ("u", 1, 1, ("p", 2), 0.0),
        ("d", 3, 3, 1.0, 0.0), ("d", 1, 3, ("p", 3), ("p", 5)), ("d", 1, 2, ("p", 4), -PI / 2)]),
    # S; u: 11, 13, 23, 33; d: 11, 12, 23, 33. u11 = pi, d23 = -2pi/3, d12 = -pi/2, d11 = pi.
    "A47977": dict(t="S", scale="MZ", k=6, entries=[
        ("u", 3, 3, 1.0, 0.0), ("u", 2, 3, ("p", 0), 0.0), ("u", 1, 3, ("p", 1), 0.0), ("u", 1, 1, ("p", 2), PI),
        ("d", 3, 3, 1.0, 0.0), ("d", 2, 3, ("p", 3), -2 * PI / 3), ("d", 1, 2, ("p", 4), -PI / 2),
        ("d", 1, 1, ("p", 5), PI)]),
}
# Generalisation check: A6842 zero pattern with both d phases free (k 8 = N: dof 0, only checks
# that the (I_f, I_c) zero-pattern class reaches chi^2 ~ 0).
TEX["A6842_freephase"] = dict(t="H", scale="MZ", k=8, entries=[
    ("u", 3, 3, 1.0, 0.0), ("u", 2, 3, ("p", 0), 0.0), ("u", 1, 3, ("p", 1), 0.0), ("u", 1, 1, ("p", 2), 0.0),
    ("d", 3, 3, 1.0, 0.0), ("d", 1, 3, ("p", 3), ("p", 6)), ("d", 1, 2, ("p", 4), ("p", 7)), ("d", 1, 1, ("p", 5), 0.0)])
for _a in TEX.values():
    _a["n_mod"] = sum(1 for e in _a["entries"] if isinstance(e[3], tuple))


def build(tex, x):
    Y = {"u": np.zeros((3, 3), complex), "d": np.zeros((3, 3), complex)}
    for s, i, j, m, ph in tex["entries"]:
        mod = 10.0 ** x[m[1]] if isinstance(m, tuple) else m
        phv = x[ph[1]] if isinstance(ph, tuple) else ph
        v = mod * np.exp(1j * phv)
        i, j = i - 1, j - 1
        if tex["t"] == "H":
            if i == j:
                Y[s][i, i] = v.real
            else:
                Y[s][i, j], Y[s][j, i] = v, np.conj(v)
        else:
            Y[s][i, j] = Y[s][j, i] = v
    return Y["u"], Y["d"]


def my_obs(Yu, Yd):
    """Own observable map: V = U_u^dag U_d from Y Y^dag, PDG angles, delta from
    Q = V_us V_cb V_ub^* V_cs^* = s12 s23 s13 c13^2 (c12 c23 e^{i delta} - s12 s23 s13)."""
    wu, Uu = np.linalg.eigh(Yu @ Yu.conj().T)
    wd, Ud = np.linalg.eigh(Yd @ Yd.conj().T)
    mu, md = np.sqrt(np.abs(wu)), np.sqrt(np.abs(wd))
    V = Uu.conj().T @ Ud
    s13 = abs(V[0, 2])
    c13 = math.sqrt(1 - s13 ** 2)
    s12, s23 = abs(V[0, 1]) / c13, abs(V[1, 2]) / c13
    c12, c23 = math.sqrt(1 - s12 ** 2), math.sqrt(1 - s23 ** 2)
    Q = V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])
    e = (Q / (s12 * s23 * s13 * c13 ** 2) + s12 * s23 * s13) / (c12 * c23)
    d = math.atan2(e.imag, e.real) % (2 * PI)
    return {"ckm_s12": s12, "ckm_s13": s13, "ckm_s23": s23, "ckm_delta_rad": d,
            "mr_u_c": mu[0] / mu[1], "mr_c_t": mu[1] / mu[2], "mr_d_s": md[0] / md[1], "mr_s_b": md[1] / md[2]}


def off_obs(Yu, Yd):
    o = official_obs(Yu, Yd, np.diag([1e-3, 1e-1, 1.0]).astype(complex), np.diag([1e-3, 1e-2, 1.0]).astype(complex),
                     ordering="NO")
    return {k: float(o[k]) for k in Q8}


def box(tex):
    k = tex["k"]
    lo = np.array([-6.0] * tex["n_mod"] + [-2 * PI] * (k - tex["n_mod"]))
    hi = np.array([1.0] * tex["n_mod"] + [2 * PI] * (k - tex["n_mod"]))
    return lo, hi


class Obj:
    def __init__(self, tex, scale):
        self.tex, self.sc = tex, ev.Scorer(Q8, "NO", mass_floor=0.02, scale=SCALES[scale])
        self.n = 0

    def chi2(self, x):
        self.n += 1
        try:
            c = self.sc.chi2(my_obs(*build(self.tex, x)))
        except Exception:
            return 1e6
        return c if np.isfinite(c) else 1e6

    def resid(self, x):
        """Signed residuals whose sum of squares is the official chi^2 (CKM pulls + whitened mass block)."""
        try:
            p = self.sc._prep(my_obs(*build(self.tex, x)))
            chi2, terms, pulls = self.sc._score(p)
        except Exception:
            return np.full(8, 1e3)
        if not np.isfinite(chi2):
            return np.full(8, 1e3)
        r = [pulls[k] for k in Q8[:4]]
        r += list(np.linalg.solve(self.sc._mr_L, np.array([p[k] for k in self.sc.mr_use]) - self.sc._mr_value))
        return np.array(r)

    def full(self, x):
        p = my_obs(*build(self.tex, x))
        r = self.sc(p, k=self.tex["k"])
        return r, p


def fit(tex, scale, warm, n_random, rng):
    ob = Obj(tex, scale)
    lo, hi = box(tex)
    starts = []
    if warm is not None:
        w = np.asarray(warm, float).copy()
        w[tex["n_mod"]:] = (w[tex["n_mod"]:] + PI) % (2 * PI) - PI  # phases into the box interior
        starts.append(("warm", w))
        for _ in range(20):
            starts.append(("perturb", np.clip(w + rng.normal(0, 0.3, len(w)), lo, hi)))
    # random starts: the best n_random of 50 n_random log-uniform box samples (pure random starts in the
    # 7-decade box almost never reach the basin; first trial: best random-start chi2 ~ 1e3-1e4)
    X = rng.uniform(lo, hi, size=(50 * n_random, len(lo)))
    c = np.array([ob.chi2(x) for x in X])
    for i in np.argsort(c)[:n_random]:
        starts.append(("random", X[i]))
    res = []
    for kind, x0 in starts:
        if kind == "random":  # Levenberg-Marquardt-type trust region on the residuals first
            x0 = least_squares(ob.resid, x0, bounds=(lo, hi), method="trf", max_nfev=400).x
        r = minimize(ob.chi2, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options={"maxiter": 500, "eps": 1e-7})
        res.append((float(r.fun), kind, r.x))
    res.sort(key=lambda t: t[0])
    best = res[0]
    for c, kind, x in res[:5]:  # polish
        r = minimize(ob.chi2, x, method="Nelder-Mead", options={"maxfev": 4000, "xatol": 1e-9, "fatol": 1e-10})
        if r.fun < best[0]:
            best = (float(r.fun), kind + "+NM", r.x)
    finals = np.array([c for c, _, _ in res])
    rr, p = ob.full(best[2])
    return {"best_chi2": rr.chi2, "best_kind": best[1], "x": best[2].tolist(), "p": rr.p_global,
            "max_pull": rr.max_abs_pull, "fit_ok": rr.fit_ok, "pulls": rr.pulls, "pred": p,
            "n_starts": len(starts), "n_within_0.01": int((finals < rr.chi2 + 0.01).sum()),
            "n_random_within_0.5": int(sum(1 for c, kd, _ in res if kd == "random" and c < rr.chi2 + 0.5)),
            "best_random_chi2": float(min(c for c, kd, _ in res if kd == "random")),
            "nfev": ob.n}


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    surv = json.loads((ROOT / "experiments/20261120-044-predictive-quarks/survivors.json").read_text())
    rng = np.random.default_rng(4701)
    out = {}
    # 1. own observable map vs official engine at random points
    dev = 0.0
    for name, tex in TEX.items():
        lo, hi = box(tex)
        for _ in range(200):
            x = rng.uniform(np.maximum(lo, -3), np.minimum(hi, 0))
            a, b = my_obs(*build(tex, x)), off_obs(*build(tex, x))
            for k in Q8:
                d = abs(a[k] - b[k])
                if k == "ckm_delta_rad":
                    d = min(d, 2 * PI - d)
                dev = max(dev, d / max(abs(b[k]), 1e-12))
    out["engine_max_rel_dev"] = dev
    print(f"own observables vs src.yukawa: max relative deviation {dev:.2e}", flush=True)
    # 2. claimed points and refits
    for name, scale in (("A6842", "MZ"), ("A6624", "GUT10"), ("A47977", "MZ"), ("A6842_freephase", "MZ")):
        tex = TEX[name]
        sid = name.split("_")[0]
        rec = next((s for s in surv if s["id"] == sid and s["scale"] == scale), None)
        entry = {"scale": scale, "k": tex["k"]}
        warm = None
        if name == "A6842_freephase":
            warm = next(s for s in surv if s["id"] == "A6842")["x"] + [0.0, PI / 2]
        if rec is not None and len(rec["x"]) == tex["k"]:
            warm = rec["x"]
            r, p = Obj(tex, scale).full(np.array(warm))
            entry["claimed"] = {"chi2_044": rec["chi2"], "chi2_recomputed": r.chi2, "max_pull": r.max_abs_pull,
                                "p": r.p_global, "fit_ok": r.fit_ok,
                                "max_pred_dev": max(abs(p[k] - rec["pred"][k]) / abs(rec["pred"][k]) for k in Q8)}
            print(f"{name} {scale}: claimed chi2 {rec['chi2']:.4f}, recomputed independently {r.chi2:.4f}", flush=True)
        t0 = time.time()
        entry["refit"] = fit(tex, scale, warm, n_random, rng)
        f = entry["refit"]
        print(f"{name} {scale}: refit chi2 {f['best_chi2']:.4f} ({f['best_kind']}), fit_ok {f['fit_ok']}, "
              f"p {f['p']:.3f}, max|pull| {f['max_pull']:.2f}, best random-start chi2 {f['best_random_chi2']:.4f}, "
              f"{f['n_random_within_0.5']} random starts within 0.5, {time.time() - t0:.0f}s", flush=True)
        # u-sector branch diagnostics for the 2110.05065 comparison (H textures only)
        if tex["t"] == "H":
            Yu, Yd = build(tex, np.array(f["x"]))
            mu = np.sqrt(np.abs(np.linalg.eigvalsh(Yu @ Yu.conj().T)))
            entry["Yu_abs"] = np.abs(Yu).round(6).tolist()
            entry["Yd_abs"] = np.abs(Yd).round(6).tolist()
            entry["u11_over_mt"] = float(abs(Yu[0, 0]) / mu[2])
            entry["mu_over_mt"] = (mu / mu[2]).tolist()
        out[name] = entry
    (HERE / "textures.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

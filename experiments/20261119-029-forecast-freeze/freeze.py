"""Exp 029: freeze the simple CKM <-> PMNS relations that survive today, with predictions.

Builds frozen_relations.csv (the pre-registration artifact). Forecasts are in forecast.py.
Run from this folder (after symbolic_survivors.py):
    ../../.venv/Scripts/python freeze.py > freeze.log

Families (selection rules in README):
  K  known relations of the src.baselines registry that take CKM inputs, official scorer
     src.evaluate (evaluate_baseline), kept if max |pull| <= 2 in at least one ordering
  S  exp 010 symbolic survivors, C <= 4, >= 1 CKM leaf, |pull| <= 2 in both orderings
     (symbolic_survivors_all.csv), capped at CAP_PER_TARGET per PMNS target
  F  exp 009 F2 LO-tier best configurations (GR2 and TBM with theta12e = theta_C,
     theta23e = theta_cb, std order, phi fitted)
  B  exp 027 bi-large CDSV-I, CDSV-II and RMSV types 1, 2, 4 (type 3 = type 1)
Every measured number comes from data/ through src (src.data, src.baselines.data_context).
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "20261119-027-verified-scout"))
sys.path.insert(0, str(ROOT / "experiments" / "20261118-009-f2-clebsch"))

from src import data  # noqa: E402
from src.baselines import BASELINES, CKM_INPUTS, lo_matrix, obs_from_matrix, standard_matrix  # noqa: E402
from src.evaluate import Scorer, evaluate_baseline, fit_and_evaluate  # noqa: E402
from src.mixing import wolfenstein_to_standard  # noqa: E402
from verify_bilarge import cdsv, rmsv  # noqa: E402  (exp 027 model code, import only)
from f2lib import v_e  # noqa: E402  (exp 009 model code, import only)

PMNS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
ORDS = ("NO", "IO")
TWO_PI = 2 * np.pi
CAP_PER_TARGET = 6
N_CKM_SAMPLES = 2000
SEED = 29
K_BITS_SYMBOLIC_C4 = 15  # ceil(log2 27051) distinct functions at C <= 4 (exp 010 grammar)


# ---------------------------------------------------------------- CKM context and sampling

def ckm_records():
    c = data.load_ckm()
    return c


def ctx_from(std, wolf):
    """Context dict for every model: standard angles, CKM matrix, Wolfenstein values."""
    t12, t13, t23, d = std
    return {"ckm_s12": np.sin(t12), "ckm_s13": np.sin(t13), "ckm_s23": np.sin(t23), "ckm_delta_rad": d,
            "theta_C": t12, "theta_ub": t13, "theta_cb": t23, "delta_q": d,
            "V_ckm": standard_matrix(t12, t13, t23, d), "wolf": dict(wolf)}


def central_ctx():
    c = ckm_records()
    std = (np.arcsin(c["s12"]["value"]), np.arcsin(c["s13"]["value"]), np.arcsin(c["s23"]["value"]),
           c["delta_rad"]["value"])
    wolf = {k: c[k]["value"] for k in ("lambda", "A", "rhobar", "etabar")}
    return ctx_from(std, wolf)


def sample_ctxs(n, seed=SEED, scale_err=None):
    """CKM draws: independent Gaussians on the Wolfenstein parameters (PDG gives no correlation
    matrix), symmetric error = mean of the asymmetric ones. The standard angles are shifted by the
    Wolfenstein -> standard map of the draw minus that of the centre, so the central point equals
    the official standard-parametrization records exactly. scale_err: optional {name: factor}."""
    c = ckm_records()
    names = ("lambda", "A", "rhobar", "etabar")
    mu = np.array([c[k]["value"] for k in names])
    sig = np.array([0.5 * (c[k]["err_plus"] + c[k]["err_minus"]) for k in names])
    if scale_err:
        sig = sig * np.array([scale_err.get(k, 1.0) for k in names])
    rng = np.random.default_rng(seed)
    draws = mu + sig * rng.standard_normal((n, 4))
    std0 = np.array(wolfenstein_to_standard(*mu), dtype=float)
    cen = central_ctx()
    base = np.array([cen["theta_C"], cen["theta_ub"], cen["theta_cb"], cen["delta_q"]])
    out = []
    for d in draws:
        s = base + (np.array(wolfenstein_to_standard(*d), dtype=float) - std0)
        out.append(ctx_from(tuple(s), dict(zip(names, d))))
    return out


# ---------------------------------------------------------------- model wrappers

def pmns_of(U):
    o = obs_from_matrix(U)
    return {k: float(o[k]) for k in PMNS}


def f2_model(pattern):
    def f(th, c):
        Ve = v_e(c["theta_C"], 0.0, c["theta_cb"], c["delta_q"], "std")
        D = np.diag([1.0, np.exp(1j * th["phi"]), np.exp(1j * th["phi"])])
        return pmns_of(Ve.conj().T @ D @ lo_matrix(pattern))
    return f


def cdsv_model(pat):
    def f(th, c):
        return pmns_of(cdsv(c["wolf"], th["phi"], pat))
    return f


def rmsv_model(ut):
    def f(th, c):
        return pmns_of(rmsv(c["wolf"], th["psi"], th["phi"], ut))
    return f


def baseline_model(b):
    def f(th, c):
        return {k: float(v) for k, v in b.model(th, c).items() if k in PMNS}
    return f


# ---------------------------------------------------------------- uncertainty helpers

def circ_std_deg(x):
    a = np.radians(x)
    R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def ckm_spread(model, th, keys, ctxs):
    vals = {k: [] for k in keys}
    for c in ctxs:
        p = model(th, c)
        for k in keys:
            vals[k].append(p[k])
    return {k: (circ_std_deg(np.array(v)) if k == "dcp_deg" else float(np.std(v))) for k, v in vals.items()}


def gauss_sigma(ordering):
    p = data.load_pmns(ordering)
    return {k: 0.5 * (p[k]["err_plus"] + p[k]["err_minus"]) for k in PMNS}


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def param_spread(model, th, bounds, keys, ctx, ordering):
    """Linear propagation of today's PMNS-data uncertainty on the fitted parameters into the
    predictions: Gaussian surrogate chi^2 (NuFIT symmetrized 1 sigma), numerical Hessian at the
    official best fit, C = 2 H^+ (pseudo-inverse: flat directions do not move predictions)."""
    names = list(bounds)
    if not names:
        return {k: 0.0 for k in keys}
    sig = gauss_sigma(ordering)
    best = data.load_pmns(ordering)
    x0 = np.array([th[n] for n in names], dtype=float)

    def pred(x):
        return model(dict(zip(names, x)), ctx)

    def chi2(x):
        p = pred(x)
        r = 0.0
        for k in keys:
            d = p[k] - best[k]["value"]
            if k == "dcp_deg":
                d = wrap_deg(d)
            r += (d / sig[k]) ** 2
        return r

    n = len(names)
    h = 1e-3
    H = np.zeros((n, n))
    f0 = chi2(x0)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (chi2(x0 + ei + ej) - chi2(x0 + ei - ej) - chi2(x0 - ei + ej)
                                 + chi2(x0 - ei - ej)) / (4 * h * h)
    del f0
    # the official best fit sits on the piecewise-linear NuFIT tables, not exactly at the surrogate
    # minimum, so H can have non-positive eigenvalues: keep only the curved (positive) directions
    w, Q = np.linalg.eigh(H)
    keep = w > 1e-8 * max(abs(w).max(), 1e-300)
    C = 2 * (Q[:, keep] / w[keep]) @ Q[:, keep].T
    J = np.zeros((len(keys), n))
    for i in range(n):
        e = np.eye(n)[i] * h
        pp, pm = pred(x0 + e), pred(x0 - e)
        for a, k in enumerate(keys):
            d = pp[k] - pm[k]
            J[a, i] = (wrap_deg(d) if k == "dcp_deg" else d) / (2 * h)
    V = J @ C @ J.T
    return {k: float(np.sqrt(V[a, a])) for a, k in enumerate(keys)}


# ---------------------------------------------------------------- build

def row_base(rid, family, source, formula, inputs, scale, ordering, use, k_cont, k_bits_pub, k_honest, note=""):
    return dict(id=rid, family=family, source=source, formula=formula, inputs=" ".join(inputs), scale=scale,
                ordering=ordering, predicts=" ".join(use), N=len(use), k_cont=k_cont, k_bits_published=k_bits_pub,
                k_honest=k_honest, note=note)


def add_fitted(rows, specs, ctxs):
    cen = central_ctx()
    for sp in specs:
        for o in ORDS:
            if sp.get("baseline") is not None:
                fr = evaluate_baseline(sp["baseline"], o, ctx=cen, n_starts=32, seed=0)
            else:
                fr = fit_and_evaluate(lambda t, m=sp["model"]: m(t, cen), sp["bounds"], o, PMNS,
                                      inputs=sp["inputs"], k_discrete=0, periodic=sp["periodic"],
                                      n_starts=32, seed=0)
            r = fr.result
            th = dict(fr.params)
            pred = sp["model"](th, cen)
            sck = ckm_spread(sp["model"], th, PMNS, ctxs)
            spa = param_spread(sp["model"], th, sp["bounds"], PMNS, cen, o)
            row = row_base(sp["id"], sp["family"], sp["source"], sp["formula"], sp["inputs"], "M_Z", o, PMNS,
                           len(sp["bounds"]), sp["k_bits_pub"], sp["k_honest"], sp.get("note", ""))
            row.update(params_today=json.dumps({k: round(float(v), 6) for k, v in th.items()}),
                       chi2_today=r.chi2, chi2_global_today=r.chi2_global, p_global_today=r.p_global,
                       max_abs_pull_today=r.max_abs_pull, survives_today=bool(r.max_abs_pull <= 2.0))
            for k in PMNS:
                row[f"pred_{k}"] = pred[k]
                row[f"sig_ckm_{k}"] = sck[k]
                row[f"sig_fit_{k}"] = spa[k]
                row[f"pull_today_{k}"] = r.pulls.get(k, np.nan)
            rows.append(row)
            print(f"{sp['id']:14s} {o} chi2 {r.chi2:7.3f} p_glob {r.p_global:.3g} max|pull| {r.max_abs_pull:.2f} "
                  f"pred {[round(pred[k], 5) for k in PMNS]} params {row['params_today']}", flush=True)


def main():
    ctxs = sample_ctxs(N_CKM_SAMPLES)
    rows = []

    # ---- K: registry baselines with CKM inputs; screen with the official scorer
    known_specs, screened = [], []
    cen = central_ctx()
    for bid, b in BASELINES.items():
        if not set(b.inputs) & set(CKM_INPUTS) or not set(b.predicts) <= set(PMNS):
            continue
        res = {o: evaluate_baseline(b, o, ctx=cen, n_starts=32, seed=0).result for o in ORDS}
        mx = {o: res[o].max_abs_pull for o in ORDS}
        screened.append((bid, mx["NO"], mx["IO"]))
        print(f"screen {bid:10s} max|pull| NO {mx['NO']:.2f} IO {mx['IO']:.2f}", flush=True)
        if min(mx.values()) > 2.0:
            continue
        if b.predicts != PMNS:  # partial-observable baselines (A1, A3): none survive today
            raise RuntimeError(f"unexpected partial survivor {bid}")
        known_specs.append(dict(id=f"K-{bid}", family="K", source="exp 003/006 registry (src.baselines)",
                                formula=b.relation, inputs=b.inputs, baseline=b, model=baseline_model(b),
                                bounds=dict(b.params), periodic=b.periodic, k_bits_pub=b.k_discrete,
                                k_honest=b.k, note=b.note))
    with open(HERE / "known_screen.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["baseline", "max_abs_pull_NO", "max_abs_pull_IO"])
        w.writerows(screened)

    # ---- F: exp 009 LO tier best
    f_specs = [dict(id=f"F2LO-{pat}", family="F", source="exp 009 tier_best.csv",
                    formula=f"U = V_e^dag diag(1,e^iphi,e^iphi) U_{pat}, V_e = R23(theta_cb) R12(theta_C) "
                            f"(c12 = c23 = 1, std)",
                    inputs=("ckm_s12", "ckm_s23"), model=f2_model(pat), bounds={"phi": (0.0, TWO_PI)},
                    periodic=("phi",), k_bits_pub=15, k_honest=16,
                    note="k = 1 (phi) + ceil(log2 17220) LO-tier configs") for pat in ("GR2", "TBM")]

    # ---- B: exp 027 bi-large
    b_specs = []
    for pat, desc in (("I", "U_nu: s13 = lam, s12 = 2 lam, s23 = 1 - lam, delta_nu = pi; U_l = R23 Phi R12 Phi*"),
                      ("II", "U_nu: s13 = lam, s12 = 2 lam, s23 = 3 lam, delta_nu = pi; U_l = Phi R12^T Phi* R23^T")):
        b_specs.append(dict(id=f"BL-CDSV-{pat}", family="B", source="exp 027 (arXiv:1902.08962)",
                            formula=f"U = U_l(lam, A lam^2, phi)^dag U_nu; {desc}", inputs=("ckm_lambda", "ckm_A"),
                            model=cdsv_model(pat), bounds={"phi": (0.0, TWO_PI)}, periodic=("phi",),
                            k_bits_pub=1, k_honest=12, note="honest bits: 2048-config Cabibbo-seed grammar"))
    for ut in (1, 2, 4):
        b_specs.append(dict(id=f"BL-RMSV-{ut}", family="B", source="exp 027 (arXiv:1410.3658)",
                            formula=f"U = U_l(type {ut}; lam, A, phi)^dag U_BL, s13 = lam, s12 = s23 = psi lam",
                            inputs=("ckm_lambda", "ckm_A"), model=rmsv_model(ut),
                            bounds={"psi": (1.0, 4.4), "phi": (0.0, TWO_PI)}, periodic=("phi",),
                            k_bits_pub=2, k_honest=7, note="type 3 identical to type 1"))

    add_fitted(rows, known_specs + f_specs + b_specs, ctxs)

    # ---- S: exp 010 symbolic survivors
    sym = list(csv.DictReader(open(HERE / "symbolic_survivors_all.csv")))
    for r in sym:
        r["mx"] = max(abs(float(r["pull_official_NO"])), abs(float(r["pull_official_IO"])))
    n_by_t = {}
    for t in PMNS:
        g = sorted([r for r in sym if r["target"] == t], key=lambda r: (int(r["complexity"]), r["mx"]))
        n_by_t[t] = len(g)
        for i, r in enumerate(g[:CAP_PER_TARGET]):
            rid = f"S-{t.replace('sq', '').replace('_deg', '')}-{i + 1}"
            ckm_in = [x for x in r["leaves"].split() if x in ("t12q", "t13q", "t23q", "dq")]
            inputs = tuple(sorted({{"t12q": "ckm_s12", "t13q": "ckm_s13", "t23q": "ckm_s23",
                                    "dq": "ckm_delta_rad"}[x] for x in ckm_in}))
            other = [x for x in r["leaves"].split() if x not in ckm_in and not x[0].isdigit()
                     and x not in ("pi", "pi/4", "pi/2", "sqrt2", "sqrt3")]
            angle = "delta" if t == "dcp_deg" else "theta" + t[1:3]
            for o in ORDS:
                row = row_base(rid, "S", "exp 010 grammar C<=4 (symbolic_survivors_all.csv)",
                               f"{angle} = {r['expression']} (rad, M_Z)", inputs + tuple(other), "M_Z", o, (t,),
                               0, K_BITS_SYMBOLIC_C4, K_BITS_SYMBOLIC_C4,
                               f"complexity {r['complexity']}; uses mass ratio: {r['uses_mass_ratio']}")
                pull = float(r[f"pull_official_{o}"])
                sc = Scorer((t,), o, inputs=inputs)
                er = sc({t: float(r["pred"])})
                row.update(params_today="{}", chi2_today=er.chi2, chi2_global_today=er.chi2_global,
                           p_global_today=er.p_global, max_abs_pull_today=abs(pull), survives_today=True)
                for k in PMNS:
                    row[f"pred_{k}"] = float(r["pred"]) if k == t else np.nan
                    row[f"sig_ckm_{k}"] = float(r["sigma_pred"]) if k == t else np.nan
                    row[f"sig_fit_{k}"] = 0.0 if k == t else np.nan
                    row[f"pull_today_{k}"] = pull if k == t else np.nan
                rows.append(row)
    keys = list(rows[0])
    with open(HERE / "frozen_relations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    n_rel = len({r["id"] for r in rows})
    n_surv = {o: sum(1 for r in rows if r["ordering"] == o and r["survives_today"]) for o in ORDS}
    print(f"symbolic survivors per target (before cap): {n_by_t}, total {sum(n_by_t.values())}")
    print(f"frozen relations: {n_rel}; rows {len(rows)}; surviving per ordering {n_surv}")


if __name__ == "__main__":
    main()

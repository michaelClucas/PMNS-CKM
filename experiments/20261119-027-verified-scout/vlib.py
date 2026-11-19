"""Shared helpers for the exp 027 verify_*.py scripts.

Every model is written as a function model(ckm, **params) -> 3x3 complex U_PMNS, where
`ckm` is a dict of Wolfenstein inputs (lam, A, rhobar, etabar) read through src.data.
CKM dependence is measured by central finite differences of the PMNS observables with
respect to each Wolfenstein input at fixed model parameters, in units of the input's
PDG 1-sigma error and of the NuFIT 1-sigma width of the observable, plus the shift from
replacing the real CKM by a random hierarchical CKM (exp 007 N1-style draw).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_ckm  # noqa: E402
from src.mixing import extract_angles, wolfenstein_matrix, wolfenstein_to_standard  # noqa: E402
from src.baselines import r12, r13, r23, lo_matrix, LO_TH12  # noqa: E402,F401

WOLF = ("lambda", "A", "rhobar", "etabar")
PMNS_KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
# rough NuFIT 6.1 NO 1-sigma widths, only to express derivatives in sigma units (not used for scoring)
SIG = {"s12sq": 0.012, "s13sq": 0.00056, "s23sq": 0.02, "dcp_deg": 30.0}


def ckm_central():
    c = load_ckm()
    return {k: c[k]["value"] for k in WOLF}


def ckm_errors():
    c = load_ckm()
    return {k: 0.5 * (c[k]["err_plus"] + c[k]["err_minus"]) for k in WOLF}


def V_of(ckm):
    """Exact CKM in standard-form phases."""
    return wolfenstein_matrix(ckm["lambda"], ckm["A"], ckm["rhobar"], ckm["etabar"])


def std_angles(ckm):
    t12, t13, t23, d = wolfenstein_to_standard(ckm["lambda"], ckm["A"], ckm["rhobar"], ckm["etabar"])
    return dict(t12=float(t12), t13=float(t13), t23=float(t23), delta=float(d))


def P(*ph):
    return np.diag(np.exp(1j * np.asarray(ph, dtype=float)))


def U13d(t, d):
    """R13 with the PDG phase: (1,3) entry s e^{-i d}."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s * np.exp(-1j * d)], [0, 1, 0], [-s * np.exp(1j * d), 0, c]], dtype=complex)


def std_U(t12, t13, t23, d):
    return r23(t23) @ U13d(t13, d) @ r12(t12)


def obs(U):
    t12, t13, t23, d = (float(x) for x in extract_angles(np.asarray(U, dtype=complex)))
    return {"s12sq": np.sin(t12) ** 2, "s13sq": np.sin(t13) ** 2, "s23sq": np.sin(t23) ** 2,
            "dcp_deg": np.degrees(d)}


def _dwrap(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def ckm_dependence(model, params, n_rand=200, seed=27):
    """Derivatives d obs / d(Wolfenstein) in (obs sigma) per (input sigma), and the spread of
    each observable when the CKM is replaced by a random hierarchical one
    (lambda U[0.05, 0.5], A U[0.3, 2], rhobar, etabar U[-1, 1]; STATE N1 ranges)."""
    c0, e = ckm_central(), ckm_errors()
    o0 = obs(model(c0, **params))
    der = {}
    for k in WOLF:
        h = e[k]
        cp, cm = dict(c0), dict(c0)
        cp[k] += h
        cm[k] -= h
        op, om = obs(model(cp, **params)), obs(model(cm, **params))
        der[k] = {q: (_dwrap(op[q], om[q]) if q == "dcp_deg" else op[q] - om[q]) / 2 / SIG[q] for q in PMNS_KEYS}
    rng = np.random.default_rng(seed)
    shifts = {q: [] for q in PMNS_KEYS}
    for _ in range(n_rand):
        cr = {"lambda": rng.uniform(0.05, 0.5), "A": rng.uniform(0.3, 2.0),
              "rhobar": rng.uniform(-1, 1), "etabar": rng.uniform(-1, 1)}
        try:
            orr = obs(model(cr, **params))
        except Exception:  # noqa: BLE001
            continue
        for q in PMNS_KEYS:
            d = _dwrap(orr[q], o0[q]) if q == "dcp_deg" else orr[q] - o0[q]
            shifts[q].append(abs(d) / SIG[q])
    rand = {q: float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan") for q, v in shifts.items()}
    rand["frac_invalid"] = float(np.mean(~np.isfinite(shifts["s13sq"]))) if shifts["s13sq"] else 1.0
    # also a "CKM -> 1" limit (lambda -> 0): what PMNS is without quark mixing
    c1 = {"lambda": 1e-6, "A": c0["A"], "rhobar": c0["rhobar"], "etabar": c0["etabar"]}
    try:
        o1 = obs(model(c1, **params))
    except Exception as exc:  # noqa: BLE001  (some models are undefined without quark mixing)
        o1 = {"undefined": str(exc)[:80]}
    return {"obs": o0, "d_per_sigma": der, "median_shift_random_ckm_sigma": rand, "obs_ckm_to_1": o1}


def report(name, model, params, extra=None):
    res = ckm_dependence(model, params)
    res["model"] = name
    res["params"] = params
    if extra:
        res.update(extra)
    o, d = res["obs"], res["d_per_sigma"]
    print(f"== {name}  params={params}")
    print("   obs:", {k: round(v, 5) for k, v in o.items()})
    print("   CKM->1 limit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res["obs_ckm_to_1"].items()})
    for k in WOLF:
        print(f"   d obs/d {k:7s} (sigma_obs per sigma_in):", {q: f"{v:+.3g}" for q, v in d[k].items()})
    print("   median |shift| for random hierarchical CKM (sigma_obs):",
          {q: round(v, 2) for q, v in res["median_shift_random_ckm_sigma"].items()})
    return res


def dump(path, obj):
    def conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    pathlib.Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def same_up_to_rephasing(U1, U2, tol=1e-10):
    """True if |U1_ij| = |U2_ij| and J agree (same physical mixing)."""
    from src.mixing import jarlskog
    return bool(np.allclose(np.abs(U1), np.abs(U2), atol=tol) and abs(jarlskog(U1) - jarlskog(U2)) < tol)

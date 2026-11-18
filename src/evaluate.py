"""The program's single scoring path for candidate relations.

    evaluate(pred, ordering, analysis="IC23_woSKatm", k=0, use=None, inputs=(), k_discrete=0)
    fit_and_evaluate(model_fn, param_bounds, ordering, use, ...)
    evaluate_baseline(baseline, ordering, analysis, ...)

Observable keys in `pred`
-------------------------
  PMNS  s12sq, s13sq, s23sq, dcp_deg      scored with NuFIT 6.1 Delta chi^2 tables
  CKM   ckm_<name>, <name> any record of src.data.load_ckm() (ckm_s12, ckm_Vus, ckm_J, ...)
  mass  any record of src.data.load_masses() (m_tau, m_d, m_c_over_m_s, ...)
CKM and mass terms are Gaussian with asymmetric errors (no correlations: PDG publishes none).

PMNS term
---------
All predicted PMNS parameters go into ONE call of src.data.pmns_delta_chi2 (stats.Objective
with pmns_mode="joint"). When s23sq and dcp_deg are both predicted, the 2D (s23sq, delta)
table is used; the remaining parameters add their 1D profiles. Other PMNS correlations
(s13-s23, s13-delta) are ignored, as documented in src.data.

The per-observable PMNS "pulls" are sign * sqrt(1D Delta chi^2) of each parameter alone.
They are informational (max |pull|, the 2 sigma criterion); the chi^2 is the joint value,
so sum(pull^2) != chi^2 when s23sq and dcp_deg are both predicted.

Ordering offset (IO vs NO)
--------------------------
  chi2         "within-ordering": Delta chi^2 relative to the best fit of the requested
               ordering. This is the fit quality of the relation given that ordering.
  chi2_global  chi2 + ordering_offset(ordering, analysis), added exactly once, and only
               when at least one PMNS parameter is predicted (otherwise the ordering is
               irrelevant and chi2_global = chi2). Offset = 0 for NO, 1.486 (IC23_woSKatm)
               or 5.908 (IC24_wSKatm) for IO. Use it to compare NO and IO results.
  Both get a p-value with the same dof = N - k. The ordering is not counted as an extra
  observable. If a result is quoted for the better of the two orderings chosen after
  looking at the scores, that choice is one more k_discrete; the caller adds it.

N and inputs
------------
  N = number of observables scored (the `use` set). Each one is a prediction compared
  to data.
  `inputs` = measured numbers the relation consumed at their central values (e.g.
  theta_C for QLC). They are not scored and not counted in N, and an observable cannot
  be both an input and a prediction (ValueError). This is only honest when the input
  error is small next to the predicted observable's error; otherwise fit the input as a
  nuisance parameter: make it a fitted parameter AND a scored observable (N + 1, k + 1,
  dof unchanged), which propagates its error and uses the data's own chi^2 for it.
  k = total fitted free parameters including data-selected discrete choices
  (CLAUDE.md). k_discrete = how many of the k are discrete; p_fit uses only the
  continuous part (dof = N - k + k_discrete) and is reported for information. The
  breakthrough criterion uses p (dof = N - k).
  dof <= 0 gives p = nan.

p-values use stats.p_value (asymptotic chi^2; see its caveat on PMNS non-Gaussianity).
No RG running is applied anywhere in this module: predictions are compared to low-scale
data as given.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, asdict
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from . import data as _data
from .stats import Objective, p_value

PMNS_KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
CHI2_CAP = 1e4  # floor added to a Gaussian fallback when a point is outside the NuFIT tables (minimizer only)


# ---------------------------------------------------------------- registry of observables


@functools.lru_cache(maxsize=None)
def _gauss_records():
    """name -> (group, record) for every Gaussian observable."""
    out = {}
    for name, r in _data.load_ckm().items():
        if isinstance(r, dict) and "value" in r and "err_plus" in r:
            out["ckm_" + name] = ("ckm", r)
    for name, r in _data.load_masses().items():
        if isinstance(r, dict) and "value" in r and "err_plus" in r:
            out[name] = ("masses", r)
    return out


def observable_keys():
    """All observable names evaluate() understands."""
    return PMNS_KEYS + tuple(_gauss_records())


def _check_names(names, what):
    known = set(observable_keys())
    bad = [n for n in names if n not in known]
    if bad:
        raise KeyError(f"unknown {what}: {bad}. Known: PMNS {PMNS_KEYS}, ckm_<load_ckm name>, "
                       f"<load_masses name>")


def pmns_best(ordering, analysis=_data.DEFAULT_ANALYSIS):
    """NuFIT best-fit point {s12sq, s13sq, s23sq, dcp_deg} for one ordering."""
    p = _data.load_pmns(ordering, analysis)
    return {k: p[k]["value"] for k in PMNS_KEYS}


def _wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- result types


@dataclass
class EvalResult:
    chi2: float
    chi2_global: float
    ordering_offset: float
    N: int
    k: int
    k_discrete: int
    dof: int
    p: float
    p_global: float
    p_fit: float
    max_abs_pull: float
    pulls: dict
    terms: dict
    ordering: str
    analysis: str
    use: tuple
    inputs: tuple
    pred: dict = field(default_factory=dict)

    def row(self):
        """Flat dict for CSV output."""
        d = {k: v for k, v in asdict(self).items() if k not in ("pulls", "terms", "pred", "use", "inputs")}
        d["use"] = " ".join(self.use)
        d["inputs"] = " ".join(self.inputs)
        for name, v in self.pulls.items():
            d[f"pull_{name}"] = v
        for name, v in self.pred.items():
            d[f"pred_{name}"] = v
        return d


@dataclass
class FitResult:
    params: dict
    result: EvalResult
    start_chi2: list
    n_starts: int
    nfev: int

    def n_distinct_minima(self, tol=1e-3):
        """Number of starts whose final chi^2 lies within tol of the best (convergence check)."""
        best = min(self.start_chi2)
        return sum(abs(c - best) < tol for c in self.start_chi2)


# ---------------------------------------------------------------- scorer


class Scorer:
    """Precomputed stats.Objective for a fixed (use, ordering, analysis). Call with pred."""

    def __init__(self, use: Sequence[str], ordering: str, analysis: str = _data.DEFAULT_ANALYSIS):
        if ordering not in _data.ORDERINGS:
            raise ValueError(f"ordering must be one of {_data.ORDERINGS}")
        if analysis not in _data.ANALYSES:
            raise ValueError(f"analysis must be one of {tuple(_data.ANALYSES)}")
        use = tuple(dict.fromkeys(use))
        if not use:
            raise ValueError("nothing to score")
        _check_names(use, "observables")
        self.use, self.ordering, self.analysis = use, ordering, analysis
        g = _gauss_records()
        ckm = {n: g[n][1] for n in use if n in g and g[n][0] == "ckm"}
        masses = {n: g[n][1] for n in use if n in g and g[n][0] == "masses"}
        self.pmns_use = tuple(p for p in PMNS_KEYS if p in use)
        self.best = pmns_best(ordering, analysis)
        pmns_fn = functools.partial(_data.pmns_delta_chi2, analysis=analysis, relative_to="local")
        self.objective = Objective(
            ckm=ckm, masses=masses,
            pmns_delta_chi2=pmns_fn if self.pmns_use else None,
            pmns_ordering=ordering, pmns_params=self.pmns_use or PMNS_KEYS,
            pmns_mode="joint", pmns_best=self.best)
        self.offset = _data.ordering_offset(ordering, analysis) if self.pmns_use else 0.0

    def _prep(self, pred):
        missing = [n for n in self.use if n not in pred]
        if missing:
            raise KeyError(f"pred lacks {missing}")
        p = {n: float(pred[n]) for n in self.use}
        bad = [n for n, v in p.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite predictions {bad}")
        if "dcp_deg" in p:  # put delta within 180 deg of the best fit so the pull sign is right
            p["dcp_deg"] = self.best["dcp_deg"] + _wrap_deg(p["dcp_deg"] - self.best["dcp_deg"])
        return p

    def chi2(self, pred) -> float:
        return self.objective(self._prep(pred)).chi2

    def chi2_soft(self, pred) -> float:
        """chi2, or (outside the NuFIT tables, where chi2 = inf) CHI2_CAP + a Gaussian
        estimate from the NuFIT 1 sigma errors. Keeps the minimizer's landscape sloped
        back toward the tables instead of flat. Never used for reported numbers."""
        p = self._prep(pred)
        c = self.objective(p).chi2
        if np.isfinite(c):
            return c
        recs = _data.load_pmns(self.ordering, self.analysis)
        g = 0.0
        for n in self.pmns_use:
            r = recs[n]
            d = p[n] - r["value"]
            g += (d / (r["err_plus"] if d > 0 else r["err_minus"])) ** 2
        return CHI2_CAP + g

    def __call__(self, pred, k=0, inputs=(), k_discrete=0) -> EvalResult:
        p = self._prep(pred)
        r = self.objective(p)
        N = len(self.use)
        k, k_discrete = int(k), int(k_discrete)
        if not 0 <= k_discrete <= k:
            raise ValueError("need 0 <= k_discrete <= k")
        dof = N - k

        def pv(c2, kk):
            if N - kk <= 0:
                return float("nan")
            return p_value(c2, N, kk) if np.isfinite(c2) else 0.0

        chi2 = float(r.chi2)
        chi2_g = chi2 + self.offset
        return EvalResult(
            chi2=chi2, chi2_global=chi2_g, ordering_offset=self.offset,
            N=N, k=k, k_discrete=k_discrete, dof=dof,
            p=pv(chi2, k), p_global=pv(chi2_g, k), p_fit=pv(chi2, k - k_discrete),
            max_abs_pull=float(r.max_abs_pull()), pulls=dict(r.pulls), terms=dict(r.terms),
            ordering=self.ordering, analysis=self.analysis, use=self.use, inputs=tuple(inputs),
            pred=p)


def evaluate(pred: Mapping, ordering: str, analysis: str = _data.DEFAULT_ANALYSIS, k: int = 0,
             use: Sequence[str] | None = None, inputs: Sequence[str] = (), k_discrete: int = 0) -> EvalResult:
    """Score one prediction dict against data. See the module docstring for conventions.

    pred: observable name -> predicted value (extra unknown keys are ignored when use=None).
    use: observables to score (N = len(use)). Default: every known observable key in pred.
    inputs: measured observables the relation consumed (not scored, must not overlap use).
    k: total free parameters (continuous fitted + data-selected discrete).
    k_discrete: how many of k are discrete (for p_fit only).
    """
    known = set(observable_keys())
    if use is None:
        use = [n for n in pred if n in known]
    inputs = tuple(inputs)
    _check_names(inputs, "inputs")
    overlap = set(use) & set(inputs)
    if overlap:
        raise ValueError(f"{sorted(overlap)} are both inputs and predictions; an input taken from "
                         "data cannot also be scored. Fit it as a nuisance parameter instead.")
    return Scorer(use, ordering, analysis)(pred, k=k, inputs=inputs, k_discrete=k_discrete)


# ---------------------------------------------------------------- fitting


def fit_and_evaluate(model_fn: Callable[[dict], Mapping], param_bounds: Mapping[str, tuple],
                     ordering: str, use: Sequence[str], analysis: str = _data.DEFAULT_ANALYSIS,
                     inputs: Sequence[str] = (), k_discrete: int = 0, periodic: Sequence[str] = (),
                     n_starts: int = 32, seed: int = 0, extra_starts: Sequence[Mapping] = (),
                     tol: float = 1e-9) -> FitResult:
    """Minimize the within-ordering chi^2 over continuous parameters, then evaluate.

    model_fn(theta: dict) -> pred dict. param_bounds: name -> (lo, hi).
    periodic: parameter names with period hi - lo (phases). They are optimized unbounded
        and wrapped into [lo, hi) afterwards; others are bounded.
    k = len(param_bounds) + k_discrete. chi2_global differs from chi2 by a constant, so
    the same best fit serves both.
    Multi-start: n_starts points from a scrambled Sobol sequence over the box (seeded),
    plus extra_starts, each run with Nelder-Mead then polished with a second Nelder-Mead
    from the best point.
    """
    from scipy.stats import qmc

    names = list(param_bounds)
    lo = np.array([param_bounds[n][0] for n in names], dtype=float)
    hi = np.array([param_bounds[n][1] for n in names], dtype=float)
    per = np.array([n in periodic for n in names])
    scorer = Scorer(use, ordering, analysis)
    nfev = 0

    def theta_of(x):
        x = np.asarray(x, dtype=float).copy()
        x[per] = lo[per] + np.mod(x[per] - lo[per], hi[per] - lo[per])
        return dict(zip(names, x))

    def f(x):
        nonlocal nfev
        nfev += 1
        try:
            c = scorer.chi2_soft(model_fn(theta_of(x)))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return 10 * CHI2_CAP
        return c if np.isfinite(c) else 10 * CHI2_CAP

    if not names:
        res = scorer(model_fn({}), k=k_discrete, inputs=inputs, k_discrete=k_discrete)
        return FitResult({}, res, [res.chi2], 0, 1)

    bounds = [(None, None) if p else (a, b) for p, a, b in zip(per, lo, hi)]
    m = 1 << max(int(np.ceil(np.log2(max(n_starts, 1)))), 0)
    starts = qmc.Sobol(len(names), scramble=True, seed=seed).random(m)[:n_starts]
    starts = [lo + s * (hi - lo) for s in starts]
    starts += [np.array([s[n] for n in names], dtype=float) for s in extra_starts]
    opts = dict(xatol=1e-10, fatol=tol, maxiter=4000 * len(names), maxfev=4000 * len(names))
    finals, best = [], None
    for x0 in starts:
        r = minimize(f, x0, method="Nelder-Mead", bounds=bounds, options=opts)
        finals.append(float(r.fun))
        if best is None or r.fun < best.fun:
            best = r
    r = minimize(f, best.x, method="Nelder-Mead", bounds=bounds, options=opts)
    xbest = r.x if r.fun <= best.fun else best.x
    theta = theta_of(xbest)
    k = len(names) + int(k_discrete)
    res = scorer(model_fn(theta), k=k, inputs=inputs, k_discrete=k_discrete)
    return FitResult({n: float(v) for n, v in theta.items()}, res, finals, len(starts), nfev)


def evaluate_baseline(b, ordering: str, analysis: str = _data.DEFAULT_ANALYSIS, ctx=None,
                      n_starts: int = 32, seed: int = 0) -> FitResult:
    """Score a src.baselines.Baseline (fits its params if it has any)."""
    from .baselines import data_context

    ctx = data_context() if ctx is None else ctx
    return fit_and_evaluate(lambda th: b.model(th, ctx), b.params, ordering, b.predicts,
                            analysis=analysis, inputs=b.inputs, k_discrete=b.k_discrete,
                            periodic=b.periodic, n_starts=n_starts, seed=seed)

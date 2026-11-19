"""Statistics: chi^2 with asymmetric errors, combined objective, p-values, trials factors.

Data entries follow the src.data format: name -> {value, err_plus, err_minus, ...}.
err_minus may be stored with either sign; its absolute value is used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from scipy import stats as _st

__all__ = [
    "asym_pull",
    "chi2_asym",
    "chi2_cov",
    "Objective",
    "ObjectiveResult",
    "PMNS_PARAMS",
    "p_value",
    "discrete_bits",
    "pvalue_to_sigma",
    "sigma_to_pvalue",
    "trials_factor_pvalue",
    "clopper_pearson",
    "fake_universe_rate",
]

PMNS_PARAMS = ("s12sq", "s13sq", "s23sq", "dcp_deg")


# ------------------------------------------------------------------ chi^2


def asym_pull(pred, value, err_plus, err_minus):
    """Pull (pred - value) / sigma with sigma = err_plus if pred > value else err_minus.

    Broadcasts. Positive pull means the prediction is above the measurement.
    """
    pred, value = np.asarray(pred, dtype=float), np.asarray(value, dtype=float)
    ep, em = np.abs(np.asarray(err_plus, dtype=float)), np.abs(np.asarray(err_minus, dtype=float))
    diff = pred - value
    return diff / np.where(diff > 0, ep, em)


def chi2_asym(pred, value, err_plus, err_minus, axis=-1):
    """Sum of squared asymmetric pulls along `axis`. Returns (chi2, pulls)."""
    pulls = asym_pull(pred, value, err_plus, err_minus)
    return np.sum(pulls**2, axis=axis), pulls


def chi2_cov(pred, value, cov):
    """Gaussian chi^2 = r^T C^{-1} r with r = pred - value (last axis). Returns (chi2, r).

    Uses a Cholesky solve, so `cov` must be symmetric positive definite.
    """
    r = np.asarray(pred, dtype=float) - np.asarray(value, dtype=float)
    L = np.linalg.cholesky(np.asarray(cov, dtype=float))
    y = np.linalg.solve(L, r[..., None])[..., 0]
    return np.sum(y**2, axis=-1), r


# ---------------------------------------------------------------- objective


@dataclass
class ObjectiveResult:
    chi2: float
    terms: dict = field(default_factory=dict)  # name -> chi^2 contribution
    pulls: dict = field(default_factory=dict)  # name -> signed pull (informational for PMNS)
    n_obs: int = 0

    def p_value(self, k):
        return p_value(self.chi2, self.n_obs, k)

    def max_abs_pull(self):
        return max((abs(v) for v in self.pulls.values()), default=0.0)


class Objective:
    """Total chi^2 = CKM Gaussian chi^2 + PMNS Delta chi^2 + mass Gaussian chi^2.

    Parameters
    ----------
    ckm : dict name -> {value, err_plus, err_minus}, observables to include. Pass only the
        ones you want (e.g. {k: d for k, d in load_ckm().items() if k in wanted}).
    pmns_delta_chi2 : callable(ordering, s12sq=None, s13sq=None, s23sq=None, dcp_deg=None)
        -> float, e.g. src.data.pmns_delta_chi2. None disables the PMNS term.
    pmns_ordering : "NO" or "IO".
    pmns_params : which of PMNS_PARAMS enter the fit (and N).
    pmns_mode : default "sum1d" (a UserWarning is raised when it is left implicit and both
        s23sq and dcp_deg are scored, exp 024). "sum1d" sums the 1D profiled Delta chi^2 of each parameter (ignores
        correlations between PMNS parameters; this is what NuFIT's published 1D tables
        support). "joint" makes one call with all parameters and trusts the callable
        to combine them (use it if the loader provides a joint or 2D-projection chi^2).
    pmns_best : optional dict param -> best-fit value, used only to sign the PMNS pulls.
    masses : dict like `ckm` for mass observables.
    ckm_cov : optional (names, cov). Those CKM names use a correlated Gaussian with this
        covariance instead of independent asymmetric errors.
    pmns_ordering_offset : chi^2 penalty of this ordering relative to the global best
        (src.data.ordering_offset), added once. Pass the callable without relative_to="global".
        The callable gets no `analysis` argument: bind it (functools.partial) if not default.

    Calling the objective with a dict `pred` (observable name -> predicted value, PMNS
    parameters under the PMNS_PARAMS names) returns an ObjectiveResult. A missing
    prediction raises KeyError rather than being silently skipped.

    Pulls: for Gaussian terms, the signed pull. For PMNS, sign * sqrt(Delta chi^2_1D),
    which is a pull only in the sense of Wilks' theorem.
    """

    def __init__(
        self,
        ckm: Mapping | None = None,
        pmns_delta_chi2: Callable | None = None,
        pmns_ordering: str = "NO",
        pmns_params=PMNS_PARAMS,
        pmns_mode: str | None = None,
        pmns_best: Mapping | None = None,
        masses: Mapping | None = None,
        ckm_cov=None,
        pmns_ordering_offset: float = 0.0,
    ):
        if pmns_ordering not in ("NO", "IO"):
            raise ValueError("pmns_ordering must be 'NO' or 'IO'")
        if pmns_mode is None:  # default kept at sum1d for backward compatibility (exp 024)
            pmns_mode = "sum1d"
            if pmns_delta_chi2 is not None and {"s23sq", "dcp_deg"} <= set(pmns_params):
                warnings.warn("Objective: pmns_mode defaults to 'sum1d', which drops the s23sq-delta "
                              "correlation that src.evaluate keeps (exp 012 L2); pass pmns_mode='joint' "
                              "to match the official score, or 'sum1d' explicitly to silence this.",
                              UserWarning, stacklevel=2)
        if pmns_mode not in ("sum1d", "joint"):
            raise ValueError("pmns_mode must be 'sum1d' or 'joint'")
        bad = set(pmns_params) - set(PMNS_PARAMS)
        if bad:
            raise ValueError(f"unknown PMNS params {bad}")
        self.ckm = dict(ckm or {})
        self.masses = dict(masses or {})
        self.pmns = pmns_delta_chi2
        self.ordering = pmns_ordering
        self.pmns_params = tuple(pmns_params) if pmns_delta_chi2 is not None else ()
        self.pmns_mode = pmns_mode
        self.pmns_best = dict(pmns_best or {})
        # IO-vs-NO penalty, added once. The callable must return within-ordering Delta chi^2:
        # a loader wrapped with relative_to="global" would add the offset once per call.
        kw = getattr(pmns_delta_chi2, "keywords", None) or {}
        if pmns_mode == "sum1d" and kw.get("relative_to") == "global":
            raise ValueError("sum1d with relative_to='global' counts the ordering offset per "
                             "parameter; use pmns_ordering_offset instead")
        self.pmns_ordering_offset = float(pmns_ordering_offset)
        if ckm_cov is not None:
            names, cov = ckm_cov
            self.cov_names = list(names)
            self.cov = np.asarray(cov, dtype=float)
            if self.cov.shape != (len(self.cov_names),) * 2:
                raise ValueError("ckm_cov shape does not match names")
            missing = set(self.cov_names) - set(self.ckm)
            if missing:
                raise ValueError(f"ckm_cov names not in ckm: {missing}")
        else:
            self.cov_names, self.cov = [], None

    @property
    def n_obs(self):
        return len(self.ckm) + len(self.masses) + len(self.pmns_params)

    def _gauss(self, data, pred, res, skip=()):
        for name, d in data.items():
            if name in skip:
                continue
            p = float(asym_pull(pred[name], d["value"], d["err_plus"], d["err_minus"]))
            res.pulls[name] = p
            res.terms[name] = p * p

    def __call__(self, pred: Mapping) -> ObjectiveResult:
        res = ObjectiveResult(chi2=0.0, n_obs=self.n_obs)
        self._gauss(self.ckm, pred, res, skip=set(self.cov_names))
        if self.cov is not None:
            pv = np.array([pred[n] for n in self.cov_names], dtype=float)
            mv = np.array([self.ckm[n]["value"] for n in self.cov_names], dtype=float)
            c2, r = chi2_cov(pv, mv, self.cov)
            res.terms["ckm_cov"] = float(c2)
            sig = np.sqrt(np.diag(self.cov))
            for n, ri, si in zip(self.cov_names, r, sig):
                res.pulls[n] = float(ri / si)  # marginal pull, informational
        self._gauss(self.masses, pred, res)
        if self.pmns_params:
            vals = {p: float(pred[p]) for p in self.pmns_params}
            pulls = {}
            for p, v in vals.items():
                d1 = max(float(self.pmns(self.ordering, **{p: v})), 0.0)
                if p in self.pmns_best:
                    diff = v - self.pmns_best[p]
                    if p == "dcp_deg":
                        diff = (diff + 180.0) % 360.0 - 180.0
                    sgn = np.sign(diff)
                else:
                    sgn = 1.0
                pulls[p] = float(sgn * np.sqrt(d1))
                if self.pmns_mode == "sum1d":
                    res.terms[p] = d1
            if self.pmns_mode == "joint":
                res.terms["pmns_joint"] = float(self.pmns(self.ordering, **vals))
            if self.pmns_ordering_offset:
                res.terms["pmns_ordering_offset"] = float(self.pmns_ordering_offset)
            res.pulls.update(pulls)
        res.chi2 = float(sum(res.terms.values()))
        return res


# ---------------------------------------------------------------- p-values


def _as_int(x, name):
    """x as an int; raises ValueError for non-integral values instead of truncating."""
    xf = float(x)
    if not np.isfinite(xf) or xf != np.floor(xf):
        raise ValueError(f"{name} = {x!r} is not an integer. Parameter counts are integers: "
                         "a discrete choice among n options costs discrete_bits(n) = ceil(log2 n).")
    return int(xf)


def discrete_bits(n_options):
    """Parameter cost of one data-selected choice among n_options: ceil(log2 n_options).

    The program's single convention for discrete choices (exp 012 M3). 1 option costs 0,
    2 options 1, 15 options 4. Several independent choices: pass the product of their
    option counts (log2 of a product is the sum, then ceil once).
    """
    n = _as_int(n_options, "n_options")
    if n < 1:
        raise ValueError("n_options must be >= 1")
    return int(np.ceil(np.log2(n) - 1e-12)) if n > 1 else 0


def p_value(chi2, N, k):
    """Upper-tail chi^2 p-value with dof = N - k.

    N: number of measured observables in the fit. k: the parameters that reduce the
    chi^2 degrees of freedom, i.e. the CONTINUOUS fitted parameters. Data-selected discrete
    choices are not a chi^2 dof: they count toward compression (criterion 2) as
    discrete_bits(n_options) and are paid for in the fake-universe null (criterion 4),
    which reruns the same choice (exp 012 M3). src.evaluate passes k = k_continuous.

    N and k must be integers (integral floats such as 2.0 are accepted). A non-integer
    k raises ValueError; it is never truncated. Raises ValueError if dof <= 0.

    Caveat: the PMNS contribution is a profiled Delta chi^2 from NuFIT tables. Those are
    not exactly chi^2-distributed (physical boundaries, theta23 octant degeneracy,
    periodic delta_CP, non-Gaussian tails), so this p-value is approximate. NuFIT's own
    Monte Carlo studies show the deviation matters most for delta_CP and the ordering.
    Where it matters, calibrate with pseudo-experiments.
    """
    dof = _as_int(N, "N") - _as_int(k, "k")
    if dof <= 0:
        raise ValueError(f"dof = N - k = {dof} <= 0; p-value undefined")
    return float(_st.chi2.sf(chi2, dof))


def pvalue_to_sigma(p):
    """Two-sided Gaussian-equivalent significance of a p-value."""
    return _st.norm.isf(np.asarray(p, dtype=float) / 2)


def sigma_to_pvalue(z):
    """Two-sided p-value for a z-sigma deviation."""
    return 2 * _st.norm.sf(np.abs(np.asarray(z, dtype=float)))


def trials_factor_pvalue(p_local, n_trials):
    """Sidak global p-value 1 - (1 - p_local)^n_trials (independent trials assumed).

    Computed as -expm1(n log1p(-p)) to stay accurate for tiny p_local. For correlated
    trials this is conservative; the fake-universe rate is the real calibration.
    """
    p = np.asarray(p_local, dtype=float)
    n = np.asarray(n_trials, dtype=float)
    if np.any((p < 0) | (p > 1)) or np.any(n < 1):
        raise ValueError("need 0 <= p_local <= 1 and n_trials >= 1")
    with np.errstate(divide="ignore"):  # p = 1 -> log1p(-1) = -inf -> result 1
        return -np.expm1(n * np.log1p(-p))


def clopper_pearson(k, n, cl=0.95):
    """Exact (Clopper-Pearson) two-sided binomial confidence interval for k successes in n."""
    if not 0 <= k <= n or n < 1:
        raise ValueError("need 0 <= k <= n, n >= 1")
    a = 1 - cl
    lo = 0.0 if k == 0 else float(_st.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def fake_universe_rate(n_hits, n_trials, cl=0.95):
    """Rate of fake universes in which the search found an equally good relation.

    Returns dict(rate, lo, hi, upper) where (lo, hi) is the two-sided Clopper-Pearson
    interval at `cl` and `upper` is the one-sided upper limit at `cl`. The breakthrough
    criterion (rate < 1%) should be judged on `upper`: with 0 hits in 100 trials the
    95% one-sided upper limit is ~3%, so ~300 clean fake universes are needed to push
    it below 1%.
    """
    lo, hi = clopper_pearson(n_hits, n_trials, cl)
    upper = 1.0 if n_hits == n_trials else float(_st.beta.ppf(cl, n_hits + 1, n_trials - n_hits))
    return {"rate": n_hits / n_trials, "lo": lo, "hi": hi, "upper": upper}

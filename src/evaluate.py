"""The program's single scoring path for candidate relations.

    evaluate(pred, ordering, analysis="IC23_woSKatm", k=0, use=<required>, inputs=(), k_discrete=0,
             ordering_selected=False)
    evaluate_model(model_fn, ordering, use, ..., leak_check=True)
    fit_and_evaluate(model_fn, param_bounds, ordering, use, ...)
    evaluate_baseline(baseline, ordering, analysis, ...)
    check_observables(use, inputs)            alias / independence rules (raises ValueError)
    alias_atoms(key)                          the measurement atoms a key depends on

ordering is "NO", "IO" or "best". "best" scores both and keeps the lower chi2_global, and
adds 1 to k and k_discrete (the ordering bit, exp 012 M2). Pass ordering_selected=True
instead when you picked the ordering yourself after looking at both.

Observable keys in `pred`
-------------------------
  PMNS  s12sq, s13sq, s23sq, dcp_deg      scored with NuFIT 6.1 Delta chi^2 tables
  CKM   ckm_<name>, <name> any record of src.data.load_ckm() (ckm_s12, ckm_Vus, ckm_J, ...)
  mass  any record of src.data.load_masses() (m_tau, m_d, m_c_over_m_s, ...)
CKM and mass terms are Gaussian with asymmetric errors (no correlations: PDG publishes none).
ckm_delta_rad is periodic (2 pi), like dcp_deg (360 deg).

Alias rules (exp 012 H2, M1), enforced by check_observables on every score
---------------------------------------------------------------------------
Each key maps to a set of measurement atoms (alias_atoms). Two keys that share an atom
are the same measurement, or correlated measurements without a published covariance:
  CKM global fit  ckm:lambda {lambda, s12, Vus, Vcd, Vud, Vcs}, ckm:A {A, s23, Vcb, Vts,
                  Vtb}, ckm:s13 {s13, Vub}, and rhobar, etabar, delta_rad, J, Vtd alone.
  CKM direct      each direct_<V> is its own atom (independent measurements).
  masses          one atom per flavour: m_c_over_m_s = {c, s}; m_t_direct, m_t_pole_xsec,
                  m_t_msbar_xsec = {t}. So m_c with m_c_over_m_s, or two top masses, collide.
  PMNS            one atom per parameter.
Rules (ValueError on violation):
  1. no two keys of `use` share an atom (no N inflation by aliases);
  2. no key of `use` shares an atom with a key of `inputs` (an input cannot be scored
     back as a "prediction" under another name);
  3. CKM global-fit keys and CKM direct keys never appear together in use + inputs (the
     global fit already contains the direct measurements);
  4. the CKM global fit has 4 parameters (lambda, A, rhobar, etabar): at most 4 global
     keys in `use`, and every global key in `use` must add an independent direction to
     the inputs' global keys. Tested on the Jacobian d key / d(Wolfenstein, in sigma
     units) with unit-normalised rows: rank needs singular values > 0.05 (aliases sit
     below 0.007, independent sets above 0.2).
`use` must be given explicitly. use=None raises ValueError (exp 012 M6: it used to score
every known key in pred, including echoed inputs).

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
               ordering.
  chi2_global  chi2 + ordering_offset(ordering, analysis), added exactly once, and only
               when at least one PMNS parameter is predicted (otherwise the ordering is
               irrelevant and chi2_global = chi2). Offset = 0 for NO, 1.486 (IC23_woSKatm)
               or 5.908 (IC24_wSKatm) for IO.

Parameter counting and p-values (one convention, exp 012 M3)
------------------------------------------------------------
  N            number of observables scored (len(use)).
  inputs       measured numbers the relation consumed at their central values (not scored,
               not in N). Honest only when the input error is small next to the predicted
               observable's error; otherwise fit the input as a nuisance parameter (fitted
               parameter AND scored observable: N + 1, k + 1, dof unchanged).
  k_discrete   data-selected discrete choices in BITS: a choice among n options costs
               stats.discrete_bits(n) = ceil(log2 n). Must be an integer.
  k            k_continuous + k_discrete (all integers; non-integers raise, never truncate).
  Criterion 2  compression: k <= N - 2  (EvalResult.compression_ok).
  dof          N - k_continuous. Discrete choices are not chi^2 dof; the null pays for them.
  p            chi^2 p-value of chi2 with dof = N - k_continuous (fit quality in this ordering).
  p_fit        same as p (kept for old callers).
  p_global     chi^2 p-value of chi2_global, same dof. This is the criterion-3 p
               (EvalResult.p_crit3): it charges the IO penalty, so picking IO after seeing
               both orderings cannot hide it.
  dof <= 0 gives p = nan.
Before exp 013, p and p_global used dof = N - k (discrete choices subtracted). That had no
statistical meaning and is gone.

p-values use stats.p_value (asymptotic chi^2; see its caveat on PMNS non-Gaussianity).
No RG running is applied anywhere in this module: predictions are compared to low-scale
data as given.

Universes. Scorer accepts optional data overrides (ckm_records, mass_records, pmns_fn,
pmns_centre) so src.nulls can score fake universes with exactly the same code path.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, asdict
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from . import data as _data
from .mixing import jarlskog as _jarlskog, standard_matrix as _standard_matrix
from .mixing import wolfenstein_to_standard as _w2s
from .stats import Objective, _as_int, p_value

PMNS_KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
CHI2_CAP = 1e4  # floor added to a Gaussian fallback when a point is outside the NuFIT tables (minimizer only)
ORDERING_CHOICES = _data.ORDERINGS + ("best",)

WOLFENSTEIN = ("lambda", "A", "rhobar", "etabar")
CKM_MAGNITUDES = ("Vud", "Vus", "Vub", "Vcd", "Vcs", "Vcb", "Vtd", "Vts", "Vtb")
CKM_GLOBAL = WOLFENSTEIN + ("s12", "s13", "s23", "delta_rad", "J") + CKM_MAGNITUDES
_CKM_GLOBAL_ATOM = {
    **{n: "ckm:lambda" for n in ("lambda", "s12", "Vus", "Vcd", "Vud", "Vcs")},
    **{n: "ckm:A" for n in ("A", "s23", "Vcb", "Vts", "Vtb")},
    **{n: "ckm:s13" for n in ("s13", "Vub")},
    **{n: "ckm:" + n for n in ("rhobar", "etabar", "delta_rad", "J", "Vtd")},
}
RANK_TOL = 0.05  # min singular value of unit-normalised Jacobian rows (see module docstring)


class AliasError(ValueError):
    """Two observables are the same measurement, or an input is scored back as a prediction."""


class LeakError(ValueError):
    """A model reads measured data it did not declare as an input (src.leakprobe)."""


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


def _wrap_rad(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- alias groups (H2, M1)

_FLAVOURS = ("u", "d", "s", "c", "b", "t", "e", "mu", "tau")


def _mass_flavours(key):
    """Flavours a mass key depends on: m_c -> {c}, m_c_over_m_s -> {c, s}, m_t_* -> {t}."""
    out = set()
    for part in key.split("_over_"):
        if not part.startswith("m_"):
            return None
        f = part[2:].split("_")[0]
        if f not in _FLAVOURS:
            return None
        out.add(f)
    return out


def ckm_sector(key):
    """'global', 'direct' or None for an observable key."""
    if not key.startswith("ckm_"):
        return None
    name = key[4:]
    if name.startswith("direct_"):
        return "direct"
    return "global" if name in CKM_GLOBAL else None


def alias_atoms(key) -> frozenset:
    """Measurement atoms a key depends on. Keys sharing an atom are aliases (module docstring)."""
    if key in PMNS_KEYS:
        return frozenset({"pmns:" + key})
    sec = ckm_sector(key)
    if sec == "global":
        return frozenset({_CKM_GLOBAL_ATOM[key[4:]]})
    if sec == "direct":
        return frozenset({"ckm_direct:" + key[len("ckm_direct_"):]})
    fl = _mass_flavours(key)
    if fl:
        return frozenset("mass:" + f for f in fl)
    return frozenset({"other:" + key})


def ckm_from_wolfenstein(lam, A, rhobar, etabar) -> dict:
    """Every CKM global-fit key (load_ckm names, no ckm_ prefix) from exact Wolfenstein parameters."""
    t12, t13, t23, d = (float(x) for x in _w2s(lam, A, rhobar, etabar))
    V = _standard_matrix(t12, t13, t23, d)
    out = {"lambda": lam, "A": A, "rhobar": rhobar, "etabar": etabar,
           "s12": np.sin(t12), "s13": np.sin(t13), "s23": np.sin(t23), "delta_rad": d,
           "J": _jarlskog(V)}
    out.update(zip(CKM_MAGNITUDES, np.abs(V).ravel()))
    return {k: float(v) for k, v in out.items()}


@functools.lru_cache(maxsize=None)
def _ckm_jacobian_rows():
    """key -> unit row of d key / d(Wolfenstein params in sigma units) at the PDG central values."""
    c = _data.load_ckm()
    x0 = np.array([c[k]["value"] for k in WOLFENSTEIN])
    sig = np.array([(c[k]["err_plus"] + abs(c[k]["err_minus"])) / 2 for k in WOLFENSTEIN])
    rows = {k: np.zeros(4) for k in CKM_GLOBAL}
    for i in range(4):
        h = np.zeros(4)
        h[i] = 1e-3 * sig[i]
        fp, fm = ckm_from_wolfenstein(*(x0 + h)), ckm_from_wolfenstein(*(x0 - h))
        for k in CKM_GLOBAL:
            rows[k][i] = (fp[k] - fm[k]) / 2e-3
    return {k: r / np.linalg.norm(r) for k, r in rows.items()}


def _numrank(names):
    if not names:
        return 0
    rows = _ckm_jacobian_rows()
    sv = np.linalg.svd(np.array([rows[n] for n in names]), compute_uv=False)
    return int(np.sum(sv > RANK_TOL))


@functools.lru_cache(maxsize=4096)
def _check_observables_cached(use, inputs):
    for a in range(len(use)):
        for b in range(a + 1, len(use)):
            common = alias_atoms(use[a]) & alias_atoms(use[b])
            if common:
                raise AliasError(
                    f"{use[a]} and {use[b]} are the same measurement or correlated measurements "
                    f"without a published covariance (shared {sorted(common)}); score only one of them")
    for u in use:
        for i in inputs:
            common = alias_atoms(u) & alias_atoms(i)
            if common:
                raise AliasError(
                    f"prediction {u} and input {i} share {sorted(common)}: the input would be scored "
                    "back as a prediction. Fit it as a nuisance parameter instead.")
    sectors = {ckm_sector(k) for k in use + inputs} - {None}
    if sectors == {"global", "direct"}:
        raise AliasError("CKM global-fit keys and direct_* keys cannot be mixed (the global fit "
                         "already contains the direct measurements)")
    g_use = [k[4:] for k in use if ckm_sector(k) == "global"]
    g_in = [k[4:] for k in inputs if ckm_sector(k) == "global"]
    if len(g_use) > 4:
        raise AliasError(f"{len(g_use)} CKM global-fit keys scored; the fit has 4 parameters (max 4)")
    if g_use and _numrank(g_use + g_in) != _numrank(g_in) + len(g_use):
        raise AliasError(f"CKM global-fit keys {g_use} (with inputs {g_in}) are not independent "
                         "functions of (lambda, A, rhobar, etabar)")
    return True


def check_observables(use: Sequence[str], inputs: Sequence[str] = ()):
    """Raise AliasError (a ValueError) if use/inputs break an alias rule (module docstring)."""
    _check_observables_cached(tuple(use), tuple(inputs))


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
    k_continuous: int = 0
    ordering_selected: bool = False

    @property
    def p_crit3(self):
        """The criterion-3 p-value: p_global (dof = N - k_continuous)."""
        return self.p_global

    @property
    def compression_ok(self):
        """Criterion 2: k <= N - 2 (k counts discrete choices in bits)."""
        return self.k <= self.N - 2

    @property
    def fit_ok(self):
        """Criterion 3: every pull within 2 sigma and p_global > 0.05."""
        return bool(self.max_abs_pull < 2 and np.isfinite(self.p_global) and self.p_global > 0.05)

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
    """Precomputed stats.Objective for a fixed (use, ordering, analysis). Call with pred.

    inputs: declared inputs; the alias rules are checked against them here.
    Data overrides (keyword-only, used by src.nulls for fake universes):
      ckm_records   name -> record for CKM keys (load_ckm names, no ckm_ prefix)
      mass_records  name -> record for mass keys
      pmns_fn       callable(ordering, **params) -> within-ordering Delta chi^2
      pmns_centre   {s12sq, s13sq, s23sq, dcp_deg}: the data centre (pull signs, delta wrap)
      offset        ordering offset added to chi2_global (default: the real NuFIT one)
    """

    def __init__(self, use: Sequence[str], ordering: str, analysis: str = _data.DEFAULT_ANALYSIS,
                 inputs: Sequence[str] = (), *, ckm_records: Mapping | None = None,
                 mass_records: Mapping | None = None, pmns_fn: Callable | None = None,
                 pmns_centre: Mapping | None = None, offset: float | None = None):
        if ordering not in _data.ORDERINGS:
            raise ValueError(f"ordering must be one of {_data.ORDERINGS} (evaluate and fit_and_evaluate "
                             "also take 'best')")
        if analysis not in _data.ANALYSES:
            raise ValueError(f"analysis must be one of {tuple(_data.ANALYSES)}")
        use = _require_use(use)
        use = tuple(dict.fromkeys(use))
        if not use:
            raise ValueError("nothing to score")
        inputs = tuple(inputs)
        _check_names(use, "observables")
        _check_names(inputs, "inputs")
        check_observables(use, inputs)
        self.use, self.ordering, self.analysis, self.inputs = use, ordering, analysis, inputs
        g = _gauss_records()

        def rec(n):
            grp, r = g[n]
            if grp == "ckm" and ckm_records is not None:
                return ckm_records[n[4:]]
            if grp == "masses" and mass_records is not None:
                return mass_records[n]
            return r

        ckm = {n: rec(n) for n in use if n in g and g[n][0] == "ckm"}
        masses = {n: rec(n) for n in use if n in g and g[n][0] == "masses"}
        self._ckm_delta_centre = ckm["ckm_delta_rad"]["value"] if "ckm_delta_rad" in ckm else None
        self.pmns_use = tuple(p for p in PMNS_KEYS if p in use)
        self.best = dict(pmns_centre) if pmns_centre is not None else pmns_best(ordering, analysis)
        if pmns_fn is None:
            pmns_fn = functools.partial(_data.pmns_delta_chi2, analysis=analysis, relative_to="local")
        self.objective = Objective(
            ckm=ckm, masses=masses,
            pmns_delta_chi2=pmns_fn if self.pmns_use else None,
            pmns_ordering=ordering, pmns_params=self.pmns_use or PMNS_KEYS,
            pmns_mode="joint", pmns_best=self.best)
        if not self.pmns_use:
            self.offset = 0.0
        else:
            self.offset = _data.ordering_offset(ordering, analysis) if offset is None else float(offset)

    def _prep(self, pred):
        missing = [n for n in self.use if n not in pred]
        if missing:
            raise KeyError(f"pred lacks {missing}")
        p = {n: float(pred[n]) for n in self.use}
        bad = [n for n, v in p.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite predictions {bad}")
        if "dcp_deg" in p:  # put delta within 180 deg of the data centre so the pull sign is right
            p["dcp_deg"] = self.best["dcp_deg"] + _wrap_deg(p["dcp_deg"] - self.best["dcp_deg"])
        if self._ckm_delta_centre is not None:  # the CKM phase is periodic too (exp 012 L1)
            c = self._ckm_delta_centre
            p["ckm_delta_rad"] = c + _wrap_rad(p["ckm_delta_rad"] - c)
        return p

    def chi2(self, pred) -> float:
        return self.objective(self._prep(pred)).chi2

    def chi2_soft(self, pred) -> float:
        """chi2, or (outside the NuFIT tables, where chi2 = inf) CHI2_CAP + a Gaussian
        estimate from the NuFIT 1 sigma errors around the data centre. Keeps the
        minimizer's landscape sloped back toward the tables instead of flat. Never used
        for reported numbers."""
        p = self._prep(pred)
        c = self.objective(p).chi2
        if np.isfinite(c):
            return c
        recs = _data.load_pmns(self.ordering, self.analysis)
        g = 0.0
        for n in self.pmns_use:
            r = recs[n]
            d = p[n] - self.best[n]
            g += (d / (r["err_plus"] if d > 0 else r["err_minus"])) ** 2
        return CHI2_CAP + g

    def __call__(self, pred, k=0, inputs=None, k_discrete=0, ordering_selected=False) -> EvalResult:
        inputs = self.inputs if inputs is None else tuple(inputs)
        if inputs != self.inputs:
            _check_names(inputs, "inputs")
            check_observables(self.use, inputs)
        p = self._prep(pred)
        r = self.objective(p)
        N = len(self.use)
        k, k_discrete = _as_int(k, "k"), _as_int(k_discrete, "k_discrete")
        if not 0 <= k_discrete <= k:
            raise ValueError("need 0 <= k_discrete <= k")
        if ordering_selected:
            k, k_discrete = k + 1, k_discrete + 1
        k_cont = k - k_discrete
        dof = N - k_cont

        def pv(c2):
            if dof <= 0:
                return float("nan")
            return p_value(c2, N, k_cont) if np.isfinite(c2) else 0.0

        chi2 = float(r.chi2)
        chi2_g = chi2 + self.offset
        pw = pv(chi2)
        return EvalResult(
            chi2=chi2, chi2_global=chi2_g, ordering_offset=self.offset,
            N=N, k=k, k_discrete=k_discrete, dof=dof,
            p=pw, p_global=pv(chi2_g), p_fit=pw,
            max_abs_pull=float(r.max_abs_pull()), pulls=dict(r.pulls), terms=dict(r.terms),
            ordering=self.ordering, analysis=self.analysis, use=self.use, inputs=inputs,
            pred=p, k_continuous=k_cont, ordering_selected=bool(ordering_selected))


def _check_ordering_arg(ordering):
    if ordering not in ORDERING_CHOICES:
        raise ValueError(f"ordering must be one of {ORDERING_CHOICES}")


def _require_use(use):
    if use is None:
        raise ValueError(
            "use=None is not accepted: pass use=<tuple of the observables this relation predicts> "
            "explicitly (exp 012 M6: the old default scored every known key in pred, including "
            "echoed inputs).")
    if isinstance(use, str):
        raise ValueError("use must be a sequence of observable names, not a single string")
    return tuple(use)


def _scorer_kw(scorer_kwargs, ordering):
    """scorer_kwargs is either one override dict or {"NO": {...}, "IO": {...}}."""
    kw = dict(scorer_kwargs or {})
    if set(kw) <= set(_data.ORDERINGS) and kw:
        return dict(kw.get(ordering) or {})
    return kw


def evaluate(pred: Mapping, ordering: str, analysis: str = _data.DEFAULT_ANALYSIS, k: int = 0,
             use: Sequence[str] | None = None, inputs: Sequence[str] = (), k_discrete: int = 0,
             ordering_selected: bool = False, scorer_kwargs: Mapping | None = None) -> EvalResult:
    """Score one prediction dict against data. See the module docstring for conventions.

    pred: observable name -> predicted value (keys not in `use` are ignored).
    use: observables to score (N = len(use)). Required; None raises ValueError.
    inputs: measured observables the relation consumed (not scored; alias rules apply).
    k: total free parameters (continuous fitted + discrete bits).
    k_discrete: how many of k are discrete bits.
    ordering: "NO", "IO", or "best" (scores both, keeps the lower chi2_global, adds the
        ordering bit to k and k_discrete). With no PMNS key in use the ordering is
        irrelevant and "best" returns the NO result with no extra bit.
    ordering_selected: True if the caller picked this ordering after seeing both (adds 1 bit).
    scorer_kwargs: Scorer data overrides, either one dict or per ordering {"NO": .., "IO": ..}.
    """
    use = _require_use(use)
    _check_ordering_arg(ordering)
    inputs = tuple(inputs)
    if ordering == "best":
        if not any(u in PMNS_KEYS for u in use):
            return evaluate(pred, "NO", analysis, k, use, inputs, k_discrete, ordering_selected,
                            _scorer_kw(scorer_kwargs, "NO"))
        res = {o: Scorer(use, o, analysis, inputs, **_scorer_kw(scorer_kwargs, o))
               for o in _data.ORDERINGS}
        scored = [res[o](pred, k=k, k_discrete=k_discrete) for o in _data.ORDERINGS]
        best = min(scored, key=lambda r: r.chi2_global)
        return res[best.ordering](pred, k=k, k_discrete=k_discrete, ordering_selected=True)
    sc = Scorer(use, ordering, analysis, inputs, **_scorer_kw(scorer_kwargs, ordering))
    return sc(pred, k=k, k_discrete=k_discrete, ordering_selected=ordering_selected)


def check_leaks(model_fn: Callable[[], Mapping], inputs: Sequence[str] = (), **probe_kwargs):
    """Run src.leakprobe.leak_probe; raise LeakError if an undeclared record moves the output."""
    from .leakprobe import leak_probe

    flagged = leak_probe(model_fn, declared_inputs=tuple(inputs), **probe_kwargs)
    if flagged:
        lines = "; ".join(f"{f.record} -> moves {list(f.moved)}" for f in flagged[:8])
        raise LeakError(f"model reads {len(flagged)} undeclared measured record(s): {lines}. "
                        "Declare them in inputs (alias rules then apply) or remove the read.")


def evaluate_model(model_fn: Callable[[], Mapping], ordering: str, use: Sequence[str] | None = None,
                   analysis: str = _data.DEFAULT_ANALYSIS, k: int = 0, inputs: Sequence[str] = (),
                   k_discrete: int = 0, ordering_selected: bool = False, leak_check: bool = True,
                   **probe_kwargs) -> EvalResult:
    """evaluate(model_fn(), ...) after a leak probe of model_fn (exp 012 H3). model_fn() -> pred."""
    use = _require_use(use)
    if leak_check:
        check_leaks(model_fn, inputs, **probe_kwargs)
    return evaluate(model_fn(), ordering, analysis, k=k, use=use, inputs=inputs, k_discrete=k_discrete,
                    ordering_selected=ordering_selected)


# ---------------------------------------------------------------- fitting


def fit_and_evaluate(model_fn: Callable[[dict], Mapping], param_bounds: Mapping[str, tuple],
                     ordering: str, use: Sequence[str], analysis: str = _data.DEFAULT_ANALYSIS,
                     inputs: Sequence[str] = (), k_discrete: int = 0, periodic: Sequence[str] = (),
                     n_starts: int = 32, seed: int = 0, extra_starts: Sequence[Mapping] = (),
                     tol: float = 1e-9, ordering_selected: bool = False, leak_check: bool = False,
                     scorer_kwargs: Mapping | None = None) -> FitResult:
    """Minimize the within-ordering chi^2 over continuous parameters, then evaluate.

    model_fn(theta: dict) -> pred dict. param_bounds: name -> (lo, hi).
    periodic: parameter names with period hi - lo (phases). They are optimized unbounded
        and wrapped into [lo, hi) afterwards; others are bounded.
    k = len(param_bounds) + k_discrete (k_discrete in bits). chi2_global differs from chi2
    by a constant within one ordering, so the same best fit serves both.
    ordering="best": fit both orderings, keep the lower chi2_global, add the ordering bit.
    leak_check: run the leak probe on model_fn at the best-fit parameters (LeakError).
    scorer_kwargs: Scorer data overrides (one dict, or per ordering).
    Multi-start: n_starts points from a scrambled Sobol sequence over the box (seeded),
    plus extra_starts, each run with Nelder-Mead then polished with a second Nelder-Mead
    from the best point.
    """
    from scipy.stats import qmc

    use = _require_use(use)
    _check_ordering_arg(ordering)
    inputs = tuple(inputs)
    if ordering == "best":
        if not any(u in PMNS_KEYS for u in use):
            return fit_and_evaluate(model_fn, param_bounds, "NO", use, analysis, inputs, k_discrete,
                                    periodic, n_starts, seed, extra_starts, tol, ordering_selected,
                                    leak_check, _scorer_kw(scorer_kwargs, "NO"))
        fits = [fit_and_evaluate(model_fn, param_bounds, o, use, analysis, inputs, k_discrete, periodic,
                                 n_starts, seed, extra_starts, tol, False, False,
                                 _scorer_kw(scorer_kwargs, o))
                for o in _data.ORDERINGS]
        fr = min(fits, key=lambda f: f.result.chi2_global)
        o = fr.result.ordering
        sc = Scorer(use, o, analysis, inputs, **_scorer_kw(scorer_kwargs, o))
        if leak_check:
            check_leaks(lambda: model_fn(fr.params), inputs)
        res = sc(model_fn(fr.params), k=fr.result.k, k_discrete=fr.result.k_discrete, ordering_selected=True)
        return FitResult(fr.params, res, fr.start_chi2, fr.n_starts, sum(f.nfev for f in fits))

    names = list(param_bounds)
    lo = np.array([param_bounds[n][0] for n in names], dtype=float)
    hi = np.array([param_bounds[n][1] for n in names], dtype=float)
    per = np.array([n in periodic for n in names])
    scorer = Scorer(use, ordering, analysis, inputs, **_scorer_kw(scorer_kwargs, ordering))
    k_discrete = _as_int(k_discrete, "k_discrete")
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
        if leak_check:
            check_leaks(lambda: model_fn({}), inputs)
        res = scorer(model_fn({}), k=k_discrete, k_discrete=k_discrete, ordering_selected=ordering_selected)
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
    k = len(names) + k_discrete
    if leak_check:
        check_leaks(lambda: model_fn(theta), inputs)
    res = scorer(model_fn(theta), k=k, k_discrete=k_discrete, ordering_selected=ordering_selected)
    return FitResult({n: float(v) for n, v in theta.items()}, res, finals, len(starts), nfev)


def evaluate_baseline(b, ordering: str, analysis: str = _data.DEFAULT_ANALYSIS, ctx=None,
                      n_starts: int = 32, seed: int = 0, ordering_selected: bool = False) -> FitResult:
    """Score a src.baselines.Baseline (fits its params if it has any). ordering may be "best"."""
    from .baselines import data_context

    ctx = data_context() if ctx is None else ctx
    return fit_and_evaluate(lambda th: b.model(th, ctx), b.params, ordering, b.predicts,
                            analysis=analysis, inputs=b.inputs, k_discrete=b.k_discrete,
                            periodic=b.periodic, n_starts=n_starts, seed=seed,
                            ordering_selected=ordering_selected)

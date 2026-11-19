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
                  mr_x_y = {x, y} (same-scale ratios, below).
  PMNS            one atom per parameter; dm21sq_eV2 {dm21}, dm3l_sq_eV2 {dm3l},
                  dm_ratio {dm21, dm3l}.
Rules (ValueError on violation):
  1. no two keys of `use` share an atom (no N inflation by aliases), except two mr_* keys
     (scored together with their covariance);
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
Without `scale` no RG running is applied to predictions: they are compared to low-scale
data as given.

Same-scale mass ratios (exp 019)
--------------------------------
  mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau   m_u/m_c, m_c/m_t, ...
Running-mass ratios at mu = M_Z (SM, MSbar; src.rg.mass_ratios: quarks four-loop QCD to
M_Z, charged leptons one-loop QED pole -> MSbar at M_Z, top via y_t(M_Z)). The seven are
independent in log space. They are scored as ONE correlated Gaussian block,
chi^2 = r^T C^-1 r, where C is the linear propagation of the PDG errors of the nine masses
and alpha_s(M_Z) (src.rg.propagate; the shared m_c, m_s, m_mu, m_b and alpha_s correlate
the ratios). Pulls are marginal, r_i / sqrt(C_ii). Alias atoms: mr_x_y = {mass:x, mass:y},
so a ratio key never combines with a raw mass key or a PDG ratio record of the same
flavour (m_u, m_c_over_m_s, ...), nor with such an input. Ratio keys may share flavours
with each other (the covariance accounts for it).
Under mass_records overrides (src.nulls universes) the central values are recomputed from
the overridden masses and the covariance keeps the real data's relative (log) covariance.

Clebsch double ratio (exp 024)
------------------------------
  R_clebsch     R = (m_e/m_mu) / (m_d/m_s) at M_Z = mr_e_mu / mr_d_s (exp 021). RG-invariant
                at one loop in the SM and threshold-invariant for the first two generations,
                so GUT Clebsch pairs can be screened on it before any fit.
A "derived" key of the mass-ratio block: log R = log mr_e_mu - log mr_d_s
(DERIVED_RATIO_KEYS gives the log coefficients). It joins the same correlated Gaussian block,
its log variance and covariances follow by linear propagation of cov_log, and mass_floor
applies to the R key itself (f^2 added to its log variance, like any mr_* key). Alias atoms:
{mass:e, mass:mu, mass:d, mass:s}. It may be scored with other mr_* keys, but the block's log
coefficient rows must be linearly independent: R with both mr_e_mu and mr_d_s raises
AliasError (that would score the same information twice). Exp 021 quoted R = 0.0938 +- 0.0027
with a 2% floor on each of m_d and m_s (sqrt(2) * 2% on R, PDG errors replaced); this module
adds the floor in quadrature to the PDG-propagated error instead (mass_floor=0.02 gives
+-0.0024; mass_floor={"R_clebsch": 0.02 * sqrt(2)} gives +-0.0031).
Scale translation: multiplicative, factor = factor(mr_e_mu) / factor(mr_d_s).

Neutrino mass splittings (exp 019)
----------------------------------
  dm21sq_eV2    Delta m^2_21 (eV^2), NuFIT 1D profile (DMS table)
  dm3l_sq_eV2   Delta m^2_31 (NO, > 0) or Delta m^2_32 (IO, < 0) (eV^2), DMA profile
  dm_ratio      Delta m^2_21 / |Delta m^2_3l|, profile chi^2(r) = min_x [DMS(r |x|) + DMA(x)]
Each is a 1D profile added to the PMNS term. Correlation approximation: the splittings'
profiles are treated as independent of each other and of the angles (Delta m^2_21 comes
from solar + KamLAND, Delta m^2_3l from atmospheric + long-baseline; the neglected
correlations, e.g. Delta m^2_3l-s23sq, are the same kind as those src.data already
ignores). dm_ratio aliases both splittings. The splittings are never translated in
src.nulls fake universes. Predicting a splitting makes chi2_global carry the ordering offset.

Theory floor (exp 019)
----------------------
mass_floor (Scorer / evaluate / fit_and_evaluate kwarg, default 0): a relative theory error
added in quadrature to every mass observable (load_masses records and mr_* keys):
sigma -> sqrt(sigma^2 + (f |value|)^2). A float applies to all; a mapping gives per-key
values with "default" for the rest. Reported in EvalResult.mass_floor.

Scoring predictions made at a high scale (exp 019)
--------------------------------------------------
scale = {"mu": 2e16, "model": "SM" | "MSSM", "tan_beta": .., "m_susy": 3000.0,
         "m_lightest": 0.0, "majorana": (0, 0), "sys_frac": 0.05}
declares that pred holds values at mu. Each prediction is moved to the data's scale by the
data's own running, computed with src.rg from this universe's central values:
  additive  (s12sq, s13sq, s23sq, dcp_deg, ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad):
            x_low = x_mu - [X(mu) - X(M_Z)]
  multiplicative (ckm_J, mr_*, dm21sq_eV2, dm3l_sq_eV2, dm_ratio):
            x_low = x_mu * X(M_Z) / X(mu)
and is then scored at low scale exactly as above (errors, tables, correlations unchanged).
This equals running the prediction down to first order in (prediction - data); the error
is second order and small when the prediction is within a few sigma. Other keys raise.
Running systematic (EvalResult.scale_systematic; reported, not added to chi^2):
sys_frac * |shift| for additive keys and sys_frac * |ln factor| (relative) for multi-
plicative ones, for the missing two-loop Yukawa / kappa running (default 5% of the running
itself), plus for MSSM half the change of the shift when m_susy is varied by x2 and /2
(proxy for the missing threshold corrections), in quadrature. NOT covered: tan-beta-
enhanced SUSY threshold corrections to y_d, y_s, y_b (10-50% at large tan beta), seesaw
thresholds, and the PMNS running dependence on m_lightest / Majorana phases beyond the
values in `scale`. EvalResult.scale_shift holds the shift or factor per key.

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
from . import rg as _rg
from .mixing import jarlskog as _jarlskog, standard_matrix as _standard_matrix
from .mixing import wolfenstein_to_standard as _w2s
from .stats import Objective, _as_int, p_value

PMNS_KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
DM_KEYS = ("dm21sq_eV2", "dm3l_sq_eV2", "dm_ratio")
MASS_RATIO_KEYS = tuple(_rg.MASS_RATIO_KEYS)
# log-linear combinations of the mr_* ratios scored in the same correlated block (exp 024)
DERIVED_RATIO_KEYS = {"R_clebsch": {"mr_e_mu": 1.0, "mr_d_s": -1.0}}
_MR_FLAVOURS = {k: frozenset(k[3:].split("_")) for k in MASS_RATIO_KEYS}
_MR_FLAVOURS.update({k: frozenset().union(*(_MR_FLAVOURS[b] for b in c)) for k, c in DERIVED_RATIO_KEYS.items()})
MR_BLOCK_KEYS = MASS_RATIO_KEYS + tuple(DERIVED_RATIO_KEYS)
SCALE_ADDITIVE = PMNS_KEYS + ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")
SCALE_MULTIPLICATIVE = ("ckm_J",) + MR_BLOCK_KEYS + DM_KEYS
SCALE_DEFAULTS = {"mu": None, "model": "SM", "tan_beta": None, "m_susy": 3000.0, "m_lightest": 0.0,
                  "majorana": (0.0, 0.0), "sys_frac": 0.05}
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
    return PMNS_KEYS + DM_KEYS + MR_BLOCK_KEYS + tuple(_gauss_records())


def _check_names(names, what):
    known = set(observable_keys())
    bad = [n for n in names if n not in known]
    if bad:
        raise KeyError(f"unknown {what}: {bad}. Known: PMNS {PMNS_KEYS}, {DM_KEYS}, {MR_BLOCK_KEYS}, "
                       f"ckm_<load_ckm name>, <load_masses name>")


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
    if key == "dm21sq_eV2":
        return frozenset({"pmns:dm21"})
    if key == "dm3l_sq_eV2":
        return frozenset({"pmns:dm3l"})
    if key == "dm_ratio":
        return frozenset({"pmns:dm21", "pmns:dm3l"})
    if key in _MR_FLAVOURS:
        return frozenset("mass:" + f for f in _MR_FLAVOURS[key])
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
            if use[a] in _MR_FLAVOURS and use[b] in _MR_FLAVOURS:
                continue  # one correlated block with a full covariance
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
    _check_mr_rank(use)
    return True


def mr_coefficients(key) -> np.ndarray:
    """Log coefficients of a mass-ratio block key over MASS_RATIO_KEYS (unit row for mr_*)."""
    c = DERIVED_RATIO_KEYS.get(key, {key: 1.0})
    return np.array([c.get(m, 0.0) for m in MASS_RATIO_KEYS])


def _check_mr_rank(use):
    blk = [u for u in use if u in MR_BLOCK_KEYS]
    if any(u in DERIVED_RATIO_KEYS for u in blk):
        C = np.array([mr_coefficients(u) for u in blk])
        if np.linalg.matrix_rank(C) < len(blk):
            raise AliasError(f"mass-ratio keys {blk} are not independent in log space (e.g. R_clebsch with "
                             "both mr_e_mu and mr_d_s); score at most two of the three")


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
    mass_floor: object = 0.0          # float or {key: frac, "default": frac}
    scale: dict | None = None         # the high-scale spec the prediction was made at, or None
    scale_shift: dict = field(default_factory=dict)        # key -> additive shift or factor
    scale_systematic: dict = field(default_factory=dict)   # key -> running systematic (module docstring)

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
        d = {k: v for k, v in asdict(self).items()
             if k not in ("pulls", "terms", "pred", "use", "inputs", "scale_shift", "scale_systematic")}
        if isinstance(d["mass_floor"], Mapping):
            d["mass_floor"] = repr(dict(d["mass_floor"]))
        d["scale"] = "" if self.scale is None else repr(self.scale)
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


# ---------------------------------------------------------------- helpers for the exp 019 observables


def _floor_of(mass_floor, key):
    if isinstance(mass_floor, Mapping):
        f = mass_floor.get(key, mass_floor.get("default", 0.0))
    else:
        f = mass_floor
    f = float(f or 0.0)
    if not np.isfinite(f) or f < 0:
        raise ValueError(f"mass_floor must be >= 0, got {f!r} for {key}")
    return f


def _floored(rec, f):
    if not f:
        return rec
    t = f * abs(rec["value"])
    return dict(rec, err_plus=float(np.hypot(rec["err_plus"], t)), err_minus=float(np.hypot(rec["err_minus"], t)))


def _mass_override(mass_records):
    """mass_records -> {name: value} for src.rg (None = real data)."""
    if mass_records is None:
        return None
    return {k: float(r["value"]) for k, r in mass_records.items() if isinstance(r, Mapping) and "value" in r}


@functools.lru_cache(maxsize=1)
def _mr_real():
    """Real-data mass-ratio values, names and log covariance (src.rg.mass_ratios at M_Z)."""
    r = _rg.mass_ratios(errors=True)
    return r["value"], tuple(r["names"]), r["cov_log"]


def mass_ratio_targets(mass_records=None) -> tuple:
    """(values {mr key: value}, names, cov_log) of the mr_* data, real or under overrides."""
    vals, names, cov_log = _mr_real()
    ov = _mass_override(mass_records)
    if ov is not None:
        vals = _rg.mass_ratios(masses=ov, errors=False)["value"]
    return dict(vals), names, cov_log


def mr_block_values(vals) -> dict:
    """{mr_* and derived key: value} from {mr_*: value} (DERIVED_RATIO_KEYS)."""
    out = dict(vals)
    for k, c in DERIVED_RATIO_KEYS.items():
        out[k] = float(np.exp(sum(w * np.log(vals[b]) for b, w in c.items())))
    return out


@functools.lru_cache(maxsize=None)
def _dm_ratio_grid(ordering, analysis):
    x, y = _data.pmns_profile_1d(ordering, "dm3l_sq_eV2", analysis=analysis)  # x in 1e-3 eV^2
    return x, y, np.linspace(x[0], x[-1], 4001)


def dm_ratio_delta_chi2(ordering, r, analysis=_data.DEFAULT_ANALYSIS) -> float:
    """Profile Delta chi^2 of Delta m^2_21 / |Delta m^2_3l| from the two NuFIT 1D profiles
    (treated as independent): min over Delta m^2_3l of DMS(r |dm3l|) + DMA(dm3l).
    The minimum is found on a 4001-point grid over the DMA table, then on a 2001-point grid
    across the best two coarse cells (resolution ~1e-3 of the DMA 1 sigma)."""
    r = float(r)
    if not np.isfinite(r) or r <= 0:
        return np.inf
    x, y, g = _dm_ratio_grid(ordering, analysis)

    def c(xs):
        return _data.pmns_profile_1d(ordering, "dm21sq_eV2", r * np.abs(xs) * 1e-3, analysis=analysis) \
            + np.interp(xs, x, y)

    cg = c(g)
    i = int(np.argmin(cg))
    if not np.isfinite(cg[i]):
        return np.inf
    fine = np.linspace(g[max(i - 1, 0)], g[min(i + 1, len(g) - 1)], 2001)
    return float(min(cg[i], np.min(c(fine))))


def _dm_best(ordering, analysis):
    p = _data.load_pmns(ordering, analysis)
    d21 = p["dm21sq_1e-5eV2"]["value"] * 1e-5
    d3l = p["dm3l_sq_1e-3eV2"]["value"] * 1e-3
    return {"dm21sq_eV2": d21, "dm3l_sq_eV2": d3l, "dm_ratio": d21 / abs(d3l)}


def _dm_sigma(ordering, analysis):
    """Symmetrised 1 sigma of the splitting keys (for the minimizer fallback only)."""
    p = _data.load_pmns(ordering, analysis)
    s21 = 0.5 * (p["dm21sq_1e-5eV2"]["err_plus"] + p["dm21sq_1e-5eV2"]["err_minus"]) * 1e-5
    s3l = 0.5 * (p["dm3l_sq_1e-3eV2"]["err_plus"] + p["dm3l_sq_1e-3eV2"]["err_minus"]) * 1e-3
    b = _dm_best(ordering, analysis)
    rr = b["dm_ratio"] * np.hypot(s21 / b["dm21sq_eV2"], s3l / abs(b["dm3l_sq_eV2"]))
    return {"dm21sq_eV2": s21, "dm3l_sq_eV2": s3l, "dm_ratio": rr}


def _freeze_spec(scale):
    spec = dict(SCALE_DEFAULTS)
    bad = set(scale) - set(spec)
    if bad:
        raise ValueError(f"unknown scale keys {sorted(bad)}; known {sorted(spec)}")
    spec.update(scale)
    if spec["mu"] is None or not float(spec["mu"]) > 0:
        raise ValueError("scale needs mu > 0 (GeV)")
    if spec["model"] not in ("SM", "MSSM"):
        raise ValueError("scale model must be 'SM' or 'MSSM'")
    if spec["model"] == "MSSM" and spec["tan_beta"] is None:
        raise ValueError("MSSM scale needs tan_beta")
    return (float(spec["mu"]), spec["model"], None if spec["tan_beta"] is None else float(spec["tan_beta"]),
            float(spec["m_susy"]), float(spec["m_lightest"]), tuple(float(x) for x in spec["majorana"]),
            float(spec["sys_frac"]))


def _spec_dict(fs):
    return dict(zip(("mu", "model", "tan_beta", "m_susy", "m_lightest", "majorana", "sys_frac"), fs))


def _running(ordering, analysis, mu, model, tb, msusy, ml, maj, fckm, fmass, fcentre):
    """Data's running between M_Z and mu: {key: shift (additive) or factor (multiplicative)}."""
    mz = _rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"]
    masses = dict(fmass) or None
    lo = _rg.ckm_at(mz, ckm=fckm, masses=masses)
    hi = _rg.ckm_at(mu, model, tb, msusy, ckm=fckm, masses=masses)
    c = dict(fcentre)
    pm = (float(np.arcsin(np.sqrt(c["s12sq"]))), float(np.arcsin(np.sqrt(c["s13sq"]))),
          float(np.arcsin(np.sqrt(c["s23sq"]))), float(np.radians(c["dcp_deg"])))
    kw = dict(ordering=ordering, m_lightest=ml, majorana=maj, analysis=analysis, pmns=pm, ckm=fckm, masses=masses)
    plo = _rg.pmns_at(mz, **kw)
    phi = _rg.pmns_at(mu, model=model, tan_beta=tb, m_susy=msusy, **kw)
    out = {}
    for i, k in ((12, "s12"), (13, "s13"), (23, "s23")):
        out["ckm_" + k] = float(np.sin(hi[f"theta{i}"]) - np.sin(lo[f"theta{i}"]))
        out[k + "sq"] = float(np.sin(phi[f"theta{i}"]) ** 2 - np.sin(plo[f"theta{i}"]) ** 2)
    out["ckm_delta_rad"] = float(_wrap_rad(hi["delta"] - lo["delta"]))
    out["dcp_deg"] = float(_wrap_deg(np.degrees(phi["delta"] - plo["delta"])))
    out["ckm_J"] = float(lo["J"] / hi["J"])
    out["dm21sq_eV2"] = float(plo["dm21sq_eV2"] / phi["dm21sq_eV2"])
    out["dm3l_sq_eV2"] = float(plo["dm3l_sq_eV2"] / phi["dm3l_sq_eV2"])
    out["dm_ratio"] = float(plo["dm21_over_dm3l"] / phi["dm21_over_dm3l"])
    rlo = _rg.mass_ratios(None, masses=masses, errors=False)["value"]
    rhi = _rg.mass_ratios(mu, model, tb, msusy, ckm=fckm, masses=masses, errors=False)["value"]
    for k in MASS_RATIO_KEYS:
        out[k] = float(rlo[k] / rhi[k])
    for k, c in DERIVED_RATIO_KEYS.items():
        out[k] = float(np.prod([out[b] ** w for b, w in c.items()]))
    return out


@functools.lru_cache(maxsize=256)
def _scale_table(ordering, analysis, fs, fckm, fmass, fcentre):
    """({key: shift or factor}, {key: systematic}) for a frozen scale spec (module docstring)."""
    mu, model, tb, msusy, ml, maj, sys_frac = fs
    base = _running(ordering, analysis, mu, model, tb, msusy, ml, maj, fckm, fmass, fcentre)
    if model == "MSSM":
        up = _running(ordering, analysis, mu, model, tb, 2 * msusy, ml, maj, fckm, fmass, fcentre)
        dn = _running(ordering, analysis, mu, model, tb, msusy / 2, ml, maj, fckm, fmass, fcentre)
    sys = {}
    for k, v in base.items():
        if k in SCALE_ADDITIVE:
            s = sys_frac * abs(v)
            if model == "MSSM":
                s = float(np.hypot(s, 0.5 * abs(up[k] - dn[k])))
        else:
            s = sys_frac * abs(np.log(v))
            if model == "MSSM":
                s = float(np.hypot(s, 0.5 * abs(np.log(up[k] / dn[k]))))
        sys[k] = float(s)
    return base, sys


# ---------------------------------------------------------------- scorer


class Scorer:
    """Precomputed stats.Objective for a fixed (use, ordering, analysis). Call with pred.

    inputs: declared inputs; the alias rules are checked against them here.
    Data overrides (keyword-only, used by src.nulls for fake universes):
      ckm_records   name -> record for CKM keys (load_ckm names, no ckm_ prefix)
      mass_records  name -> record for mass keys (also sets the mr_* central values)
      pmns_fn       callable(ordering, **params) -> within-ordering Delta chi^2
      pmns_centre   {s12sq, s13sq, s23sq, dcp_deg}: the data centre (pull signs, delta wrap)
      offset        ordering offset added to chi2_global (default: the real NuFIT one)
    Scoring options (keyword-only, exp 019; module docstring):
      mass_floor    relative theory floor on mass observables (float or mapping), default 0
      scale         high-scale spec: pred is given at scale["mu"] and translated down
    """

    def __init__(self, use: Sequence[str], ordering: str, analysis: str = _data.DEFAULT_ANALYSIS,
                 inputs: Sequence[str] = (), *, ckm_records: Mapping | None = None,
                 mass_records: Mapping | None = None, pmns_fn: Callable | None = None,
                 pmns_centre: Mapping | None = None, offset: float | None = None,
                 mass_floor=0.0, scale: Mapping | None = None):
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
        self.mass_floor = dict(mass_floor) if isinstance(mass_floor, Mapping) else float(mass_floor or 0.0)
        g = _gauss_records()

        def rec(n):
            grp, r = g[n]
            if grp == "ckm" and ckm_records is not None:
                return ckm_records[n[4:]]
            if grp == "masses" and mass_records is not None:
                return mass_records[n]
            return r

        ckm = {n: rec(n) for n in use if n in g and g[n][0] == "ckm"}
        masses = {n: _floored(rec(n), _floor_of(self.mass_floor, n)) for n in use if n in g and g[n][0] == "masses"}
        self._ckm_delta_centre = ckm["ckm_delta_rad"]["value"] if "ckm_delta_rad" in ckm else None
        self.pmns_use = tuple(p for p in PMNS_KEYS if p in use)
        self.dm_use = tuple(p for p in DM_KEYS if p in use)
        self.mr_use = tuple(p for p in MR_BLOCK_KEYS if p in use)
        self.best = dict(pmns_centre) if pmns_centre is not None else pmns_best(ordering, analysis)
        if pmns_fn is None:
            pmns_fn = functools.partial(_data.pmns_delta_chi2, analysis=analysis, relative_to="local")
        self._pmns_fn = pmns_fn
        self.objective = Objective(
            ckm=ckm, masses=masses,
            pmns_delta_chi2=pmns_fn if self.pmns_use else None,
            pmns_ordering=ordering, pmns_params=self.pmns_use or PMNS_KEYS,
            pmns_mode="joint", pmns_best=self.best)
        if not (self.pmns_use or self.dm_use):
            self.offset = 0.0
        else:
            self.offset = _data.ordering_offset(ordering, analysis) if offset is None else float(offset)
        if self.dm_use:
            self._dm_best = _dm_best(ordering, analysis)
        if self.mr_use:
            vals, names, cov_log = mass_ratio_targets(mass_records)
            if any(n in DERIVED_RATIO_KEYS for n in self.mr_use):
                vals = mr_block_values(vals)
                C = np.array([[DERIVED_RATIO_KEYS.get(n, {n: 1.0}).get(m, 0.0) for m in names] for n in self.mr_use])
                cl = C @ cov_log @ C.T
            else:  # unchanged pre-exp-024 path (bit-identical scores)
                idx = [names.index(n) for n in self.mr_use]
                cl = cov_log[np.ix_(idx, idx)].copy()
            cl[np.diag_indices_from(cl)] += np.array([_floor_of(self.mass_floor, n) ** 2 for n in self.mr_use])
            v = np.array([vals[n] for n in self.mr_use])
            self._mr_value = v
            self._mr_cov = cl * np.outer(v, v)
            self._mr_L = np.linalg.cholesky(self._mr_cov)
            self._mr_sig = np.sqrt(np.diag(self._mr_cov))
        self.scale = None
        self.scale_shift, self.scale_systematic = {}, {}
        if scale is not None:
            bad = [u for u in use if u not in SCALE_ADDITIVE + SCALE_MULTIPLICATIVE]
            if bad:
                raise ValueError(f"scale translation is not defined for {bad}; supported: "
                                 f"{SCALE_ADDITIVE + SCALE_MULTIPLICATIVE}")
            fs = _freeze_spec(scale)
            ck = ckm_records if ckm_records is not None else _data.load_ckm()
            fckm = (float(np.arcsin(ck["s12"]["value"])), float(np.arcsin(ck["s13"]["value"])),
                    float(np.arcsin(ck["s23"]["value"])), float(ck["delta_rad"]["value"]))
            ov = _mass_override(mass_records)
            fmass = tuple(sorted(ov.items())) if ov else ()
            fcentre = tuple(sorted((k, float(self.best[k])) for k in PMNS_KEYS))
            shift, sys = _scale_table(ordering, analysis, fs, fckm, fmass, fcentre)
            self.scale = _spec_dict(fs)
            self.scale_shift = {k: shift[k] for k in use}
            self.scale_systematic = {k: sys[k] for k in use}

    def _prep(self, pred):
        missing = [n for n in self.use if n not in pred]
        if missing:
            raise KeyError(f"pred lacks {missing}")
        p = {n: float(pred[n]) for n in self.use}
        bad = [n for n, v in p.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite predictions {bad}")
        for n, s in self.scale_shift.items():  # high-scale prediction -> data scale
            p[n] = p[n] - s if n in SCALE_ADDITIVE else p[n] * s
        if "dcp_deg" in p:  # put delta within 180 deg of the data centre so the pull sign is right
            p["dcp_deg"] = self.best["dcp_deg"] + _wrap_deg(p["dcp_deg"] - self.best["dcp_deg"])
        if self._ckm_delta_centre is not None:  # the CKM phase is periodic too (exp 012 L1)
            c = self._ckm_delta_centre
            p["ckm_delta_rad"] = c + _wrap_rad(p["ckm_delta_rad"] - c)
        return p

    def _dm_term(self, n, v):
        if n == "dm_ratio":
            return dm_ratio_delta_chi2(self.ordering, v, self.analysis)
        return max(float(self._pmns_fn(self.ordering, **{n: v})), 0.0)

    def _score(self, p):
        """(chi2, terms, pulls) of a prepared prediction."""
        r = self.objective(p)
        terms, pulls = dict(r.terms), dict(r.pulls)
        for n in self.dm_use:
            d1 = self._dm_term(n, p[n])
            terms[n] = d1
            pulls[n] = float(np.sign(p[n] - self._dm_best[n]) * np.sqrt(d1))
        if self.mr_use:
            res = np.array([p[n] for n in self.mr_use]) - self._mr_value
            y = np.linalg.solve(self._mr_L, res)
            terms["mass_ratio_cov"] = float(y @ y)
            for n, ri, si in zip(self.mr_use, res, self._mr_sig):
                pulls[n] = float(ri / si)
        return float(sum(terms.values())), terms, pulls

    def chi2(self, pred) -> float:
        return self._score(self._prep(pred))[0]

    def chi2_soft(self, pred) -> float:
        """chi2, or (outside the NuFIT tables, where chi2 = inf) CHI2_CAP + a Gaussian
        estimate from the NuFIT 1 sigma errors around the data centre. Keeps the
        minimizer's landscape sloped back toward the tables instead of flat. Never used
        for reported numbers."""
        p = self._prep(pred)
        c = self._score(p)[0]
        if np.isfinite(c):
            return c
        recs = _data.load_pmns(self.ordering, self.analysis)
        g = 0.0
        for n in self.pmns_use:
            r = recs[n]
            d = p[n] - self.best[n]
            g += (d / (r["err_plus"] if d > 0 else r["err_minus"])) ** 2
        if self.dm_use:
            sig = _dm_sigma(self.ordering, self.analysis)
            for n in self.dm_use:
                g += ((p[n] - self._dm_best[n]) / sig[n]) ** 2
        return CHI2_CAP + g

    def __call__(self, pred, k=0, inputs=None, k_discrete=0, ordering_selected=False) -> EvalResult:
        inputs = self.inputs if inputs is None else tuple(inputs)
        if inputs != self.inputs:
            _check_names(inputs, "inputs")
            check_observables(self.use, inputs)
        p = self._prep(pred)
        chi2, terms, pulls = self._score(p)
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

        chi2_g = chi2 + self.offset
        pw = pv(chi2)
        return EvalResult(
            chi2=chi2, chi2_global=chi2_g, ordering_offset=self.offset,
            N=N, k=k, k_discrete=k_discrete, dof=dof,
            p=pw, p_global=pv(chi2_g), p_fit=pw,
            max_abs_pull=float(max((abs(v) for v in pulls.values()), default=0.0)), pulls=pulls, terms=terms,
            ordering=self.ordering, analysis=self.analysis, use=self.use, inputs=inputs,
            pred=p, k_continuous=k_cont, ordering_selected=bool(ordering_selected),
            mass_floor=self.mass_floor, scale=None if self.scale is None else dict(self.scale),
            scale_shift=dict(self.scale_shift), scale_systematic=dict(self.scale_systematic))


def _ordering_dependent(use):
    """True if the ordering matters for this set of observables (PMNS angles or splittings)."""
    return any(u in PMNS_KEYS or u in DM_KEYS for u in use)


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


def _scorer_kw(scorer_kwargs, ordering, mass_floor=0.0, scale=None):
    """scorer_kwargs is either one override dict or {"NO": {...}, "IO": {...}}; the exp 019
    scoring options are added when not at their defaults."""
    kw = dict(scorer_kwargs or {})
    if set(kw) <= set(_data.ORDERINGS) and kw:
        kw = dict(kw.get(ordering) or {})
    if isinstance(mass_floor, Mapping) or mass_floor:
        kw["mass_floor"] = mass_floor
    if scale is not None:
        kw["scale"] = scale
    return kw


def evaluate(pred: Mapping, ordering: str, analysis: str = _data.DEFAULT_ANALYSIS, k: int = 0,
             use: Sequence[str] | None = None, inputs: Sequence[str] = (), k_discrete: int = 0,
             ordering_selected: bool = False, scorer_kwargs: Mapping | None = None, *,
             mass_floor=0.0, scale: Mapping | None = None) -> EvalResult:
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
    mass_floor: relative theory floor on mass observables (module docstring), default 0.
    scale: high-scale spec, pred given at scale["mu"] (module docstring), default None.
    """
    use = _require_use(use)
    _check_ordering_arg(ordering)
    inputs = tuple(inputs)
    if ordering == "best":
        if not _ordering_dependent(use):
            return evaluate(pred, "NO", analysis, k, use, inputs, k_discrete, ordering_selected,
                            _scorer_kw(scorer_kwargs, "NO", mass_floor, scale))
        res = {o: Scorer(use, o, analysis, inputs, **_scorer_kw(scorer_kwargs, o, mass_floor, scale))
               for o in _data.ORDERINGS}
        scored = [res[o](pred, k=k, k_discrete=k_discrete) for o in _data.ORDERINGS]
        best = min(scored, key=lambda r: r.chi2_global)
        return res[best.ordering](pred, k=k, k_discrete=k_discrete, ordering_selected=True)
    sc = Scorer(use, ordering, analysis, inputs, **_scorer_kw(scorer_kwargs, ordering, mass_floor, scale))
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
                   mass_floor=0.0, scale: Mapping | None = None, **probe_kwargs) -> EvalResult:
    """evaluate(model_fn(), ...) after a leak probe of model_fn (exp 012 H3). model_fn() -> pred."""
    use = _require_use(use)
    if leak_check:
        check_leaks(model_fn, inputs, **probe_kwargs)
    return evaluate(model_fn(), ordering, analysis, k=k, use=use, inputs=inputs, k_discrete=k_discrete,
                    ordering_selected=ordering_selected, mass_floor=mass_floor, scale=scale)


# ---------------------------------------------------------------- fitting


def fit_and_evaluate(model_fn: Callable[[dict], Mapping], param_bounds: Mapping[str, tuple],
                     ordering: str, use: Sequence[str], analysis: str = _data.DEFAULT_ANALYSIS,
                     inputs: Sequence[str] = (), k_discrete: int = 0, periodic: Sequence[str] = (),
                     n_starts: int = 32, seed: int = 0, extra_starts: Sequence[Mapping] = (),
                     tol: float = 1e-9, ordering_selected: bool = False, leak_check: bool = False,
                     scorer_kwargs: Mapping | None = None, *, mass_floor=0.0,
                     scale: Mapping | None = None) -> FitResult:
    """Minimize the within-ordering chi^2 over continuous parameters, then evaluate.

    model_fn(theta: dict) -> pred dict. param_bounds: name -> (lo, hi).
    periodic: parameter names with period hi - lo (phases). They are optimized unbounded
        and wrapped into [lo, hi) afterwards; others are bounded.
    k = len(param_bounds) + k_discrete (k_discrete in bits). chi2_global differs from chi2
    by a constant within one ordering, so the same best fit serves both.
    ordering="best": fit both orderings, keep the lower chi2_global, add the ordering bit.
    leak_check: run the leak probe on model_fn at the best-fit parameters (LeakError).
    scorer_kwargs: Scorer data overrides (one dict, or per ordering).
    mass_floor, scale: as in evaluate (module docstring).
    Multi-start: n_starts points from a scrambled Sobol sequence over the box (seeded),
    plus extra_starts, each run with Nelder-Mead then polished with a second Nelder-Mead
    from the best point.
    """
    from scipy.stats import qmc

    use = _require_use(use)
    _check_ordering_arg(ordering)
    inputs = tuple(inputs)
    if ordering == "best":
        if not _ordering_dependent(use):
            return fit_and_evaluate(model_fn, param_bounds, "NO", use, analysis, inputs, k_discrete,
                                    periodic, n_starts, seed, extra_starts, tol, ordering_selected,
                                    leak_check, _scorer_kw(scorer_kwargs, "NO", mass_floor, scale))
        fits = [fit_and_evaluate(model_fn, param_bounds, o, use, analysis, inputs, k_discrete, periodic,
                                 n_starts, seed, extra_starts, tol, False, False,
                                 _scorer_kw(scorer_kwargs, o, mass_floor, scale))
                for o in _data.ORDERINGS]
        fr = min(fits, key=lambda f: f.result.chi2_global)
        o = fr.result.ordering
        sc = Scorer(use, o, analysis, inputs, **_scorer_kw(scorer_kwargs, o, mass_floor, scale))
        if leak_check:
            check_leaks(lambda: model_fn(fr.params), inputs)
        res = sc(model_fn(fr.params), k=fr.result.k, k_discrete=fr.result.k_discrete, ordering_selected=True)
        return FitResult(fr.params, res, fr.start_chi2, fr.n_starts, sum(f.nfev for f in fits))

    names = list(param_bounds)
    lo = np.array([param_bounds[n][0] for n in names], dtype=float)
    hi = np.array([param_bounds[n][1] for n in names], dtype=float)
    per = np.array([n in periodic for n in names])
    scorer = Scorer(use, ordering, analysis, inputs, **_scorer_kw(scorer_kwargs, ordering, mass_floor, scale))
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

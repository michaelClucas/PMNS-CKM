"""Criterion-4 nulls (STATE.md "Criterion-4 protocol", exp 012): fake universes, hits, decision rule.

The search under test is passed in as a callable:

    pipeline(universe) -> best result

`universe` is a Universe (below). The pipeline must read every measured number from it
(universe.context(), universe.load_ckm(), universe.load_pmns(o), ...) and score with
universe.evaluate(...) / universe.fit_and_evaluate(...), never through src.data directly,
so that the same code runs on the real data and on every fake. It returns the result of
its own selection (an src.evaluate.EvalResult, a mapping with the same fields, or None
when it finds nothing). The pipeline must be a module-level function (it is pickled to
worker processes) and must rerun every data-dependent choice, including discrete choices
and the ordering choice, on each universe.

    run_null(pipeline, null="N1", ordering="NO", n_fakes=1000, ...)  -> NullResult
    power_check(pipeline, inject_fn, n_inject=20, n_fakes=300, ok_fn=candidate_ok, ...) -> PowerResult
    make_injector(pred)                                               -> inject_fn
    is_hit(fake, ref), candidate_ok(res), max_hits_to_pass(n), decide(n_hits, n)
    is_hit_fit_only(fake, ref), candidate_ok_fit_only(res), FIT_ONLY   (informational rule)

Nulls
-----
N1  CKM-side hierarchical null. PMNS data and tables are the base universe's (real NuFIT
    for run_null on real data). The CKM is replaced by the exact Wolfenstein matrix with
    lambda ~ U[0.05, 0.5], A ~ U[0.3, 2], rhobar ~ U[-1, 1], etabar ~ U[-1, 1]
    (N1_PRIOR). Every CKM record (global fit and direct_*) is centred on the fake value
    and keeps the real record's absolute errors. Masses stay at the base values. This is
    the test of a CKM-PMNS link.
N2  Shifted-data null. CKM and masses are the base universe's. For each ordering the PMNS
    data centre is redrawn from a smooth prior that is not built from pattern values:
    s12sq, s13sq, s23sq uniform on the NuFIT 3 sigma range of that ordering widened x3
    about its midpoint (N2_WIDEN, clipped to (0, 1)), dcp_deg uniform on [0, 360). The
    fake is scored with the NuFIT tables translated to the fake centre (below).
N3  Mass-side null (exp 019). CKM and PMNS are the base universe's. The seven same-scale
    mass ratios of src.evaluate (u/c, c/t, d/s, s/b, b/tau, e/mu, mu/tau) get independent
    log-uniform factors 10^U[-w, w]: w = N3_PRIOR["decades"] = 1 for the light-over-heavy
    ratios, w = N3_PRIOR["heavy_decades"] = 0.2 for c_t and b_tau. Only the ratios listed
    in N3_PRIOR["ratios"] (default u_c, d_s, s_b, e_mu, mu_tau) are moved; c_t and b_tau
    move m_c and m_b, which are also the QCD decoupling scales of src.rg, so when either is
    listed their factors are clipped so that m_b(m_b) stays in N3_MB_RANGE = [2.5, 50] GeV
    and m_c(m_c) in [N3_MC_MIN = 1.0 GeV, 0.8 m_b] (the four-loop running of src.rg is not
    trustworthy outside); truth["factors"] holds the clipped factors. The heaviest mass of each sector is the anchor
    (t for up quarks, tau for down quarks via b_tau and for leptons); each lighter mass is
    moved by the product of the factors on its chain (u by u_c * c_t, d by d_s * s_b *
    b_tau, e by e_mu * mu_tau, ...), so each realised M_Z ratio moves by its own factor
    (exactly for the light quarks and leptons, whose running is linear in the mass; up to
    threshold effects for c and b). Each fake record keeps the real record's RELATIVE
    error, and the PDG ratio records (m_u_over_m_d, ...) are moved consistently; top
    records are never moved. Pipelines pass Universe.rg_masses() to src.rg as masses=, and
    the scorer gets the records through mass_records (mr_* targets are recomputed from
    them). truth["factors"] holds the drawn factors, truth["masses"] the fake values.
Haar-PMNS is not implemented here: per exp 012 M5 it is informational only.

Translated NuFIT tables (exp 012 M4)
------------------------------------
Let t_real be the NuFIT best fit of the ordering (src.evaluate.pmns_best) and t_fake the
fake centre. The fake's within-ordering Delta chi^2 at a prediction x is the real one at
a mapped point x':
  s12sq, s13sq   x' = x - t_fake + t_real
  dcp_deg        x' = x - t_fake + t_real, wrapped to [-180, 180) by src.data (periodic,
                 so every x' is inside the table)
  s23sq          same octant (t_fake - 0.5 and t_real - 0.5 of equal sign, or t_fake = 0.5):
                 x' = x - t_fake + t_real. Other octant: x' = t_real - (x - t_fake), i.e.
                 the real shape reflected about the fake centre, so the fake's octant-
                 partner minimum sits on the 0.5 side of its best fit as the real one does.
                 Universe.s23_mirror[o] overrides the octant rule (injected universes use
                 a pure translation; see pmns_noise).
  dm21sq_eV2, dm3l_sq_eV2 are not translated.
The 2D (s23sq, dcp) table is used when both are predicted, exactly as for real data, with
both coordinates mapped. Points mapped outside a table give inf (as for real data). The
fake keeps the real ordering offset (chi2_global = chi2 + the real NuFIT IO penalty).
The 1D pull signs and the delta wrap use the fake centre.

Hit and decision (protocol steps 3-4)
-------------------------------------
A fake is a hit when its best result r has k <= N - 2 (k counts discrete bits, as
src.evaluate does), max|pull| < 2, and p_global >= the real candidate's p_global
(is_hit). p_global is compared with a relative tolerance, p_fake >= p_ref * (1 - P_RTOL)
with P_RTOL = 1e-5, so a relation the null does not touch still counts as a hit when the
fake's p moves by integrator noise (exp 018: 1e-7 RG noise flipped exact comparisons).

Fit-only rule (exp 019; INFORMATIONAL). For a lead with k > N - 2 the strict rule is
vacuous: no fake can be a hit and no injection passes candidate_ok, so criterion 4 says
nothing. is_hit_fit_only drops the compression clause (max|pull| < 2 and p_global >=
p_ref (1 - P_RTOL)); candidate_ok_fit_only drops it from criteria 2-3. Use them together
(FIT_ONLY = dict(hit_fn=..., ok_fn=...) for run_null / power_check). A null passed under
the fit-only rule says the search does not find equally good fits in fake universes. It
does NOT satisfy criterion 4, which still needs a result that passes criterion 2.
NullResult.hit_rule and PowerResult.hit_rule record which rule was used.
The null passes when the one-sided 95% Clopper-Pearson upper limit on the hit rate is
below 1%: at most 4 hits in 1000 fakes, 0 in 300 (max_hits_to_pass). Fakes run in index
order; with early_stop the run stops as soon as the hit count exceeds that bound (only a
fail can be declared early, never a pass).

Power check (protocol step 5)
-----------------------------
inject_fn(rng, base) -> Universe in which the candidate's structure is true (make_injector
builds one from a best-fit prediction plus measurement noise). For each of n_inject
injections the pipeline is run on the injected universe (its result is the reference);
if that result passes ok_fn (default candidate_ok, criteria 2-3) the null is run with
n_fakes fakes built around the injected universe. PowerResult.fraction is the fraction of
injections where both hold; below 0.5 the null cannot confirm or exclude the family. With
300 fakes the null passes only with 0 hits. With workers > 1 one process pool serves all
injections (run_null(pool=...)): fakes go to it in blocks of 4 x workers and are counted
in index order, so the results equal the serial ones.

Seeds: fake i of a run uses np.random.default_rng([*seed, null code, ordering code, i]),
so results do not depend on the worker count.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

import numpy as np

from . import data as _data
from . import evaluate as _ev
from .mixing import standard_matrix as _standard_matrix
from .stats import fake_universe_rate

PMNS_KEYS = _ev.PMNS_KEYS
NULLS = ("N1", "N2", "N3")
N1_PRIOR = {"lambda": (0.05, 0.5), "A": (0.3, 2.0), "rhobar": (-1.0, 1.0), "etabar": (-1.0, 1.0)}
N2_WIDEN = 3.0
N3_PRIOR = {"decades": 1.0, "heavy_decades": 0.2, "ratios": ("u_c", "d_s", "s_b", "e_mu", "mu_tau")}
N3_RATIOS = ("u_c", "c_t", "d_s", "s_b", "b_tau", "e_mu", "mu_tau")
# mass record -> the ratio factors that move it (chains anchored at t and tau)
_N3_CHAIN = {"m_c": ("c_t",), "m_u": ("u_c", "c_t"), "m_b": ("b_tau",), "m_s": ("s_b", "b_tau"),
             "m_d": ("d_s", "s_b", "b_tau"), "m_mu": ("mu_tau",), "m_e": ("e_mu", "mu_tau")}
_N3_PDG_RATIOS = {"m_u_over_m_d": ("m_u", "m_d"), "m_c_over_m_s": ("m_c", "m_s"),
                  "m_b_over_m_c": ("m_b", "m_c"), "m_b_over_m_s": ("m_b", "m_s")}
P_RTOL = 1e-5
N3_MC_MIN = 1.0             # GeV, lower clip of a fake m_c(m_c)
N3_MB_RANGE = (2.5, 50.0)   # GeV, clip of a fake m_b(m_b); m_c is also kept below 0.8 m_b
RATE_LIMIT = 0.01
CL = 0.95
_NULL_CODE = {"N1": 1, "N2": 2, "inject": 3, "N3": 4}
_ORD_CODE = {"NO": 0, "IO": 1}


# ---------------------------------------------------------------- universes


def _ckm_records_real():
    return {k: v for k, v in _data.load_ckm().items() if isinstance(v, dict) and "value" in v}


def _mass_records_real():
    return {k: v for k, v in _data.load_masses().items() if isinstance(v, dict) and "value" in v}


@dataclass
class Universe:
    """One data set: CKM records, mass records, and a PMNS data centre per ordering.

    ckm / masses: flat name -> record dicts (load_ckm / load_masses names). None = real.
    pmns_centre: {ordering: {s12sq, s13sq, s23sq, dcp_deg}}; an ordering missing or None
        means the real NuFIT data (untranslated tables).
    ordering: the ordering this universe is being scored in (pipelines may also use
        ordering="best" in evaluate, which pays the ordering bit).
    """
    kind: str = "real"
    ordering: str = "NO"
    analysis: str = _data.DEFAULT_ANALYSIS
    ckm: dict | None = None
    masses: dict | None = None
    pmns_centre: dict = field(default_factory=dict)
    truth: dict = field(default_factory=dict)
    index: int = -1
    s23_mirror: dict = field(default_factory=dict)  # {ordering: bool}; missing = octant rule

    @classmethod
    def real(cls, ordering="NO", analysis=_data.DEFAULT_ANALYSIS):
        return cls(kind="real", ordering=ordering, analysis=analysis)

    # ---- data access
    def load_ckm(self) -> dict:
        return dict(self.ckm) if self.ckm is not None else _ckm_records_real()

    def load_masses(self) -> dict:
        return dict(self.masses) if self.masses is not None else _mass_records_real()

    def rg_masses(self) -> dict | None:
        """masses= override for src.rg ({record: value}); None for the real masses."""
        if self.masses is None:
            return None
        return {k: float(r["value"]) for k, r in self.masses.items() if isinstance(r, dict) and "value" in r}

    def pmns_real_best(self, ordering):
        return _ev.pmns_best(ordering, self.analysis)

    def pmns_data_centre(self, ordering) -> dict:
        c = self.pmns_centre.get(ordering)
        return dict(c) if c is not None else self.pmns_real_best(ordering)

    def load_pmns(self, ordering) -> dict:
        """load_pmns-like records with the four PMNS values moved to the data centre
        (errors and ranges are the real ones, ranges shifted with the centre)."""
        recs = _data.load_pmns(ordering, self.analysis)
        c = self.pmns_centre.get(ordering)
        if c is None:
            return recs
        out = dict(recs)
        for k in PMNS_KEYS:
            r = dict(recs[k])
            shift = c[k] - r["value"]
            r["value"] = c[k]
            if "range_3sigma" in r:
                r["range_3sigma"] = [x + shift for x in r["range_3sigma"]]
            out[k] = r
        return out

    def context(self) -> dict:
        """src.baselines.data_context() computed from this universe's records."""
        from .baselines import INPUT_RECORDS

        src = {"ckm": self.load_ckm(), "masses": self.load_masses()}
        ctx = {name: src[grp][rec]["value"] for name, (grp, rec) in INPUT_RECORDS.items()}
        ctx["theta_C"] = float(np.arcsin(ctx["ckm_s12"]))
        ctx["theta_ub"] = float(np.arcsin(ctx["ckm_s13"]))
        ctx["theta_cb"] = float(np.arcsin(ctx["ckm_s23"]))
        ctx["delta_q"] = float(ctx["ckm_delta_rad"])
        ctx["V_ckm"] = _standard_matrix(ctx["theta_C"], ctx["theta_ub"], ctx["theta_cb"], ctx["delta_q"])
        return ctx

    # ---- translated NuFIT tables
    def map_to_real(self, ordering, s12sq=None, s13sq=None, s23sq=None, dcp_deg=None) -> dict:
        """The point x' of the real tables that scores like x in this universe (module docstring)."""
        c = self.pmns_centre.get(ordering)
        out = {"s12sq": s12sq, "s13sq": s13sq, "s23sq": s23sq, "dcp_deg": dcp_deg}
        if c is None:
            return out
        t = self.pmns_real_best(ordering)
        for k in ("s12sq", "s13sq", "dcp_deg"):
            if out[k] is not None:
                out[k] = out[k] - c[k] + t[k]
        if s23sq is not None:
            mirror = self.s23_mirror.get(ordering)
            same = (c["s23sq"] - 0.5) * (t["s23sq"] - 0.5) >= 0 if mirror is None else not mirror
            out["s23sq"] = s23sq - c["s23sq"] + t["s23sq"] if same else t["s23sq"] - (s23sq - c["s23sq"])
        return out

    def pmns_delta_chi2(self, ordering, s12sq=None, s13sq=None, s23sq=None, dcp_deg=None,
                        dm21sq_eV2=None, dm3l_sq_eV2=None) -> float:
        """Within-ordering Delta chi^2 of this universe's PMNS data (translated NuFIT tables)."""
        m = self.map_to_real(ordering, s12sq, s13sq, s23sq, dcp_deg)
        return _data.pmns_delta_chi2(ordering, dm21sq_eV2=dm21sq_eV2, dm3l_sq_eV2=dm3l_sq_eV2,
                                     analysis=self.analysis, relative_to="local", **m)

    # ---- scoring
    def scorer_kwargs(self, ordering) -> dict:
        kw = {}
        if self.ckm is not None:
            kw["ckm_records"] = self.ckm
        if self.masses is not None:
            kw["mass_records"] = self.masses
        if self.pmns_centre.get(ordering) is not None:
            kw["pmns_fn"] = self.pmns_delta_chi2
            kw["pmns_centre"] = self.pmns_data_centre(ordering)
        return kw

    def _all_kwargs(self):
        return {o: self.scorer_kwargs(o) for o in _data.ORDERINGS}

    def scorer(self, use, ordering=None, inputs=(), **opts):
        """src.evaluate.Scorer on this universe; opts: mass_floor, scale."""
        o = ordering or self.ordering
        return _ev.Scorer(use, o, self.analysis, inputs, **self.scorer_kwargs(o), **opts)

    def evaluate(self, pred, use, ordering=None, **kw) -> _ev.EvalResult:
        """src.evaluate.evaluate on this universe (ordering defaults to self.ordering)."""
        return _ev.evaluate(pred, ordering or self.ordering, self.analysis, use=use,
                            scorer_kwargs=self._all_kwargs(), **kw)

    def fit_and_evaluate(self, model_fn, param_bounds, use, ordering=None, **kw) -> _ev.FitResult:
        """src.evaluate.fit_and_evaluate on this universe."""
        return _ev.fit_and_evaluate(model_fn, param_bounds, ordering or self.ordering, use,
                                    analysis=self.analysis, scorer_kwargs=self._all_kwargs(), **kw)


# ---------------------------------------------------------------- fake generators


def n1_universe(base: Universe, rng, prior: Mapping | None = None) -> Universe:
    """N1: base PMNS and masses, CKM from the broad hierarchical prior."""
    pr = dict(N1_PRIOR, **(prior or {}))
    w = {k: float(rng.uniform(*pr[k])) for k in ("lambda", "A", "rhobar", "etabar")}
    vals = _ev.ckm_from_wolfenstein(w["lambda"], w["A"], w["rhobar"], w["etabar"])
    real = _ckm_records_real()
    ckm = {}
    for name, rec in real.items():
        key = name[len("direct_"):] if name.startswith("direct_") else name
        if key in vals:
            ckm[name] = dict(rec, value=vals[key])
        else:
            ckm[name] = rec
    return replace(base, kind="N1", ckm=ckm, truth={"wolfenstein": w})


def n2_centre(ordering, rng, analysis=_data.DEFAULT_ANALYSIS, widen=N2_WIDEN) -> dict:
    """N2 prior draw of a PMNS data centre for one ordering."""
    recs = _data.load_pmns(ordering, analysis)
    c = {}
    for k in ("s12sq", "s13sq", "s23sq"):
        lo, hi = recs[k]["range_3sigma"]
        mid, half = (lo + hi) / 2, widen * (hi - lo) / 2
        c[k] = float(rng.uniform(max(mid - half, 1e-6), min(mid + half, 1 - 1e-6)))
    c["dcp_deg"] = float(rng.uniform(0.0, 360.0))
    return c


def n2_universe(base: Universe, rng, prior: Mapping | None = None) -> Universe:
    """N2: base CKM and masses, PMNS centres from the smooth prior (both orderings)."""
    widen = float((prior or {}).get("widen", N2_WIDEN))
    centres = {o: n2_centre(o, rng, base.analysis, widen) for o in _data.ORDERINGS}
    return replace(base, kind="N2", pmns_centre=centres, truth={"pmns_centre": centres})


def n3_masses(rng, base_masses: Mapping, prior: Mapping | None = None):
    """N3 draw: (fake mass records, factors). See the module docstring."""
    pr = dict(N3_PRIOR, **(prior or {}))
    bad = set(pr["ratios"]) - set(N3_RATIOS)
    if bad:
        raise ValueError(f"unknown N3 ratios {sorted(bad)}; known {N3_RATIOS}")
    fac = {}
    for r in N3_RATIOS:  # all seven are drawn, so the stream does not depend on the subset
        w = pr["heavy_decades"] if r in ("c_t", "b_tau") else pr["decades"]
        x = float(10 ** rng.uniform(-w, w))
        fac[r] = x if r in pr["ratios"] else 1.0
    base = {k: dict(v) for k, v in base_masses.items() if isinstance(v, dict) and "value" in v}
    mb0, mc0 = base["m_b"]["value"], base["m_c"]["value"]
    if fac["b_tau"] != 1.0:  # keep the QCD thresholds of src.rg sensible (module docstring)
        fac["b_tau"] = float(np.clip(fac["b_tau"], N3_MB_RANGE[0] / mb0, N3_MB_RANGE[1] / mb0))
    if fac["c_t"] != 1.0:
        fac["c_t"] = float(np.clip(fac["c_t"], N3_MC_MIN / mc0, 0.8 * mb0 * fac["b_tau"] / mc0))
    val = {k: base[k]["value"] for k in base}
    for m, chain in _N3_CHAIN.items():
        val[m] = base[m]["value"] * float(np.prod([fac[c] for c in chain]))
    for k, (a, b) in _N3_PDG_RATIOS.items():
        if k in base:
            val[k] = base[k]["value"] * (val[a] / base[a]["value"]) / (val[b] / base[b]["value"])
    out = {}
    for k, r in base.items():
        f = val[k] / r["value"]
        out[k] = r if f == 1.0 else dict(r, value=val[k], err_plus=r["err_plus"] * f,
                                          err_minus=r["err_minus"] * f)
    return out, fac


def n3_universe(base: Universe, rng, prior: Mapping | None = None) -> Universe:
    """N3: base CKM and PMNS, mass ratios from the log-uniform prior (module docstring)."""
    masses, fac = n3_masses(rng, base.load_masses(), prior)
    return replace(base, kind="N3", masses=masses,
                   truth={"factors": fac, "masses": {k: v["value"] for k, v in masses.items()}})


_MAKERS = {"N1": n1_universe, "N2": n2_universe, "N3": n3_universe}


def make_fake(null: str, base: Universe, rng, prior: Mapping | None = None) -> Universe:
    if null not in _MAKERS:
        raise ValueError(f"null must be one of {NULLS}")
    return _MAKERS[null](base, rng, prior)


# ---------------------------------------------------------------- injection


def _sample_1d(x, chi2, rng):
    """Draw from the density exp(-chi2/2) on a (non-uniform) 1D grid, linear in between."""
    w = np.exp(-0.5 * (chi2 - chi2.min()))
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x))))
    return float(np.interp(rng.uniform(0, cdf[-1]), cdf, x))


def _sample_2d(x, y, z, rng):
    w = np.exp(-0.5 * (z - z.min()))
    cell = 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])
    cell = cell * np.outer(np.diff(x), np.diff(y))
    i = rng.choice(cell.size, p=(cell / cell.sum()).ravel())
    a, b = np.unravel_index(i, cell.shape)
    return float(rng.uniform(x[a], x[a + 1])), float(rng.uniform(y[b], y[b + 1]))


def pmns_noise(ordering, rng, analysis=_data.DEFAULT_ANALYSIS) -> dict:
    """Measurement-noise offsets (pseudo-data centre minus truth) for the four PMNS parameters.

    A point s is drawn from the NuFIT likelihood shape exp(-Delta chi^2 / 2) (s12sq and
    s13sq from their 1D profiles, (s23sq, dcp_deg) jointly from the 2D table) and the
    offset is t_real - s. With the data centre at truth + offset and the tables translated
    without mirroring, the truth is scored at the real-table point s, so Delta chi^2(truth)
    has the same (non-Gaussian, octant-bimodal) distribution as for the real measurement."""
    t = _ev.pmns_best(ordering, analysis)
    out = {}
    for k, tag in (("s12sq", "T12"), ("s13sq", "T13")):
        x, y = _data._shifted_profile_1d(analysis, ordering, tag)
        out[k] = t[k] - _sample_1d(x, y, rng)
    x, y, z = _data._shifted_grid_2d(analysis, ordering, "T23-DCP")
    s23, d = _sample_2d(x, y, z, rng)
    out["s23sq"] = t["s23sq"] - s23
    out["dcp_deg"] = _ev._wrap_deg(t["dcp_deg"] - d)
    return out


def make_injector(pred: Mapping, noise: bool = True) -> Callable:
    """inject_fn(rng, base) -> Universe in which `pred` is the truth.

    pred: the candidate's best-fit prediction (src.evaluate keys). PMNS keys set the data
    centre of base.ordering to pred + pmns_noise, scored with purely translated tables
    (no octant mirror, see pmns_noise); the other ordering keeps the base data;
    a PMNS key absent from pred keeps the base centre plus noise. CKM keys (ckm_<name>)
    and mass keys set that record's value to pred + Gaussian noise with the record's
    asymmetric errors. Records not in pred keep the base values. noise=False gives the
    Asimov universe (pred exactly).
    """
    pred = {k: float(v) for k, v in pred.items()}

    def inject_fn(rng, base: Universe) -> Universe:
        o = base.ordering
        centre = base.pmns_data_centre(o)
        nz = pmns_noise(o, rng, base.analysis) if noise else {k: 0.0 for k in PMNS_KEYS}
        new_c = {k: pred.get(k, centre[k]) + nz[k] for k in PMNS_KEYS}
        new_c["dcp_deg"] = new_c["dcp_deg"] % 360.0
        new_c["s23sq"] = float(np.clip(new_c["s23sq"], 1e-6, 1 - 1e-6))
        ckm, masses = base.load_ckm(), base.load_masses()
        touched_ckm = touched_m = False

        def noisy(rec, v):
            if not noise:
                return v
            z = rng.standard_normal()
            return v + z * (rec["err_plus"] if z > 0 else abs(rec["err_minus"]))

        for k, v in pred.items():
            if k.startswith("ckm_") and k[4:] in ckm:
                ckm[k[4:]] = dict(ckm[k[4:]], value=noisy(ckm[k[4:]], v))
                touched_ckm = True
            elif k in masses:
                masses[k] = dict(masses[k], value=noisy(masses[k], v))
                touched_m = True
        centres = dict(base.pmns_centre)
        centres[o] = new_c
        return replace(base, kind="inject", pmns_centre=centres, s23_mirror={**base.s23_mirror, o: False},
                       ckm=ckm if touched_ckm else base.ckm, masses=masses if touched_m else base.masses,
                       truth={"pred": dict(pred), "pmns_centre": new_c})

    return inject_fn


# ---------------------------------------------------------------- hits and decision

SUMMARY_FIELDS = ("N", "k", "k_discrete", "max_abs_pull", "p", "p_global", "chi2", "chi2_global", "ordering")


def summarize(res):
    """Picklable summary of a pipeline result (None stays None)."""
    if res is None:
        return None
    if isinstance(res, Mapping):
        return dict(res)
    out = {f: getattr(res, f) for f in SUMMARY_FIELDS if hasattr(res, f)}
    if hasattr(res, "pred"):
        out["pred"] = dict(res.pred)
    return out


def _get(r, name):
    return r[name] if isinstance(r, Mapping) else getattr(r, name)


def _p_at_least(fake, ref, rtol):
    pf, pr = float(_get(fake, "p_global")), float(_get(ref, "p_global"))
    if not np.isfinite(pf):
        return False
    return bool(pf >= pr * (1.0 - rtol) or not np.isfinite(pr))


def is_hit(fake, ref, rtol: float = P_RTOL) -> bool:
    """Protocol step 3: k <= N - 2, max|pull| < 2, p_global >= the reference's * (1 - rtol)."""
    if fake is None:
        return False
    return bool(_get(fake, "k") <= _get(fake, "N") - 2 and _get(fake, "max_abs_pull") < 2
                and _p_at_least(fake, ref, rtol))


def is_hit_fit_only(fake, ref, rtol: float = P_RTOL) -> bool:
    """INFORMATIONAL fit-only hit (module docstring): max|pull| < 2 and p_global >= ref * (1 - rtol).
    No compression clause, so usable when k > N - 2. It does not establish criterion 4."""
    if fake is None:
        return False
    return bool(_get(fake, "max_abs_pull") < 2 and _p_at_least(fake, ref, rtol))


def candidate_ok(res, p_min=0.05) -> bool:
    """Criteria 2-3 on a result: k <= N - 2, max|pull| < 2, p_global > p_min."""
    if res is None:
        return False
    p = float(_get(res, "p_global"))
    return bool(_get(res, "k") <= _get(res, "N") - 2 and _get(res, "max_abs_pull") < 2
                and np.isfinite(p) and p > p_min)


def candidate_ok_fit_only(res, p_min=0.05) -> bool:
    """INFORMATIONAL: criterion 3 only (max|pull| < 2, p_global > p_min), no compression clause."""
    if res is None:
        return False
    p = float(_get(res, "p_global"))
    return bool(_get(res, "max_abs_pull") < 2 and np.isfinite(p) and p > p_min)


FIT_ONLY = {"hit_fn": is_hit_fit_only, "ok_fn": candidate_ok_fit_only}


def hit_rule_name(hit_fn) -> str:
    if hit_fn is is_hit:
        return "strict"
    if hit_fn is is_hit_fit_only:
        return "fit_only (informational; criterion 4 needs a compressed result)"
    return "custom:" + getattr(hit_fn, "__name__", repr(hit_fn))


def max_hits_to_pass(n, limit=RATE_LIMIT, cl=CL) -> int:
    """Largest hit count in n fakes whose one-sided Clopper-Pearson upper limit is < limit (-1: none)."""
    h = -1
    while h + 1 <= n and fake_universe_rate(h + 1, n, cl)["upper"] < limit:
        h += 1
    return h


def decide(n_hits, n, limit=RATE_LIMIT, cl=CL) -> dict:
    r = fake_universe_rate(n_hits, n, cl)
    return {**r, "n_hits": n_hits, "n": n, "passed": bool(r["upper"] < limit)}


# ---------------------------------------------------------------- runners


@dataclass
class NullResult:
    null: str
    ordering: str
    n_fakes: int          # planned
    n_run: int            # actually run (smaller only after an early fail)
    n_hits: int
    rate: float
    upper: float
    passed: bool
    stopped_early: bool
    max_hits: int
    ref: dict | None
    fakes: list           # per fake: {"index", "hit", "result": summary or None}
    hit_rule: str = "strict"

    def row(self):
        return {k: getattr(self, k) for k in ("null", "ordering", "n_fakes", "n_run", "n_hits", "rate",
                                              "upper", "passed", "stopped_early", "max_hits")}


@dataclass
class PowerResult:
    null: str
    ordering: str
    n_inject: int
    n_pass: int
    fraction: float
    adequate: bool        # fraction >= pass_fraction
    nulls: list           # NullResult per injection (None where the candidate itself failed)
    refs: list            # summary of the pipeline result on each injected universe
    n_candidate_ok: int = 0  # injections where the pipeline's result passed ok_fn
    hit_rule: str = "strict"


def _seed_list(seed):
    return [int(s) for s in (seed if isinstance(seed, (tuple, list)) else (seed,))]


def _fake_task(args):
    pipeline, null, base, seed, i, prior = args
    rng = np.random.default_rng(_seed_list(seed) + [_NULL_CODE[null], _ORD_CODE[base.ordering], i])
    fake = make_fake(null, base, rng, prior)
    fake.index = i
    return i, summarize(pipeline(fake))


def run_null(pipeline: Callable, null: str = "N1", ordering: str = "NO", n_fakes: int = 1000,
             ref=None, base: Universe | None = None, analysis: str = _data.DEFAULT_ANALYSIS,
             seed=0, workers: int = 1, early_stop: bool = True, hit_fn: Callable = is_hit,
             prior: Mapping | None = None, limit: float = RATE_LIMIT, cl: float = CL,
             keep_fakes: bool = True, pool=None) -> NullResult:
    """Run `pipeline` on n_fakes fake universes of one null and ordering; count hits.

    base: the universe the fakes are built from (default: real data in `ordering`).
    ref: the candidate's result on base (default: pipeline(base)).
    workers: processes (1 = in-process). prior: N1 overrides {"lambda": (lo, hi), ...},
    N2 {"widen": x} or N3 {"decades", "heavy_decades", "ratios"}. hit_fn(fake_summary,
    ref_summary) -> bool; summaries are dicts with N, k, max_abs_pull, p_global, ... (or the
    pipeline's own mapping). pool: an existing multiprocessing pool (not closed here);
    fakes then go in blocks of 4 x workers so an early stop wastes little work.
    """
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}")
    base = Universe.real(ordering, analysis) if base is None else replace(base, ordering=ordering)
    ref_s = summarize(pipeline(base) if ref is None else ref)
    if ref_s is None:
        raise ValueError("the pipeline found nothing on the base universe; nothing to test")
    max_hits = max_hits_to_pass(n_fakes, limit, cl)
    tasks = ((pipeline, null, base, seed, i, prior) for i in range(n_fakes))
    fakes, hits, stopped = [], 0, False

    def consume(it):
        nonlocal hits, stopped
        for i, s in it:
            h = bool(hit_fn(s, ref_s))
            hits += h
            fakes.append({"index": i, "hit": h, "result": s if keep_fakes else None})
            if early_stop and hits > max_hits:
                stopped = len(fakes) < n_fakes
                return

    if pool is not None:
        block = 4 * max(int(workers), 1)
        tasks = list(tasks)
        for a in range(0, len(tasks), block):
            consume(pool.imap(_fake_task, tasks[a:a + block], chunksize=1))
            if early_stop and hits > max_hits:
                break
    elif workers <= 1:
        consume(map(_fake_task, tasks))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(int(workers)) as pool:
            consume(pool.imap(_fake_task, tasks, chunksize=1))
            pool.terminate()
    n_run = len(fakes)
    d = decide(hits, max(n_run, 1), limit, cl)  # rate and upper limit over the fakes actually run
    passed = bool(d["passed"]) and n_run == n_fakes and hits <= max_hits
    return NullResult(null=null, ordering=ordering, n_fakes=n_fakes, n_run=n_run, n_hits=hits,
                      rate=hits / max(n_run, 1), upper=d["upper"], passed=passed, stopped_early=stopped,
                      max_hits=max_hits, ref=ref_s, fakes=fakes, hit_rule=hit_rule_name(hit_fn))


def power_check(pipeline: Callable, inject_fn: Callable, n_inject: int = 20, n_fakes: int = 300,
                null: str = "N1", ordering: str = "NO", analysis: str = _data.DEFAULT_ANALYSIS,
                seed=0, workers: int = 1, pass_fraction: float = 0.5, base: Universe | None = None,
                ok_fn: Callable = candidate_ok, **null_kwargs) -> PowerResult:
    """Protocol step 5: fraction of injected-truth universes on which the candidate would pass.

    inject_fn(rng, base) -> Universe with the candidate's structure true (make_injector).
    An injection passes when the pipeline's result on it passes ok_fn (default
    candidate_ok: k <= N - 2, max|pull| < 2, p_global > 0.05) AND the null passes with
    n_fakes fakes built around the injected universe. Requiring criteria 2-3 matters: a
    result that fails them is never a hit, so a null over a family the CKM does not
    affect would otherwise "pass" whenever measurement noise spoils the injected fit.
    Each injection j uses rng = default_rng([*seed, 3, ordering code, j]) and fake seeds
    [*seed, 1000 + j]. Early stop applies inside each null run. ok_fn(summary) -> bool:
    pass candidate_ok_fit_only together with hit_fn=is_hit_fit_only (FIT_ONLY) for the
    informational fit-only rule. With workers > 1 one pool serves every injection.
    """
    base = Universe.real(ordering, analysis) if base is None else replace(base, ordering=ordering)
    results, refs, n_pass, n_ok = [], [], 0, 0
    pool = mp.get_context("spawn").Pool(int(workers)) if workers > 1 else None
    try:
        for j in range(n_inject):
            rng = np.random.default_rng(_seed_list(seed) + [_NULL_CODE["inject"], _ORD_CODE[ordering], j])
            u = inject_fn(rng, base)
            ref = summarize(pipeline(u))
            refs.append(ref)
            if not ok_fn(ref):
                results.append(None)
                continue
            n_ok += 1
            nr = run_null(pipeline, null, ordering, n_fakes, ref=ref, base=u, analysis=analysis,
                          seed=_seed_list(seed) + [1000 + j], workers=workers, pool=pool, **null_kwargs)
            results.append(nr)
            n_pass += nr.passed
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    frac = n_pass / n_inject if n_inject else float("nan")
    return PowerResult(null=null, ordering=ordering, n_inject=n_inject, n_pass=n_pass, fraction=frac,
                       adequate=bool(frac >= pass_fraction), nulls=results, refs=refs, n_candidate_ok=n_ok,
                       hit_rule=hit_rule_name(null_kwargs.get("hit_fn", is_hit)))

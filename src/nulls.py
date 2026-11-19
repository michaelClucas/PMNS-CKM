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
    power_check(pipeline, inject_fn, n_inject=20, n_fakes=300, ...)  -> PowerResult
    make_injector(pred)                                               -> inject_fn
    is_hit(fake, ref), candidate_ok(res), max_hits_to_pass(n), decide(n_hits, n)

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
src.evaluate does), max|pull| < 2, and p_global >= the real candidate's p_global.
The null passes when the one-sided 95% Clopper-Pearson upper limit on the hit rate is
below 1%: at most 4 hits in 1000 fakes, 0 in 300 (max_hits_to_pass). Fakes run in index
order; with early_stop the run stops as soon as the hit count exceeds that bound (only a
fail can be declared early, never a pass).

Power check (protocol step 5)
-----------------------------
inject_fn(rng, base) -> Universe in which the candidate's structure is true (make_injector
builds one from a best-fit prediction plus measurement noise). For each of n_inject
injections the pipeline is run on the injected universe (its result is the reference);
if that result passes criteria 2-3 the null is run with n_fakes fakes built around the
injected universe. PowerResult.fraction is the fraction of injections where both hold;
below 0.5 the null cannot confirm or exclude the family. With 300 fakes the null passes
only with 0 hits.

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
NULLS = ("N1", "N2")
N1_PRIOR = {"lambda": (0.05, 0.5), "A": (0.3, 2.0), "rhobar": (-1.0, 1.0), "etabar": (-1.0, 1.0)}
N2_WIDEN = 3.0
RATE_LIMIT = 0.01
CL = 0.95
_NULL_CODE = {"N1": 1, "N2": 2, "inject": 3}
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

    def scorer(self, use, ordering=None, inputs=()):
        o = ordering or self.ordering
        return _ev.Scorer(use, o, self.analysis, inputs, **self.scorer_kwargs(o))

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


_MAKERS = {"N1": n1_universe, "N2": n2_universe}


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


def is_hit(fake, ref) -> bool:
    """Protocol step 3: k <= N - 2, max|pull| < 2, p_global >= the reference's p_global."""
    if fake is None:
        return False
    pf, pr = float(_get(fake, "p_global")), float(_get(ref, "p_global"))
    if not np.isfinite(pf):
        return False
    return bool(_get(fake, "k") <= _get(fake, "N") - 2 and _get(fake, "max_abs_pull") < 2
                and (pf >= pr or not np.isfinite(pr)))


def candidate_ok(res, p_min=0.05) -> bool:
    """Criteria 2-3 on a result: k <= N - 2, max|pull| < 2, p_global > p_min."""
    if res is None:
        return False
    p = float(_get(res, "p_global"))
    return bool(_get(res, "k") <= _get(res, "N") - 2 and _get(res, "max_abs_pull") < 2
                and np.isfinite(p) and p > p_min)


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
    n_candidate_ok: int = 0  # injections where the pipeline's result passed criteria 2-3


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
             keep_fakes: bool = True) -> NullResult:
    """Run `pipeline` on n_fakes fake universes of one null and ordering; count hits.

    base: the universe the fakes are built from (default: real data in `ordering`).
    ref: the candidate's result on base (default: pipeline(base)).
    workers: processes (1 = in-process). prior: N1 overrides {"lambda": (lo, hi), ...} or
    N2 {"widen": x}. hit_fn(fake_summary, ref_summary) -> bool; summaries are dicts with
    N, k, max_abs_pull, p_global, ... (or the pipeline's own mapping).
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

    if workers <= 1:
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
                      max_hits=max_hits, ref=ref_s, fakes=fakes)


def power_check(pipeline: Callable, inject_fn: Callable, n_inject: int = 20, n_fakes: int = 300,
                null: str = "N1", ordering: str = "NO", analysis: str = _data.DEFAULT_ANALYSIS,
                seed=0, workers: int = 1, pass_fraction: float = 0.5, base: Universe | None = None,
                **null_kwargs) -> PowerResult:
    """Protocol step 5: fraction of injected-truth universes on which the candidate would pass.

    inject_fn(rng, base) -> Universe with the candidate's structure true (make_injector).
    An injection passes when the pipeline's result on it passes criteria 2-3
    (candidate_ok: k <= N - 2, max|pull| < 2, p_global > 0.05) AND the null passes with
    n_fakes fakes built around the injected universe. Requiring criteria 2-3 matters: a
    result that fails them is never a hit, so a null over a family the CKM does not
    affect would otherwise "pass" whenever measurement noise spoils the injected fit.
    Each injection j uses rng = default_rng([*seed, 3, ordering code, j]) and fake seeds
    [*seed, 1000 + j]. Early stop applies inside each null run.
    """
    base = Universe.real(ordering, analysis) if base is None else replace(base, ordering=ordering)
    results, refs, n_pass, n_ok = [], [], 0, 0
    for j in range(n_inject):
        rng = np.random.default_rng(_seed_list(seed) + [_NULL_CODE["inject"], _ORD_CODE[ordering], j])
        u = inject_fn(rng, base)
        ref = summarize(pipeline(u))
        refs.append(ref)
        if not candidate_ok(ref):
            results.append(None)
            continue
        n_ok += 1
        nr = run_null(pipeline, null, ordering, n_fakes, ref=ref, base=u, analysis=analysis,
                      seed=_seed_list(seed) + [1000 + j], workers=workers, **null_kwargs)
        results.append(nr)
        n_pass += nr.passed
    frac = n_pass / n_inject if n_inject else float("nan")
    return PowerResult(null=null, ordering=ordering, n_inject=n_inject, n_pass=n_pass, fraction=frac,
                       adequate=bool(frac >= pass_fraction), nulls=results, refs=refs, n_candidate_ok=n_ok)

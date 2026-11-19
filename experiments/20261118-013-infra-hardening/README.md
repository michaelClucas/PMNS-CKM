# 20261118-013 Infrastructure hardening (closes exp 012 findings)

## Hypothesis

The bookkeeping holes found by the exp 012 skeptic audit (H2, H3, M1, M2, M3, M6, L1, and the M4/M7 null-protocol issues) can be closed in `src/` without changing any χ² value the scorer computes, so that every later candidate is scored, counted and null-tested by one code path.

## Method

Changes in `src/` (all backward compatible except `use=None`, which now raises):

- `src/evaluate.py`
  - Alias atoms (`alias_atoms`, `check_observables`, `AliasError`). CKM global fit: `ckm:lambda` {lambda, s12, Vus, Vcd, Vud, Vcs}, `ckm:A` {A, s23, Vcb, Vts, Vtb}, `ckm:s13` {s13, Vub}; rhobar, etabar, delta_rad, J, Vtd alone. Each `direct_*` is its own atom. Masses: one atom per flavour (a ratio carries two; the three top masses share `t`). Rules, checked on every `Scorer`: no two `use` keys share an atom; no `use` key shares an atom with an `inputs` key; global and direct CKM keys never together in use + inputs; at most 4 global keys in `use`, and each must add an independent direction to the inputs' global keys (rank of the Jacobian d key / d(λ, A, ρ̄, η̄) in σ units, unit rows, singular values > 0.05; aliases sit below 0.007, independent sets above 0.2).
  - `use` is required (`use=None` raises `ValueError` with the reason).
  - One counting convention: `k_discrete` in bits (`stats.discrete_bits(n) = ceil(log2 n)`), integers only; `dof = N − k_continuous`; `p` and `p_global` use that dof; `p_fit = p`. `EvalResult` gains `k_continuous`, `ordering_selected`, and the properties `p_crit3` (= `p_global`), `compression_ok` (k ≤ N − 2), `fit_ok` (max|pull| < 2 and p_global > 0.05).
  - Ordering choice: `ordering="best"` in `evaluate` / `fit_and_evaluate` / `evaluate_baseline` scores both orderings, keeps the lower `chi2_global` and adds 1 to k and k_discrete. `ordering_selected=True` does the same for a caller that picked the ordering itself. No PMNS key in `use` means no ordering choice and no bit.
  - `ckm_delta_rad` is wrapped to within π of the data value (like `dcp_deg`).
  - Leak check: `evaluate_model(model_fn, ..., leak_check=True)`, `fit_and_evaluate(..., leak_check=False)` (runs the probe at the best fit when True), `check_leaks`, `LeakError`.
  - `Scorer` takes keyword-only data overrides (`ckm_records`, `mass_records`, `pmns_fn`, `pmns_centre`, `offset`) and `evaluate`/`fit_and_evaluate` pass them through `scorer_kwargs`, so `src.nulls` scores fakes with the same code. `ckm_from_wolfenstein` gives every global CKM key from exact Wolfenstein parameters.
- `src/stats.py`: `p_value` raises on a non-integer N or k instead of truncating (integral floats accepted) and documents that k is the continuous count; `discrete_bits`.
- `src/baselines.py`: `Baseline.n_options`; k_discrete from `discrete_bits(n_options)`: A4 family 2 → 4 (5 X × 3 factor orders = 15), B (LO patterns) and B1 (solar sum rule) 1 → 3 (5 patterns); the rest unchanged (A1s, C1: 1 option; the others: 2 options).
- `src/leakprobe.py`: the exp 012 probe as a library function, `leak_probe(model_fn, declared_inputs=(), files=..., rtol, n_sigma) -> [LeakFlag(key, record, moved)]`. It also clears every `lru_cache` in loaded `src.*` modules before each shifted call and again at the end, so leaks through cached derived quantities (`rg.quark_masses_mz`, `rg.ckm_at`) are caught and no shifted value survives the probe.
- `src/nulls.py`: the STATE criterion-4 protocol (API below).
- `src/rg.py`: `masses=` / `sm_inputs=` overrides on `quark_masses_mz`, `derived_yukawas_mz`, `build_mz_state`, `ckm_at` (cached per override set; no-argument calls unchanged, `quark_masses_mz.__wrapped__()` and `_derived_yukawas_tuple.__wrapped__()` kept for exp 010's monkeypatch). `ckm_at` results memoised by argument (`cache=False` bypasses, `ckm_at.cache_clear()`), returned as deep copies.
- `src/data.py`: `pmns_profile_1d(ordering, param, values=None, analysis, relative_to)` (vectorized; `values=None` returns the shifted table); `pmns_delta_chi2` caches the shifted 1D/2D tables and finds the 2D cell by direct uniform-grid arithmetic with a searchsorted-equivalent correction, so its output is bitwise unchanged.

Tests: `tests/test_evaluate_audit.py` (the 7 former strict xfails of exp 012, adapted, plus alias/ordering/counting tests), `tests/test_leakprobe.py`, `tests/test_nulls.py`, `tests/test_data_profile.py`, `tests/test_rg_overrides.py`; `tests/test_evaluate.py` updated for the explicit `use` and the new dof.

Re-score: `score_baselines.py` (copy of exp 006's with the new columns and "best" rows), then `diff_vs_006.py`. Timing: `bench.py`.

```
.venv/Scripts/python -m pytest -q
.venv/Scripts/python experiments/20261118-013-infra-hardening/score_baselines.py   # ~7 min, 2 processes
.venv/Scripts/python experiments/20261118-013-infra-hardening/diff_vs_006.py
.venv/Scripts/python experiments/20261118-013-infra-hardening/bench.py
```

**Parameter count:** N/A (infrastructure).

## Result

### Finding → fix → test

| exp 012 finding | fix | tests |
|---|---|---|
| H2 N inflation by aliases | alias atoms; ≤ 4 independent CKM global keys (Jacobian rank); global and direct never mixed | `test_evaluate_audit.py::test_F1_alias_observables_rejected`, `::test_F1_mass_ratio_alias_rejected`, `::test_ckm_global_rules`, `::test_global_and_direct_never_mixed`, `::test_mass_alias_rules`, `::test_every_baseline_passes_alias_rules` |
| H3 leakage undetectable | `src/leakprobe.py`; `evaluate_model(leak_check=True)`, `fit_and_evaluate(leak_check=...)`, `LeakError` | `test_evaluate_audit.py::test_F3_evaluate_detects_leak`, `::test_fit_and_evaluate_leak_check`; `test_leakprobe.py` (leaky model flagged, QLC clean with its input, undeclared read flagged, alias read flagged, leak through an `rg` lru cache flagged and caches restored) |
| M1 overlap by name only | input/prediction check on alias atoms and on CKM rank | `test_evaluate_audit.py::test_F2_input_alias_rejected`, `::test_scorer_rechecks_inputs_given_at_call` |
| M6 `use=None` | `use` required, ValueError explains why | `test_evaluate_audit.py::test_F4_use_none_does_not_count_echoed_inputs` |
| M2 free ordering choice | `p_crit3 = p_global`; `ordering="best"` / `ordering_selected=True` add 1 bit | `test_evaluate_audit.py::test_best_ordering_adds_one_bit_and_uses_p_global`, `::test_ordering_selected_flag_and_pred_best` |
| M3 three counting conventions | k_d = ceil(log2 #options) (`stats.discrete_bits`, `Baseline.n_options`); dof = N − k_cont for p and p_global; `p_value` rejects non-integer k | `test_evaluate_audit.py::test_F5_k_discrete_counts_bits`, `::test_discrete_bits_and_integer_counts`, `::test_registry_k_discrete_from_options`, `::test_p_uses_continuous_dof_and_criteria_flags`; `test_evaluate.py::test_dof_k_and_p_fit` (updated) |
| L1 CKM δ not wrapped | wrap to within π of the data value | `test_evaluate_audit.py::test_F9_ckm_delta_wraps` |
| M4 Gaussian-scored fakes | `src/nulls.py` scores fakes with NuFIT tables translated to the fake centre (δ periodic, s23 octant mirror) | `test_nulls.py::test_translated_tables` |
| M7 / protocol | `src/nulls.py`: N1, N2, hit rule, Clopper-Pearson rule (≤ 4/1000, 0/300), early stop only to fail, `power_check` | `test_nulls.py` (11 tests) |

The 7 former strict xfails are the first seven tests of `tests/test_evaluate_audit.py`. The exp 012 file was not edited; run directly, its xfail markers would now report XPASS(strict) failures.

Not addressed (out of scope): L2 (`stats.Objective` default `pmns_mode="sum1d"`), L3 (Δm² keys in `evaluate`), L4.

### Baseline re-score (`results.csv`, `run.log`, `diffs_vs_006.csv`, `diff_summary.log`)

148 fits, 442 s, 2 processes. Against exp 006, 580 cells changed. No χ², χ²_global, max|pull|, fitted parameter, N, k_c or input-sensitivity value changed. What changed:

- k_d (and k): A4a/A4b/A4c × 5 patterns 2 → 4 (k 4 → 6); B-* and B1-* 1 → 3 (B: k 1 → 3; B1: k 4 → 6).
- dof = N − k_cont, so p and p_global changed for 30 baselines (A1±, A2, A3, A3k, all A4, B1-*, B2-TM1/TM2, C2, C3, E1); dof also changed for B-* (whose p was already 0). p_fit is unchanged everywhere (it already used N − k_cont). Several p's that were nan (dof ≤ 0) are now finite.
- Key values (IC23_woSKatm, p_global old → new): TM1 NO 0.024 → 0.078, IO 0.069 → 0.191 (k = 3, N = 4: still fails compression). A4b-BM (lead L1) NO nan → 0.053, IO nan → 0.059 (k = 6 > N − 2). B1-TBM NO nan → 0.063, IO nan → 0.150. C3 Koide nan → 0.667 (N = 1, k = 1).
- New `best:<ordering>` rows (ordering picked by χ²_global, +1 bit): same p, k one larger.
- **No row passes criteria 2 and 3 together** (`compression_ok and fit_ok` false in all 222 rows). STATE's "no known relation has k ≤ N−2 with acceptable p" stands; its TM1 p-values (0.024 / 0.18) read p_global 0.078 (NO) / 0.191 (IO) under the new convention.

### `src/nulls.py` API

```
Universe(kind, ordering, analysis, ckm, masses, pmns_centre, truth, index, s23_mirror)
  .real(ordering, analysis); .load_ckm(); .load_masses(); .load_pmns(o); .context()
  .pmns_delta_chi2(o, ...); .evaluate(pred, use, ordering=None, **kw); .fit_and_evaluate(...)
pipeline(universe) -> EvalResult | mapping with N, k, max_abs_pull, p_global | None
run_null(pipeline, null="N1"|"N2", ordering, n_fakes=1000, ref=None, base=None, seed=0,
         workers=1, early_stop=True, hit_fn=is_hit, prior=None) -> NullResult
power_check(pipeline, inject_fn, n_inject=20, n_fakes=300, null="N1", ordering, seed, workers)
         -> PowerResult(fraction, adequate, n_candidate_ok, nulls, refs)
make_injector(pred, noise=True) -> inject_fn(rng, base)
is_hit, candidate_ok, max_hits_to_pass, decide, n1_universe, n2_universe, pmns_noise
```

A power-check injection passes only if the pipeline's result on the injected universe passes criteria 2-3 and the null passes. Without the first condition a CKM-blind family "passed" 30% of injections (when noise spoiled its own fit, no fake could be a hit). Injection noise is drawn from the NuFIT likelihood shape (1D s12, s13; 2D s23-δ) and placed so the truth lands at a likelihood-distributed point of the translated table (Asimov χ² 0.012; with noise mean Δχ²(truth) 5.2 for 4 parameters; 80% of injections have all pulls < 2 and p > 0.05).

Toy results (`tests/test_nulls.py`):
- Planted CKM-dependent relation (2 variants, 1 bit, N = 4, k = 1), truth injected: N1 with 1000 fakes passes (0 hits). Power check (20 × 300, N1): 18/20 injected fits pass criteria 2-3, 12/20 nulls have 0 hits, fraction 0.60, adequate.
- CKM-independent family made true by injection: N1 fails at the 5th fake (5/5 hits, early stop); power check fraction 0.00.
- N2 with 1 and 2 workers gives identical per-fake results.

### Performance (`bench.py`, `bench.log`; machine shared with other agents, absolute numbers noisy)

| item | before | after |
|---|---|---|
| `pmns_delta_chi2` joint call (2D + two 1D) | 96 µs | 50 µs (x1.9), bitwise identical on 12,000 points incl. grid nodes |
| 1D profile on 1e5 points | 1.35 s (scalar loop) | 12.8 ms `pmns_profile_1d` (x106), bitwise identical |
| `ckm_at(2e16)` repeat call | 289 ms | 39 µs (memoised) |
| `ckm_at` first call in a process | 0.2-0.7 s (y_t solve; STATE's "~10 s" not reproduced) | unchanged |
| toy `run_null`, 1000 N1 fakes, 1 process | n/a | 2.1 s |

### Test suite

`.venv/Scripts/python -m pytest -q`: 458 passed (401 before, 57 new).

### Notes for other experiments

- `evaluate(..., use=None)` raises. All `src/` and `tests/` callers updated.
- `EvalResult.p` and `p_global` now use dof = N − k_cont. Code that reads them (e.g. exp 015 `finalize.py`) gets the new convention; values with k_discrete > 0 go up.
- Non-integer `k`, `k_discrete`, or `p_value` k raise instead of truncating.
- `Scorer(use, ordering, analysis)` still works; the 4th positional argument is now `inputs`.

## Verdict

Hypothesis supported. Every exp 012 finding in scope has a fix in `src/` and a passing test, and the scorer's χ² values are unchanged: in the re-score only parameter counts, dof and the p-values that depend on them moved. No baseline passes criteria 2 and 3 together under the new rules, so no past negative verdict changes. Criterion-4 nulls now have one implementation (`src/nulls.py`) whose power check reports no power for a CKM-blind family. Remaining: L2-L4 from exp 012; the leak probe still cannot see hardcoded numbers or direct CSV reads.

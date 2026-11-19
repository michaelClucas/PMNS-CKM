# 20261119-019 infra-scoring

## Hypothesis

The open infrastructure gaps listed in STATE ("Known infrastructure gaps (open)") and in the src-gap sections of exps 014-018 can be closed in `src/` without changing any existing call signature or any existing score. Afterwards criterion 4 has a documented informational rule for k > N − 2 and a mass-side null, and the official scorer covers same-scale mass ratios, Δm², a theory floor and GUT-scale predictions. Exps 014-017 scored all of these outside the scorer.

## Method

Edits to `src/nulls.py`, `src/evaluate.py` and `src/rg.py`, each with tests on toy pipelines or analytic checks: `tests/test_nulls_019.py`, `tests/test_evaluate_019.py`, `tests/test_rg_019.py`. Backward compatibility was checked three ways:
- the full existing suite;
- exp 017's local tests;
- the old-vs-new comparison below.

Exp 012's local test file has the same 12 failures before and after this change. They come from rules adopted in exp 013 (`use=None` refused, alias rules); STATE notes that file is superseded by `tests/test_evaluate_audit.py`.

Free parameters: N/A (infrastructure).

## Result: gap → fix → test

| Gap (source) | Fix | Test |
|---|---|---|
| `power_check` hardcodes `candidate_ok` (018 #1) | `power_check(..., ok_fn=candidate_ok)` | `test_power_check_ok_fn_is_used` |
| Criterion 4 vacuous for k > N − 2 (018 #1) | `is_hit_fit_only`, `candidate_ok_fit_only`, `FIT_ONLY = {hit_fn, ok_fn}`. Labelled INFORMATIONAL in the docstring and in `NullResult.hit_rule` / `PowerResult.hit_rule` (`"fit_only (informational; criterion 4 needs a compressed result)"`). Criterion 4 proper still needs compression. | `test_fit_only_power_detects_planted_relation_with_k_above_n_minus_2`: planted CKM-PMNS family counted with k = 3, N = 4. Strict power 0 by construction. Fit-only power adequate, with every injection whose fit passes also passing N1. Flat family has fit-only power 0. |
| `is_hit` compares p_global exactly (018 #2) | `p_fake >= p_ref (1 - P_RTOL)`, `P_RTOL = 1e-5` (documented), `rtol=` argument on both hit rules | `test_p_tolerance_and_fit_only_rule` (1e-7 tie is a hit; 1e-3 is not; rtol=0 restores the exact rule) |
| No mass-side null (018 #3) | N3 null, described below | `test_n3_changes_mass_inputs`, `test_n3_heavy_ratios_optional_and_clipped`, `test_n3_null_passes_mass_relation_and_fails_flat_family` |
| `power_check` spawns a pool per injection (018 #4) | one spawn pool for the whole power check; `run_null(pool=...)` feeds it blocks of 4 × workers and checks early stop in index order | `test_shared_pool_matches_serial` (per-fake hits identical to serial). `bench_pool.py`: 6 injections × 24 fakes, 2 workers: 19.1 s shared vs 55.1 s with a pool per injection |
| `src.evaluate` cannot score same-scale mass ratios (016, 017, STATE) | `mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau` at M_Z (SM, MSbar, from `src.rg.mass_ratios`). Scored as one correlated Gaussian block with the linearly propagated covariance. Alias atoms `{mass:x, mass:y}` block any raw-mass or PDG-ratio key or input of the same flavour; mr keys may share flavours with each other. | `test_mass_ratio_block_chi2_and_pulls` (χ² = rᵀC⁻¹r, 1σ pull), `test_mass_ratio_alias_rules`, `test_mass_ratio_targets_follow_mass_overrides` |
| No Δm² or Δm² ratio (015, 016, 017, exp 012 L2-L4) | `dm21sq_eV2`, `dm3l_sq_eV2` (NuFIT 1D profiles DMS, DMA) and `dm_ratio` = Δm²21/\|Δm²3ℓ\| (profile min over Δm²3ℓ of DMS(r\|x\|) + DMA(x)). Approximation, documented: the splittings are taken as independent of each other and of the angles. They are not translated in N2 fakes. Predicting a splitting charges the ordering offset and the ordering bit under "best". | `test_splitting_observables` (equals `pmns_delta_chi2`, signed pulls, offset; ratio equals a 400k-point brute-force profile to 2e-4), `test_splitting_alias_ordering_bit_and_universes` |
| No theory floor on masses (016) | `mass_floor=` (float, or mapping with `"default"`) on `Scorer`, `evaluate`, `evaluate_model`, `fit_and_evaluate`. Added in quadrature as a relative error to load_masses records and mr keys. Reported in `EvalResult.mass_floor` (default 0). | `test_mass_floor_on_ratios_and_raw_masses` |
| Cannot score GUT-scale predictions (014, 015, 016, STATE) | `scale={"mu", "model", "tan_beta", "m_susy", "m_lightest", "majorana", "sys_frac"}`: the prediction is moved to M_Z by the data's own `src.rg` running, additive for angles and phases and multiplicative for J, mr, Δm². It is then scored at low scale. Running systematic is reported in `EvalResult.scale_systematic`: `sys_frac` (5%) of the running, plus for MSSM half the change for M_SUSY ×2 and ÷2. Unsupported keys raise. | `test_data_run_up_scores_like_data_at_low_scale` (SM and MSSM tanβ = 10; Δχ² < 1e-6), `test_high_scale_prediction_off_by_one_sigma_scores_one_sigma` (exact run of a +1σ CKM input or m_u gives pull 1 / the same χ² to 1e-4), `test_high_scale_rejects_unsupported_keys_and_bad_specs`, `test_mssm_systematic_includes_threshold_variation_and_fit_passes_options` |
| `ckm_at` builds κ from PMNS data: 1e-12 crosstalk (018 #5, STATE) | `build_mz_state(neutrinos=False)` (κ = 0, no PMNS or splitting read) used by `ckm_at`, the y_t solve in `derived_yukawas_mz`, and `mass_ratios` | `test_quark_only_model_has_no_pmns_flags`: leak probe over pmns.json + masses.json, 0 PMNS/splitting flags (quark inputs still flagged). Control (not a test): the old path gives 4 PMNS flags (θ12, θ13, θ23, δ). `test_ckm_at_equals_old_path_with_neutrinos` (1e-9). |
| `pmns_at` assumes m_lightest = 0 (016, STATE) | m_lightest ≥ 0 validated. New overrides `pmns=`, `splittings=`, `ckm=`, `masses=`, `sm_inputs=`. New outputs `dm21sq_eV2`, `dm3l_sq_eV2`, `sum_m`, `m_lightest` at μ. Memoised. The GUT path passes `m_lightest` and Majorana phases through. The old code already handled m_lightest ≠ 0 numerically; what was missing was the overrides, the validation, and the plumbing into scoring. | `test_pmns_at_m_lightest` (NO/IO × {0, 0.015, 0.08} eV reproduce NuFIT at M_Z), `test_pmns_at_m_lightest_changes_mssm_running_and_validates` |
| `derived_yukawas_mz` has no uncertainties (017, STATE) | `rg.propagate(fn)`: linear propagation of the 9 mass records and α_s(M_Z), central differences with step max(σ, 1e-6\|x\|), symmetrised errors, inputs independent. Built on it: `derived_yukawas_mz_unc()` and `mass_ratios(mu, ..., errors=True)` (value, err, cov, cov_log, jac). Not included: the error of the QED pole → MSbar conversion (~0.2%) and higher-order running. Those go in `mass_floor`. | `test_propagate_ratio_is_exact_for_a_simple_function`, `test_yukawa_and_ratio_uncertainties` |

### N3 null (`src.nulls`)

- Real CKM and PMNS. The seven mass ratios get independent log-uniform factors 10^U[−w, w], with w = 1 decade for the light-over-heavy ratios. By default only u/c, d/s, s/b, e/μ, μ/τ are moved (`N3_PRIOR`).
- c/t and b/τ are optional, with w = 0.2 and clipping. They move m_c and m_b, which are also `src.rg`'s QCD decoupling scales: m_b is kept in [2.5, 50] GeV and m_c in [1.0, 0.8 m_b].
- Anchors are t and τ. Lighter masses follow their chain, so each realised M_Z ratio moves by its own factor:
  - exact (to 1e-7) for the light quarks;
  - to 2% for the leptons, where the one-loop QED log term is not linear in M.
- Fake records keep the real relative errors. PDG ratio records move consistently, and top records never move.
- Pipelines read the fakes through `Universe.rg_masses()` → `src.rg` `masses=`. The scorer gets them through `mass_records`, and mr targets are recomputed. Under overrides the mr covariance keeps the real log covariance.

### Reference numbers (`ref_numbers.py` → `ref_numbers.json`)

mr observables at M_Z (relative 1σ from propagation):

| key | value | rel. err |
|---|---|---|
| mr_u_c | 1.975e-3 | 3.3% |
| mr_c_t | 3.722e-3 | 1.5% |
| mr_d_s | 5.059e-2 | 1.7% |
| mr_s_b | 1.872e-2 | 0.88% |
| mr_e_mu | 4.744e-3 | 2.2e-8 |
| mr_mu_tau | 5.887e-2 | 4.9e-5 |
| mr_b_tau | 1.640 | 0.53% |

- α_s(M_Z) dominates the m_c, m_b running and correlates c/t, s/b and b/τ (log correlation c/t-b/τ 0.89).
- The lepton ratios are at ppb-ppm: anything predicting them needs `mass_floor`.

Running from M_Z to 2×10^16 GeV:
- SM: m_b/m_τ factor 2.61 (sys 4.8%), s23^q shift +0.0056 (sys 0.0003 = 0.4σ of the CKM error). PMNS shifts are ≤ 2e-5.
- MSSM tanβ = 10: m_b/m_τ factor 2.10 (sys 3.7%), s12² shift −4.3e-4, δ shift −0.011°.

dm_ratio Δχ² at the record central values is 5e-4 (NO) and 4e-3 (IO), from the table minima.

## Changed numbers in existing results

- `src.rg.derived_yukawas_mz`: bit-identical (max rel diff 0).
- `ckm_at` angles/J: ≤ 7e-13 relative. `ckm_at` ratios: ≤ 6.5e-12 relative (SM and MSSM, 2e16 GeV).
- `pmns_at`: identical.
- No printed number in any experiment moves at these levels.
- `src.evaluate`: identical χ² and pulls for all existing keys (the term sum is unchanged; existing suite passes). `EvalResult` has four new fields with defaults (`mass_floor`, `scale`, `scale_shift`, `scale_systematic`), so `row()` has two new CSV columns (`mass_floor`, `scale`).
- `src.nulls.is_hit` tolerance: recounting exp 018's stored per-fake summaries with the fit-only rule plus tolerance (`recount_018.py`):
  - **T1 under N1**: fixed-length rate NO goes from 490/1000 to **967/1000** hits. The early-stopped null runs reach 5 hits sooner (all 10 / 7 fakes are hits). These were the RG-noise ties exp 018 described: T1 does not depend on the CKM, so every N1 fake is the same relation.
  - All other exp 018 null and rate files: unchanged.
  - The power files keep only the first fakes, so they cannot be fully recounted. The tolerance can only add hits, so power can only fall. Every power check was already 0 except F2-LO/N2/NO at 1/20 (0.05), and all were already below 0.5.
  - Verdicts unchanged: every one of these nulls already failed.

## Still open (not in this experiment's scope)

- tanβ-enhanced SUSY threshold corrections and two-loop MSSM Yukawa running in `src.rg`. These are documented as not covered by `scale_systematic`; at large tanβ they dominate m_b/m_τ.
- Exp 012 L2-L4 leftovers: `stats.Objective` default `sum1d`; undefined δ scored as 0.
- The fit-only rule does not make criterion 4 reachable for k > N − 2. It only gives an informative null. STATE should say so.

## Verdict

Every gap assigned to this experiment is closed and tested; the full suite (`.venv/Scripts/python -m pytest -q`) passes, 488 tests. Existing scores are unchanged. The one changed exp 018 number is the T1/N1 hit rate, which rises because of the noise-tie tolerance, and no verdict changes.

## Files / rerun

From the repo root:
```
.venv/Scripts/python -m pytest -q tests/test_rg_019.py tests/test_evaluate_019.py tests/test_nulls_019.py
.venv/Scripts/python experiments/20261119-019-infra-scoring/ref_numbers.py     # ref_numbers.json, ~20 s
.venv/Scripts/python experiments/20261119-019-infra-scoring/recount_018.py     # exp 018 hits with the tolerance
.venv/Scripts/python experiments/20261119-019-infra-scoring/bench_pool.py      # ~75 s, 2 cores
```
No data files were added.

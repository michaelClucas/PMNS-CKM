# 20261119-024 Infra: R double ratio, CKM-information metric, region helper, FN prior-minimum discrepancy

## Hypothesis

Infrastructure task (no physics hypothesis). Close four open STATE gaps: R and the exp 021 information metric are not in `src`, there is no reusable fixed-τ region constraint (exp 020), and the FN prior-predictive minimum χ² disagrees between exp 014 (216) and exp 021 (79).

## Method and changes

All changes are additive. Any code path without the new key or module is unchanged.

### 1. Clebsch double ratio `R_clebsch` in `src.evaluate`

- New observable key `R_clebsch` = (m_e/m_μ)/(m_d/m_s) at M_Z = `mr_e_mu / mr_d_s`. It is defined in `DERIVED_RATIO_KEYS` as log-coefficients over the `mr_*` ratios.
- It is scored inside the existing correlated mass-ratio block. Its log variance and its covariances with any `mr_*` key come from linear propagation of `rg.mass_ratios()["cov_log"]`.
- Alias atoms are {e, μ, d, s}, so it collides with raw masses and PDG ratio records of those flavours, in `use` or in `inputs`.
- It may be scored with other `mr_*` keys. The block's log-coefficient rows must be independent, so `R_clebsch` + `mr_e_mu` + `mr_d_s` raises `AliasError` (that would count the same information twice).
- `mass_floor` applies to the R key itself, added in quadrature to its log variance like any `mr_*` key. A per-key mapping works too.
- `mass_records` overrides (src.nulls universes) recompute R from the overridden masses.
- Scale translation is multiplicative, with factor = factor(mr_e_mu)/factor(mr_d_s).
- Numbers (`src.evaluate`, real data):
  - R = 0.093772. This equals exp 021's 0.0937717 to 1e-12.
  - PDG error ± 0.00157 (1.67%, as in exp 021).
  - `mass_floor=0.02` gives ± 0.00244.
  - Running factor M_Z → 2e16: 1.00023 (SM), 0.99992 (MSSM tanβ 10).
- Convention difference: exp 021 quoted ± 0.0027 by replacing the PDG errors of m_d and m_s with 2% each (√2 · 2%). The src convention adds the floor in quadrature to the propagated error. To match exp 021's spirit, use `mass_floor={"R_clebsch": 0.02*sqrt(2)}` (± 0.0031). This is documented in the module docstring.

### 2. `src/infogain.py` (from exp 021 `bound.py`)

API:
- `pmns_data_bits(ordering)`: information budget vs Haar.
- `null_info(real_chi2, fake_chi2s)`: B_ckm, S_rank, rate with a CP interval.
- `predictive_gain(post, prior, x_obs, sigma, periodic, sigma_scale)`: G_pred plus min χ².
- `kl_shift(post, prior)`.
- `i_gauss(r2)`.
- `sharing_net_bits(...)`.
- Helpers `logmeanexp` and `cp_interval`.

Definitions are in the module docstring. `tests/test_infogain.py` reproduces exp 021's `bound.json` to rel 1e-12 from the committed exp 007/009/014/018 files:
- data budget;
- all 4 F1 and 24 F2 null entries;
- the 4 exp 018 rate runs;
- every I_gauss value;
- modular sharing.

It also reproduces MN_ours joint-4 G_pred (σ×1, σ×3), min χ² and s13² KL_shift to 1e-9 when the gitignored exp 014 PT samples are present (skipped otherwise). Analytic tests cover G_pred (Gaussian vs uniform), periodic wrapping and KL.

### 3. `src/region.py` (fixed-parameter region)

- `Region.from_grid(points, chi2, level, names, step)` takes complex τ or (n, d) arrays.
- Methods: `save`/`load` (JSON), `contains`, `dchi2_at` (grid-cell lookup), `bounds()` (for `param_bounds`), `restrict(level)`, `representatives(k)` (exp 020's farthest-point sample).
- `profile_over_region(fit_fn, region, points, add_dchi2=True)` minimizes over region points the χ² of the new data, and by default adds the region's Δχ² so the joint χ² charges for using the region's edge.
- Counting note (docstring): a parameter fixed by an independent dataset is not in k. Its Δχ² belongs in the joint χ².
- `tests/test_region.py`: 5 tests.

### 4. FN prior-predictive minimum χ²: 216 (exp 014) vs 79 (exp 021)

**Both are correct. They are different random draws, and the statistic is too noisy to compare.**

- Exp 014's `analyze_heldout.py` is run once per model, with a fresh `default_rng(5)`, so FN's 200k-sample prior uses the first draw of seed 5.
- Exp 021's `bound.py` creates one `default_rng(5)` outside its case loop and draws MN_ours, MN_drrel, then FN_ours. FN therefore gets the third 200k block of the stream.
- The box, the PMNS χ²₅ definition (wrap on δ) and the NO filter are identical in both scripts.
- `fn_prior_min.py` → `fn_prior_min.json` checks this directly:
  - first draw of seed 5: 216.48 (= exp 014's file);
  - third draw after two MN-box draws: 79.40 (= exp 021).
- The same cross-check works for MN:
  - MN_ours (first draw in both scripts) agrees: 189.96 vs 190.
  - MN_drrel (second draw in bound.py) differs: 289 vs 190.
- The minimum over 200k draws of FN's very wide sinh-coordinate prior is an extreme-value statistic. Ten other seeds (100-109) give 13.4 to 276 (median 123).
- Consequences:
  - Exp 014's prior "min χ²₅" values are a property of one draw, not of the prior.
  - Its `prior_frac_lt_50 = 0` is draw-specific as well: seed 100 reaches 13.4.
  - No conclusion changes. The posterior minimum is ≥ 3000 either way, and every G_pred sign is unchanged.
- Exp 014 is not wrong, so no erratum was added to its README.
- Future scripts should take a fresh `default_rng(seed)` per case (or `SeedSequence.spawn`) so that per-case numbers do not depend on case order.

### 5. `Objective` default and undefined δ (exp 012 L2, L4): not fixed, reported

- **L2:** the `stats.Objective` default stays `sum1d`. Flipping it to `joint` would move no committed score: the only direct caller (exp 005) passes the mode explicitly, and `src.evaluate` passes `joint`. It could silently change the numbers of exps 022/023, which are running against `src` now. Instead, `Objective` now raises a `UserWarning` when the mode is left implicit and both s23sq and dcp_deg are scored. No number moves. Recommendation: make it `joint` (or required) once 022/023 are committed.
- **L4:** not fixed. `src.baselines` already omits `dcp_deg` when δ is undefined. In `evaluate`, profiling δ when s13 → 0 would change minimizer landscapes in `fit_and_evaluate` and could move fitted scores. Callers that build predictions from `extract_angles` must still drop δ themselves when s13 ≈ 0.

## Verification

- `.venv/Scripts/python -m pytest -q`: 512 passed, 0 warnings (`pytest.log`; was 488 after exp 019, + 24 new: 10 in test_evaluate_024, 9 in test_infogain, 5 in test_region). `tests/test_stats.py` now passes `pmns_mode="sum1d"` explicitly in the one test that relied on the default.
- `check_ref_numbers.py` reruns exp 019's `ref_numbers.py` (in a temp copy, so exp 019 stays untouched) and compares every one of the numbers in the committed `ref_numbers.json`. Max abs and rel difference: 0.0 (`ref_check.json`).
- The pre-exp-024 `mr_*` covariance path is kept verbatim when no derived key is used. A test checks it is bit-identical.

## Parameter count

N/A (infrastructure).

## Result / verdict

- R, the information metric and the region helper are in `src` with tests.
- The FN discrepancy is resolved: it came from RNG stream order plus an extreme-value statistic. Neither number is an error.
- L2 is mitigated with a warning; L4 is open.

## Files

- `fn_prior_min.py` → `fn_prior_min.json`: FN discrepancy check. ~25 s, single process, needs no PT samples.
- `check_ref_numbers.py` → `ref_check.json`: exp 019 reference regression.
- `pytest.log`: full suite output.
- Source changes:
  - `src/evaluate.py`: R_clebsch, `DERIVED_RATIO_KEYS`, `MR_BLOCK_KEYS`, `mr_coefficients`, `mr_block_values`.
  - `src/stats.py`: implicit-sum1d warning.
  - New: `src/infogain.py`, `src/region.py`.
  - New tests: `tests/test_evaluate_024.py`, `tests/test_infogain.py`, `tests/test_region.py`.

Rerun from the repo root:
- `.venv/Scripts/python experiments/20261119-024-infra-metrics/check_ref_numbers.py`
- `cd experiments/20261119-024-infra-metrics && ../../.venv/Scripts/python fn_prior_min.py`

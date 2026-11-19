# 20261119-026 Infra: SUSY threshold corrections, global optimizers, Objective default, null_info

## Hypothesis

Infrastructure task (no physics hypothesis). Close the open STATE gaps from exps 022-024: `src.rg` has no tanβ-enhanced SUSY thresholds (022 dropped mr_s_b and mr_b_tau, 023 applied η̄_b by hand), `fit_and_evaluate` is Nelder-Mead only and failed on exp 022's 9-D texture, `Objective` defaults to `sum1d`, and `null_info` is not a standard `src.nulls` output.

## Changes

All additive. With default arguments no committed score moves (see Verification).

### 1. tanβ-enhanced SUSY thresholds (`src/rg.py`, `src/evaluate.py`)

Source: Antusch-Maurer, arXiv:1306.6879, Eqs. (4)-(10), (14)-(16), with leading-order consequences in Eqs. (17)-(24). They follow Blazek-Raby-Pokorski hep-ph/9504364 and Antusch-Spinrath arXiv:0804.0717. Read from the arXiv PDF.

- New optional `thresholds={"eta_b", "eta_q", "eta_l"}` in `rg.run`, `run_mssm`, `ckm_at`, `pmns_at` and `mass_ratios`. These are the paper's barred parameters η̄_b, η̄_q, η̄_ℓ, and `tan_beta` is read as tan β̄ (the paper's Eq. (4) absorbs η'_ℓ into β). The default is None or all zero. In that case the code path is skipped and the results are bit-identical.
- Matching at M_SUSY, applied SM → MSSM going up and inverted going down:
  - Y_d^MSSM = U_u diag(1/(1+η̄_q), 1/(1+η̄_q), 1/(1+η̄_b)) U_u† Y_d^SM / cos β̄
  - Y_e^MSSM = U_e diag(1/(1+η̄_ℓ), 1/(1+η̄_ℓ), 1) U_e† Y_e^SM / cos β̄
  - U_u and U_e are the left rotations of Y_u and Y_e at M_SUSY. This is the paper's "basis where Y_u is diagonal", written covariantly.
  - Y_u and κ are at tree level.
- Consequences (paper Eq. 22): θ_i3^SM = (1+η̄_q)/(1+η̄_b) θ_i3^MSSM, and θ12 and δ are unchanged. Exp 023's linear treatment uses the same direction.
- Above M_SUSY, `observables()` reports the MSSM eigenvalues × sin β or cos β with no η factor undone. These are the quantities a GUT model predicts.
- `η ≤ −1`, unknown keys, and thresholds with model "SM" all raise.
- `src.evaluate` `scale={...}` now also accepts `eta_b`, `eta_q`, `eta_l` (MSSM only).
  - The data are run up with the threshold matching, so every translation factor includes it.
  - For `ckm_s13` and `ckm_s23` the translation becomes x_low = T·x_mu − [T·X_η(mu) − X(M_Z)] with T = (1+η̄_q)/(1+η̄_b). `EvalResult.scale_slope` / `Scorer.scale_slope` hold T for these keys.
  - Exp 019's purely additive rule has slope 1, which is wrong by a factor T at large η̄_b. At η = 0 the new rule is identical to the old one.
  - `EvalResult.scale` lists the η keys only when they are nonzero.
- **Parameter count:** η̄_b, η̄_q and η̄_ℓ are model parameters (they depend on the SUSY spectrum). Any fit that fits them, or picks a value after looking at the data (as 1705.01555 does with η̄_b = −0.8), counts each one in k. This is stated in both module docstrings.

### 2. Global / gradient optimizers in `fit_and_evaluate`

- New keyword-only arguments:
  - `method`: `"nelder-mead"` (the default, unchanged), `"lbfgsb"` or `"de"`.
  - `optimizer_options`.
- `"lbfgsb"` runs L-BFGS-B with finite-difference gradients from the same Sobol starts plus `extra_starts`.
- `"de"` runs `scipy.optimize.differential_evolution` over the box: Sobol init, seeded, with its own L-BFGS-B polish. `maxfev` is an evaluation budget, enforced through a callback. Each extra start gets a Nelder-Mead run.
- Both methods finish with the existing Nelder-Mead polish. `ordering="best"` passes the method through.
- **Landscape fix (the main finding).** Outside the NuFIT tables, `Scorer.chi2_soft` returned CHI2_CAP plus a PMNS Gaussian and dropped every finite term (CKM, mass ratios). A point with PMNS just outside the tables and an arbitrarily bad quark sector therefore scored about 1e4. Entering the tables with a bad quark sector cost 1e5-1e6. Every minimizer was pushed to stay outside the tables. This is why exp 022's DE stalled at 13777 (= CHI2_CAP + PMNS distance), and both new methods stalled there too on the first try (table below, "old landscape").
  - New `chi2_soft(pred, include_finite=True)` adds the finite terms.
  - `"lbfgsb"` and `"de"` use it. `"nelder-mead"` keeps the old landscape, so default fits are bit-identical.
  - Inside the tables the two are identical.

### 3. `Objective` default `pmns_mode` → `"joint"` (`src/stats.py`)

- The exp 024 warning is removed.
- No committed caller relied on the default:
  - exp 005 passes `pmns_mode` explicitly;
  - `src.evaluate` passes `"joint"`;
  - `tests/test_stats.py` passes it in every PMNS test.
- `tests/test_evaluate_024.py`'s warning test is replaced by a no-warning / default-joint test.

### 4. `null_info` as a standard `run_null` output (`src/nulls.py`)

- `NullResult.info` is a new last field, default None.
- It holds `src.infogain.null_info(ref chi2_global, fake chi2_global list)`: B_ckm, S_rank, chi² rate with its CP interval, and the fake median. `chi2` is used when `chi2_global` is absent.
- The fake χ² values are collected even with `keep_fakes=False`.
- `null_info_of(ref_summary, fake_chi2s)` is exposed.
- Non-finite fakes are dropped (null_info's existing exp 021 convention), so `info["n"]` can be below `n_run`.
- `NullResult.row()` is unchanged.

### 5. Undefined δ (exp 012 L4): not done

Profiling δ when s13 → 0 changes the `fit_and_evaluate` landscape for any model that passes through s13 ≈ 0, so it can move fitted scores. It was skipped, as the brief allows. Callers still drop `dcp_deg` themselves when s13 ≈ 0.

## Validation

### Thresholds (`validate_thresholds.py` → `thresholds.json`, `validate_thresholds.log`)

**Bit identity and round trip**
- η = 0 is bit-identical for `ckm_at`, `pmns_at` and `mass_ratios`. The `Scorer` shifts and χ² are identical too (tests).
- Running up to 2e16 and back down with (η̄_b, η̄_q, η̄_ℓ) = (−0.4, 0.1, 0.05) at tanβ 20: max relative deviation 1.3e-12.

**Leading-order relations (Eqs. 17-24), just above M_SUSY, tanβ 20**
- Deviations are ≤ 6e-4 for y_b, y_s and y_d, and ≤ 4e-4 for θ13, θ23 and θ12. These are O(θ²) mixing effects.
- y_μ, y_e and y_τ agree to 1e-12.

**Antusch-Maurer Fig. 6b, y_τ/y_b at M_GUT** (M_SUSY 1 TeV; tanβ̄ 15, where the contours are nearly flat; contours read off the figure by eye, about ±0.02 in η̄_b):

| contour | η̄_b published (read-off) | η̄_b ours | diff |
|---|---|---|---|
| 0.75 | −0.41 | −0.398 | +0.012 |
| 1.0 | −0.22 | −0.209 | +0.011 |
| 1.25 | −0.02 | −0.018 | +0.002 |
| 1.5 | +0.14 | +0.175 | +0.035 |
| 1.75 | +0.33 | +0.368 | +0.038 |
| 2.0 | +0.50 | +0.562 | +0.062 |

- At η̄_b = 0 we get y_τ/y_b = 1.272 (tanβ 20). The published contours imply about 1.27-1.29.
- In the ratio itself the differences are ≤ 1% for η̄_b ≤ 0 and 2-4% for η̄_b ≥ +0.14. That is the known accuracy of our 1-loop MSbar MSSM Yukawa running against their 2-loop DR-bar REAP (STATE: MSSM θ23^q, θ13^q about 2%).

**Fig. 5, threshold-corrected θ_i3(M_GUT)/θ_i3(M_Z) at tanβ̄ 20**
- At η̄_b = 0: ours 0.924 (θ13) and 0.924 (θ23); published read-off 0.930 and 0.930 (from 3.385e-3/3.64e-3 and 3.915e-2/4.208e-2). Difference 0.6%.
- The η̄_b dependence through the running (y_b enhanced):
  - ours: −1.5% from η̄_b = 0 to −0.4;
  - published: smaller, roughly −0.5% to −1% (the contour spacing limits the read-off).

**Eq. (25)/(26), threshold-invariant GUT ratios** (inputs differ: PDG 2012 there, 2026 here):

| quantity | published | ours (tanβ 10) |
|---|---|---|
| (1+η̄_ℓ)y_μ/((1+η̄_q)y_s) | 4.36 ± 0.23 | 4.308 |
| (1+η̄_ℓ)y_e/((1+η̄_q)y_d) | 0.41 (+0.02/−0.06) | 0.404 |
| y_μ y_d/(y_s y_e) | 10.7 (+1.8/−0.8) | 10.66 |

- The running factor GUT/M_Z is 2.247 for ours. The published figure is 2.306 for μ/s and 2.318 for e/d, using their Table 1 M_Z values. We are 2.6-3.1% lower, the same 1-loop vs 2-loop / DR-bar gap.
- The invariants change by ≤ 0.05% when (η̄_b, η̄_q, η̄_ℓ) = (−0.3, 0.2, 0.1) are switched on, and by < 0.01% between tanβ 10 and 30. The paper states they are insensitive to both.

**Against exp 023's linear treatment** (model-side s13, s23, m_s/m_b ÷ (1+η̄_b), m_b/m_τ × (1+η̄_b)). Relative difference of the translated low-scale value, full treatment vs linear, for a prediction at the data's own GUT value (the 5%-off case agrees to 5e-4):

| η̄_b | s13, s23 | m_s/m_b | m_b/m_τ |
|---|---|---|---|
| −0.01 | < 1e-5 | < 1e-5 | −1e-5 |
| −0.1 | 3e-5 | 3e-5 | −1.7e-4 |
| −0.2 | 1.0e-4 | 2.1e-4 | −4.8e-4 |
| −0.4 | 5.5e-4 | 1.4e-3 | −1.9e-3 |
| −0.8 | 1.1e-2 | 3.3e-2 | −3.3e-2 |

The two agree at small η̄_b, as required. At η̄_b = −0.8 they differ by 1-3%. That is the η̄_b dependence of the MSSM running (y_b enhanced 5×) that exp 023's README described as "O(1-2%)". With the 2% mass floor, 3.3% is 1.6σ on mr_s_b and mr_b_tau.

**Exp 023 eb08 point (η̄_b = −0.8), rescored and refit** (`refit_023.py` → `refit_023_eb08.json`):
- Committed (linear): χ² 9.53. Recomputed with the current src: 9.53337857116074, identical.
- Full treatment at the same parameters: χ² 17.91. The pulls that move are mr_s_b (0.85 → 2.39), mr_b_tau (0.14 → −1.45) and mr_c_t (0.87 → 1.54).
- Refit with the full treatment: χ² **11.59**, max pull 1.93 (δ_ℓ); s23^q −1.91, mr_d_s +1.26, mr_s_b +1.13. Six least_squares starts: the committed point plus five exp-023-style perturbations; the best came from the committed point.
- Exp 023's verdict does not change: k_cont 17 ≥ N 16, so the model is not compressive whatever χ² is.

### Optimizer on exp 022's 9-D texture (`opt_022.py` → `opt_*.json/log`)

Setup:
- Exp 022's `fit.py` model, bounds (u_ij, d_ij ∈ [−10, 10], r ∈ [0, 1]), observables (N 14), scale (MSSM tanβ 5) and 2% floor.
- Random starts only: the paper's benchmark start was not given.
- Target: χ² 22.4019 (exp 022 `fit_primary.json`).
- One process per run.

| run | landscape | budget | nfev | wall | best χ² |
|---|---|---|---|---|---|
| exp 022 DE (de.py, for reference) | old | popsize 20 | 70,610 | 152 s | 13777 |
| `lbfgsb`, 48 Sobol starts, seed 1 | old (first try) | maxfun 3000/start | 28,903 | 63 s | stuck at 14837 (cap plateau) |
| `de`, seed 1 | old (first try) | maxfev 150k | 62,220 | 128 s | stuck at 13777 (cap plateau) |
| `lbfgsb`, 48 Sobol starts, seed 1 | new | maxfun 3000/start | 115,698 | 173 s | **22.418** |
| `de`, seed 1 | new | maxfev 150k | 125,450 | 223 s | **22.418** |
| `de` on log-magnitude + sign reparametrisation (`de_log`), seed 1 | new | maxfev 150k | 155,394 | 250 s | **22.4022** |
| `lbfgsb`, 48 Sobol starts, seed 2 | new | maxfun 3000/start | 116,560 | 147 s | **22.418** |
| `de`, seed 2 | new | maxfev 150k | 150,914 | 238 s | **22.418** |

- With the old soft objective every method sits on the CHI2_CAP plateau, whatever the algorithm.
- With `include_finite=True`, both new methods reach exp 022's basin from random starts within ~1.2-1.6e5 evaluations (3-4 min on one core):
  - plain DE and L-BFGS-B end 0.016 above the published minimum in all four runs (22.418 vs 22.402, same pulls to 0.001: a shallow neighbouring point the final Nelder-Mead polish does not leave);
  - DE in log-magnitude coordinates hits 22.4022.
- Recommendation: use `method="de"` (or `"lbfgsb"` with ≥ 48 starts) for 9-17-D texture fits, and give hierarchical couplings in log-magnitude coordinates when possible.
- The "old landscape" rows were produced before `include_finite` existed (the method code was otherwise the same). Their logs and JSON are kept as `opt_*_s1_oldsoft.*`. To reproduce them, call `chi2_soft` without `include_finite`.

## Verification

- `check_committed.py` → `committed_check.json`: exp 022 (primary, tb10, alt, eta_minus, v1) and exp 023 (eb0, eb08) rescored at their stored points with default arguments. Max |Δχ²| = 0.0.
- `check_ref_numbers.py` → `ref_check.json`: a copy of exp 024's script, rerunning exp 019's `ref_numbers.py` and comparing it with the committed `ref_numbers.json`. Max abs / rel difference 0.0, `pass_1e-9` true.
- The default `fit_and_evaluate` path is untouched (the test compares an explicit `method="nelder-mead"` run with the default: same params, starts and nfev).
- `.venv/Scripts/python -m pytest -q`: **529 passed** (`pytest.log`). That is 512 before, plus 17 in the new `tests/test_infra_026.py`; one exp 024 test was rewritten.

## Parameter count

N/A (infrastructure). For users: each η̄ that is fitted or chosen after seeing the data adds 1 to k. η̄_q and η̄_ℓ only enter m_d/m_s-type ratios through the double ratios, which are invariant (Eq. 25), so they mostly matter for absolute Yukawas and mr_s_b, mr_mu_tau.

## Result / verdict

- tanβ-enhanced thresholds are in `src.rg` and in `src.evaluate` `scale=` scoring, cited and validated:
  - they agree with Antusch-Maurer to ≤ 1% on y_τ/y_b for η̄_b ≤ 0, 0.6% on the θ_i3 running and 2.6-3.1% on the invariant running factors (1-loop vs 2-loop);
  - they agree with exp 023's linear form at small η̄_b.
- Exp 023's χ² 9.53 at η̄_b = −0.8 becomes 11.59 after a refit with the full treatment. Its conclusion is unchanged.
- The optimizer failure on exp 022 was mostly a landscape bug in `chi2_soft`, which hid the quark and mass terms outside the NuFIT tables. With that fixed for the new methods, DE and multi-start L-BFGS-B find exp 022's minimum from random starts.
- `Objective` defaults to joint and `run_null` reports `info` (B_ckm, S_rank).
- Remaining gaps:
  - 2-loop MSSM Yukawa RGEs and DR-bar conversion: our GUT values sit 1-3% from REAP's.
  - The old `nelder-mead` default still uses the PMNS-only soft objective. Existing pipelines should switch to `method="de"` or `"lbfgsb"` for new texture fits.
  - Undefined δ (L4) is not addressed.
  - `derived_yukawas_mz` has no uncertainties.
  - η̄ values must be supplied by the user; no SUSY-spectrum calculator.

## Files

- `validate_thresholds.py` → `thresholds.json`, `validate_thresholds.log` (~3 min, 1 process).
- `opt_022.py <lbfgsb|de|de_log> <seed>` → `opt_<method>_s<seed>.json/.log` (3-4 min each, 1 process); `opt_*_s1_oldsoft.*` are the first attempts without `include_finite`.
- `refit_023.py [n_starts]` → `refit_023_eb08.json`, `refit_023.log` (~1 min, 1 process).
- `check_committed.py` → `committed_check.json`.
- `check_ref_numbers.py` → `ref_check.json`, `check_ref_numbers.log`.
- `pytest.log`.
- Source changes:
  - `src/rg.py`: `thresholds=`, `THRESHOLD_KEYS`, `_freeze_thresholds`, `_threshold_rot`.
  - `src/evaluate.py`: scale `eta_*`, `scale_slope`, `chi2_soft(include_finite=)`, `method=`, `optimizer_options=`, `FIT_METHODS`, `_de_search`.
  - `src/stats.py`: default joint.
  - `src/nulls.py`: `NullResult.info`, `null_info_of`.
  - Tests: `tests/test_infra_026.py` (new), `tests/test_evaluate_024.py` (one test rewritten).

Run the scripts from the repo root (`validate_thresholds.py`, `check_*.py`) or from this folder (`opt_022.py`, `refit_023.py`) with the repo's `.venv` Python.

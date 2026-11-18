# 20261118-004 core-lib

Shared math and statistics code: `src/mixing.py`, `src/haar.py`, `src/stats.py`, with tests in `tests/test_mixing.py`, `tests/test_haar.py`, `tests/test_stats.py`.

## Hypothesis

N/A, infrastructure.

## Method

- `src/mixing.py`: PDG standard parametrization; angle extraction from rephasing invariants (`|U_e2|`, `|U_e3|`, `|U_mu3|` via atan2 forms, and delta = atan2(J / c13², (|U_mu1|² - s12²c23² - c12²s23²s13²)/2), both numerators equal D sin δ and D cos δ with D = s12c12s23c23s13); Jarlskog; rephasing U = P · U_std · Q via a maximum-weight spanning tree on the entries plus one weighted least-squares step; exact Wolfenstein map (PDG) and its inverse. All functions take `(..., 3, 3)` batches.
- `src/haar.py`: Mezzadri QR sampler for U(3) and SU(3); direct sampler of Haar-induced angles (sin²θ12, cos⁴θ13, sin²θ23, δ uniform); `fake_universe_pair`; `shifted_data_universe` (split-normal shifts of data dicts within asymmetric errors).
- `src/stats.py`: asymmetric-error pulls and χ², covariance χ²; `Objective` combining CKM Gaussian terms, a PMNS Δχ² callable (e.g. `src.data.pmns_delta_chi2`, imported by the caller, not by this module), and mass terms; `p_value(chi2, N, k)`; Šidák trials factor; Clopper-Pearson interval and `fake_universe_rate` with one-sided upper limit.

Tests check: round trip standard_matrix -> extract_angles on 20k random points and on a 6×6×6 grid of edge angles (0, 1e-9, 1e-5, 0.3, π/2 - 1e-9, π/2); invariance of angles and J under random row/column/overall phases; reconstruction P·U_std·Q = U for generic, edge-case and QR-random matrices; Wolfenstein exactness (unitarity, ρ̄ + iη̄ = -V_ud V_ub*/(V_cd V_cb*) to 1e-13, inverse map); Haar unitarity and KS tests (1e5 samples, fixed seed, threshold p > 1e-3) that sin²θ12, cos⁴θ13, sin²θ23, δ/2π are uniform, plus power checks that a wrong measure and an un-phase-fixed QR are rejected; χ², p-value, Šidák and Clopper-Pearson values against closed forms and a coverage simulation.

## Free parameters

N/A.

## Result

- `tests.log`: 351 passed (`.venv/Scripts/python -m pytest tests/test_mixing.py tests/test_haar.py tests/test_stats.py -q`, about 8 s).
- `run.log` (`bench.py`, numpy 2.5.5, OMP_NUM_THREADS=2, best of 3):

| operation | time |
|---|---|
| haar_unitary, 1e6 matrices | 2.5 s |
| sample_haar_angles, 1e6 | 0.11 s |
| standard_matrix, 1e6 | 0.56 s |
| extract_angles, 1e6 | 0.68 s |
| jarlskog, 1e6 | 0.07 s |
| rephase_to_standard, 1e5 | 2.5 s |

## Verdict

Library is ready for use by later experiments.

## Known limitations

- δ precision is about 1e-16 / D with D = s12c12s23c23s13. Below D = 1e-15 δ is returned as 0. In extreme corners (two angles below ~1e-7) `rephase_to_standard` reconstructs U only to ~1e-9. None of this matters at measured CKM/PMNS values (D ~ 1e-3 for CKM).
- When c13 = 0, t12 is set to 0 and t23 is read from the 2×2 block; when D = 0, δ = 0. These are conventions for unphysical parameters.
- Majorana phases are not returned (they sit inside Q from `rephase_to_standard`, mixed with column rephasing).
- `rephase_to_standard` is ~40× slower per matrix than `extract_angles` (Python loop over 9 edges, batched pinv); fine for 1e5, use `extract_angles` for bulk work.
- `fake_universe_pair` draws non-hierarchical Haar CKMs. A look-elsewhere null that keeps the observed quark hierarchy needs a conditioned sampler or `shifted_data_universe`.
- `shifted_data_universe` uses independent split-normal shifts. It ignores correlations and the non-Gaussian PMNS likelihood (θ23 octant, δ_CP); sampling from the NuFIT Δχ² tables would be more faithful.
- `Objective` in `pmns_mode="sum1d"` adds 1D profiled Δχ² values, which ignores correlations among PMNS parameters and double-counts nothing only if the callable's 1D profiles are independent. `pmns_mode="joint"` trusts the callable. In both cases the ordering offset (IO vs NO) is whatever the callable returns; in sum1d mode an IO offset baked into each 1D profile would be counted once per parameter, so check the loader's convention before using IO with sum1d.
- `p_value` assumes Wilks; profiled Δχ² from NuFIT is not exactly χ²-distributed.

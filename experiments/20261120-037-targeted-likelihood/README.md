# 20261120-037 Targeted exact-likelihood tests of the frozen CKM-linked relations

**Pre-registration:** `prereg.md`, SHA-256 `e4a90deca60bf3b82fecd6b0f21b88d146f0093a668c1e446059da9bd1c8e15c`, logged in `run.log` (2026-11-20T16:36:40Z) before any real-data rank was computed. Frozen list: exp 029 `frozen_relations.csv`, SHA-256 `04d6574e…4217`, verified on load, not modified.

## Hypothesis

For each relation in the exp 029 frozen list, in each ordering: the real CKM makes the measured PMNS (NuFIT 6.1) more likely under the relation than fake CKMs do (N1, N1lam), and the measured PMNS is fitted better than shifted PMNS data (N2). Iterations 9-11 showed that the NN pipeline has no power against such links; an exact-density oracle reaches ~50% power near 7 bits (exp 034). This experiment replaces the NN with the per-relation likelihood statistic, which for a named relation is the most powerful statistic available, and measures its power before reading the real-data ranks.

## What changed relative to exp 030 (bi-large under the old protocol)

Exp 030 ran the whole 2052-configuration bi-large grammar as the pipeline and used the fit-only hit rule (max|pull| < 2 and p_global ≥ the reference). Any fake that reached a good fit with any grammar entry was a hit, so power was 0-5%. Here each relation is fixed (no grammar search per fake), the statistic is the profiled log-likelihood of that one relation, and a hit is a fake with a likelihood at least as high. The search history enters only through the stated history factor (bits). Power comparison: see Result.

## Method

Full rules in `prereg.md`; code in `lib37.py` (statistic, data sets, relations), `run.py` (runner), `analyze.py`.

- **Statistic.** T = −½ min_θ χ²_PMNS(pred(θ, CKM)), profiled over the relation's own continuous parameters (k_cont), with the official within-ordering NuFIT 6.1 Δχ² (4-key rows: 1D s12², s13² profiles + 2D (s23², δ) table; symbolic rows: 1D profile of their one observable). The vectorised χ² equals `src.evaluate.Scorer` to 3.6e-14 on real and N2-translated tables (`check.log`). The grid-plus-local-refinement profile reproduces the frozen χ²_today of every fitted row within +0.004 (`check.log`); symbolic predictions within 5e-6. The same procedure runs on real data, injected worlds and every fake.
- **Nulls** (fake j shared across all rows and data sets, `default_rng([37, code, j])`, banks of 50,000 per null): N1 (`src.nulls.n1_universe`), N1lam (`n1lam_universe`, λ and A from the PDG posterior), N2 (`n2_universe`, real CKM, shifted PMNS with translated tables). Hit: fake χ²_min ≤ real χ²_min.
- **Multiplicity.** 74 rows × 3 nulls = 222 tests, global 1%, Šidák local 4.53e-5. Sequential Monte Carlo p̂ = (k+1)/(n+1): ≥ 1000 fakes, extended while k ≤ 1 up to 50,000; pass iff k ≤ 1 at 50,000. Secondary history factor 2^bits (bits = k_honest − k_cont from exp 029).
- **Power first.** 20 injected worlds per row (truth = frozen best-fit prediction at the real CKM; data = truth + `pmns_noise`, translated tables), 300 fakes each per null. P300 = fraction with 0/300 hits (protocol rule). Global-rule power: exact for symbolic rows (worlds run to 50,000); for fitted rows bounded by the fraction with k ≤ 1 at 2,000 (budget cap, pre-registered). Projected: 20 worlds per row at exp 029 nominal JUNO/DUNE/Hyper-K precision (Gaussian likelihood), plus optimistic precision for symbolic rows.
- **Compute.** 4 worker processes, 1 thread each, 16:37-17:09 UTC (6,315 worker-seconds), before the CPU-pinning rule; banks took 47 s. The deadline truncated no row. Every real row stopped at 1000 fakes with k ≥ 2 already reached, so no real-data extension to 50,000 was needed.

## Free-parameter count

Per row (from exp 029, `results.csv`): N = 4 observables for fitted families, 1 for symbolic rows. k_cont: A4a/A4b 2 phases, E1 3 angles, F2LO 1, CDSV 1, RMSV 2, symbolic 0. Data-selected bits: A4 4, E1 1, F2LO 15, CDSV 11, RMSV 5, symbolic 15. No row has k ≤ N − 2 under honest counting. This experiment fits nothing new to measured data; its own choices (grids, nulls, thresholds, world counts) are fixed in `prereg.md`.

## Result

### Power (computed before the real ranks were read; `analyze.log`, "power summary")

| precision | null | cells | median P300 | cells P300 ≥ 0.5 | max P300 | max power at the global rule |
|---|---|---|---|---|---|---|
| today | N1 | 74 | 0.00 | 0 | 0.30 (F2LO-TBM NO) | ≤ 0.30 fitted (bound); 0.05 symbolic |
| today | N1lam | 74 | 0.00 | 0 | 0.20 (S-s23-1 IO) | 0.00 |
| today | N2 | 74 | 0.00 | 6 | 0.65 (F2LO-TBM NO) | ≤ 0.20 fitted (bound); 0.05 symbolic |
| nominal future | N1 | 74 | 0.05 | 10 | 0.85 (S-s12-6 NO) | ≤ 0.60 (BL-CDSV-I NO, bound); 0.10 symbolic |
| nominal future | N1lam | 74 | 0.00 | 0 | 0.25 | 0.00 |
| nominal future | N2 | 74 | 0.05 | 12 | 1.00 (BL-CDSV-I NO) | ≤ 0.80 (BL-CDSV-I NO, bound); 0.05 symbolic |
| optimistic (symbolic only) | N1 / N1lam / N2 | 48 each | 0.08 / 0.00 / 0.05 | 7 / 0 / 1 | 0.95 (S-s12-6 NO) | 0.05 |

- **Today no cell has ≥ 50% power at the pre-registered global threshold.** The best is ≤ 30%, an upper bound (F2LO-TBM NO, N1). At the protocol 1% rule (0/300), 6 of 222 cells reach ≥ 50%, all under N2 (F2LO, CDSV-I, A4a-TBM).
- **Symbolic one-observable relations cannot reach the global threshold**, today or at projected precision (≤ 0.10 in every symbolic cell, even where P300 is 0.85-0.95). A fake matches a single predicted number with probability of about (measurement width)/(range the fakes sweep), here 1e-2 to 1e-3, far above 4.5e-5. JUNO makes the s12² relations testable at the 1% level (N1 P300 0.5-0.85), but not at the 222-test level.
- **N1lam has no power anywhere** (max P300 0.25). Relations that use only λ (and A) are blind to it by construction, and the ones that use δ_q or θ13^q are not sharp enough.
- **Versus exp 030** (bi-large through the grammar pipeline with the fit-only rule, power 0-5%): CDSV-I now has P300 today of 0.05 / 0.15 under N1 (NO / IO) and 0.55 / 0.60 under N2. At nominal future precision it is 0.65 / 0.40 (N1) and 1.00 / 1.00 (N2). Fixing the relation and using its own likelihood removes the dilution from searching the whole grammar on each fake. 1%-level power rises, mostly under N2, but stays below 50% at the global level today.

### Real data

**No row passes any null at the global threshold** (`results.json` "passes": []). Smallest p̂ per null: N1, S-s12-3 (θ12 = cos(θ_C/√(m_μ/m_τ))), rank 0.003 (Šidák p 0.59, history-corrected 1.0); N1lam, S-s13-4 NO, rank 0.008 (Šidák 0.87); N2, S-s13-1 IO, rank 0.018 (Šidák 0.99). Over the 222 cells, 5.0% have p̂ < 0.01 and 22% have p̂ < 0.05. Some excess over uniform is expected, because the list was selected for fitting the real data, and that pushes real ranks down (Caveats).

Under N1, the relations built on θ_C sit near the 1% level: F2LO 0.011-0.012, CDSV-I 0.005-0.010, S-s12-2/3 0.003-0.008, S-s13-1/2 0.004-0.010. For formulas picked because they fit with the real Cabibbo angle, that angle is a better input than most of the N1 box (λ ∈ [0.05, 0.5]). With λ and A fixed at their measured values (N1lam), the same rows rank 0.48-0.51. So whatever the real CKM contributes comes through λ alone, as exps 025-034 found.

The pre-registered "excluded" rule (P300 ≥ 0.5 and no injected world as unremarkable as the real data) applies to three N2 cells: **K-A4a-TBM NO, F2LO-TBM NO, BL-CDSV-I NO**. Their real N2 ranks are 0.20, 0.045 and 0.074. In all 20 worlds where the relation is exactly true, the data fit better than the real data do (injected median rank 0.000). This is goodness of fit restated as a rank: these rows already have χ² 7.1-8.2 for N = 4 (p_global 0.03-0.06; CDSV-I NO is a fit failure with max|pull| 2.70). They are excluded as exact relations in NO at about the 95% level (1/21), not as approximate ones.

### Result table (all 74 rows; `analyze.log`, `results.csv`)

Ranks are k/1000 under each null on real data. P300 is the power at the protocol 1% rule over 20 worlds. "min Šidák p" is the minimum over the three nulls.

| relation | ord | k_cont + bits | N | χ² | P300 today N1 / N1lam / N2 | P300 nominal N1 / N1lam / N2 | rank N1 / N1lam / N2 | min Šidák p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| K-A4a-TBM | NO | 2 + 4 | 4 | 7.08 | 0.00 / 0.00 / 0.55 | 0.00 / 0.00 / 0.75 | 0.065 / 0.696 / 0.200 | 1.00 | no evidence; excluded as exact link by N2 (fit-quality driven) |
| K-A4a-TBM | IO | 2 + 4 | 4 | 6.02 | 0.00 / 0.00 / 0.10 | 0.05 / 0.00 / 0.05 | 0.074 / 0.714 / 0.151 | 1.00 | no evidence |
| K-A4a-GR1 | NO | 2 + 4 | 4 | 7.41 | 0.00 / 0.00 / 0.10 | 0.00 / 0.00 / 0.15 | 0.063 / 0.695 / 0.191 | 1.00 | no evidence |
| K-A4a-GR1 | IO | 2 + 4 | 4 | 6.55 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.10 | 0.077 / 0.725 / 0.161 | 1.00 | no evidence |
| K-A4a-GR2 | NO | 2 + 4 | 4 | 6.87 | 0.00 / 0.00 / 0.30 | 0.10 / 0.00 / 0.60 | 0.066 / 0.698 / 0.193 | 1.00 | no evidence |
| K-A4a-GR2 | IO | 2 + 4 | 4 | 6.17 | 0.05 / 0.00 / 0.10 | 0.00 / 0.00 / 0.20 | 0.074 / 0.714 / 0.154 | 1.00 | no evidence |
| K-A4a-HEX | NO | 2 + 4 | 4 | 7.26 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.30 | 0.065 / 0.696 / 0.188 | 1.00 | no evidence |
| K-A4a-HEX | IO | 2 + 4 | 4 | 7.67 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.05 | 0.076 / 0.729 / 0.175 | 1.00 | no evidence |
| K-A4b-BM | NO | 2 + 4 | 4 | 5.89 | 0.05 / 0.00 / 0.00 | 0.05 / 0.00 / 0.30 | 0.141 / 0.859 / 0.133 | 1.00 | no evidence |
| K-A4b-BM | IO | 2 + 4 | 4 | 4.17 | 0.00 / 0.00 / 0.10 | 0.10 / 0.00 / 0.60 | 0.132 / 0.846 / 0.070 | 1.00 | no evidence |
| K-E1 | NO | 3 + 1 | 4 | 3.82 | 0.00 / 0.05 / 0.00 | 0.00 / 0.00 / 0.00 | 0.636 / 0.637 / 0.486 | 1.00 | no evidence |
| K-E1 | IO | 3 + 1 | 4 | 0.32 | 0.00 / 0.05 / 0.00 | 0.00 / 0.00 / 0.00 | 0.046 / 0.053 / 0.077 | 1.00 | no evidence |
| F2LO-GR2 | NO | 1 + 15 | 4 | 7.21 | 0.10 / 0.00 / 0.40 | 0.15 / 0.00 / 0.75 | 0.012 / 0.494 / 0.045 | 0.95 | no evidence |
| F2LO-GR2 | IO | 1 + 15 | 4 | 7.34 | 0.15 / 0.00 / 0.50 | 0.20 / 0.00 / 0.75 | 0.011 / 0.481 / 0.050 | 0.93 | no evidence; fit failure today (max pull > 2) |
| F2LO-TBM | NO | 1 + 15 | 4 | 7.41 | 0.30 / 0.00 / 0.65 | 0.35 / 0.00 / 0.95 | 0.012 / 0.494 / 0.045 | 0.95 | no evidence; excluded as exact link by N2 (fit-quality driven) |
| F2LO-TBM | IO | 1 + 15 | 4 | 7.16 | 0.20 / 0.00 / 0.50 | 0.50 / 0.00 / 0.90 | 0.011 / 0.481 / 0.049 | 0.93 | no evidence; fit failure today (max pull > 2) |
| BL-CDSV-I | NO | 1 + 11 | 4 | 8.21 | 0.05 / 0.00 / 0.55 | 0.65 / 0.00 / 1.00 | 0.010 / 0.507 / 0.074 | 0.91 | no evidence; excluded as exact link by N2 (fit-quality driven); fit failure today (max pull > 2) |
| BL-CDSV-I | IO | 1 + 11 | 4 | 4.71 | 0.15 / 0.00 / 0.60 | 0.40 / 0.00 / 1.00 | 0.005 / 0.500 / 0.020 | 0.74 | no evidence |
| BL-CDSV-II | NO | 1 + 11 | 4 | 7.97 | 0.10 / 0.00 / 0.15 | 0.40 / 0.00 / 0.60 | 0.024 / 0.507 / 0.062 | 1.00 | no evidence |
| BL-CDSV-II | IO | 1 + 11 | 4 | 9.38 | 0.10 / 0.00 / 0.35 | 0.50 / 0.00 / 0.85 | 0.028 / 0.521 / 0.083 | 1.00 | no evidence |
| BL-RMSV-1 | NO | 2 + 5 | 4 | 4.79 | 0.00 / 0.00 / 0.00 | 0.20 / 0.00 / 0.00 | 0.078 / 0.494 / 0.195 | 1.00 | no evidence |
| BL-RMSV-1 | IO | 2 + 5 | 4 | 12.16 | 0.00 / 0.00 / 0.05 | 0.05 / 0.00 / 0.10 | 0.345 / 0.494 / 0.292 | 1.00 | no evidence |
| BL-RMSV-2 | NO | 2 + 5 | 4 | 20.02 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.482 / 0.474 / 0.460 | 1.00 | no evidence; fit failure today (max pull > 2) |
| BL-RMSV-2 | IO | 2 + 5 | 4 | 30.12 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.05 | 0.473 / 0.473 / 0.428 | 1.00 | no evidence; fit failure today (max pull > 2) |
| BL-RMSV-4 | NO | 2 + 5 | 4 | 5.14 | 0.00 / 0.00 / 0.05 | 0.00 / 0.00 / 0.60 | 0.050 / 0.477 / 0.213 | 1.00 | no evidence |
| BL-RMSV-4 | IO | 2 + 5 | 4 | 8.78 | 0.10 / 0.00 / 0.00 | 0.10 / 0.00 / 0.05 | 0.136 / 0.447 / 0.266 | 1.00 | no evidence |
| S-s12-1 | NO | 0 + 15 | 1 | 3.28 | 0.20 / 0.00 / 0.00 | 0.50 / 0.00 / 0.00 | 0.022 / 0.494 / 0.202 | 0.99 | no evidence |
| S-s12-1 | IO | 0 + 15 | 1 | 3.29 | 0.20 / 0.00 / 0.00 | 0.60 / 0.00 / 0.05 | 0.022 / 0.494 / 0.220 | 0.99 | no evidence |
| S-s12-2 | NO | 0 + 15 | 1 | 0.19 | 0.00 / 0.00 / 0.00 | 0.45 / 0.00 / 0.00 | 0.008 / 0.494 / 0.063 | 0.87 | no evidence |
| S-s12-2 | IO | 0 + 15 | 1 | 0.19 | 0.10 / 0.00 / 0.00 | 0.45 / 0.05 / 0.30 | 0.008 / 0.494 / 0.056 | 0.87 | no evidence |
| S-s12-3 | NO | 0 + 15 | 1 | 0.27 | 0.10 / 0.00 / 0.05 | 0.60 / 0.00 / 0.05 | 0.003 / 0.494 / 0.068 | 0.59 | no evidence |
| S-s12-3 | IO | 0 + 15 | 1 | 0.27 | 0.20 / 0.00 / 0.00 | 0.50 / 0.00 / 0.05 | 0.003 / 0.494 / 0.067 | 0.59 | no evidence |
| S-s12-4 | NO | 0 + 15 | 1 | 0.61 | 0.15 / 0.00 / 0.05 | 0.50 / 0.00 / 0.00 | 0.025 / 0.506 / 0.101 | 1.00 | no evidence |
| S-s12-4 | IO | 0 + 15 | 1 | 0.62 | 0.05 / 0.00 / 0.00 | 0.40 / 0.00 / 0.15 | 0.025 / 0.506 / 0.091 | 1.00 | no evidence |
| S-s12-5 | NO | 0 + 15 | 1 | 0.94 | 0.00 / 0.00 / 0.00 | 0.15 / 0.00 / 0.05 | 0.067 / 0.535 / 0.106 | 1.00 | no evidence |
| S-s12-5 | IO | 0 + 15 | 1 | 0.94 | 0.00 / 0.00 / 0.05 | 0.10 / 0.00 / 0.15 | 0.067 / 0.535 / 0.124 | 1.00 | no evidence |
| S-s12-6 | NO | 0 + 15 | 1 | 0.96 | 0.10 / 0.00 / 0.00 | 0.85 / 0.25 / 0.10 | 0.017 / 0.018 / 0.106 | 0.98 | no evidence |
| S-s12-6 | IO | 0 + 15 | 1 | 0.96 | 0.00 / 0.00 / 0.00 | 0.60 / 0.25 / 0.15 | 0.017 / 0.018 / 0.125 | 0.98 | no evidence |
| S-s13-1 | NO | 0 + 15 | 1 | 0.12 | 0.15 / 0.00 / 0.00 | 0.10 / 0.00 / 0.00 | 0.010 / 0.503 / 0.034 | 0.91 | no evidence |
| S-s13-1 | IO | 0 + 15 | 1 | 0.03 | 0.20 / 0.05 / 0.05 | 0.20 / 0.00 / 0.00 | 0.004 / 0.398 / 0.018 | 0.67 | no evidence |
| S-s13-2 | NO | 0 + 15 | 1 | 0.09 | 0.05 / 0.00 / 0.00 | 0.10 / 0.00 / 0.00 | 0.006 / 0.016 / 0.038 | 0.79 | no evidence |
| S-s13-2 | IO | 0 + 15 | 1 | 0.26 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.017 / 0.034 / 0.057 | 0.98 | no evidence |
| S-s13-3 | NO | 0 + 15 | 1 | 0.37 | 0.20 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.021 / 0.506 / 0.068 | 0.99 | no evidence |
| S-s13-3 | IO | 0 + 15 | 1 | 0.18 | 0.10 / 0.00 / 0.00 | 0.10 / 0.05 / 0.10 | 0.017 / 0.506 / 0.054 | 0.98 | no evidence |
| S-s13-4 | NO | 0 + 15 | 1 | 0.21 | 0.30 / 0.00 / 0.05 | 0.30 / 0.10 / 0.10 | 0.009 / 0.008 / 0.056 | 0.87 | no evidence |
| S-s13-4 | IO | 0 + 15 | 1 | 0.46 | 0.25 / 0.10 / 0.05 | 0.40 / 0.15 / 0.15 | 0.015 / 0.009 / 0.071 | 0.89 | no evidence |
| S-s13-5 | NO | 0 + 15 | 1 | 0.75 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.05 | 0.095 / 0.275 / 0.098 | 1.00 | no evidence |
| S-s13-5 | IO | 0 + 15 | 1 | 0.45 | 0.05 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.074 / 0.196 / 0.086 | 1.00 | no evidence |
| S-s13-6 | NO | 0 + 15 | 1 | 0.79 | 0.00 / 0.00 / 0.00 | 0.05 / 0.00 / 0.00 | 0.097 / 0.290 / 0.102 | 1.00 | no evidence |
| S-s13-6 | IO | 0 + 15 | 1 | 0.48 | 0.05 / 0.05 / 0.05 | 0.00 / 0.00 / 0.00 | 0.079 / 0.203 / 0.090 | 1.00 | no evidence |
| S-s23-1 | NO | 0 + 15 | 1 | 3.10 | 0.15 / 0.00 / 0.00 | 0.35 / 0.05 / 0.00 | 0.045 / 0.042 / 0.341 | 1.00 | no evidence |
| S-s23-1 | IO | 0 + 15 | 1 | 1.63 | 0.30 / 0.20 / 0.15 | 0.15 / 0.10 / 0.05 | 0.015 / 0.008 / 0.081 | 0.87 | no evidence |
| S-s23-2 | NO | 0 + 15 | 1 | 1.35 | 0.00 / 0.00 / 0.00 | 0.05 / 0.00 / 0.00 | 0.082 / 0.535 / 0.161 | 1.00 | no evidence |
| S-s23-2 | IO | 0 + 15 | 1 | 0.71 | 0.00 / 0.00 / 0.00 | 0.05 / 0.00 / 0.05 | 0.110 / 0.535 / 0.048 | 1.00 | no evidence |
| S-s23-3 | NO | 0 + 15 | 1 | 1.73 | 0.00 / 0.00 / 0.05 | 0.15 / 0.05 / 0.45 | 0.050 / 0.062 / 0.219 | 1.00 | no evidence |
| S-s23-3 | IO | 0 + 15 | 1 | 1.34 | 0.05 / 0.05 / 0.00 | 0.20 / 0.05 / 0.00 | 0.016 / 0.028 / 0.073 | 0.98 | no evidence |
| S-s23-4 | NO | 0 + 15 | 1 | 1.01 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.05 | 0.075 / 0.494 / 0.080 | 1.00 | no evidence |
| S-s23-4 | IO | 0 + 15 | 1 | 2.96 | 0.00 / 0.00 / 0.05 | 0.05 / 0.00 / 0.10 | 0.167 / 0.494 / 0.277 | 1.00 | no evidence |
| S-s23-5 | NO | 0 + 15 | 1 | 0.69 | 0.00 / 0.00 / 0.00 | 0.05 / 0.00 / 0.00 | 0.171 / 0.465 / 0.047 | 1.00 | no evidence |
| S-s23-5 | IO | 0 + 15 | 1 | 3.02 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.217 / 0.465 / 0.250 | 1.00 | no evidence |
| S-s23-6 | NO | 0 + 15 | 1 | 1.81 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.475 / 0.882 / 0.244 | 1.00 | no evidence |
| S-s23-6 | IO | 0 + 15 | 1 | 3.43 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.459 / 0.882 / 0.357 | 1.00 | no evidence |
| S-dcp-1 | NO | 0 + 15 | 1 | 3.89 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.534 / 0.497 / 0.504 | 1.00 | no evidence |
| S-dcp-1 | IO | 0 + 15 | 1 | 0.17 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.039 / 0.060 / 0.066 | 1.00 | no evidence |
| S-dcp-2 | NO | 0 + 15 | 1 | 2.21 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.05 | 0.285 / 0.118 / 0.234 | 1.00 | no evidence |
| S-dcp-2 | IO | 0 + 15 | 1 | 1.72 | 0.00 / 0.00 / 0.05 | 0.00 / 0.00 / 0.00 | 0.715 / 0.882 / 0.183 | 1.00 | no evidence |
| S-dcp-3 | NO | 0 + 15 | 1 | 2.25 | 0.00 / 0.00 / 0.05 | 0.05 / 0.00 / 0.00 | 0.193 / 0.179 / 0.237 | 1.00 | no evidence |
| S-dcp-3 | IO | 0 + 15 | 1 | 1.68 | 0.05 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.169 / 0.187 / 0.181 | 1.00 | no evidence |
| S-dcp-4 | NO | 0 + 15 | 1 | 2.25 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.235 / 0.232 / 0.237 | 1.00 | no evidence |
| S-dcp-4 | IO | 0 + 15 | 1 | 1.68 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.169 / 0.187 / 0.181 | 1.00 | no evidence |
| S-dcp-5 | NO | 0 + 15 | 1 | 2.38 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.10 | 0.349 / 0.465 / 0.247 | 1.00 | no evidence |
| S-dcp-5 | IO | 0 + 15 | 1 | 1.53 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.651 / 0.535 / 0.172 | 1.00 | no evidence |
| S-dcp-6 | NO | 0 + 15 | 1 | 1.61 | 0.05 / 0.05 / 0.05 | 0.00 / 0.00 / 0.00 | 0.193 / 0.179 / 0.206 | 1.00 | no evidence |
| S-dcp-6 | IO | 0 + 15 | 1 | 2.49 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.234 / 0.237 / 0.215 | 1.00 | no evidence |

## Verdict

No evidence for a CKM → PMNS link in any of the 37 frozen relations (74 rows), under N1, N1lam or N2, at the pre-registered global 1% level (minimum Šidák p 0.59). The test has no demonstrated power at that level today: no cell reaches 50% (best ≤ 30%). So a non-detection does not exclude the relations as CKM links. At the 1% protocol level, power reaches ≥ 50% in 6 N2 cells today. At nominal JUNO/DUNE/Hyper-K precision it does so in 22 cells (N1 or N2), mostly bi-large, F2LO and θ12 relations. N2 excludes three NO rows (A4a-TBM, F2LO-TBM, CDSV-I) as exact relations with power 0.55-0.65; this restates exp 029's fit quality as a rank. At the 1% level the per-relation likelihood has much more power than exp 030's grammar pipeline (CDSV-I under N2: 0-5% → 55-60% today, 100% at nominal future precision). But with 222 tests and 11-15 bits of selection history for the symbolic, F2LO and bi-large rows, no global claim is reachable with present data. Nothing here is a candidate.

## Caveats

- **Selection bias.** The frozen relations were chosen because they fit the real data with the real CKM, which biases the real rank downward. "No pass" and "excluded" are both conservative under this bias, but a small rank is not evidence. For every row near the top, the history-corrected p is 1.0 to two digits.
- **Invalid fakes count as non-hits** (pre-registered). For S-dcp-3, S-dcp-6, S-s12-6 and S-s13-4, about half of the N1/N1lam fakes are invalid under exp 010's range rules (δ_q > π). For S-s12-1/3, S-s13-5/6 and CDSV-II (N1) the share is 2-28%. This lowers those rows' real ranks, which is anti-conservative. Counting invalid fakes as hits would only raise the ranks, so the no-pass verdict does not depend on this choice.
- **Global power for fitted rows is an upper bound** (worlds stop at 2,000 fakes).
- Injected worlds use the real CKM records without CKM measurement noise. PDG errors are ≪ the N1 spread, and N1lam fakes are centred on the same records.
- Future worlds use a Gaussian likelihood with exp 029's projected σ (s13² at today's error), with no octant or δ non-Gaussianity.
- The profile is a grid with nested refinement. At the real best fits it is within 0.004 of the official multi-start fits. `check_margin.log` shows the coarse grid alone can be off by up to ~70 in χ² (E1, RMSV), so refinement runs for every data set and every fake.
- With 20 worlds, P300 has a binomial CP95 width of about ±0.2 near 0.5.

## Skeptic checklist (no row passed; use when rerunning on JUNO/DUNE data)

1. Recompute the frozen-list and prereg hashes. 2. Confirm `run.py` reads PMNS only through `TableData`/`GaussData` and the CKM only through bank contexts. 3. Rerun `check.py` (χ² agreement, χ²_today reproduction). 4. Recount M as rows × nulls actually reported. 5. For any small rank, count invalid fakes and redo the rank with invalid = hit. 6. Apply the history factor.

## Reproduce

From this folder, prefix commands with `../../.venv/Scripts/python ../../tools/pin.py ../../.venv/Scripts/python`: `check.py > check.log`; `check_margin.py > check_margin.log`; `run.py banks`; `run.py worker <HH:MM UTC deadline> <relation ids>` (four workers as in `run.log`, ~32 min wall; resumable, one JSON per relation in gitignored `artifacts/037/`); `analyze.py > analyze.log` with `PYTHONIOENCODING=utf-8`.

## Files

`prereg.md` (hashed), `lib37.py`, `run.py`, `check.py`, `check_margin.py`, `analyze.py`. Logs: `run.log` (prereg hash, launch, worker logs), `check.log`, `check_margin.log`, `analyze.log`. Results: `results.csv`, `results.json` (per row: χ², k and n per null, ranks, p̂, Šidák and history p, P300 and global-rule power for today / nominal / optimistic). Raw per-relation outputs: `artifacts/037/rel_*.json` (gitignored). No `src/` changes, so no new tests.


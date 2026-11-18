# 20261118-009 F2: GUT-Clebsch charged-lepton corrections on symmetry-fixed U_ν

## Hypothesis

Family F2 (exp 003 `model_families.md`): U_PMNS = V_e† · D(φ) · U_ν, with U_ν fixed by a flavour symmetry and V_e a small CKM-like charged-lepton rotation whose angles are the quark angles times GUT Clebsch factors. If lepton mixing is tied to quark mixing this way, some lattice point (pattern, Clebsch factors, phase) should fit (s12², s13², s23², δ) with few parameters, and the real CKM should do better than random hierarchical CKMs fed through the same search. Unlike F1, the e-sector angles are not free: θ13 and the solar-angle shift are set by θ_C × Clebsch, and δ follows from sum rules once φ is set.

## Method

**Model.** `U_PMNS = V_e(θ12e, θ13e, θ23e, δ_e)† · diag(1, e^{iφ}, e^{iφ}) · U_ν` (`f2lib.py`).
- θ12e = c12 θ_C, θ23e = c23 θ_cb, θ13e = c13 θ_ub, δ_e = δ_q (CKM phase inherited).
- c ∈ C0 = {0} ∪ C, C = {1/6, 2/9, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 9/2, 6}. C is the SU(5)/Pati-Salam Yukawa-ratio Clebsch list {1, 3/2, 2, 3, 9/2, 6} of Antusch-Spinrath arXiv:0902.4644 plus the inverses (messenger-mass Clebsches enter inversely: Antusch-King-Spinrath arXiv:1311.0877). The set is closed under c → 1/c, so θ_e = θ_q·c and θ_e = θ_q/c give the same lattice. IDs checked on the arXiv abstract pages. The TBM + θ12e = θ_C model with θ13 = θ_C/√2 is Antusch-Gross-Maurer-Sluka arXiv:1305.6612 (the brief's "Antusch-Maurer" ID; the paper has four authors).
- V_e order: "std" = R23 U13(δ_e) R12 (PDG, equal to `src.mixing.standard_matrix`) or "rev" = R12 U13(δ_e) R23.
- U_ν: TBM, BM, GR1, GR2, HEX (`src.baselines.lo_matrix` = R23(π/4) R12(θ12ν)); TM-type U_TBM R23(θν, φν) (TM1) or U_TBM R13(θν, φν) (TM2) with θν fitted and φν ∈ {0, 90, 180, 270}°.
- One relative phase φ between the e row and the (μ, τ) rows. The μ-τ relative phase is fixed at 0 (it matters only when θ23e ≠ 0). The tested U_e† U_ν ordering is the physical one; U_ν U_e† was not enumerated.
- Phase variants: φ fitted (1 continuous parameter), or φ ∈ {0, 90, 180, 270}° (structured, 0 continuous parameters).

**Discrete set (nested tiers, `configs()`):**

| tier | content | configs |
|---|---|---|
| A | 5 LO patterns × c12 ∈ C0 (θ23e = θ13e = 0) | 60 |
| B | 5 × c12 ∈ C0 × c23 ∈ C × 2 orders | 1320 |
| C | 5 × c12, c23 ∈ C0 × c13 ∈ C × 2 orders | 15840 |
| TM | 8 TM-type U_ν × c12 ∈ C0 | 96 |

Cumulative: A 60, AB 1380, ABC 17220, ALL **17316 configs (14.08 bits)** with fitted φ. Structured φ: ×4 = **69264 (16.08 bits)**. Counts are raw (some configs are equivalent: φ is irrelevant when c12 = 0, and TM2@0 with θν < 0 equals TM2@180). Raw counts make the Šidák correction conservative.

**Parameter count.** N = 4 (s12², s13², s23², δ). Fitted φ: k = 1 continuous (+1 for θν in TM) + log2(17316) = 15.1 to 16.1. Structured φ: k = 0 continuous (1 for TM) + log2(69264) = 16.1 to 17.1. Local p uses dof = 4 − k_cont; p_trials = Šidák over K (`src.stats.trials_factor_pvalue`). CKM angles are data inputs at central values (`inputs` in `src.evaluate`). Their error is propagated for the top configs (`ckm_spread.csv`, `ckm_spread_lo.csv`) and moves χ² by ≲ ±0.6.

**Scales.** M_Z: PDG 2026 standard-parametrization (s12, s13, s23, δ) from `data/ckm.json`. GUT: the same inputs run to 2e16 GeV with `src.rg.ckm_at` (SM). Ratios: θ_C × 1.00017, θ_ub × 1.1336, θ_cb × 1.1337, δ − 5e-5 rad (`gut_ckm.json`). PMNS is compared to the low-scale NuFIT data in both cases (SM PMNS running ≤ 0.12°, per STATE).

**Scoring.** The fit uses the NuFIT 6.1 tables via `f2lib.TableChi2`, a vectorized copy of `src.data.pmns_delta_chi2` (2D s23-δ plus 1D s12, s13, default analysis IC23 w/o SK-atm). It matches the loader exactly on 3000 random points per ordering (`sanity.json`). The full fit in `run_real.py` uses a 0.5° φ grid, then golden-section refinement of every config. TM configs get a θν grid and a joint Nelder-Mead polish. Top 30 per (scale, ordering, variant) are rescored with `src.evaluate.evaluate`.

**Null pipeline** (`f2lib.search`, identical for real data and fakes, `pipeline_real.py` → `pipeline_real.json`): 2° φ grid over all rows, golden refinement of the best 100 rows, exact structured phases. Per fake and per (scale, ordering, variant, tier set) it records the best χ². Hit = fake best ≤ real best from the same pipeline.
- **(b) hierarchy-preserving fake CKM, real PMNS** (seed 2009): Wolfenstein λ ~ U[0.10, 0.35], A ~ U[0.5, 1.2], ρ̄, η̄ ~ U[−0.5, 0.5] (as exp 007), scored with the NuFIT tables. At the GUT scale the fake gets the real-CKM SM running ratios. This approximation holds because the ratios are set by y_t, y_b and gauge couplings, not by the angles.
- **(a) Haar PMNS target, real CKM** (seed 1009): fake target = angles of a Haar U(3), scored with a Gaussian χ² using the NuFIT asymmetric 1σ widths (NO and IO widths separately) centred on the fake values, as exp 007 null (a). The real comparison uses the same Gaussian χ² centred on NuFIT central values. The Gaussian ignores octant structure and tails.

**Neural part** (`nn_emulator.py`).
- **Emulator.** MLP 13 → 3×128 SiLU → 5, torch on CPU. It ran with 1 thread instead of the requested 2 to stay within the 3-core cap while 2 null workers ran (`F2_TORCH_THREADS`). Training data: 300k synthetic samples of continuous F2 with θ12e ~ U[0, 0.5], θ23e ~ U[0, 0.3], θ13e ~ U[0, 0.03], φ, δ_e ~ U[0, 2π), and U_ν one-hot over the 5 LO patterns plus an order flag. Outputs are s12², s13², s23² and the smooth δ invariants (n_cos, n_sin). No measured number enters training.
- **Scan.** Likelihood scan over (θ12e, θ23e, φ) for each pattern and order, against the NuFIT tables. The allowed region (χ² ≤ 4) and the continuous best fit are checked with the exact model. Each Clebsch lattice point gets a min-over-φ emulator χ², compared with the exact enumeration.
- **Inversion.** 400 shifted-data targets per ordering (split-normal NuFIT draws, Gaussian χ²) for TBM std. Method: grid argmin, then 400 Adam steps on the emulator inputs.
- **PySR** (`pysr_distill.py`, output in `artifacts/009/pysr`):
  - (a) synthetic TBM forward map (θ12e, φ) → s13².
  - (b) inverted best-fit θ12e against (s12², s13², s23², δ, θ_C, θ_cb), with θ_C and θ_cb drawn within PDG errors.

## Result

### Pipeline sanity (`check_pipeline.py` → `sanity.json`)
All known special cases come out exactly:
- **TBC / θ13 = θ_C/√2:** TBM with c12 = 1 gives s13² = sin²θ_C/2 = 0.025351 for every φ (deviation 1e-17). This is the exp 006 A3/A3k prediction.
- **Solar sum rule:** every std-order config with θ13e = 0 satisfies `src.baselines.solar_sum_rule_cosdelta` with its pattern's θ12ν to 3e-14. The reason is that R23(θ23e) commutes with D(φ) and merges into θ23ν = π/4 − θ23e. The rev order breaks the rule, as it should.
- **TM1 / TM2 sum rules:** `src.baselines.tm1/tm2` (s12² and cos δ) hold to 2e-15 for the TM tier with θ12e = 0. With φν fixed at 270° TM1 gives χ² 10.1 (NO) / 5.4 (IO). That is consistent with exp 006 B2-TM1 (5.1 / 1.8 with φν fitted).
- **PySR (a)** recovers `s13² = 0.5 sin²(θ12e)` exactly (loss 1e-17), i.e. the TBC formula.
- fast_obs vs `extract_angles` agree to 3e-11, and the vectorized V_e/PQ build matches the slow path exactly.

### Real data, fitted φ (top 5; full lists in `top30_{MZ,GUT}_fit_{NO,IO}.csv`)
The overall top 5 are all TM-type (k_cont = 2, k = 2 + 14.08 bits = 16.08, dof 2, K = 17316):

| scale | ord | config | χ² | k | local p | trials p |
|---|---|---|---|---|---|---|
| M_Z | NO | TM2@90, c12 = 2 | 3.27 | 16.08 | 0.19 | 1 |
| M_Z | NO | TM2@180, c12 = 1/3 | 3.32 | 16.08 | 0.19 | 1 |
| M_Z | NO | TM2@180, c12 = 1/2 | 3.85 | 16.08 | 0.15 | 1 |
| M_Z | NO | TM2@0, c12 = 1 | 4.09 | 16.08 | 0.13 | 1 |
| M_Z | NO | TM2@180, c12 = 1 (θν < 0 duplicate) | 4.09 | 16.08 | 0.13 | 1 |
| M_Z | IO | TM1@90, c12 = 3 (θν = 0.66, θ12e = 0.68) | 0.11 | 16.08 | 0.95 | 1 |
| M_Z | IO | TM2@180 / TM2@0, c12 = 1 | 2.73 | 16.08 | 0.26 | 1 |
| M_Z | IO | TM1@270, c12 = 2/3 | 2.95 | 16.08 | 0.23 | 1 |
| M_Z | IO | TM1@270 / TM1@90, c12 = 1 | 3.26 | 16.08 | 0.20 | 1 |
| GUT | NO | same order as M_Z: 3.27, 3.32, 3.85, 4.10, 4.10 | | 16.08 | 0.19-0.13 | 1 |
| GUT | IO | same order as M_Z: 0.11, 2.74, 2.74, 2.95, 3.26 | | 16.08 | 0.95-0.20 | 1 |

These fits are driven by the free θν and φ: TM-type U_ν already fits θ13 by itself, and the Clebsch e-rotation is an extra knob (even c12 = 3, θ12e = 0.68 rad, fits IO).

The physically relevant sub-search is the LO patterns (tiers ABC, k = 1 + 14.07 bits, dof 3, K = 17220). There the CKM sets θ13:

| scale | ord | config (LO patterns) | χ² | φ | local p | trials p |
|---|---|---|---|---|---|---|
| M_Z | NO | GR2, c12 = 1, c23 = 1, std | 7.21 | 287° | 0.065 | 1 |
| M_Z | NO | TBM, c12 = 1, c23 = 1, std | 7.41 | 283° | 0.060 | 1 |
| M_Z | NO | GR2, c12 = 1, c23 = 1, c13 = 1/6 | 7.49 | 287° | 0.058 | 1 |
| M_Z | NO | GR2, c12 = 1, c23 = 1, c13 = 2/9 | 7.58 | 287° | 0.055 | 1 |
| M_Z | NO | HEX, c12 = 1, c23 = 1 | 7.63 | 252° | 0.054 | 1 |
| M_Z | IO | TBM, c12 = 1, c23 = 1, std | 7.16 | 282° | 0.067 | 1 |
| M_Z | IO | GR2, c12 = 1, c23 = 1 | 7.34 | 286° | 0.062 | 1 |
| M_Z | IO | TBM, c12 = 1, c23 = 1, c13 = 1/6 | 7.39 | 283° | 0.060 | 1 |
| M_Z | IO | TBM, c12 = 1, c23 = 1, c13 = 2/9 | 7.47 | 283° | 0.058 | 1 |
| M_Z | IO | GR2, c12 = 1, c23 = 1, c13 = 1/6 | 7.56 | 286° | 0.056 | 1 |
| GUT | NO | GR2 / TBM / GR2 c13=1/6 / GR2 c13=2/9 / TBM c13=1/6, all c12 = c23 = 1 | 7.95, 8.15, 8.17, 8.25, 8.37 | | 0.047-0.039 | 1 |
| GUT | IO | TBM / GR2 / GR1 / TBM c13=1/6 / TBM c13=2/9, all c12 = 1, c23 = 2/3 | 7.63, 7.81, 8.01, 8.09, 8.25 | | 0.054-0.041 | 1 |

The LO winner is θ12e = θ_C, θ23e = θ_cb (std order). It is the TBC-type relation with the atmospheric correction. With θ23ν_eff = π/4 − θ_cb it predicts s13 = sin θ_C sin(π/4 − θ_cb), so s13² = 0.0232 (+1.3σ) and s23² = 0.445 (NO −1.8σ; IO best fit 0.555, so in IO this config sits in the first-octant region of the 2D s23-δ table). δ ≈ 260-265°, and φ tunes s12² onto the data. Tier A alone (θ12e only) gives χ² ≥ 24.6: θ12e = θ_C makes s13² too large by about 5σ.

### Real data, structured φ ∈ {0, 90, 180, 270}° (`top30_*_struct_*.csv`)
- ALL: best are TM-type with θν fitted (k_cont = 1, k = 1 + 16.08 bits, dof 3). M_Z NO: TM2@180 c12 = 2/9 χ² 4.36 (p 0.23), then TM2@270 c12 = 2/9 7.36, TM2@180 c12 = 1/6 8.06, TM2@270 c12 = 1/6 8.50, TM1@270 c12 = 1/6 9.87. M_Z IO: TM2@270 c12 = 2/9 4.26 (p 0.23), TM2@270 c12 = 1/6 4.88, then pure TM1@270 (c12 = 0) 5.45 three times (φ irrelevant). GUT values agree to ≤ 0.003. Trials p = 1 throughout.
- LO patterns only (k_cont = 0, k = 16.07 bits, dof 4): best is GR1 c12 = c23 = 1, φ = 270°, χ² 17.0 (NO) / 16.5 (IO), p ≈ 0.002, trials p = 1 (`tier_best.csv`).

### Look-elsewhere null (300 fakes each, 95% Clopper-Pearson, `null_summary.json`)
Selected rows (all 64 rows are in the JSON). Real = best χ² of the real data through the same pipeline.

| null | scale | ord | variant | tier set | real | hits/300 | rate | 95% CI |
|---|---|---|---|---|---|---|---|---|
| (b) hier. CKM | M_Z | NO | fit | ABC (LO) | 7.21 | 138 | 0.46 | [0.40, 0.52] |
| (b) | M_Z | IO | fit | ABC | 7.20 | 135 | 0.45 | [0.39, 0.51] |
| (b) | GUT | NO | fit | ABC | 7.95 | 150 | 0.50 | [0.44, 0.56] |
| (b) | GUT | IO | fit | ABC | 7.68 | 144 | 0.48 | [0.42, 0.54] |
| (b) | M_Z | NO | fit | AB | 7.21 | 80 | 0.27 | [0.22, 0.32] |
| (b) | M_Z | NO | fit | ALL | 3.29 | 125 | 0.42 | [0.36, 0.48] |
| (b) | M_Z | IO | fit | ALL | 2.75 | 250 | 0.83 | [0.79, 0.87] |
| (b) | M_Z | NO | struct | ABC | 17.0 | 125 | 0.42 | [0.36, 0.48] |
| (b) | M_Z | NO | struct | ALL | 5.23 | 225 | 0.75 | [0.70, 0.80] |
| (b) | M_Z | IO | struct | ALL | 4.38 | 20 | 0.067 | [0.041, 0.101] |
| (b) | GUT | IO | struct | ALL | 4.38 | 21 | 0.070 | [0.044, 0.105] |
| (a) Haar PMNS | M_Z | NO | fit | ABC | 10.10 | 9 | 0.030 | [0.014, 0.056] |
| (a) | M_Z | IO | fit | ABC | 39.4 | 29 | 0.097 | [0.066, 0.136] |
| (a) | M_Z | NO | fit | ALL | 7.20 | 53 | 0.18 | [0.14, 0.23] |
| (a) | M_Z | NO | struct | AB | 21.5 | 3 | 0.010 | [0.002, 0.029] |
| (a) | GUT | NO | fit | ABC | 10.66 | 12 | 0.040 | [0.021, 0.069] |

Null (b) is the test of the CKM link: it keeps the real PMNS data and swaps the CKM for random hierarchical ones. Random CKMs do as well as the real one 27-50% of the time in the LO tiers and 7-86% overall. The Clebsch lattice {1/6, ..., 6} × θ_q is dense enough that most λ ∈ [0.10, 0.35] land some θ12e near the ~0.21 rad that θ13 needs. The lowest (b) rate (IO, structured, ALL, 6.7%, CI [4.1, 10.1]%) comes from a TM-type config with fitted θν and is still far above 1%. Null (a) rates are lower (1-18%) only because Haar targets are rarely near-tribimaximal. It tests whether the PMNS data look like "symmetry pattern + small correction", which is already known, not a quark-lepton link.

### What the NN and PySR part contributed (`emulator_report.json`, `emulator_lattice_*.csv`, `pysr_report.json`)
- **Emulator accuracy** (100k held-out): RMSE s13² 1.2e-4 (9e-5 near the data), s12² 5.5e-4, s23² 2.4e-4, median |Δδ| 0.12°. All are well under the NuFIT 1σ widths (5.7e-4, 6.7e-3, 1.7e-2, 23°).
- **Continuous allowed regions** (χ² ≤ 4, θ13e = 0): with θ12e, θ23e and φ all free (k = 3, N = 4), the best continuous fits are χ² ≈ 2.5-4 for TBM, GR1, GR2 and HEX (BM ≥ 22). The allowed regions are narrow:
  - θ12e/θ_C ∈ [0.93, 0.99] (M_Z and GUT alike);
  - θ23e/θ_cb ∈ [0.0, 0.84] at M_Z, [0.0, 0.74] at GUT (IO allows θ23e = 0).
- **Lattice comparison:** no Clebsch lattice point lies inside the χ² ≤ 4 region (exact: 0 of 1808 at M_Z, 0 of 1698 at GUT). c23 ∈ {1/3, 1/2, 2/3} falls inside the θ23e band, but c12 = 1 sits 1-7% above the θ12e band and the next lattice values (2/3, 3/2) are far outside. The emulator reproduces the exact classification (χ² ≤ 9: exact 11, emulator 13, overlap 11 at M_Z; 12 / 15 / 12 at GUT; median |Δχ²| 0.8 for points with χ² < 25). So the NN gives the same answer as direct enumeration plus a picture of the continuous region. It adds no new candidate.
- **Inversion + PySR (b):** emulator-inverted best θ12e for NO targets: θ12e/θ_C = 0.961 ± 0.020 (spread over targets within NuFIT errors). PySR's selected formula is θ12e ≈ sqrt(s13²/s23²), i.e. the known charged-lepton sum rule sin θ13 = sin θ12e sin θ23. θ_C and θ_cb appear only at complexity ≥ 13 with no loss gain (overfit). PySR finds no CKM dependence. That is expected by construction: the inverted e-angles depend only on the PMNS target, and the CKM enters only through the lattice choice c12 = θ12e/θ_C, which comes out at 0.96 ± 0.02 and is not a listed Clebsch.

## Verdict

**No candidate.** F2 over 17316 Clebsch configurations (14.1 bits; 16.1 bits with structured phases) reproduces the known special cases exactly: TBC θ13 = θ_C/√2, the solar sum rule, and TM1/TM2. But it fails on its own terms:
1. With the LO patterns, where the CKM actually fixes θ13, the best fit is θ12e = θ_C, θ23e = θ_cb on GR2/TBM. That is χ² 7.2 (NO) / 7.2 (IO) at M_Z and 8.0 / 7.6 at GUT, with k = 1 + 14 bits, local p ≈ 0.05-0.07 and trials p = 1.
2. The best data-preferred continuous charged-lepton angle is θ12e = (0.96 ± 0.02) θ_C, which is not on the Clebsch lattice.
3. 27-50% of random hierarchical CKMs do as well as the real one (null b). The real V_CKM is not singled out.
4. The good fits (χ² 0.1-3.3) come from TM-type U_ν with a fitted θν, where the CKM is irrelevant: the F1 lesson again, with θν playing the role of the free phases.

F2 with k ≥ 1 + 14 bits cannot meet k ≤ N − 2 = 2 unless a theory fixes the pattern and Clebsch pair before looking at the data. The pre-declarable minimal version (tier A: θ12e = c θ_C only, 60 configs) fits at best χ² 30.8 (NO) / 24.6 (IO). What would change this: a model in which θ12e = 0.96 θ_C arises naturally (e.g. a Clebsch acting on sin θ, or GUT-threshold corrections to θ_C), stated before the fit and tested against null (b); or observables the free knobs cannot absorb (mass ratios m_μ/m_s, m_e/m_d tied to the same Clebsch).

## src issues
None found that affect results. Notes for the overseer:
- `src.rg.ckm_at` takes about 10 s per call (it re-solves y_t(M_Z) by secant each process). That is fine for one call but argues for caching the derived Yukawas to disk if nulls ever need per-fake running. This experiment caches the running ratios in `gut_ckm.json`.
- `src.data._grid_2d` tables are uniform grids. `pmns_delta_chi2` could use direct indexing instead of `searchsorted` for speed (done here in `f2lib.TableChi2`, checked equal).

## Files
- `f2lib.py`: model, config enumeration, vectorized observables and χ², search pipeline, GUT running cache.
- `check_pipeline.py` → `sanity.json` (checks and known-result tests).
- `run_real.py` → `fits_{MZ,GUT}_{NO,IO}.csv` (all 17316 configs), `top30_{MZ,GUT}_{fit,struct}_{NO,IO}.csv` (src.evaluate rescoring, k, local p, trials p), `ckm_spread.csv`, `real_pipeline.json`, `gut_ckm.json`.
- `pipeline_real.py` → `pipeline_real.json` (real-data values of the null pipeline; null comparisons use these).
- `run_null.py --kind {a,b} --n 300 --workers 2 [--start i]` → `null_{a,b}.csv`. These are resumable. `null_a.csv` has 5 duplicate rows from a split run (identical seeds), removed by index in `summarize.py`.
- `summarize.py` → `tier_best.csv`, `ckm_spread_lo.csv`, `null_summary.json`.
- `nn_emulator.py` → `emulator_report.json`, `emulator_lattice_{MZ,GUT}.csv`, `artifacts/009/emulator.pt`, `artifacts/009/inversions.csv` (gitignored).
- `pysr_distill.py` → `pysr_report.json` (PySR run files in `artifacts/009/pysr`).
- `run.log`: all runs. Lines from parallel jobs interleave. The first null-b run (2 fakes at a 1° grid) was discarded, then restarted after vectorizing; the emulator was restarted once with a smaller net for time.

Rerun order: `check_pipeline.py`, `run_real.py`, `pipeline_real.py`, `run_null.py --kind b`, `run_null.py --kind a`, `nn_emulator.py`, `pysr_distill.py`, `summarize.py`. Run all from this folder with `../../.venv/Scripts/python`.

## Note (overseer)
The full per-config dumps `fits_{MZ,GUT}_{NO,IO}.csv` (11 MB total) were moved to gitignored `artifacts/009/`. Regenerate with `run_real.py`. The committed `top30_*.csv` and `tier_best.csv` carry every number quoted above.

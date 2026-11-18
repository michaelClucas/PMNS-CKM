# 20261118-007 F1 shared-rotation enumeration

## Hypothesis

Family F1 (exp 003 `model_families.md`): U_PMNS = f(V_CKM, Φ, X) with Φ = diag(1, e^{iα}, e^{iβ}) and X a symmetry-motivated fixed matrix. If lepton mixing is quark mixing combined with a group-theory matrix, some configuration from a finite catalog should fit (sin²θ12, sin²θ13, sin²θ23, δ) better than the same search does on fake universes.

## Method

**Compositions** (14). Every one is U = A Φ B, so U_ij = Σ_k A_ik B_kj φ_k:
- left: W Φ X, W ∈ {V, V†, V^T, V*}
- right: X Φ W, same four W
- sandwich: V Φ X V†, V X Φ V†, V† Φ X V, V† X Φ V, V^T Φ X V*, V* Φ X V^T

Φ on the outside of a product is a rephasing and is not enumerated.

**X catalog** (10): I (with the permutations, all six 3x3 permutation matrices), TBM, BM, GR1 (tan θ12 = 1/φ), GR2 (cos θ12 = φ/2), HEX (θ12 = 30°), TM1 = TBM·R23(π/12), TM2 = TBM·R13(π/12), DEM (diagonalizes the democratic matrix), UW (A4/Δ(27) magic matrix (1/√3)[[1,1,1],[1,ω,ω²],[1,ω²,ω]]). TM1/TM2 use θ = π/12, picked once before any fit and not tuned. X^T and X† variants are not included.

**Permutations**: X_eff = P_r X P_c, 6 × 6.

**Trials count.** Raw configurations 14 × 10 × 36 = 5040. Many are identical for continuous (α, β) because Φ P = P Φ' and sign flips are π shifts of the phases. Distinct classes were found by hashing the multiset of rephasing invariants (|U_ij|², J) over a 12 × 12 phase grid with a fixed Haar-random V (seed 0); integer-affine reparametrizations of (α, β) map that grid to itself. Result: **K = 2784 distinct configurations (log2 K = 11.44 bits)**, the same with a second Haar V, a 24 × 24 grid, and the real CKM.

**Fit.** For each distinct config and ordering: χ² on a 48 × 48 (α, β) grid (7.5°), then Nelder-Mead from the 3 best grid points at least 30° apart, scored by `src.data.pmns_delta_chi2(ordering, s12sq, s13sq, s23sq, dcp_deg)` (2D s23-δ table + 1D s12, s13 profiles, `relative_to="local"`, default analysis IC23 w/o SK-atm). V_CKM = `wolfenstein_matrix` at the PDG 2026 central (λ, A, ρ̄, η̄). The grid uses a vectorized copy of the same χ² (`f1lib.TableChi2`), identical to the loader to machine precision on 4000 random points per ordering (`check_scorer.py`).

**Parameter count.** N = 4. k = 2 continuous (α, β) + log2(2784) = 11.4 bits of discrete choice. The p-value column uses dof = N − k_cont = 2 and is a local p-value; `p_trials` is the Šidák correction 1 − (1 − p)^K. F1 with fitted phases cannot meet k ≤ N − 2 by construction.

**Structured-phase variant.** α, β ∈ {0, 90°, 180°, 270°, 120°, 240°} (Z4 ∪ Z3, 36 pairs), no continuous fit, over all 5040 raw configs (the continuous-phase equivalences do not preserve this set, so no dedup by class). Trials K_s = number of distinct predicted (s12², s13², s23², δ) tuples = **114059 (16.8 bits)**. k_cont = 0, dof = 4.

**CKM uncertainty.** Top 10 continuous and top 5 structured configs per ordering, phases fixed at the fit, 4000 CKM draws with split-normal (λ, A, ρ̄, η̄) from PDG errors (seed 3003, no correlations since PDG publishes none) → `ckm_spread.csv`.

**Look-elsewhere null** (`run_null.py`, pipeline `f1lib.search`, identical for real data and fakes): 48² grid over the 2784 distinct configs, Nelder-Mead polish (3 starts) of the 20 best configs by grid χ², plus the exact 36-point structured scan over the 5040 raw configs. Per fake the best χ² over all configs is recorded. Hit = fake best χ² ≤ real best χ² from the same pipeline.
- **(a) Haar PMNS target, real CKM**: fake PMNS = Haar U(3) (seed 1001, fake i uses `default_rng([1001, i])`). NuFIT tables exist only around the real data, so fakes are scored with a Gaussian χ² using the NuFIT 1σ asymmetric widths (NO widths and IO widths separately) centred on the fake values, δ difference wrapped. The real comparison value is the same Gaussian χ² centred on the NuFIT central values (`gauss_NO/IO` in `real_pipeline.json`), so the comparison is like for like. Approximation: Gaussian widths in sin² ignore the non-Gaussian tails, the octant structure and the physical boundaries, and Haar targets near sin² = 0 or 1 get the same absolute widths as the real data.
- **(b) hierarchy-preserving fake CKM, real PMNS**: V = exact Wolfenstein matrix with λ ~ U[0.10, 0.35], A ~ U[0.5, 1.2], ρ̄, η̄ ~ U[−0.5, 0.5] (seed 2002), scored with the NuFIT tables exactly like the real data.

## Result

### Real data, continuous phases (k = 2 + 11.4 bits, dof 2 for the local p)

Pipeline best equals the full-fit best in both orderings (`real_pipeline.json` vs `top30_*.csv`).

| ordering | rank | config | χ² | p (dof 2) | α, β (deg) | s12², s13², s23², δ |
|---|---|---|---|---|---|---|
| NO | 1 | V X Φ V†, X = DEM r4c5 | 0.785 | 0.68 | 205.8, 319.7 | 0.310, 0.0224, 0.460, 199° |
| NO | 2 | V† X Φ V, GR2 r1c2 | 0.865 | 0.65 | 213.8, 142.4 | 0.308, 0.0224, 0.457, 207° |
| NO | 3 | V† X Φ V, TBM r1c2 | 0.913 | 0.63 | 207.9, 136.6 | 0.308, 0.0224, 0.457, 205° |
| NO | 4 | V X Φ V†, HEX r1c0 | 1.045 | 0.59 | 194.8, 138.4 | 0.308, 0.0225, 0.550, 181° |
| NO | 5 | V† X Φ V, BM r0c0 | 1.063 | 0.59 | 258.2, 185.8 | 0.308, 0.0224, 0.456, 219° |
| IO | 1 | V* Φ X V^T, BM r0c0 | 2.699 | 0.26 | 85.1, 143.3 | 0.308, 0.0223, 0.475, 278° |
| IO | 2 | V Φ X V†, BM r0c0 | 4.168 | 0.12 | 81.5, 8.6 | 0.314, 0.0232, 0.465, 274° |
| IO | 3 | V† Φ X, TBM r0c0 | 6.024 | 0.049 | 283.7, 240.7 | 0.308, 0.0235, 0.457, 266° |
| IO | 4 | V^T Φ X, TBM r0c0 | 6.131 | 0.047 | 281.4, 324.4 | 0.310, 0.0235, 0.457, 263° |
| IO | 5 | V X Φ V†, BM r0c2 | 6.163 | 0.046 | 75.8, 33.3 | 0.308, 0.0226, 0.559, 214° |

NO: 11 of 2784 configs reach χ² < 2, 20 reach χ² < 6. IO: 0 below 2, 2 below 6. Šidák-corrected p over K = 2784 is 1.0 for all of them. The many near-ties (TBM/GR2/BM/HEX in the same sandwich slots) show the result is set by the two fitted phases, not by X: a sandwich V X Φ V† with two free phases can place s12², s13² and δ almost anywhere near a symmetric X. The `pull_*` columns use 1D Gaussian errors and overstate the s23² tension for second-octant fits (e.g. HEX rank 4); χ² uses the 2D table, which allows the second octant in NO at Δχ² ≈ 1.

The exp 003 lead L1 (V Φ BM V†) with a joint 4-observable fit: NO χ² = 5.89 (rank 20, δ = 274°), IO χ² = 4.17 (rank 2). Its angle-only fit (Σpull² 0.20) does not survive adding δ.

### Structured phases (α, β ∈ {0, 90, 120, 180, 240, 270}°, k = 0 + 16.8 bits, dof 4)

| ordering | rank | config | α, β | χ² | p (dof 4) |
|---|---|---|---|---|---|
| NO | 1 | V X Φ V†, HEX r1c0 | 180, 240 | 2.94 | 0.57 |
| NO | 2 | V X Φ V†, HEX r1c0 | 180, 120 | 4.65 | 0.33 |
| NO | 3 | V^T Φ X V*, HEX r0c0 | 0, 240 | 4.77 | 0.31 |
| NO | 4 | V† Φ X V, HEX r0c0 | 0, 120 | 4.86 | 0.30 |
| NO | 5 | V† Φ X V, HEX r0c0 | 0, 180 | 5.53 | 0.24 |
| IO | 1 | V X Φ V†, TBM r1c0 | 120, 90 | 8.40 | 0.078 |
| IO | 2 | V* Φ X V^T, BM r0c0 | 90, 120 | 9.01 | 0.061 |
| IO | 3 | V* Φ X V^T, BM r0c0 | 90, 180 | 9.11 | 0.058 |

NO best: s12² = 0.312, s13² = 0.0218, s23² = 0.545, δ = 175°. Local p = 0.57 but K_s = 114059 distinct predictions, so the trials-corrected p is 1. This variant has k_cont = 0 and could meet k ≤ N − 2 only if the discrete choice were fixed by theory in advance; here it is data-selected out of 2^16.8 options.

### CKM uncertainty (`ckm_spread.csv`)

Negligible against the PMNS errors. For the top NO config the 68% half-widths from CKM are 0.001 (s12²), 0.00014 (s13²), 0.0006 (s23²), 0.17° (δ); χ² 16-84% range 0.77-1.02. Same order for all top configs.

### Look-elsewhere null (300 fakes each, 95% Clopper-Pearson, `null_summary.json`)

| null | ordering | variant | real best χ² | hits / 300 | rate | 95% CI | fake median best χ² |
|---|---|---|---|---|---|---|---|
| (a) Haar PMNS, Gaussian χ² | NO | continuous | 0.674 | 25 | 0.083 | [0.055, 0.121] | 12.9 |
| (a) | NO | structured | 5.631 | 15 | 0.050 | [0.028, 0.081] | 48.0 |
| (a) | IO | continuous | 6.121 | 109 | 0.363 | [0.309, 0.421] | 12.1 |
| (a) | IO | structured | 8.356 | 23 | 0.077 | [0.049, 0.113] | 45.8 |
| (b) hierarchical fake CKM, NuFIT tables | NO | continuous | 0.785 | 108 | 0.360 | [0.306, 0.417] | 1.78 |
| (b) | NO | structured | 2.941 | 56 | 0.187 | [0.144, 0.235] | 9.95 |
| (b) | IO | continuous | 2.699 | 58 | 0.193 | [0.150, 0.243] | 7.19 |
| (b) | IO | structured | 8.403 | 47 | 0.157 | [0.117, 0.203] | 23.1 |

Null (b) is the relevant one: it keeps the real PMNS data and a quark-like hierarchy, and 36% of random hierarchical CKMs give an F1 fit at least as good as the real CKM in NO. Null (a) rates are lower mainly because Haar targets often sit near sin² = 0 or 1 or in regions where no catalog X reaches, so it is a weaker test (and uses a different χ²); it is still far above 1%. Runtime: 11 s per fake (a), 19 s per fake (b), single thread.

## Verdict

No candidate. F1 over this catalog (14 compositions × 10 X × 36 permutations, 2784 distinct configurations) fits the NO data well (best χ² 0.79 with two fitted phases), but the same search on hierarchy-preserving fake CKMs does as well 36% of the time (CI 31-42%), and the structured-phase variant 19% of the time. The good fits come from the two free phases, which can absorb any small CKM-sized rotation; the real V_CKM is not singled out. F1 with fitted phases also fails compression (k = 2 + 11.4 bits > N − 2 = 2). The structured-phase variant has no continuous parameters but 16.8 bits of data-selected choice and a null rate of 16-19%. Lead L1 (V Φ BM V†) should be dropped: joint χ² 5.9 in NO.

What would change this: a theory that fixes the composition, X and Φ before looking at the data (one configuration, not 2^11), tested against the same null; or adding observables the phases cannot absorb (masses, Majorana phases, RG-consistent GUT-scale comparison).

## Files

- `f1lib.py`: catalog, compositions, dedup fingerprint, vectorized observables and χ², search pipeline.
- `common.py`: config list and dedup. `check_scorer.py`: scorer and observable checks.
- `run_real.py` → `configs.csv`, `real_pipeline.json`, `fits_{NO,IO}.csv` (all 2784), `top30_{NO,IO}.csv`, `struct_top30_{NO,IO}.csv`, `ckm_spread.csv`.
- `run_null.py --kind {a,b} --n 300 --workers 3` → `null_{a,b}.csv` (resumable; rerun with larger `--n` to extend). `summarize_null.py` → `null_summary.json`.
- `run.log`: all runs (lines from the parallel jobs interleave).

Config names: `form|X|r<row perm>c<col perm>`, form written with `P` for Φ, `Vd` = V†, `VT` = V^T, `Vc` = V*. Permutation index into `itertools.permutations(range(3))`; X_eff = P_r X P_c with P[i, p[i]] = 1.

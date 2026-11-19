# Compressive-family proposals (exp 021)

Goal: families in which quark and lepton observables share parameters by construction and in which k ≤ N − 2 is reachable. Every number below comes from `checks.py` (C1-C3), `checks2.py` (C4-C6), `checks3.py` (C7) or the cited abstract/PDF.

## Counting conventions

N counts dimensionless observables from the menu:
- CKM: s12, s13, s23, δ_q (4)
- PMNS: s12², s13², s23², δ_ℓ (4)
- in-sector mass ratios: m_u/m_c, m_c/m_t, m_d/m_s, m_s/m_b, m_e/m_μ, m_μ/m_τ (6)
- Δm²21/Δm²31 (1)

A ratio that a family uses as an input (fixed by a free Yukawa entry) is not counted in N and its parameter is not counted in k. Every discrete choice that is not fixed by a symmetry argument or a published model chosen before looking at the data counts as ceil(log2 #options) bits.

## One new tool used throughout: the Clebsch double ratio R

R = (m_e/m_μ)/(m_d/m_s) = **0.0938**, from `src.rg.derived_yukawas_mz`. Uncertainties: ±0.0027 with the program's 2% per-mass floor, ±0.0016 with PDG 2026 errors.

R is the sharpest light-quark test of any quark-lepton Clebsch structure:
- Light-generation Yukawa ratios do not run in the SM at the per-mille level.
- The tan β-enhanced SUSY threshold corrections are generation-universal for the first two generations, so they cancel in m_d/m_s.

So R is a scale-independent, threshold-independent prediction of any fixed 1-2 Clebsch texture. For a diagonal Clebsch pair (m_e/m_d = c1, m_μ/m_s = c2), R = c1/c2. For a texture with zeros, R follows from |det| and the trace (C1).

| Clebsch structure | R predicted | pull (2% floor) | pull (PDG) |
|---|---|---|---|
| minimal SU(5), Y_e = Y_d^T | 1 | +342 | +579 |
| Georgi-Jarlskog (1/3, 3), diagonal | 0.111 | +6.5 | +11.1 |
| Georgi-Jarlskog in the [[0,a],[a,c]] texture (c12 = c21 = 1, c22 = 3) | | ≥ +7.8 | |
| (1/2, 6) diagonal | 0.0833 | −3.9 | −6.7 |
| 1503.03306 (4/9, 9/2) diagonal | 0.0988 | **+1.9** | **+3.2** |
| 1305.6612 (c12 = 6, c21 = −1/2, c22 = 6), texture | passes (min pull 0.002) | | |

1305.6612's own m_s/m_d prediction is 18.95 (+0.33/−0.24). Against PDG 2026 (m_s/m_d = 19.77 ± 0.33) that is −2.0σ, or −1.3σ with the 2% floor.

In the texture Y_d = [[0, b], [a, c]], Y_e = (C∘Y_d)^T with c_ij ∈ {1/6, 2/9, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 9/2, 6}, matched to m_d/m_s and θ12d^L within θ_C ± 0.05: **29 of 1331 Clebsch triples (2.2%) pass R at 2σ** (C1).
- 4 of them give the θ12e ≈ 0.22 that TBM-type U_ν needs for θ13.
- 15 give θ12e < 0.05.

The lattice is sparse enough that R is a real filter. A model whose Clebsches were fixed in advance and that passes R has earned about 5.5 bits.

---

## Ranked table

Score = (viability estimate) × (N − k), with k including bits.

| rank | family | N | k (cont + bits) | N − k | genuine predictions | viability check | literature |
|---|---|---|---|---|---|---|---|
| 1 | **P1** SU(5) sparse Clebsch textures + CSD3 seesaw | 13 | ≈ 7 + 3 (Clebsch, n, η fixed by a published model) = 10; ≈ 7 + 13 if scanned | +3 (fixed) / −7 (scanned) | R (m_e/m_μ), PMNS 4 + Δm² ratio from 1 ν parameter, θ12e and θ23e from Y_d | R +1.9σ (floor) / +3.2σ (PDG) for 4/9, 9/2. CSD3 alone χ² 12.0 for 5 obs (p ≈ 0.017 at 4 dof); a θ12e correction lowers this only to 10.2 and must be ≤ 0.03 rad. Needs θ23e to fix s23² (0.514) and δ (273°). | 1503.03306 (A4×SU(5), 16 inputs with 3 fixed, 18 obs, χ² 7.98 with old errors); 1512.07531 (Littlest Seesaw); 2309.15901 (modular SU(5) LS) |
| 2 | **P3** S4×SO(10), all Yukawas from the same three CSD3 flavons + 45_H Clebsches | ≈ 15 | published: 13 real + 6 phases = 19 (k > N). Proposed: phases fixed to Z_n values ⇒ ≈ 10 + ~6-10 bits | −1 to +5 | δ_ℓ in two windows (225-253° or 281-308°), R, PMNS from shared alignments | δ windows vs NuFIT 6.1 (NO, angles at best fit): Δχ² 0.7-1.8 (lower window), 3.9-4.3 (upper). Fit quality with 2% light-quark floor unknown. | 1705.01555 (χ² 3.4, 19 inputs for 18 observables, "we cannot reliably define χ²_red") |
| 3 | **P5** modular SU(5) or PS, one τ, minimal-weight assignment fixed a priori | 15 | 2 (τ) + ~6 couplings + ~3 bits ≈ 11 | +4 | R, PMNS + CKM from τ | Y_e = Y_d^T is dead by R (+342σ). Needs a 45_H or weight-dependent Clebsch inside the R-passing 2%. Exp 017: quarks never fit at a common τ (best quark-only χ² 255). | 2101.12724 (A4 SU(5), k 18), 2309.15901 |
| 4 | **P1b** SU(5)×A4, TBM + θ12e = θ_C via Clebsch (6, −1/2, 6) | 15 | 14 published (dimensionful); ≈ 10 dimensionless + Clebsch selection ~7 bits | ≈ −2 | θ13 ≈ θ_C/√2, δ_ℓ ≈ 269°, m_s/m_d, δ_q | R passes. Published PMNS point vs NuFIT 6.1: **χ² 120** (θ13 9.44°: Δχ² 60; θ23 38.5°: Δχ² 58). m_s/m_d −2.0σ. | 1305.6612 (χ² 8.1 for 18 obs, 14 params, tan β = 40, thresholds free) |
| 5 | **P4** universal four-zero Hermitian texture, one shape ratio r = B̃/A shared by u, d, e, ν | 8 (+ mass ratios as inputs) | 1 (r) + 4 phases + m1 + 1 bit = 7 | +1 | CKM (1 dof), PMNS (1 dof) | **Quark side alone χ² 77.8 for 1 dof** (δ_q +5.6σ, m_s −5.2, m_b +3.6 with masses as 2%-floor nuisances). Separate r_u = 0.20, r_d = 0.23 fits exactly (k = N). Leptons with the quark r: χ² 67.9; with free r: 3.7. Dead. | 1004.3781 (universal four-zero texture, S3); hep-ph/0212195 (Fritzsch-Xing four-zero quarks) |
| 6 | **P2** Pati-Salam A4×Z5, CSD4 ("A to Z") | ? (count not in abstract) | ? | ? | θ23 45°, θ13 9°, δ 260°, θ12 31-34°, θ_C ≈ 1/4 | CSD4 lepton sector with U_e = 1: χ² 1080. Abstract benchmarks vs NuFIT 6.1: χ² 29 (mid) to 55. LO θ_C = 1/4 (sin = 0.247 vs 0.225). Dead at LO. | 1406.7005 |
| 7 | **P6** QLC from symmetry: U = V_CKM† Φ U_BM, Φ ∈ {1, i, −1, −i}² | 4 | 0 + 5 bits (16 phases × 2 orders) | −1 | all 4 PMNS | **Best χ²_global 25.3** (NO, Φ = 1): s12² = 0.336. Dead. | hep-ph/0405088 (Minakata-Smirnov QLC); exp 007 covers the fitted-phase version |

Viability estimates are subjective priors from the checks: P1 0.15, P3 0.15, P5 0.03, P1b 0.02, P4 < 0.01, P2 < 0.01, P6 0. Ranked by score: P1 (0.45), P3 (~0.15-0.75 depending on phase counting), P5 (0.12), P1b (−), P4, P2, P6.

---

## P1. SU(5) sparse Clebsch textures + CSD3 seesaw (rank 1)

**Construction.** SU(5) × A4 (or S4) with spontaneous CP. At M_GUT:

- Y_d = [[0, d12, 0], [d21, d22, d23], [0, d32, d33]]. This is the 1503.03306 pattern: five texture zeros counted across Y_d and Y_e.
- Y_e = (C ∘ Y_d)^T with C fixed by 24_H/45_H messenger contractions, e.g. C11 → 4/9-type, C22 = 9/2, C33 = 1 (b-τ).
- Y_u: hierarchical, entries u_ij.
- Seesaw with two RH neutrinos: m_ν = m_a φ_a φ_aᵀ + m_b e^{iη} φ_b φ_bᵀ, with φ_a = (0, 1, 1), φ_b = (1, 3, 1) (CSD3) and η = 2π/3 from Z3 roots of unity.
- U_PMNS = U_e† U_ν, where U_e is the left rotation of Y_e. Since Y_e is fixed by Y_d and C, U_e is fixed by the down sector, which the CKM and m_d/m_s, m_s/m_b pin.

The CKM → PMNS link runs through θ12e and θ23e. These are, up to Clebsches, the right-handed d-sector angles, which the quark data fix once the texture is fixed.

**Parameter count.**
- Quark side: about 6 dimensionless parameters (d-sector 3 ratios, up-sector 2-3 mixing-relevant ratios) for CKM 4 + m_d/m_s + m_s/m_b (6 observables). k_q ≈ N_q, so the quark side predicts nothing on its own.
- Lepton side, continuous: 0 for e (Clebsch-fixed) and 1 for ν (m_b/m_a).
- Lepton side, observables: m_e/m_μ (through R), m_μ/m_τ (excluded: needs tan β and thresholds, +1 parameter), PMNS 4, Δm² ratio. That gives N_ℓ = 6 from 1 parameter.
- Discrete:
  - n = 3 (CSD(n), n ∈ {1..5}): 2.3 bits
  - η (9 roots): 3.2 bits
  - Clebsch set: 0 if taken verbatim from 1503.03306; 10.4 bits if scanned on the 11-value lattice for three entries
- Total, with the model fixed a priori: N = 13, k ≈ 7 + ~3 (choosing among ~6 published LS GUT variants) = 10 ≤ N − 2 = 11.
- The Clebsch and η of 1503.03306 were chosen against 2015 data. Strictly that is data selection. This is the main counting risk, and the reason for pre-registration.

**Genuine predictions:**
- R (scale-free)
- m1 = 0
- 5 lepton observables from 1 parameter
- θ12e and θ23e, which shift θ23 and δ off the pure-CSD3 values. These shifts are the only CKM-dependent part.

**Checks.**
- (C6) R for (4/9, 9/2): +1.9σ (floor), +3.2σ (PDG). Off-diagonal entries shift it. The exact texture value needs the full fit.
- (C4) Pure CSD3 against NuFIT 6.1: χ² 12.0 for 5 observables, 1 parameter (p ≈ 0.017). Predictions: s23² = 0.514 (Δχ² mostly from the 2D s23-δ table), δ = 273°, s12² = 0.318, s13² = 0.0228, Δm² ratio fits.
- (C7) A charged-lepton θ12e with a free phase lowers χ² only to 10.2, with θ12e ≤ 0.03 rad allowed. It cannot fix s23 or δ. A θ23e correction is needed, and in this family it is tied to d23/d33, i.e. to V_cb through the Clebsch C23.
- CSD4 (C4): χ² 1080, dead.

**Verdict.** The only family here where the CKM enters the PMNS prediction through a fixed map and the count can reach k ≤ N − 2. Lepton-sector tension (p ≈ 0.02) and R at +2-3σ make it marginal. Literature: 1503.03306, 1512.07531, 2309.15901 (all abstracts checked).

## P3. S4×SO(10) with shared CSD3 alignments (rank 2)

**Construction (1705.01555, abstract and PDF text checked).**
- All Yukawas are Y^f = Σ_i y_i^f φ_i φ_iᵀ (rank-1 sums), with S4 triplet flavons in CSD3 alignment: φ_1 ∝ (0, 1, 1)-type, φ_2 ∝ (1, 3, 1)-type, φ_3 ∝ (0, 0, 1).
- Four 45_H adjoints supply the d/e Clebsches.
- M_R from the same flavons.
- V_CKM and U_PMNS are therefore built from the same three alignment vectors. The difference between them is the coefficient hierarchy y_i^f, not independent unitary matrices.

**Count as published:** 13 real (y_i^u, y_i^d, y_i^e, M_i, y_P) + 6 phases = 19 inputs for 18 observables, χ² ≈ 3.4. Not compressive.

**Proposed compressive variant.**
- Impose spontaneous CP with the flavon VEV phases fixed to Z_n roots, as η already is in the paper. The 6 phases become discrete: ≈ 6 × log2 3 ≈ 9.5 bits if each phase is chosen from 3 values, or 0 if a symmetry fixes them.
- Drop the 3 absolute scales.
- Result: k ≈ 10 + bits for N ≈ 15 dimensionless.
- Reachable only if the phases follow from the symmetry.

**Checks.**
- (C5) Its δ windows vs NuFIT 6.1 NO: lower window 225-253° gives Δχ² 0.7-1.8; upper 281-308° gives 3.9-4.3.
- R depends on the 45_H Clebsches (not extracted), so the C1/C6 filter is pending.
- With PDG 2026 light-quark masses and a 2% floor it has not been fitted. Exps 014 and 016 show published SO(10) and modular fits degrade by 30-130 χ² when their 12-53% light-quark errors are replaced.

**Verdict.** Best-specified "same alignments" model in the literature. Worth a reproduction + refit (like exp 016 did for 2O), then the fixed-phase variant.

## P5. Modular SU(5)/PS with a minimal-weight principle (rank 3)

**Construction.**
- One modulus τ.
- Matter weights fixed a priori by the rule "smallest weights for which every Yukawa entry has at least one invariant". This replaces the 22-bit scan of exp 017 with ≤ 3 bits (group choice).
- SU(5) ties Y_e to Y_d. With only 5_H, Y_e = Y_d^T, which C6 kills at 342σ. So the construction must use 45_H or weight-split couplings so that R lands in the surviving 2%.

**Count:** τ (2) + ~6 coupling ratios + ~3 bits ≈ 11 for N = 15.

**Checks:** R kills the minimal version. Exp 017 shows quarks do not fit at a common τ even with independent e couplings (best quark-only χ² 255), and exp 016 shows 2O's ms/mb and θ23^q tension. Low viability.

**Literature:** 2101.12724, 2309.15901.

## P1b. AGMS SU(5)×A4, TBM + θ12e ≈ θ_C (rank 4)

1305.6612 (PDF text checked):
- Clebsches c12 = c22 = 6, c21 = −1/2 for Y_e.
- 14 parameters for 18 observables; χ² 8.1 with 2013 data.
- tan β = 40, with the threshold parameters η_q12, η_q3 free in [−0.5, 0.5].

Its texture passes R (C1). But its printed PMNS predictions (θ13 = 9.44°, θ23 = 38.49°, δ = 268.8°, θ12 = 34.29°) give χ² 120 against NuFIT 6.1, and m_s/m_d = 18.95 is now −2.0σ. Dead as published.

This is the F2-LO structure again (exp 009: θ12e = θ_C on TBM gives s13² +5σ unless θ23e = θ_cb). A refit would need the 2-3 block Clebsch to absorb that. Same class as F2, which nulls showed has no CKM specificity.

## P4. Universal four-zero Hermitian textures (rank 5)

**Construction.**
- M_f = P_f† [[0, C_f, 0], [C_f, B̃_f, B_f], [0, B_f, A_f]] P_f for f = u, d, e, ν.
- Eigenvalues (−m1, m2, m3).
- "Universal" is defined here as a shared shape ratio r = B̃_f/A_f in all four sectors.
- V_CKM = O_uᵀ diag(1, e^{iφ1}, e^{iφ2}) O_d, and likewise for PMNS with m1 free.

**Count:** k = 1 + 4 + 1 (+1 bit for the choice of universal ratio) for N = 8.

**Check (C2):**
- Quark side, 2% mass nuisances: shared r gives χ² 77.8 (1 dof); separate r_u, r_d fit exactly.
- Leptons with the quark r: χ² 67.9.
- Dead at PDG 2026 precision. The two sectors want r = 0.20 vs 0.23, a 15% difference that 2% masses resolve.

**Literature:** 1004.3781, hep-ph/0212195, 1404.5726 (abstracts/search listing checked).

## P2. Pati-Salam A4×Z5 CSD4 (rank 6)

1406.7005 (abstract checked):
- Y_ν = Y_u, Y_e ≈ Y_d.
- Predicts θ23 = 45 ± 0.5°, θ13 = 9 ± 0.5°, δ = 260 ± 5°, θ12 = 31-34°, θ_C ≈ 1/4.

Against NuFIT 6.1: χ² 29 at mid-range values, 55 at the edge that fixes θ13 (θ12 = 31° costs 47). Pure CSD4 (C4): χ² 1080. Parameter count not in the abstract. Dead at the published level.

## P6. QLC with symmetry-fixed phases (rank 7)

U = V_CKM† diag(1, e^{ia}, e^{ib}) U_BM (or U_BM Φ V_CKM†), with a, b ∈ {0, π/2, π, 3π/2}: 32 configurations, 5 bits, k_cont = 0, N = 4.

Best χ²_global 25.3 (C3). The root cause is s12² = 0.336, the known QLC solar miss (hep-ph/0405088; −4.3σ in exp 003). Dead. The fitted-phase version is inside exp 007's F1 catalog.

---

## Considered and not proposed

- **"Lepton = fixed O(1) matrix × quark sector" (anarchy reversal).** Every version with an unfixed O(1) matrix has 4+ free lepton parameters (exps 007/009 show these absorb the CKM). Versions with a fixed matrix are F1's structured variant (exp 007: 16.8 bits, null rate 16-19%).
- **Dirac ν = up quark (SO(10)) with M_R ∝ 1.** Gives m_ν ∝ M_uᵀM_u, so Δm² ratio ~ (m_c/m_t)⁴ ~ 1e-9. Dead by inspection.
- **Common-τ modular without GUT relations.** Exp 017.

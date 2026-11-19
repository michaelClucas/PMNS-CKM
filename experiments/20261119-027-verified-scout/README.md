# 20261119-027 Verified scout: model classes where the CKM enters PMNS by construction

## Hypothesis

Some published model class computes U_PMNS from the CKM (or from quark-sector parameters the CKM fixes), not from a lepton-only seesaw, and is not already covered by F1 (exp 007), F2 (exp 009) or the exp 003 rough pulls. Exp 022 showed that exp 021 misdescribed SU(5)+CSD3 (no CKM → PMNS link in the paper's matrices). So no model below is listed as "CKM-linked" until its mass/mixing matrices, as written in the paper, have been built numerically and the PMNS dependence on the CKM inputs measured.

## Method

1. Literature: arXiv search plus full PDF text (pdftotext, kept in the session scratchpad, not committed) for hep-ph/0404046 (Raidal), hep-ph/0405088 (Minakata-Smirnov), hep-ph/0407139 (Frampton-Mohapatra), hep-ph/0504007 (Antusch-King-Mohapatra), hep-ph/0503222 (Datta-Everett-Ramond, "Cabibbo haze"), 1008.5061 (Patel SO(10)×S4), 1106.3284 (Qin-Ma), 1107.3728 (Antusch-Maurer), 1205.0506 (King, TBC), 1410.3658 (Roy-Morisi-Singh-Valle), 1902.08962 (Chen-Ding-Srivastava-Valle), 1808.09364 (Antusch-Hohl-Khosa-Susič, SU(5)+CSD2), hep-ph/0605032 and 1711.08796 (correlation matrix), 1102.0879 (Ahn-Cheng-Oh, abstract only). Searches for 2020-2026 papers claiming PMNS from CKM found none beyond these classes (search index coverage of recent preprints was poor; see Caveats).
2. Each model is written as `model(ckm, **params) -> U_PMNS` from the paper's stated structure. The scripts print the equation numbers used. `vlib.ckm_dependence` reports:
   - central finite-difference derivatives of (s12², s13², s23², δ) with respect to each Wolfenstein input (λ, A, ρ̄, η̄), in units of (NuFIT σ of the observable) per (PDG σ of the input), at fixed model parameters;
   - the median shift (in NuFIT σ) when the real CKM is replaced by a random hierarchical one (STATE N1 ranges);
   - the λ → 0 limit.
   CKM inputs come from `src.data.load_ckm`. The NuFIT widths in `vlib.SIG` are rough and only express derivatives in σ units; they are never used for scoring.
3. Coverage: models are rebuilt as the matching F1/F2 configuration and compared through |U_ij| and J (`same_up_to_rephasing`).
4. For the models not covered: fit with `src.evaluate.fit_and_evaluate` / `evaluate_model`, imported only, with `leak_check=True` and declared `inputs` (`quickfit.py`). There is also a pilot look-elsewhere check for the bi-large grammar (`grammar_pilot.py`, 1 core, ~2 min).

## Parameter count

Per model in the tables below.
- k_pub counts only the alternatives the paper itself lists.
- k_honest adds the pattern grammar the authors picked from after seeing data.
- Bits are ceil(log2 #options).

## Result

### A. Verification: does the CKM enter PMNS, and is the model covered?

"CKM enters" = nonzero ∂PMNS/∂CKM at fixed model parameters in the paper's own matrices (logs: `verify_*.log`). Every model below has this, so the column is "yes" throughout. The live question is coverage.

| model (paper) | structure as written | CKM enters? | evidence | covered? | status vs data |
|---|---|---|---|---|---|
| Raidal hep-ph/0404046 | U = U_BM V_CKM† (BM from SU(5) right-handed d rotations; Cabibbo via SO(10) Y_ν = Y_u) | yes | ∂s12²/∂λ = −0.053 σ/σ; random CKM moves s12² by 9.9σ | F1 "X Φ W", X = BM | s12² 0.280, s13² 0.00056: dead (θ13 ≈ θ13^CKM as the paper says) |
| Minakata-Smirnov hep-ph/0405088, eq. 27 | U = V_CKM† U_BM | yes | ∂s13²/∂λ = +0.235 σ/σ | F1 "W Φ X" | s12² 0.336, s13² 0.0236: QLC −4.3σ (exp 003), P6 χ² 25 (exp 021) |
| Minakata-Smirnov eq. 40 | U = (R12 R23)† V_CKM | yes | random CKM moves s12² by 9.9σ | F1 (checked: equals X V with X = (R12 R23)†) | s13² 0.0008: dead |
| Frampton-Mohapatra hep-ph/0407139 | G_422, M_d ≈ M_l in the 1-2 block, U_ν = BM; θ12^l ≈ θ_C at their benchmark | yes | identical to F2 point | F2 tier A (BM, c12 = 1), exact match | s13² 0.0254: +6σ |
| Antusch-King-Mohapatra hep-ph/0504007 | G_422, Clebsch −3 (12), 2 (22): θ12e = 1.5 θ_C; U_ν: θ12 = π/4, θ13 = 0 | yes | random CKM moves s13² by 79σ | F2 tier A/B (BM, c12 = 3/2). θ23e = λ² is off the c23 lattice; s13 does not care. | s13² 0.051 (θ13 = 1.06 θ_C): dead |
| King TBC 1205.0506 / Antusch-Maurer 1107.3728 | TBM + θ12e = c θ_C from PS/SU(5) Clebsch ratios | yes | identical to F2 point | F2 tier A (TBM, c12 ∈ lattice), exact match | θ13 = θ_C/√2: +4.9σ (exp 003) |
| Patel 1008.5061, SO(10)×S4 type-II | V_PMNS = U_e^T U_BM, U_e = R23(−3θ_cb) R13(−3θ_ub) R12(θ_C) (eq. 24); "entire lepton sector is a prediction" | yes | random CKM moves s13² by 50σ, s23² by 5.5σ | F2 tier C class (BM, c12 = 1, c23 = c13 = 3). Not an exact lattice point: the paper's sign of θ23e and the real θ13e are outside F2's μ-τ phase and δ_e choices (max Δ\|U\|² 0.011-0.25). | s13² 0.0293 (matches the paper's eq. 25): +13σ, dead |
| Qin-Ma 1106.3284 | V† U_PMNS V = U_TB, i.e. U = V U_TB V†. Orientation fixed by reproducing their eq. 12: 31.77/7.89/44.64° vs 31.80/7.67/44.66°. The PDF text drops the dagger. | yes | random CKM moves s13² by 19.5σ | F1 sandwich "V Φ X V†", X = TBM, Φ = 1 | NO χ² 68.5 (s12² −4.9σ, s13² −6.1σ): dead |
| Correlation matrix (hep-ph/0605032, 1711.08796) | V_CKM Φ U_PMNS = X (BM/TBM) | yes | (algebraic) | F1 "W Φ X" with W = V† | covered by exp 007 |
| Cabibbo haze hep-ph/0503222 | U = U_CKM† F (symmetric M_d), F unspecified with two large angles; otherwise free-coefficient Wolfenstein-like expansions | yes, but F is free | (algebraic) | F1 with free X; no fixed prediction | not a model with countable k |
| **Bi-large, CDSV 1902.08962** | U = U_l† U_ν. U_l = CKM-form (λ, Aλ², free phase φ). U_ν: sin θ13 = λ, sin θ12 = 2λ, sin θ23 = 1 − λ (I) or 3λ (II), δ_ν = π. | **yes, twice** (U_l and the λ seed in U_ν) | ∂s13²/∂λ = +0.22 σ/σ, ∂s12²/∂λ = +0.16 σ/σ; random CKM moves s12² by 24σ. Published benchmarks reproduced exactly (I: 0.306/0.0216/0.573/1.305π; II: 0.320/0.0215/0.512/1.270π). | **not covered**: U_ν is λ-dependent and not a group pattern | quick fit below |
| **Bi-large, RMSV 1410.3658** | U = U_l† U_BL, sin θ13 = λ, sin θ12 = sin θ23 = ψλ, ψ and φ free, U_l types 1-4 | yes | ∂s12²/∂λ = +0.19 σ/σ | **not covered** | quick fit below. Angles reproduced (s23² 0.420 vs 0.417). Their δ is the arcsin(J) branch. |
| **SU(5)+CSD2, 1808.09364** | Y_d = [[0,z,0],[y e^{iγ},x,0],[0,0,y_b]], Y_e = [[0,c_y y e^{iω},0],[c_z z,c_x x,0],[0,0,y_τ]]. m_ν = CSD2 (TM1, m1 = 0). x, y, z fixed by m_e, m_μ and θ12^dL (from the CKM). | **yes, but weak and inverse** | θ12^eL ≈ (c_y/c_x)(m_d/m_s)/θ12^dL: d ln θ12^eL / d ln θ12^dL = −1.06. Per-σ CKM derivatives ≤ 0.03; random CKM moves s13² by 1.7σ, δ by 1.9σ. Undefined at λ → 0. | **class covered** (F2 TM1 tier: free θ_ν absorbs the CKM, STATE). Not the same points: continuous TM phase, Clebsch-ratio θ12e | quick fit below (reduced model) |
| SU(5)+CSD3 1503.03306 (exp 022) | for reference | no (θ12e = 2.7e-4, θ23e = 0) | exp 022 | n/a | dead |

### B. Quick fits of the uncovered models (official scorer, `quickfit.log`)

PMNS keys only, except CSD2. No running (low-scale comparison). Inputs declared and leak probe passed.

| model | ord | χ² | χ²_global | p_global | N | k_cont | k_pub | k_honest | max pull |
|---|---|---|---|---|---|---|---|---|---|
| CDSV-I | NO | 8.21 | 8.21 | 0.042 | 4 | 1 | 2 | 12 | δ −2.7 |
| CDSV-I | IO | 4.71 | 6.20 | 0.10 | 4 | 1 | 2 (+1 ordering bit if chosen) | 12 | δ −1.7 |
| CDSV-II | NO | 7.97 | 7.97 | 0.047 | 4 | 1 | 2 | 12 | s23² +1.6 |
| CDSV-II | IO | 9.38 | 10.86 | 0.013 | 4 | 1 | 2 | 12 | −1.9 |
| RMSV type 1 (= type 3) | NO | 4.79 | 4.79 | 0.091 | 4 | 2 | 4 | 7 | δ +1.8 |
| RMSV type 2 | NO | 20.0 | 20.0 | 4.5e-5 | 4 | 2 | 4 | 7 | s23² −3.6 |
| RMSV type 4 | NO | 5.13 | 5.13 | 0.077 | 4 | 2 | 4 | 7 | δ +1.8 |
| Qin-Ma (covered by F1) | NO | 68.5 | 68.5 | 5e-14 | 4 | 0 | 0 | 3 | s13² −6.1 |
| CSD2-GUT (3, 3/2, 1/2) | NO | 43.3 | 43.3 | 2e-9 | 6 | 3 | 4 | 10 | mr_d_s +5.7, δ −3.4 |
| CSD2-GUT (9/2, 3/2, 3/2) and (3, 1, 1) | NO | 61.8 | 61.8 | 2e-13 | 6 | 3 | 4 | 10 | mr_d_s −5.4, s13² −4.1 |

How the bits were counted:
- CDSV, k_honest: U_ν grammar {nλ, 1 − nλ, n = 1..4} per angle (8³) × δ_ν {0, π} × U_l form (2) = 2048, so 11 bits. k_pub: 2 patterns, 1 bit.
- RMSV: k_pub 2 bits (4 U_l types). k_honest adds ~8 options for the U_BL ansatz (3 bits).
- CSD2: N = 4 PMNS + Δm² ratio + m_d/m_s. m_d/m_s = (m_e/m_μ) c_x²/(c_y c_z) because the model consumes m_e and m_μ as inputs, so R_clebsch would alias. k_pub: the 102/120 variant, 1 bit. k_honest: 37 Clebsch tuples × 2 variants, 7 bits.
- The CSD2 fit is a reduced model:
  - 1-2 charged-lepton block solved exactly;
  - θ12^dL from the CKM via the paper's right-angled-UT construction;
  - no running or SUSY thresholds;
  - 24 Nelder-Mead starts.
  It is indicative only. The m_d/m_s pull is the R filter of exp 021 (R = c_y c_z/c_x² = 0.0833 or 0.111 vs 0.0938).

Compression: CDSV has k_pub = 2 = N − 2 on paper, so it is the only CKM-linked family in the program that reaches k ≤ N − 2 at all, and only before counting the grammar. With honest bits, k = 12 > 2. Fit quality: NO p_global 0.042-0.047 fails criterion 3 narrowly. IO CDSV-I passes p (0.10) but pays the ordering bit.

### C. Pilot look-elsewhere check for the bi-large grammar (`grammar_pilot.log`)

Setup: same 2048-config grammar, φ on a 3° grid, NO table χ², 100 fake hierarchical CKMs (λ U[0.05, 0.5], A U[0.3, 2]).
- Real data: grammar best 10.25 (CDSV-II on the grid). Two configs are at or below CDSV-I's grid χ² 10.81, none below 8. The grid is coarse in φ (s13² is very φ-sensitive), so grid χ² sits ~2.5 above the polished fit. Fakes use the same grid, so the comparison is like for like.
- Fakes: 35/100 reach χ² ≤ the real best (CP95 [0.26, 0.45]); 38/100 reach ≤ CDSV-I. Fake-best quantiles (5, 25, 50, 75%): 2.8, 7.9, 16.4, 147.
- So a random hierarchical CKM fed through the same Cabibbo-seed grammar does as well a third of the time. This is a pilot, not the protocol (no power check, no N2), but it already sits 25× above the 1% bar. The bi-large fits come from the grammar's flexibility, not from the real λ.

## Verdict

No candidate. Every class checked feeds the CKM into PMNS by construction (verified numerically), unlike SU(5)+CSD3.
- **Covered, dead:** the QLC realizations (Raidal, Minakata-Smirnov, Qin-Ma, correlation-matrix) are F1 configurations. The Pati-Salam/SO(10) charged-lepton-Clebsch models (Frampton-Mohapatra, AKM, TBC/Antusch-Maurer, Patel) are F2 configurations or sit in F2's class. All of them predict θ13 ≈ θ_C × O(1)/√2 and fail at +5 to +13σ. Exp 021's "P6 QLC" result and exp 003's pulls stand.
- **Genuinely new to the repo:**
  - **Bi-large Cabibbo-seed ansätze** (CDSV, RMSV). Phenomenological, no full model. Fit quality is marginal (NO p_global 0.04-0.09). Compressive only if the λ-multiple grammar is not counted. Pilot null hit rate 35%. Known baseline (published 2014/2019), no evidence of CKM specificity.
  - **SU(5)+CSD2** (1808.09364). The CKM link is real but enters as 1/θ12^dL times a mass ratio and moves PMNS by ≲ 2σ even for a random CKM. Its tuples fail the R filter, and the reduced-model fit is χ² 43-62 for N 6. Same class as F2-TM1, where free θ_ν carries the fit.
- No 2020-2026 paper with a fixed CKM → PMNS map was found.

## Recommendation for next iteration (ranked, at most 2)

1. **Bi-large grammar under the full criterion-4 protocol (1 core, < 30 min).** This is the only family in the program where the CKM enters PMNS twice by construction and k ≤ N − 2 holds before grammar bits. Running it closes the "Cabibbo-seed" class with a powered null. Expected outcome: no evidence (pilot 35%). It is worth running only because it is cheap and the power check is meaningful here (fakes move predictions by 10-35σ).
2. **CSD2-GUT R-screen of the 37 published Clebsch tuples (1 core, < 15 min), full refit only for tuples that pass.** Expected outcome: 0-2 tuples pass R at 2σ (the exp 021 lattice passes 2.2%), and any refit inherits a weak CKM link. Low value. Run only if iteration 8 has spare capacity after the NN information test (STATE task 1).

### Pre-registration text, frozen (draft for the overseer)

**PR-1 Bi-large Cabibbo-seed grammar (exp TBD).**
- Model: U_PMNS = U_l(λ, A, φ)† U_ν.
  - U_l ∈ {R23(θ23^q) Φ R12(θ12^q) Φ*, Φ R12^T(θ12^q) Φ* R23^T(θ23^q)}, with sin θ12^q = λ, sin θ23^q = Aλ², Φ = diag(e^{−iφ/2}, e^{iφ/2}, 1).
  - U_ν = R23 U13(δ_ν) R12 with each sin θ_ij^ν ∈ {nλ, 1 − nλ : n = 1..4} and δ_ν ∈ {0, π}.
  - φ continuous. 2048 configurations.
  - λ, A from `src.data` (inputs ckm_lambda, ckm_A).
- Observables: s12², s13², s23², δ_ℓ (N = 4), scored with `src.evaluate`, both orderings, ordering bit charged.
- Count: k = 1 + 11 bits (+1 ordering).
- Selection score: χ²_global after polishing φ (grid 1° then Brent).
- Hit: fake best has max |pull| < 2 and p_global ≥ the real best's p_global (fit-only rule, since k > N − 2 under honest counting; informational).
- Nulls: N1 (fake λ U[0.05, 0.5], A U[0.3, 2], ρ̄, η̄ irrelevant) and N2 (translated tables), 1000 each per ordering, after a 20-injection power check (300 fakes each).
- Kill: real best p_global < 0.01, or power < 50% (then "no evidence"), or N1 CP95 upper limit ≥ 1%.
- Wording: "candidate" only if all pass.

**PR-2 SU(5)+CSD2 R-screen (exp TBD).**
- Take the 37 (c_x, c_y, c_z) tuples of 1808.09364 Table 1 verbatim.
- For each, R_pred = c_y c_z/c_x², scored as `R_clebsch` with the program floor (`mass_floor={"R_clebsch": 0.02·√2}`).
- Pass = |pull| < 2.
- Only passing tuples are refitted with the full Y_d/Y_e/Y_u texture of their eq. 2.9-2.10. The refit uses MSSM `src.rg` running at tanβ fixed to the paper's value, N = PMNS 4 + Δm² ratio + mr_d_s + mr_s_b + CKM 4, and k counted with 7 bits for the tuple + variant choice.
- Kill: no tuple passes R, or refit p_global < 0.01.

## Files

- `vlib.py`: shared matrix builders, CKM-dependence measure, rephasing-invariant comparison.
- `verify_qlc_raidal_ms.py` → `.log`/`.json`: Raidal, Minakata-Smirnov (eq. 27, 40).
- `verify_clebsch_family.py` → `.log`/`.json`: Frampton-Mohapatra, AKM, TBC/Antusch-Maurer, Patel, with F2 equivalence checks.
- `verify_qin_ma.py` → `.log`/`.json`: Qin-Ma orientation check against the paper, CKM dependence.
- `verify_bilarge.py` → `.log`/`.json`: CDSV I/II and RMSV, published benchmarks reproduced.
- `verify_csd2_gut.py` → `.log`/`.json`: SU(5)+CSD2 reduced model, θ12^eL ∝ 1/θ12^dL.
- `quickfit.py` → `quickfit.log`/`.json`: official-scorer fits.
- `grammar_pilot.py` → `grammar_pilot.log`/`.json`: bi-large look-elsewhere pilot. It imports `TableChi2` from exp 009's `f2lib.py`, which exp 009 checked against `src.data.pmns_delta_chi2`; it gives the same χ² as the loader and `src.evaluate` at the CDSV-I fit point (8.2079 all three).

Rerun from this folder with `../../.venv/Scripts/python <script>` (each < 5 min, 1 core). No background processes.

## Caveats

- Web search found no 2020-2026 paper claiming a fixed CKM → PMNS map, but the search index covered recent preprints poorly. An arXiv listing search (INSPIRE citations of 1902.08962 and 1808.09364) is the cheapest follow-up.
- Ahn-Cheng-Oh 1102.0879 was read at abstract level only. Its charged-lepton matrices carry free phases ("phases play a crucial role"), so it is F2-like with extra freedom. Not verified numerically.
- The Frampton-Mohapatra θ12^l ≈ θ_C uses the paper's benchmark (η ≈ 1.5, m_s ≈ m_μ/2 at v_R). The model's θ12^l is a mass-ratio expression, not exactly θ_C.
- CSD2-GUT is a reduced model:
  - only the 1-2 block of the charged leptons;
  - θ12^dL from the CKM via the paper's α = 90° construction, valid only when θ_C > s13/s23;
  - m_e/m_μ at low scale;
  - no thresholds.
  The paper's own fit (NuFIT 3.2, older light-quark errors) reports χ² < 4 for 10 tuples. Our χ² 43-62 is dominated by m_d/m_s at the 2% floor, which is the same light-quark-precision effect seen in exps 014, 016 and 022.
- No RG running was applied to PMNS. Low-scale comparison throughout, as the bi-large papers do.
- The pilot null uses a 3° φ grid and a fit-only hit rule. It is indicative, not the protocol.

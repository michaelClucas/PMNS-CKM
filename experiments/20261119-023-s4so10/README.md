# 023: S4 × SO(10) with shared CSD3 alignments (arXiv:1705.01555), reproduction, refit, Z3 phases, held-out PMNS

## Hypothesis

P3 from exp 021. In the Björkeroth-de Anda-King-Perdomo model every Yukawa matrix is a sum of the same few matrices built from three CSD3 flavon alignments, so V_CKM and U_PMNS come from shared structure. If the model fits PDG 2026 + NuFIT 6.1 with a 2% light-quark floor, and fixing its six free phases to Z_n roots brings k to ≤ N − 2, it would be a compressive known model. Separately: if the shared structure is real, quark + charged-lepton data should constrain the PMNS prediction.

## Model (transcribed from the LaTeX source, Eqs. 10-17, App. C-D)

- Y11 = φ̃1φ̃1ᵀ with φ̃1 = (1,1,3), Y22 from (0,1,1), Y33 from (0,0,1), Y12 = φ̃1φ̃2ᵀ + φ̃2φ̃1ᵀ, Y_P = [[0,0,−1],[0,2,0],[−1,0,0]].
- Y^u = y1u e^{iη} Y11 + y2u Y22 + y3u e^{iη'} Y33; m_ν = μ1 e^{iη} Y11 + μ2 Y22 + μ3 e^{iη'} Y33 (seesaw keeps the form).
- Y^d = y12d e^{iη/2} Y12 + y2d e^{iα_d} Y22 + y3d e^{iβ_d} Y33 + y_P e^{iγ} Y_P; Y^e the same with y12e, y2e e^{iα_e}, y3e e^{iβ_e} and the same y_P e^{iγ}.
- η = −2π/3 fixed by the flavon potential. 13 real couplings + 6 free phases (η', α_d, β_d, α_e, β_e, γ). GUT scale, MSSM tanβ = 5, M_SUSY = 1 TeV, threshold η̄_b = −0.8 chosen "empirically" by the authors.
- Convention: left rotations from Y Y†, V = U_u† U_d, light-ν matrix in the charged-lepton basis U_eᵀ m_ν* U_e. Of the conjugation choices tried, this is the one that gives the paper's printed δ_q = 69.2°, θ23^ℓ = 41.96°, δ_ℓ = 300.9°, α21 = 67.9°, α31 = 164.2° at its Table 4 point. The other choices give δ_ℓ = 59°, 63° or 297°, or δ_q = 291°.

## Method

1. `reproduce.py`: the paper's 18 observables (Tables 2-3, GUT-scale Yukawas, angles in degrees, NuFIT 3.0), its asymmetric 1σ ranges, χ² as in its Eq. 25 (δ_ℓ not fitted). Two scores: χ² at the printed Table 4 point, and χ² after our `least_squares` refit (16 starts). The inputs are in `pub_inputs.json`; they are the paper's published inputs, used only for this step.
2. `fit_ours.py`: refit to our data with the official scorer.
   - Call: `src.evaluate.Scorer(use, "NO", mass_floor=0.02, scale={"mu": 2e16, "model": "MSSM", "tan_beta": 5, "m_susy": 1000})`. GUT-scale predictions are translated to M_Z by the data's own `src.rg` running.
   - use (N = 16): s12sq, s13sq, s23sq, dcp_deg, dm_ratio, ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau.
   - δ_ℓ is scored (the paper left it out). NuFIT 6.1 NO, IC23_woSKatm.
   - η̄_b is not in `src.rg`. When used, it is applied on the model side to linear order (`fitlib.apply_eta_b`): s13^q, s23^q and m_s/m_b are divided by (1 + η̄_b), and m_b/m_τ is multiplied by it. This neglects the η̄_b dependence of the MSSM running itself, which is O(1-2%) at η̄_b = −0.8 and larger near −0.94.
   - 24 starts per variant: the published point plus random magnitude, sign and phase perturbations.
3. `clebsch.py`: effective Clebsch factors (complex e/d coefficient ratios) and R at every fit point. R is scored two ways: against exp 021's 0.0938 ± 0.0027, and with the `src.evaluate` key `R_clebsch` (mass_floor 0.02).
4. `fit_zn.py`: the six free phases fixed to Z3 roots {0, 2π/3, 4π/3}, couplings real with either sign. All 729 assignments are fitted (warm start), then the best 20 are refitted from 7 starts. Base setting η̄_b = −0.8.
5. `heldout.py`: adaptive-Metropolis posterior (4 chains × 40k steps; samples every 20 after 1/3 burn-in) given only quark + charged-lepton data: CKM 4 + mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau (N 11), same scorer and η̄_b = −0.8. A second run adds dm_ratio. Priors: log-uniform within ±2 decades of the best full fit for every |coupling|, μ_i included; phases uniform; signs fixed to the full-fit pattern. Output: the posterior-predictive PMNS spread.

## Parameter count (honest)

- Continuous, dimensionless: 13 couplings − 3 unobservable overall scales (Y_u; m_ν; the common scale of (Y_d, Y_e)) = 10. Add 6 phases for 16, and η̄_b when fitted or chosen for 17. The paper's own count is 19 (13 + 6) for 18 dimensionful observables.
- Discrete, data-selected (estimates):
  - CSD(n) with n = 3 out of n ∈ 1..5: 3 bits.
  - η = ±2π/3, "fixed up to a sign": 1 bit.
  - Which flavon bilinears enter Y^d/Y^e: {12, 22, 33} out of 6 bilinears, 20 options: 5 bits. Including the Planck operator Y_P: 1 bit.
  - Normal ordering (the authors checked IO fits badly): 1 bit.
  - tanβ = 5 chosen empirically: 1 bit.
  - Total base ≈ 12 bits.
- Z3 variant: the 6 phases become 0 continuous + ceil(log2 3^6) = 10 bits.

## Results

### 1. Reproduction (`reproduce.json`): passes

| | ours | paper |
|---|---|---|
| χ² at the printed Table 4 point | **3.52** (quark 3.47, lepton 0.05) | 3.4 (quark 3.38, lepton 0.03) |
| χ² after our optimizer, their inputs | **3.00** (lepton 0.0002), δ_ℓ = 299.9° | — |
| predictions at the printed point | θ12,13,23^ℓ = 33.62, 8.454, 41.96°; δ_ℓ 300.9°; m_i = 0.441, 8.68, 50.24 meV; α21 67.9°, α31 164.2°; δ_q 69.24°; y_d,s,b = 2.715e-5, 5.16e-4, 0.138 | 33.62, 8.455, 41.96; 300.9; 0.441, 8.680, 50.24; 67.90, 164.2; 69.21; 2.710e-5, 5.168e-4, 0.137 |

The 0.1 χ² difference at the printed point comes from 4-digit rounding of the couplings (y_e pull −0.15 vs ~0). Largest pulls: θ23^q −0.97, y_s +0.90, y_d +0.84, y_b −0.65, as in the paper's Fig. 4.

### 2. Refit with our data (`fit_*.json`; N = 16, NO)

| variant | χ² | k_cont | dof | max pull | main pulls | δ_ℓ (GUT) |
|---|---|---|---|---|---|---|
| eb0: free phases, η̄_b = 0 (pure `src.rg`) | **212.1** | 16 | 0 | 12.0 | s23^q −12.0, m_s/m_b +7.0, s13^q +3.3, m_c/m_t +2.0 | 301° |
| eb08: free phases, η̄_b = −0.8 (paper's choice) | **9.53** | 17 | −1 | 1.93 | δ_ℓ +1.93, s23^q −1.56, m_d/m_s +1.26 | 302° |
| ebfree: free phases, η̄_b fitted | **4.42** at η̄_b = −0.94 (bound −0.95) | 17 | −1 | 1.36 | m_d/m_s +1.36, s23² −1.33, δ_ℓ +1.0 | 230° |
| real_eb0: all six phases 0 (CP only from η) | 520.1 | 10 | 6 | 16.8 | s23^q −16.8, s12² +8.7, s13^q +7.4, s23² −5.3, s13² −5.2 | 273° |
| real_ebfree | 167.9 | 11 | 5 | 7.7 | s12² +7.7, s23² −6.8, s13² −5.9, dm_ratio +3.5 | 274° |

- Without the large threshold, θ23^q and m_s/m_b cannot both be fitted. The paper's analytic estimate θ23^q ≈ √(m_s/m_b)-type fails by 12σ, and the eb0 starts agree (212.1, 213.0, 220-231).
- The good fits need η̄_b ≈ −0.8 to −0.94, a factor 5-16 enhancement of y_b at M_SUSY. In this model η̄_b is the knob that rescales θ13^q, θ23^q and m_s/m_b together. With it free, the optimizer runs to the edge of the range, y_b(GUT) ≈ 0.5, where our linear threshold treatment no longer holds.
- k_cont ≥ N in every free-phase variant, so p is undefined (dof ≤ 0). The real-phase variants have dof > 0 but p < 1e-30.
- Quark mixing is fitted by the down-sector couplings plus η̄_b. The lepton sector (s12², s13², s23², Δm² ratio) fits as well as in the paper, with δ_ℓ predicted at 230° or 302°: the paper's two δ windows, now at +1.0σ and +1.9σ against NuFIT 6.1.

### 3. Clebsch factors and R (`clebsch.json`)

| point | C12 = y12e/y12d | C22 | C33 | R(M_Z) | pull (exp 021) | pull (`R_clebsch`, floor 0.02) |
|---|---|---|---|---|---|---|
| paper Table 4 | −1.31 | 3.22 e^{−0.35iπ} | 0.24 | 0.0899 | −1.43 | −1.57 |
| our refit, paper inputs | −1.33 | 3.31 e^{−0.36iπ} | 0.24 | 0.0903 | −1.29 | −1.42 |
| eb08 | −1.33 | 3.13 e^{−0.37iπ} | 0.23 | 0.0908 | −1.12 | −1.23 |
| ebfree | −1.31 | 10.8 e^{+0.41iπ} | 0.08 | 0.0906 | −1.19 | −1.31 |
| eb0 | +1.13 | 2.39 e^{+0.54iπ} | 1.20 | 0.0918 | −0.75 | −0.81 |

- These are not group-theory Clebsches. They come from four 45_H VEVs with arbitrary SM-preserving directions (paper §2: "no fixed relationship between down-type quark and charged lepton Yukawa couplings"). C12, C22 and C33 are continuous fitted numbers.
- R is therefore not a prediction of the model. It is fitted through m_e/m_μ and m_d/m_s, and it sits at −1.1 to −1.6σ only because the fit trades it against the other mass ratios.
- The R filter of exp 021 cannot test this model.
- C22 is complex, so it can take any value. The fitted |C22| moves from 2.4 to 10.8 across the variants.

### 4. Z3-fixed phases (`fit_z3_eb08.json`)

| | value |
|---|---|
| assignments | 3^6 = 729 (η', α_d, β_d, α_e, β_e, γ each in {0, 2π/3, 4π/3}), 10 bits |
| best assignment (units of 2π/3) | (η', α_d, β_d, α_e, β_e, γ) = (1, 0, 0, 1, 1, 0) |
| χ² (N 16) | **36.96**, k_cont 11 (10 + η̄_b), dof 5, p = 6.1e-7 |
| pulls | s23^q −4.19, m_c/m_t +2.15, s13^q +1.79, δ_ℓ +1.78, s13² +1.74, m_d/m_s +1.48, dm_ratio −1.32 |
| next assignments | 41.5, 43.5, 44.6, 50.7, 55.0; 3 of 729 below 50, 10 below 100 |
| k total | 11 + 10 (phases) + ≈ 12 (base) ≈ 33 vs N − 2 = 14 |

- Fixing the phases to Z3 roots removes 6 continuous parameters (k_cont 17 → 11). It then fails fit quality (p 6e-7, θ23^q −4.2σ).
- It also fails compression on bits alone: 11 + 10 = 21 > 14 even with the ≈ 12 base bits ignored.
- Every assignment was warm-started only from the eb08 best couplings, and the top 20 were refined with 7 starts each, so the χ² is search-limited. The other assignments sit well above it (≥ 41.5). Wall time 15 min on 4 processes.
- Z2 (phases 0 or π, i.e. real couplings; `real_*` in §2) is worse: χ² 168-520.

### 5. Held-out PMNS from quark + charged-lepton data (`heldout_eb08_*.json`)

Quark + charged-lepton posterior (N 11; η̄_b = −0.8; 4 chains × 40k steps, acceptance 7-27%; best χ²_QL in chain 6.9, median 15.4):

| PMNS parameter | 2.5% | 16% | median | 84% | 97.5% | NuFIT 6.1 NO |
|---|---|---|---|---|---|---|
| sin²θ12 | 0.002 | 0.032 | 0.299 | 0.393 | 0.512 | 0.308 |
| sin²θ13 | 0.000 | 0.002 | 0.027 | 0.085 | 0.099 | 0.022 |
| sin²θ23 | 0.003 | 0.040 | 0.112 | 0.479 | 0.569 | ~0.47 |
| δ_ℓ (deg) | 2 | 46 | 227 | 333 | 356 | — |

- Inside the NuFIT 3σ range (1D): s12² 20%, s13² 1.9%, s23² 21%. All three angles at once: 0.08% of samples.
- Quark and charged-lepton data do not constrain PMNS in this model. The predictive spread covers essentially the whole prior range, and the real data sit in a 0.1% corner.
- The shared alignments do not transmit quark information. U_e is set by y12e, y2e e^{iα_e}, y3e e^{iβ_e}, which only m_e/m_μ and m_μ/m_τ constrain, and m_ν (μ1, μ2, μ3, η') is unconstrained apart from η' entering Y_u.
- This is the same pattern as SO(10) MN/FN (exp 011, 014): lepton mixing is accommodated by sector-specific couplings and phases.
- The `ql_dm` run (adding the Δm² ratio) did not mix. One chain never moved off an out-of-table start (acceptance 0), and the others ran at 6-7%. Its numbers in `heldout_eb08_ql_dm.json` are not usable. It narrows s12² toward small values only because the stuck chain is included. Indicative only, not quoted.
- The `ql` posterior is also not fully converged (low acceptance, 40k steps, signs fixed). It is indicative of the width, not a precise distribution.

### 6. Criterion-4 nulls

Not run. No variant passes both fit quality and k ≤ N − 2:
- Free-phase variants fit (χ² 4.4-9.5) but have k_cont 16-17 ≥ N = 16, plus ≈ 12 bits.
- Real-phase variants have k_cont 10-11 but χ² 168-520.
- The Z3 variant is in §4.

Under the strict hit rule nulls are vacuous for k > N − 2, and a fit-only null would not satisfy criterion 4 (STATE protocol).

## Verdict

Known baseline; no candidate. Not compressive, and with symmetry-fixed phases it fails fit quality.
- Reproduction passes. Our χ² is 3.52 at the published point (paper 3.4), and our optimizer gets 3.00 on the paper's inputs. The paper's neutrino convention is m_ν*, identified from its printed δ_ℓ.
- With our data (PDG 2026, 2% light-quark floor, official scorer at M_GUT, N = 16):
  - Free phases fit only with a large tanβ-enhanced threshold: χ² 9.53 at the paper's η̄_b = −0.8, and 4.42 with η̄_b free at the −0.94 edge. Here k_cont = 17 > N, plus ≈ 12 bits.
  - Without the threshold (pure `src.rg`) χ² = 212 (θ23^q −12σ, m_s/m_b +7σ).
  - δ_ℓ is predicted in the paper's two windows (230° / 302°), now at +1.0 / +1.9σ.
- The Clebsch factors are fitted continuous numbers (four 45_H VEVs with free directions), so R is fitted, not predicted. R pull −1.1 to −1.6σ.
- Z3-fixed phases: k_cont 11 + 10 bits (+ ≈ 12 base), χ² 37.0, p 6e-7. Real couplings: χ² 168-520. Neither compressive nor acceptable.
- Held-out: quark + charged-lepton data leave PMNS unconstrained (95% s12² 0.002-0.51, s23² 0.003-0.57, δ anywhere). The shared CSD3 alignments carry no measurable CKM → PMNS information here, consistent with exp 021's bound.
- P3 is closed: the "same alignments" model accommodates both sectors with ≥ N parameters and predicts PMNS only through lepton-specific couplings.

## Files

- `model.py`: matrices and observables (paper convention).
- `pub_inputs.json`: the paper's Tables 2-3 inputs and printed predictions.
- `reproduce.py` → `reproduce.json`.
- `fitlib.py`: scorer setup, η̄_b treatment, parameter packing, residuals.
- `fit_ours.py <variant>` → `fit_{eb0,eb08,ebfree,real_eb0,real_ebfree}.json`.
- `clebsch.py` → `clebsch.json`.
- `fit_zn.py eb08` → `fit_z3_eb08.json`. All 729 assignment χ² values are in `artifacts/023/z3_eb08_all.json` (gitignored).
- `heldout.py fit_eb08.json {ql,ql_dm} 40000` → `heldout_eb08_{ql,ql_dm}.json`. Samples are in `artifacts/023/*.npy`.
- `run.log`: the console output of every run.

Rerun from this folder with `../../.venv/Scripts/python <script> <args>`. Each script runs with ≤ 4 processes. The refits take 2-6 min per variant, the Z3 scan 15 min, and the held-out runs 2-3 min each.

## Caveats

- η̄_b is applied as a linear rescaling (Antusch-Maurer matching), not through `src.rg`. The MSSM running in `src.rg` is one-loop Yukawa with a single threshold.
- The eb08/ebfree fits are best-of-24 starts. eb08 starts land at 9.5, 10.2 and 10.4, so the minimum is probably close.
- The PMNS running from M_GUT uses m_lightest = 0. The model gives m1 ≈ 0.4 meV, negligible at tanβ = 5.
- In the held-out run the coupling signs are fixed to the best-fit pattern, and the prior box is ±2 decades. The PMNS spread depends on these priors for the neutrino couplings, which no quark datum constrains.

## Erratum (overseer, iteration 8)
The η̄_b treatment above is linear. Exp 026 added the full tanβ-enhanced threshold matching to `src.rg`; at the eb08 point the full treatment gives χ² 17.91, and a refit gives 11.59 (`experiments/20261119-026-infra-thresholds/refit_023_eb08.json`). The verdict is unchanged (k_cont 17 ≥ N 16).

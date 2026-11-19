# 20261119-022 SU(5) sparse Clebsch textures + CSD3 seesaw (P1, pre-registered)

## Pre-registration

Written before any fit of this model to measured data (timestamp in `run.log`). Nothing below is changed after the fits; deviations, if any, are listed in a separate "Deviations" section.

### Source model

Björkeroth, de Anda, de Medeiros Varzielas, King, arXiv:1503.03306 (A4 × SU(5) SUSY GUT), eqs. 2.15, 2.18-2.20 and the numerical forms 2.23-2.25. Read from the arXiv PDF (text extracted outside the repo).

Correction to the exp 021 description: in the published model Y^d has no 2-3 or 1-3 entries. The structure is (LR convention, Y_ij L_i R_j):

```
Y^u = [[1.296e-5 u11, 2.16e-4 u12, 3.6e-3 u13],
       [2.16e-4 u12,  3.6e-3 u22,  6e-2 u23  ],
       [3.6e-3 u13,   6e-2 u23,    u33       ]]      (real symmetric)

Y^d ∝ [[1.5e-5 d11, 2e-4 d12 e^{iθ}, 0        ],
       [0,          6e-4 d22,        0        ],
       [0,          0,               5e-2 d33 ]]

Y^e ∝ [[(4/9) 1.5e-5 d11, 0,                   0        ],
       [2e-4 d12 e^{iθ},  (9/2) 6e-4 d22,      0        ],
       [0,                0,                   5e-2 d33 ]]  = (C ∘ Y^d)^T

m_ν = m_a (0,1,1)(0,1,1)^T + m_b e^{iη} (1,3,1)(1,3,1)^T      (CSD3, two RH neutrinos, m1 = 0)
```

Clebsch factors (frozen): C11 = 4/9 (Y^e_11 / Y^d_11), C22 = 9/2, C12 = 1 (transposed position), C33 = 1 (b-τ). θ = π/3 (paper's choice, stated there to be irrelevant for the fit quality). η = +2π/3 (paper: one of the nine Z9 roots; sign chosen a posteriori from the δ_ℓ hint). CSD alignment n = 3.

Consequence fixed by the structure, stated before fitting: Y^e has no 2-3 or 1-3 entries, so θ23e = θ13e = 0 exactly and θ12e = O(m_e/m_μ × small). The left-handed charged-lepton rotation is nearly the identity, so the PMNS prediction is essentially pure CSD3 and does not depend on the CKM. The quark-lepton link in this model runs through masses only (m_e/m_μ, m_μ/m_τ from the down sector via Clebsches).

PMNS = U_e† U_ν with U_e the left rotation of Y^e and U_ν diagonalising m_ν. CKM = U_uL† U_dL. The complex-conjugation convention (paper footnote 7: conventions differ from MPT by Hermitian and complex conjugation) is fixed by the pipeline check, step 2: the convention that reproduces the paper's published δ_q = +69.18° (θ = +π/3) and δ_ℓ = −86.7° (η = +2π/3) at their Table 5 inputs is used. It is not chosen by fit quality.

### Free continuous parameters (k_cont = 9)

- Y^u shape: u11, u12, u13, u22, u23 with u33 = 1 (overall scale drops out of all scored ratios): 5
- Y^d shape: d11, d12, d22 with d33 = 1: 3
- neutrino: r = m_b/m_a: 1 (overall m_a drops out of dm_ratio)
- Bounds: u_ij, d_ij ∈ [−10, 10] (signs free), r ∈ [0, 1].

The paper's 16 inputs also include tan β (fixed), θ and η (fixed), η_b (fitted by hand) and the overall scales. Overall scales are not used (only ratios scored). η_b is replaced by not scoring mr_s_b (below).

### RG setup

- Predictions made at μ_GUT = 2e16 GeV and translated to the data scale with `src.evaluate`'s `scale=` (data's own running via `src.rg`): model "MSSM", tan β = 5, M_SUSY = 1000 GeV (the paper's Table 3 setup), m_lightest = 0, Majorana phases (0, 0).
- No scan over tan β or M_SUSY: 0 bits. tan β = 10 is rerun at the best fit as a sensitivity check only and is not used for selection.
- `src.rg` has no tan β-enhanced SUSY threshold corrections. b-τ unification needs one (paper: η_b = −0.244, fitted by hand), and m_s/m_b depends on (1+η_b)/(1+η_q). Hence mr_b_tau and mr_s_b are not scored in the primary score (equivalent to fitting the threshold as a free parameter on mr_s_b). The 1-2 sector ratios are threshold-safe (R argument, exp 021).

### Observables (primary, N = 14)

`use = [ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, mr_u_c, mr_c_t, mr_d_s, mr_e_mu, mr_mu_tau, s12sq, s13sq, s23sq, dcp_deg, dm_ratio]`

- Official scorer `src.evaluate.fit_and_evaluate`, ordering "NO" (the model predicts m1 = 0, normal ordering; no ordering bit), analysis IC23_woSKatm (default; IC24_wSKatm reported as sensitivity).
- mass_floor = 0.02 on every mr_* key (2% on light-quark ratios per program convention; 2% on m_e/m_μ and m_μ/m_τ as the floor for 1-loop MSSM Yukawa running and lepton thresholds, which `scale_systematic` will also report; mr_c_t gets the same floor for the 1-loop y_t running).
- Variant V1 (reported, not the verdict): add mr_s_b (N = 15, k_cont = 9), i.e. assume η_b = η_q.
- Reported alongside: χ², p, p_global, all pulls, R = (m_e/m_μ)/(m_d/m_s) pull (`R_clebsch` scored alone at the best fit, 2% floor), θ12e and θ23e of the best fit, `scale_systematic`.

### Parameter count

- k_cont = 9.
- k_discrete, primary ("as published"): 1 bit for the sign of η, which the paper says was chosen a posteriori from data. θ = π/3, n = 3, Clebsch set and texture are taken as fixed by the published model: 0 bits. k = 10 ≤ N − 2 = 12.
- k_discrete, strict (reported): η among 9 roots: 4 bits; CSD(n), n ∈ {1..5} (CSD3 was itself selected against 2013-14 data): 3 bits; Clebsch pair (C11, C22) from the ~8 standard SU(5) values each (Antusch-Spinrath list): 6 bits; total 13 bits, k = 22 > N − 2. Under the strict count the family is not compressive regardless of fit quality.

### Decision rules

- Fit quality passes if all |pull| < 2 and p_global > 0.05 (dof = N − k_cont = 5).
- Kill: p_global < 0.01.
- Nulls only if fit quality passes and k ≤ N − 2 (primary count). Then: power check first (20 injections × 300 fakes), N1 (fake CKM) and N3 (fake masses), 1000 fakes each, NO only, strict hit rule (`src.nulls.is_hit`), pass if ≤ 4 hits. Each fake refits the 9 continuous parameters with both η = ±2π/3 (the selected bit is rerun).
- Prediction recorded before fitting: the lepton sector is pure CSD3 up to a tiny θ12e. Exp 021 C4 found pure CSD3 χ² 12.0 for 5 lepton observables; if that holds here, p_global ≤ ~0.035 even with a perfect quark fit, so fit quality is expected to fail.

## Deviations

None in structure, data or scoring. Optimizer only: `fit.py` (official `fit_and_evaluate`, 48 Sobol starts + the paper's benchmark as an extra start) converged only from the paper start; the other starts stalled at χ² > 3000 (rugged landscape). A differential-evolution cross-check (`de.py`) stalled at 13777. So the fitted minimum is not proven global; `bound.py` gives a structure-only lower bound that settles the verdict anyway. A basin search around the best fit (`perturb.py`): seed 1, 40 log-normal-perturbed starts, 31 ended below χ² 30 and all of the lowest returned 22.402, the same as the primary fit. Seed 2: 40 starts, 30 below χ² 30, best also 22.402.

## Hypothesis

The published A4 × SU(5) + CSD3 structure (arXiv:1503.03306), frozen before fitting, fits CKM, mass ratios and PMNS at PDG 2026 / NuFIT 6.1 precision with k ≤ N − 2.

## Method

`model.py` builds the frozen GUT-scale Yukawas and CSD3 m_ν. `reproduce.py` checks it against the paper's tan β = 5 benchmark. `fit.py` fits the 9 shape parameters with the official scorer (scale translation MSSM, tan β 5, M_SUSY 1 TeV, μ 2e16 GeV, 2% floor on all mr_* keys, NO). `bound.py` computes the structural χ² lower bound.

## Free-parameter count

k_cont = 9, k_d = 1 bit (η sign), k = 10, N = 14 (N − 2 = 12): compressive under the as-published count. Strict count (η among 9 roots, CSD(n), Clebsch pair): k = 22, not compressive.

## Result

Pipeline check (`reproduce.log`): every published Table 5 output reproduced within 0.6% (CKM angles and δ_q ≤ 0.02%, PMNS ≤ 0.06%, Δm² ≤ 0.03%, Yukawa ratios ≤ 0.53%). Only CONJ = NUCONJ = False gives both published phase signs; that convention is used.

Structure (as pre-registered): θ23e = 0 exactly, θ12e = 2.7e-4 rad. The PMNS prediction is pure CSD3 and does not use the CKM. The scout's exp 021 description of Y^d (2-3 entries, θ23e tied to V_cb) does not match the paper.

Fits (`fit_*.json`, χ² = chi2_global, dof = N − 9):

| run | N | χ² | p_global | max abs pull |
|---|---|---|---|---|
| primary (tan β 5, IC23) | 14 | 22.40 | 4.4e-4 | 2.64 (mr_d_s) |
| tan β 10 | 14 | 22.57 | 4.1e-4 | 2.64 |
| IC24_wSKatm | 14 | 21.35 | 7.0e-4 | 2.64 |
| η = −2π/3 | 14 | 31.52 | 7.4e-6 | 3.36 |
| V1: + mr_s_b (η_b = η_q) | 15 | 95.99 | 1.7e-18 | 6.70 (mr_s_b) |

Primary pulls: ckm_s12 −0.03, ckm_s13 0.00, ckm_s23 0.00, ckm_δ −0.13, mr_u_c 0.00, mr_c_t 0.00, mr_d_s −2.64, mr_e_mu +1.82, mr_mu_tau 0.00, s12² +1.34, s13² +0.36, s23² +1.64, δ_ℓ +1.95 (273.3°), dm_ratio −0.37. χ² split: PMNS joint 11.95, dm_ratio 0.14, mass-ratio block 10.29, CKM 0.02. R at the best fit: 0.1044 (GUT), pull +4.34 (2% floor on R alone; the block-consistent √2 × 2% floor gives about +3.1). θ12e = 2.67e-4 rad, θ23e = 0. Running systematic (`scale_systematic`) is ≤ 2% on mr_c_t and negligible elsewhere. V1: m_μ/m_τ +4.98, m_s/m_b −6.70; the 2-3 Clebsch 9/2 fails without a large b threshold.

Structural lower bound (`bound.log`):
- PMNS + dm_ratio depend only on r (U_e ≈ 1). Their minimum χ² is 12.09 (IC23) or 11.04 (IC24).
- R = 2 m_s²/m_μ² ≥ 8/81 at the GUT scale. Equality needs Y^d_12 = 0, which removes the only CP phase. The mass-block χ² is at least R pull² ≥ 1.62² = 2.62.
- So χ² ≥ 14.7, p ≤ 0.012 for any parameter values, whatever the optimizer.

## Verdict

Fails fit quality; kill criterion met (p_global 4.4e-4 < 0.01). The structural bound p ≤ 0.012 holds independently of the search, so this does not depend on the fit being global. Nulls (criterion 4) were skipped, as the pre-registration requires when fit quality fails.

The model has no CKM → PMNS link: θ23e = 0 and θ12e ~ 3e-4. Lepton mixing is pure CSD3 (χ² ≈ 12 for 5 observables against NuFIT 6.1). The quark-lepton link is the Clebsch pair (4/9, 9/2), which forces R ≥ 8/81 and in practice R ≈ 0.104 (+3 to +4σ). Not a candidate. Known model, logged as a failed baseline.

## Files

`model.py`, `reproduce.py` → `reproduce.log/json`, `fit.py` → `fit_<tag>.log/json`, `de.py` → `de_primary_s1.*`, `bound.py` → `bound.log/json`, `perturb.py` (running at write-up), `run.log`. Rerun from this folder: `../../.venv/Scripts/python fit.py primary`.

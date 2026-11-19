# 20261119-029 Forecast freeze: simple CKM ↔ PMNS relations and their predictions

**Frozen on 2026-11-19.**
`frozen_relations.csv` SHA-256: `04d6574e93619ec122bc2c2b7cc48b7200283e51958e4bdac2f3af7113734217`

This is a pre-registration artifact, not a result. It fixes, before JUNO, DUNE and Hyper-K data, which simple CKM ↔ PMNS relations will be tested and what each predicts, so that a future comparison carries no selection freedom beyond the list itself. Nothing here is a candidate in the `CANDIDATES.md` sense; every entry is a known baseline or a numerological survivor that is still within 2σ.

## Hypothesis

None is tested. The question is operational: which simple relations that use CKM inputs survive today at ≤ 2σ, what do they predict for sin²θ12, sin²θ13, sin²θ23 and δ_CP in each ordering, and how many would projected JUNO/DUNE/Hyper-K precision exclude if the future central values equal today's NuFIT 6.1 best fit.

## Method

### Selection rules (fixed in the scripts, applied once)

| family | source | rule | in list |
|---|---|---|---|
| K | `src.baselines` registry (exps 003/006) | relation takes CKM inputs (`ckm_s12/s13/s23/delta_rad`) and predicts only PMNS observables; official scorer `src.evaluate.evaluate_baseline` at M_Z (no running), max \|pull\| ≤ 2 in at least one ordering | 6 of 21 screened (`known_screen.csv`): A4a-TBM, A4a-GR1, A4a-GR2, A4a-HEX (U = V† Φ X), A4b-BM (U = V Φ BM V†), E1 (KM-phase sum rule) |
| S | exp 010 grammar (`symbolic_survivors.py`) | complexity C ≤ 4 (27,051 distinct functions); ≥ 1 CKM leaf (θ12^q, θ13^q, θ23^q, δ^q); exp 010 validity rules; \|pull\| ≤ 2 in both orderings under the official scorer (1D NuFIT profile, inputs declared, central values) and under exp 010's input-error-convolved pull; at M_Z | 1008 survive (s12² 22, s13² 6, s23² 193, δ 787; all in `symbolic_survivors_all.csv`). Capped at 6 per target, ranked by complexity then max \|official pull\| over orderings: 24 frozen |
| F | exp 009 LO tier best (`tier_best.csv`) | GR2 and TBM with θ12e = θ_C, θ23e = θ_cb, std order, φ fitted, refit here with the official scorer | 2 |
| B | exp 027 bi-large | CDSV-I, CDSV-II (φ fitted), RMSV types 1, 2, 4 (ψ, φ fitted; type 3 = type 1), refit here in both orderings | 5 |

Total: **37 relations, 74 (relation, ordering) rows**. Rows with max \|pull\| > 2 today are kept in the file with `survives_today = False` (F2LO-GR2 IO, F2LO-TBM IO, CDSV-I NO, RMSV-2 NO and IO) and are not counted in the forecast. Surviving rows: **35 in NO, 34 in IO**. QLC atmospheric θ23 = π/4 ∓ θ_cb (exp 003) enters as S-s23-2 and S-s23-5; δ = −δ^q (exp 010 known relation) is S-dcp-1. RMSV-1 IO survives the pull rule but has p_global 0.0011 (and RMSV-4 IO 0.0059, A4a-HEX IO 0.010); the rule is the pull rule the brief set, p is recorded alongside.

Not included: the T1 lead (exp 010, GUT scale, up-quark mass ratios only, no CKM input) and every relation without a CKM input (TM1, TM2, solar sum rules with free θ12e, GST, Koide).

### Predictions and uncertainties (`freeze.py`)

- Fitted relations (K, F, B): refit at M_Z with `src.evaluate.fit_and_evaluate` / `evaluate_baseline` (32 starts), in each ordering separately (no ordering bit: both orderings are frozen and the ordering will be decided by data).
- `sig_ckm_*`: spread of the prediction at fixed fitted parameters over 2000 CKM draws (independent Gaussians on λ, A, ρ̄, η̄ with the PDG 2026 errors; PDG publishes no correlation matrix; standard angles shifted by the exact Wolfenstein map, centre = official standard-parametrization records).
- `sig_fit_*`: today's PMNS-data uncertainty on the fitted parameters, propagated linearly (Gaussian surrogate χ² with symmetrized NuFIT errors, numerical Hessian at the official best fit, positive-curvature directions only). When `sig_fit` is about the NuFIT 1σ width, that observable is absorbed by the fitted parameters, not predicted; the table marks it "(fit)".
- Symbolic relations: prediction and σ from exp 010's forward-mode propagation of all 11 input errors (quark masses, α_s, CKM).
- k: `k_cont` = fitted continuous parameters; `k_honest` = k_cont + data-selected discrete bits: registry k_d for K, ⌈log2 17220⌉ = 15 bits for F (LO-tier configs), 11 bits (CDSV) and 5 bits (RMSV) grammar bits from exp 027, ⌈log2 27051⌉ = 15 bits for S. No entry has k_honest ≤ N − 2.

### Projected sensitivities (`sensitivities.json`)

| observable | nominal σ | optimistic σ | source |
|---|---|---|---|
| sin²θ12 | 0.0016 | 0.0010 | JUNO 6 yr / 20 yr, Table 6 of arXiv:2204.13249 (Chin. Phys. C 46 (2022) 123001): 0.307 ± 0.0016 (0.5%) / ± 0.0010; Δm²21 0.3% / 0.2% (not used: no relation predicts it) |
| sin²θ13 | current NuFIT (≈ 0.00057) | same | no planned improvement on the reactor value (JUNO 6 yr 12%; DUNE approaches reactor precision after ~15 yr, arXiv:2006.16043) |
| sin²θ23 | 0.010 | 0.006 | interpolated from Hyper-K 10 yr: 0.017 at 0.50, 0.006 at 0.45 (arXiv:1805.04163 Table 1; arXiv:1502.05199). DUNE octant sensitivity for sin²θ23 < 0.47 or > 0.55 (arXiv:2006.16043) |
| δ_CP | NO 15°, IO 20° | NO 8°, IO 12° | interpolated: DUNE 10° at δ = 0 after 8 yr, 20° at −90° after 12 yr, 5-15° at 15 yr (arXiv:2006.16043, EPJC 80 (2020) 978, Table 11); Hyper-K 10 yr 7.2° at δ = 0, < 23° at ±90° (arXiv:1805.04163). NO best fit 207° is near CP-conserving, IO 283° near maximal |
| ordering | determined | | DUNE > 5σ for all δ within 2-3 staged yr (arXiv:2006.16043): only the true ordering's rows will be tested |
| CKM | PDG 2026 errors | ρ̄, η̄ errors × 0.25 | γ: LHCb Upgrade II ~0.35° with 300 fb⁻¹ (arXiv:1808.08865, via the CKM 2018 WG V summary arXiv:1901.03604); Belle II ~1-1.5° with 50 ab⁻¹ (arXiv:1808.10567). The ×0.25 is a crude stand-in; it only affects E1 and the A4 forms. λ and A kept at current errors |

The "interpolated" values are this experiment's reading of the published curves, not numbers printed in the sources. No PDFs are committed.

### Forecast (`forecast.py`)

Future likelihood: Gaussian in (s12², s13², s23², δ), δ residual wrapped, σ_eff² = σ_future² + σ_ckm². Future centres (`forecast_summary.json`):
- **best fit**: NuFIT 6.1 best fit of each ordering (NO: 0.3088, 0.02249, 0.470, 207°; IO: 0.3088, 0.02261, 0.555, 283°);
- **alt-octant**: the same with s23² moved to the opposite-octant minimum of the NuFIT 1D profile (NO 0.550, Δχ² 1.03; IO 0.470, Δχ² 2.27). This is reported because today's survival of many relations rests on the octant degeneracy.

Fitted relations are refit to the future likelihood (6 Nelder-Mead starts incl. today's parameters); p from χ² with dof = 4 − k_cont, converted to a two-sided Z. Symbolic relations: Z = \|pull\| on their one observable. Excluded at > 3σ: p < 2.70e-3; > 5σ: p < 5.73e-7.

A "current_gauss" scenario (today's symmetrized NuFIT errors, best-fit centre) is included as a check of the Gaussian treatment. It does not reproduce the official χ²: e.g. A4a-TBM IO 33.8 vs official 6.0, S-s23-1 NO \|pull\| 5.4 vs 1.76. The cause is the NuFIT s23² profile: both octants have local minima (NO Δχ² 1.03 at 0.55, IO 2.27 at 0.47) and the NO δ profile is nearly flat (max Δχ² 1.03). The Gaussian centred on the best fit has one octant only. So the forecast assumes the octant is resolved at the best fit (or, alt-octant, at the other minimum); both are shown.

## Result

### Frozen list (human-readable; full precision in `frozen_relations.csv`)

Prediction ± total (CKM ⊕ fitted-parameter) uncertainty. ✗ = max \|pull\| > 2 today in that ordering (kept, not counted).

| id | formula | k_cont / k_honest | NO: max\|pull\| (p_glob) | IO: max\|pull\| (p_glob) | s12² | s13² | s23² NO / IO | δ NO / IO (deg) |
|---|---|---|---|---|---|---|---|---|
| K-A4a-TBM | U = V_CKM† Φ U_TBM | 2 / 6 | 1.89 (0.029) | 1.80 (0.023) | fit | 0.0235 | 0.455 (fit) / 0.457 | 266 ± 2 / 266 ± 2 |
| K-A4a-GR1 | U = V_CKM† Φ U_GR1 | 2 / 6 | 1.99 (0.025) | 1.74 (0.018) | fit | 0.0235 | 0.455 (fit) / 0.458 | 288 / 288 |
| K-A4a-GR2 | U = V_CKM† Φ U_GR2 | 2 / 6 | 1.83 (0.032) | 1.81 (0.022) | fit | 0.0235 | 0.455 (fit) / 0.456 | 262 / 262 |
| K-A4a-HEX | U = V_CKM† Φ U_HEX | 2 / 6 | 1.95 (0.027) | 1.76 (0.010) | fit | 0.0235 / 0.0236 | 0.455 (fit) / 0.459 | 299 / 299 |
| K-A4b-BM | U = V_CKM Φ U_BM V_CKM† | 2 / 6 | 1.96 (0.053) | 1.55 (0.059) | fit | fit | fit | 274 / 274 |
| K-E1 | δ_KM^l = −δ_KM^q (3 KM angles fitted) | 3 / 4 | 1.95 (0.051) | 0.56 (0.18) | fit | fit | fit | 273 ± 1 / 268 ± 1 |
| F2LO-GR2 | U = V_e† D(φ) U_GR2, θ12e = θ_C, θ23e = θ_cb | 1 / 16 | 1.84 (0.065) | 2.42 ✗ | fit | 0.02323 ± 0.00014 | 0.4453 ± 0.0009 | 260 / 261 |
| F2LO-TBM | same, U_TBM | 1 / 16 | 1.87 (0.060) | 2.42 ✗ | fit | 0.02323 ± 0.00014 | 0.4453 ± 0.0009 | 265 / 265 |
| BL-CDSV-I | U_l(λ, Aλ², φ)† U_ν, s13 = λ, s12 = 2λ, s23 = 1 − λ | 1 / 12 | 2.70 ✗ | 1.68 (0.10) | 0.3062 ± 0.0021 | fit | 0.5717 ± 0.0015 | 125 / 235 |
| BL-CDSV-II | same, s23 = 3λ, SU(5) orientation | 1 / 12 | 1.64 (0.047) | 1.90 (0.013) | 0.3189 ± 0.0018 | fit | 0.5138 ± 0.0033 | 229 / 229 |
| BL-RMSV-1 | U_l(type 1)† U_BL, s13 = λ, s12 = s23 = ψλ | 2 / 7 | 1.79 (0.091) | 1.93 (0.0011) | fit | fit | 0.450 / 0.454 ± 0.007 | 323 ± 1 |
| BL-RMSV-2 | type 2 | 2 / 7 | 3.57 ✗ | 3.76 ✗ | fit | fit | 0.426 / 0.428 | 324 |
| BL-RMSV-4 | type 4 | 2 / 7 | 1.78 (0.077) | 1.76 (0.0059) | fit | fit | 0.486 / 0.489 ± 0.007 | 323 ± 1 |
| S-s12-1 | θ12 = √(m_s/m_b) / θ_C | 0 / 15 | 1.81 | 1.81 | 0.3211 ± 0.0030 | | | |
| S-s12-2 | θ12 = asin(π/4 − θ_C) | 0 / 15 | 0.43 | 0.43 | 0.3117 ± 0.0008 | | | |
| S-s12-3 | θ12 = cos(θ_C / √(m_μ/m_τ)) | 0 / 15 | 0.52 | 0.52 | 0.3123 ± 0.0021 | | | |
| S-s12-4 | θ12 = asin(θ_C + 1/3) | 0 / 15 | 0.78 | 0.79 | 0.3141 ± 0.0008 | | | |
| S-s12-5 | θ12 = θ23^q + cos 1 | 0 / 15 | 0.97 | 0.97 | 0.3024 ± 0.0007 | | | |
| S-s12-6 | θ12 = sin δ^q / (π/2) | 0 / 15 | 0.98 | 0.98 | 0.3023 ± 0.0059 | | | |
| S-s13-1 | θ13 = θ_C atan(π/4) | 0 / 15 | 0.34 | 0.16 | | 0.02269 ± 0.00014 | | |
| S-s13-2 | θ13 = √(θ13^q + m_s/m_b) | 0 / 15 | 0.29 | 0.51 | | 0.02232 ± 0.00018 | | |
| S-s13-3 | θ13 = √θ_C / π | 0 / 15 | 0.61 | 0.42 | | 0.02284 ± 0.00007 | | |
| S-s13-4 | θ13 = √(m_s/m_b) / sin δ^q | 0 / 15 | 0.46 | 0.68 | | 0.02222 ± 0.00053 | | |
| S-s13-5 | θ13 = cos √2 − θ13^q | 0 / 15 | 0.87 | 0.67 | | 0.02298 ± 0.00003 | | |
| S-s13-6 | θ13 = cos(θ13^q + √2) | 0 / 15 | 0.89 | 0.70 | | 0.02299 ± 0.00003 | | |
| S-s23-1 | θ23 = atan δ^q | 0 / 15 | 1.76 | 1.28 | | | 0.5711 ± 0.0106 | |
| S-s23-2 | θ23 = π/4 + θ23^q (QLC atm, +) | 0 / 15 | 1.16 | 0.84 | | | 0.5419 ± 0.0007 | |
| S-s23-3 | θ23 = atan √δ^q | 0 / 15 | 1.32 | 1.16 | | | 0.5357 ± 0.0054 | |
| S-s23-4 | θ23 = 1 − θ_C | 0 / 15 | 1.00 | 1.72 | | | 0.4875 ± 0.0007 | |
| S-s23-5 | θ23 = π/4 − θ23^q (QLC atm, −) | 0 / 15 | 0.83 | 1.74 | | | 0.4581 ± 0.0007 | |
| S-s23-6 | θ23 = π/4 − θ13^q | 0 / 15 | 1.35 | 1.85 | | | 0.4962 ± 0.0001 | |
| S-dcp-1 | δ = −δ^q | 0 / 15 | 1.97 | 0.42 | | | | 294 ± 1 |
| S-dcp-2 | δ = θ13^q − 2 | 0 / 15 | 1.49 | 1.31 | | | | 246 |
| S-dcp-3 | δ = δ^q + π | 0 / 15 | 1.50 | 1.29 | | | | 246 ± 1 |
| S-dcp-4 | δ = δ^q − π (same value as S-dcp-3) | 0 / 15 | 1.50 | 1.29 | | | | 246 ± 1 |
| S-dcp-5 | δ = θ23^q − 2 | 0 / 15 | 1.54 | 1.24 | | | | 248 |
| S-dcp-6 | δ = δ^q + 3 | 0 / 15 | 1.27 | 1.58 | | | | 238 ± 1 |

"fit" = absorbed by the relation's fitted parameters (fitted-parameter uncertainty ≥ half the NuFIT 1σ width); the frozen value is in the CSV. Symbolic angles in radians, all CKM inputs at M_Z. Per-row numbers (both orderings, all four observables, CKM and fit uncertainties separately) are in `tables.md` and `frozen_relations.csv`. S-dcp-3 and S-dcp-4 are the same function mod 2π; exp 010's 8-point fingerprint dedups on the raw value, not mod 2π, so both are in the grammar count. They are kept as frozen (the list is frozen as generated) and count as one prediction.

What is actually predicted:
- **sin²θ12** (JUNO): only the six S-s12 expressions and the bi-large CDSV forms (0.3062, 0.3189) predict it; every other fitted relation absorbs it with a phase.
- **sin²θ13**: the S-s13 expressions, A4a (0.0235), F2-LO (0.02323). The bi-large forms absorb it with φ. No improvement in σ(s13²) is projected, so the S-s13 rows stay untested (Z ≤ 0.9 in every scenario).
- **sin²θ23**: F2-LO 0.4453, CDSV-I 0.5717, CDSV-II 0.5138, RMSV 0.45-0.49, A4a IO 0.456-0.459, and six S-s23 expressions (three per octant).
- **δ_CP**: all K, F, B forms (238-323°, plus CDSV-I NO 125°) and the six S-dcp expressions.

### Forecast: projected significance if the future centre = today's best fit

Z for each row surviving today (`forecast.csv`; `tables.md` has all rows):

| id | ord | nominal | optimistic | nominal alt-oct | optimistic alt-oct | worst obs (nominal) |
|---|---|---|---|---|---|---|
| K-A4a-TBM / GR1 / GR2 / HEX | NO | 4.1 / 5.5 / 3.8 / 6.2 | 7.5 / 10.1 / 6.9 / 11.5 | 7.3-8.8 | 10.1-13.7 | δ +3.6 to +6.1 |
| K-A4a-TBM / GR1 / GR2 / HEX | IO | 6.7-6.8 | 8.3-8.5 | 1.5-1.8 | 1.7-2.3 | s13² +5.1 |
| K-A4b-BM | NO / IO | 4.3 / 6.4 | 8.1 / 10.3 | 7.6 / 1.1 | 12.9 / 1.2 | δ +4.4 / s23² −6.1 |
| K-E1 | NO / IO | 4.4 / **0.7** | 8.2 / **1.2** | 4.1 / 0.5 | 7.7 / 0.8 | δ +4.4 / δ −0.7 |
| F2LO-GR2 / TBM | NO | 3.8 / 4.0 | 7.4 / 7.9 | 10.7 / 10.8 | 18.2 / 18.5 | δ +3.5 / +3.8 |
| BL-CDSV-I | IO | 2.3 | 4.3 | 9.9 | 16.6 | δ −2.4 |
| BL-CDSV-II | NO / IO | 5.5 / 5.7 | 7.9 / 8.5 | 5.0 / 5.8 | 7.3 / 8.6 | s23² ±4 |
| BL-RMSV-1 | NO / IO | 7.7 / 9.9 | 14.6 / 15.2 | 12.0 / 2.5 | 20.1 / 4.3 | δ +7.7 / s23² −9.6 |
| BL-RMSV-4 | NO / IO | 7.6 / 6.1 | 14.6 / 9.4 | 9.3 / 2.1 | 16.5 / 3.8 | δ +7.7 / s23² −5.8 |
| S-s12-1 / 2 / 3 / 4 / 5 / 6 | both | 3.6 / 1.6 / 1.3 / 3.0 / 3.7 / 1.1 | 3.9 / 2.3 / 1.5 / 4.2 / 5.3 / 1.1 | same | same | s12² |
| S-s13-1 … 6 | both | 0.1-0.9 | 0.1-0.9 | same | same | untestable (σ(s13²) unchanged) |
| S-s23-1 / 2 / 3 (upper octant) | NO / IO | 6.9, 7.2, 5.8 / 1.1, 1.3, 1.7 | 8.3, 11.9, 8.2 / 1.3, 2.2, 2.4 | 1.4, 0.8, 1.3 / 6.9, 7.2, 5.8 | | s23² |
| S-s23-4 / 5 / 6 (lower octant) | NO / IO | 1.7, 1.2, 2.6 / 6.7, 9.7, 5.9 | 2.9, 2.0, 4.4 / 11.2, 16.0, 9.8 | 6.2, 9.2, 5.4 / 1.7, 1.2, 2.6 | | s23² |
| S-dcp-1 | NO / IO | 5.8 / 0.5 | 10.7 / 0.9 | same | same | δ |
| S-dcp-2 … 6 | NO / IO | 2.1-2.7 / 1.8-2.2 | 3.8-5.1 / 2.9-3.7 | same | same | δ |

Exclusion fractions over the rows surviving today in that ordering (M = 35 NO, 34 IO; `forecast_summary.json`):

| scenario | NO > 3σ | NO > 5σ | IO > 3σ | IO > 5σ |
|---|---|---|---|---|
| nominal, best-fit centre | 17/35 (49%) | 9/35 (26%) | 13/34 (38%) | 11/34 (32%) |
| optimistic, best-fit centre | 24/35 (69%) | 17/35 (49%) | 19/34 (56%) | 12/34 (35%) |
| nominal, alt-octant centre | 17/35 (49%) | 14/35 (40%) | 7/34 (21%) | 5/34 (15%) |
| optimistic, alt-octant centre | 23/35 (66%) | 17/35 (49%) | 15/34 (44%) | 6/34 (18%) |

By family (nominal, best-fit centre): all fitted forms (K, F, B) in NO are excluded at > 3σ (11/11), mostly by δ (they predict 260-323° against 207° ± 15°); in IO 8/10 are excluded at > 3σ, mainly by sin²θ23 (they sit in the lower octant, IO best fit 0.555). The exceptions are K-E1 IO (Z 0.7) and BL-CDSV-I IO (Z 2.3). Symbolic: 6/24 (NO) and 5/24 (IO) at > 3σ; the six S-s13 are untestable, and the S-s12/S-s23/S-dcp survivors are the ones whose central value sits near today's best fit, as the selection guarantees.

Two things carry most of the exclusion power: **the θ23 octant** (every fitted IO form except E1, CDSV-I and CDSV-II, and half of the S-s23 entries, flip between survived and excluded depending on which octant is confirmed) and **δ_CP in NO** (today the NO δ profile is nearly flat, so δ is a fresh test for every fitted form). JUNO alone excludes S-s12-1 (3.6σ) and S-s12-5 (3.7σ) at > 3σ (nominal) if the centre stays at 0.3088; S-s12-4 sits at 2.98σ (4.2σ optimistic).

### Inverse: future values that single out 1-2 relations (`windows.csv`, nominal σ, other observables at the best fit)

- **sin²θ12 (JUNO)**, either ordering: a measured value in [0.318, 0.325] leaves only S-s12-1 (θ12 = √(m_s/m_b)/θ_C) among the s12²-predicting entries (plus K-E1 in IO, which fits s12² freely); [0.295, 0.2985] leaves only S-s12-6 (θ12 = sin δ^q/(π/2)); NO [0.299, 0.3055] leaves S-s12-5 and S-s12-6. Near today's centre (0.307-0.312) 3-5 entries stay compatible, so a JUNO result there singles nothing out.
- **sin²θ23 (NO)**: each octant window holds 1-2 symbolic entries: [0.44, 0.465] only S-s23-5 (QLC atm, θ23 = π/4 − θ23^q); [0.48, 0.505] S-s23-4 and S-s23-6; [0.525, 0.54] S-s23-2 (QLC atm +) and S-s23-3; [0.565, 0.60] only S-s23-1. The fitted NO forms are excluded by δ at these points, so they do not appear.
- **δ_CP (NO)**: [20°, 40°] and [325°, 340°] leave only the RMSV forms; [210°, 215°] only S-dcp-6.
- **δ_CP (IO)**: [55°, 80°] only K-E1; [135°, 165°] only BL-CDSV-I; [310°, 330°] only S-dcp-1 (δ = −δ^q); [290°, 305°] K-E1 and S-dcp-1.

### Look-elsewhere for future tests (Šidák over the frozen list)

The list is frozen, so a future positive statement ("relation X agrees with the new data") pays only for the M rows that will be tested, i.e. the surviving rows of the true ordering: M = 35 (NO) or 34 (IO). Šidák: p_local = 1 − (1 − p_global)^{1/M}.

| global target | NO (M = 35): p_local, Z_local | IO (M = 34): p_local, Z_local |
|---|---|---|
| 0.05 | 1.46e-3, 3.18σ | 1.51e-3, 3.17σ |
| 3σ (2.70e-3) | 7.72e-5, 3.95σ | 7.95e-5, 3.95σ |
| 5σ (5.73e-7) | 1.64e-8, 5.65σ | 1.69e-8, 5.64σ |

Exclusions need no correction (each relation is excluded on its own). An agreement is evidence for a relation only through the predictions it makes at k_cont fixed, against a null; under a future centre equal to today's best fit, agreement is expected for the entries selected to sit near it. The Šidák factor removes the look-elsewhere over the list, not over the history that produced it (k_honest ≥ 4 bits for every entry, 15 bits for the symbolic ones).

## Parameter count

Per row in `frozen_relations.csv`: N = 4 (fitted families) or 1 (symbolic); k_cont 0-3; k_honest 4-16 including data-selected discrete choices. No frozen relation meets k ≤ N − 2 with honest counting, so none can become a breakthrough by future agreement alone; the freeze exists to test them without new selection freedom.

## Verdict

Pre-registration artifact, frozen on 2026-11-19 (SHA-256 above). 37 relations (74 ordering rows; 35 NO and 34 IO surviving at ≤ 2σ today). If future central values equal today's NuFIT 6.1 best fit, nominal JUNO 6 yr + DUNE/Hyper-K precision excludes 49% (NO) / 38% (IO) of the surviving rows at > 3σ and 26% / 32% at > 5σ; with the alternative octant the IO fractions drop to 21% / 15%. The survivors in every scenario are K-E1 (IO), the six untestable S-s13 expressions, and symbolic entries chosen to sit at today's centre. Nothing here is a candidate. A future test compares new data with `frozen_relations.csv` at the hash above and applies the Šidák thresholds in the table.

## Caveats

- Gaussian future likelihood; no octant or δ non-Gaussianity, no correlations between future measurements (s23²-δ correlation in long-baseline data ignored), δ resolution depends on the true δ and is interpolated.
- No RG running of PMNS (SM running ≤ 0.12°, STATE). Exp 010 symbolic entries are frozen at M_Z only; GUT-scale variants are not frozen.
- The fitted-parameter uncertainty uses a Gaussian surrogate around the official (table) best fit; it is a band, not a profile.
- Future CKM improvement modelled only as smaller ρ̄, η̄ errors in the optimistic scenario. λ uncertainty (driving the bi-large s12² prediction, ±0.002) is kept.
- `freeze.py` was run twice; the second run only changed the `sig_fit_*` columns (eigen-projection fix in the surrogate Hessian). `forecast.py` read the first file, but it uses only `params_today` and `sig_ckm_*`, which are identical (deterministic fits and seed). The hash is of the final file.
- A first `forecast.py` run without the alt-octant scenario was stopped and its outputs overwritten; `forecast.log` is the complete second run.

## Reproduce

From this folder, 1 core each, ~25 min total:
```
../../.venv/Scripts/python symbolic_survivors.py          # symbolic_survivors_all.csv (uses artifacts/exp010 caches if present)
../../.venv/Scripts/python freeze.py > freeze.log          # frozen_relations.csv, known_screen.csv
../../.venv/Scripts/python forecast.py > forecast.log      # forecast.csv, windows.csv, forecast_summary.json
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python tables.py > tables.md   # tables + SHA-256
```
Model code is imported from exp 009 (`f2lib.v_e`) and exp 027 (`verify_bilarge.cdsv/rmsv`); `src/` is imported only.

## Files

`symbolic_survivors.py`, `freeze.py`, `forecast.py`, `tables.py` (scripts); `sensitivities.json` (projected σ with citations); `frozen_relations.csv` (the frozen list); `symbolic_survivors_all.csv` (all 1008 symbolic survivors before the cap); `known_screen.csv` (registry screen); `forecast.csv`, `forecast_summary.json`, `windows.csv`, `tables.md` (forecast outputs); `freeze.log`, `forecast.log`.

# 014: SO(10) global fits, the 1306.4468 reproduction, held-out PMNS, real 10+126+120 ("FN")

## Hypothesis

1. Exp 011's failure to find large-lepton-mixing solutions of minimal SO(10) (10_H + 126bar_H, type I) came from its search, not from the model. A proper global optimizer should reproduce Dueck-Rodejohann arXiv:1306.4468 (χ² ≈ 1.1, "MN, no RGE").
2. If the model fits, conditioning on quark + charged-lepton data alone might predict PMNS (a held-out test), and the posterior might contain a low-dimensional PMNS-CKM relation.
3. The real-parameter 10 + 126bar + 120 model ("FN", one CP source) has fewer parameters than observables, so it might be predictive.

## Source check (1306.4468, read from the arXiv PDF v3)

- Eq. (4): Y_u = r(H + sF + i t_u G), Y_d = H + F + iG, Y_D = r(H - 3sF + i t_D G), Y_l = H - 3F + i t_l G, M_R = F/r_R. Type-I dominance throughout. Exp 011's `gen.py` already used this.
- MN = non-SUSY 10 + 126bar: 19 real parameters (H real diagonal 3, F 12, r, s (2), r_R). "No RGE" fit at M_GUT = 2e16 GeV against Table 3 GUT masses (m_d 0.00114 ± 0.000495, m_s 0.022 ± 0.0065, m_b 1.0 ± 0.04, m_u 0.00048 ± 0.000185, m_c 0.235 ± 0.0345, m_t 74.0 ± 3.85 GeV; charged leptons with an assumed 5% error) and Table 2 mixing/neutrino values (sinθ12^q 0.2246 ± 0.0011, sinθ23^q 0.042 ± 0.0013, sinθ13^q 0.0035 ± 0.0003, δ_CKM 1.2153 ± 0.0576, Δm²21 7.5e-5 ± 1.85e-6, Δm²31 2.47e-3 ± 6.85e-5 eV², sin²θ12 0.30 ± 0.013, sin²θ23 0.41 ± 0.031, sin²θ13 0.023 ± 0.0023; GGMSS arXiv:1209.3023). That makes 18 observables, with δ_PMNS not fitted. χ² = Σ pull² (Eq. 5), errors symmetrised, downhill simplex with restarts on up to 1700 cores. Published χ²_min: 1.103 (Table 5/6). Best-fit H, F, r, s, r_R are in Appendix A, Eq. (15). These DR inputs are hardcoded in `so10fit.py` as published fit inputs for the reproduction only. They are not measured data; all our-data runs go through `src/` via `data_gut.py`.
- FN (Sec. 2.3): all parameters in Eq. (4) real, motivated by spontaneous CP violation through purely imaginary 120_H VEVs (ref. [56] = Grimus-Kühböck, hep-ph/0612132; also Dutta-Mimura-Mohapatra hep-ph/0406262, and Joshipura-Patel arXiv:1102.5148 = ref. [14], which fitted the non-SUSY version). 18 parameters: H 3, F 6, G 3, r, s, t_u, t_l, t_D, r_R. Published FN no-RGE χ²: NH 6.6e-5, IH 9.0e-10 (Table 13).
- MN is non-SUSY, so SM running is the matching comparison and no MSSM run was needed.

## Method

`so10fit.py`: the exp 011 generator conventions (Weyl matrices, left rotations from M†M, m_ν = M_Dᵀ M_R⁻¹ M_D), the exp 011 quark-anchored MN parametrisation (Y_u real diagonal, Y_d = U* D U†, 17 dimensionless parameters plus log y-scale and log ν-scale for absolute-mass targets), an FN parametrisation, both targets, and a batched Levenberg-Marquardt.
- FN parametrisation (16 dimensionless): H = diag(h1, h2, 1) with the largest |h| set to 1 (basis permutation and overall Yukawa sign are unphysical), F real symmetric (6), G real antisymmetric (3), r > 0 (the sign of M_u is unphysical), s, t_u, t_D, t_l. Every entry is SC·sinh(u), so each can take either sign and spans many decades.
- Targets: "DR" = the 18 published observables above (absolute masses at M_GUT). "ours" = exp 011's 17-dimensional GUT-scale vector (mass ratios, CKM, PMNS incl. δ_ℓ, Δm² ratio; NO), regenerated with `data_gut.py` from `data/` + `src/rg.py` (SM, 2×10^16 GeV). It is identical to exp 011's `obs_gut.json`. Errors: max(PDG, 2%) per mass, CKM floors, NuFIT 6.1 symmetrised. "drrel" (diagnostic) = our central values with 1306.4468's relative errors (light-quark masses 30-40%, charged leptons 5%).
- `global_fit.py`: 3000 random starts per run (quark block at the data, phases / r / s / FN entries random), 400 LM iterations each, then 4 rounds of basin hopping (32 best minima × 12 perturbations, 500 iterations), then a 3000-iteration polish of the 16 best. 5 processes, about 5 min per run.
- `pt_sample.py`: parallel-tempering DE-MC. 32 temperatures (β geometric from 1 to 3e-5) × 24 walkers. All levels start from prior draws, so mode weights come from tempering and not from seeds. 25k iterations, second half kept (thinned 20), 5 independent ladders per case. Likelihood = the 12 quark + charged-lepton observables only (PMNS never enters). Prior = uniform on the exp 011 box (quark windows ±0.5 dex around the data, phases uniform, lg r ∈ [0, 4], lg|s| ∈ [-2, 2]); for FN, uniform in the sinh coordinates. Swap acceptance was 0.4-0.65 on every pair at the end. Ladder-to-ladder agreement is reported as the convergence check.
- `analyze_heldout.py`: PMNS posterior predictive vs a 2e5-sample prior predictive, PIT of the measured values, 68%-width ratios, joint PMNS χ² (5 observables). Step 4 regression: 5-fold grouped CV (groups = walkers) R² of each PMNS observable on the 12 CKM + mass observables, using linear, quadratic and gradient-boosted regressors, within the posterior vs on prior samples, plus on the model parameters θ for reference.

## Results

### 1. Reproduction of 1306.4468 (sanity): passes

| check | result |
|---|---|
| Their MN no-RGE point (Eq. 15) in our code, their inputs (`dr_point.py`) | χ² = 1.111 vs published 1.103. Every predicted observable matches Table 6 to the printed precision (e.g. sin²θ23 0.4063 vs 0.40631, m_d 0.000672 vs 0.00067). δ_CKM = +1.219 in our convention; the complex-conjugate convention gives -1.219 (an equivalent fit). |
| Same point mapped into the quark-anchored θ (`check_reparam.py`) | χ² = 1.1108, so the reparametrisation and the fit objective are exact |
| Our optimizer, their inputs (`fit_MN_DR`) | χ²_min = **1.097** (N = 18, k = 19). Same mode as theirs (lg r 1.839 vs 1.839, |s| 0.375 vs 0.376, arg s 0.168 vs 0.168). The best random start was 1.32, and 303 minima are below 2. |
| Their FN no-RGE NH point (Eqs. 23-24) | χ² = 0.005 (published 6.6e-5, the gap is rounding of the printed parameters); FN implementation verified |
| Their FN no-RGE IH point (Eq. 27) | angles and mass ratios at 0σ, but Δm² is ~100× too large: the printed r_R appears to be wrong by about 10× (likely typo); we did not pursue IH |
| Our optimizer, their inputs, FN (`fit_FN_DR`) | χ²_min ≈ 0 (1178 minima below 2; N = 18, k = 18) |

### 1b. With our inputs: the large-mixing mode exists, but χ² = 130

| fit (GUT scale, SM running, NO) | N | k | χ²_min | dominant pulls |
|---|---|---|---|---|
| MN, our data and errors (`fit_MN_ours`) | 17 | 17 | **129.9** | δ_ℓ +6.5, sin²θ23 -5.2 (0.390), m_d/m_s -4.4, δ_q +3.6, m_c/m_t +3.2, sinθ12^q +3.1 |
| MN, our data, δ_ℓ not fitted (`fit_MN_ours_nodl`, the DR observable set) | 16 | 17 | **79.9** | sin²θ23 -5.1, m_d/m_s -4.3, δ_q +3.4, m_c/m_t +3.0, s12^q +3.0 |
| MN, our central values with DR relative errors (`fit_MN_ours_drrel`) | 17 | 17 | **1.53** | m_d/m_s -1.1, rest < 0.5 |
| FN, our data and errors (`fit_FN_ours`) | 17 | 16 | **0.30** (p = 0.59, 1 dof) | δ_ℓ -0.54, rest < 0.05 |
| FN, δ_ℓ not fitted (`fit_FN_ours_nodl`) | 16 | 16 | 0 | exact fits exist with δ_ℓ anywhere from 40° to 317° (`fn_delta.json`) |

`fit_pulls.png` and `fit_summary.csv` have the details. The MN best fit with our data lies in the same large-mixing region as DR's (lg r = 1.84, s ≈ 0.35, nearly real). Exp 011's 131.8 was therefore close to the true optimum: **exp 011's search was not the problem. The difference from 1306.4468 is the error budget.** With our 2% per-mass floor (DR: 30-43% on m_u, m_d, m_s, 15% on m_c), minimal SO(10) cannot fit m_d/m_s, the Cabibbo angle, δ_q and θ23^ℓ at the same time. Swapping in DR's relative errors with our central values brings χ² back to 1.5. This is a fit-quality failure of MN against current data, conditional on the 2% floor and on no threshold or RGE corrections inside the fit. Those effects are larger than 2% (1306.4468's own "with RGE" fit worsens to 23). The fair statement is "MN is in > 5σ tension with present quark-mass precision at the no-RGE level", not a clean exclusion.

### 2. Held-out PMNS from quark + charged-lepton data: fails

Posterior predictive of PMNS (75,000 cold samples, 5 ladders), measured GUT-scale value, prior-only baseline (`heldout_*.json/.png`):

| | MN, our errors | MN, DR-size errors | FN, our errors | measured |
|---|---|---|---|---|
| sin²θ12 95% | [0.0005, 0.141] | [0.0002, 0.095] | [0, 0.39] | 0.309 |
| sin²θ13 95% | [0.0001, 0.012] | [0.0001, 0.0007] | [0, 0.0024] | 0.0225 |
| sin²θ23 95% | [0.0003, 0.140] | [0.0021, 0.020] | [0, 0.95] | 0.470 |
| lg Δm²21/Δm²3ℓ 95% | [-2.72, -1.22] | [-8.8, -2.2] | [-12.9, -0.77] | -1.52 |
| δ_ℓ | flat | flat | flat | 207° |
| PIT θ13 / θ23 | 0.993 / 0.998 | 1.000 / 1.000 | 0.998 / 0.86 | |
| posterior fraction with PMNS χ²₅ < 11.07 | 0 | 0 | 0 | |
| min PMNS χ²₅ in posterior (prior) | 323 (190) | 1485 (190) | 3005 (216) | |
| 68% width, posterior / prior: θ12, θ13, θ23 | 0.56, 21.7, 12.6 | 1.37, 1.67, 2.96 | 0.01, 0.00, 0.35 | |

- **MN:** conditioning on quark + charged-lepton data predicts quark-like small lepton mixing. The measured PMNS lies outside the 95% region in θ13 and θ23 and has zero posterior mass within the 95% joint region. This holds **even with DR-size errors, where a χ² = 1.5 full fit exists**: the large-mixing solution occupies a negligible fraction of the ql-allowed volume (s23² 97.5% quantile 0.017-0.034 in all 5 ladders). The large-mixing fit is an accommodation found by tuning the 5 free phases and s, not a prediction. Width ratios > 1 mean the ql data move the PMNS predictive away from the prior's extreme small-angle tail without sharpening it toward the data.
- **Convergence:** MN with our errors: the 5 ladders agree on the bulk (median s23² 0.001-0.012) but differ in the tail (97.5% quantile 0.002 in ladder 3 vs 0.14-0.19 in the other four), so the tail weight is uncertain by ~2×. MN-drrel ladders agree closely. FN ladders disagree strongly (median lg Δm² ratio from -10 to -3; cold ql χ² median 7-15), so **the FN held-out posterior is not converged and is dominated by its very wide sinh-coordinate prior**. Treat the FN column as indicative only. Even so, no FN posterior sample comes within PMNS χ²₅ < 3000.

### 3. FN (real 10 + 126bar + 120, one CP source) parameter count and fit

- k (dimensionless) = 16: h1, h2, F (6), G (3), r, s, t_u, t_D, t_l. With absolute scales (y-scale, r_R) it is 18, matching 1306.4468. Discrete choices: type-I dominance (1 bit), NO (1 bit), and the choice of which representation carries the CP phase / the reality assumption (≥ 1 bit). That gives k ≈ 16 + ~3 bits against N = 17 dimensionless observables, so it is **not compressive** (k ≤ N - 2 fails).
- Fit: χ² = 0.30 for 17 observables (p = 0.59 for 1 dof), so FN fits current GUT-scale data well. The one "prediction" (δ_ℓ) is not predicted: with δ_ℓ held out, distinct exact solutions give δ_ℓ from 40° to 317°. Among the all-17 minima found, δ_ℓ spans 191-237°, which only reflects that δ_ℓ was fitted.
- The held-out (ql-only) PMNS test fails as for MN, with the convergence caveat above.

### 4. Posterior relation between PMNS and CKM/masses: none

Cross-validated R² of PMNS observables on the 12 CKM + mass observables (best of linear / quadratic / gradient boosting):

| | s12² | s13² | s23² | cos δ_ℓ | sin δ_ℓ | lg Δm² ratio |
|---|---|---|---|---|---|---|
| MN posterior (our errors) | 0.08 | 0.02 | 0.08 | 0.10 | 0.08 | 0.09 |
| MN posterior (DR errors) | 0.12 | 0.04 | 0.08 | 0.09 | 0.04 | 0.32 |
| MN prior | 0.31 | 0.00 | 0.08 | 0.06 | 0.00 | 0.53 |
| FN posterior | 0.03 | 0.00 | 0.00 | 0.13 | 0.02 | 0.18 |
| FN prior | 0.14 | 0.44 | 0.26 | 0.08 | 0.00 | 0.28 |
| MN posterior, on θ (reference) | 0.52 | 0.17 | 0.54 | 0.71 | 0.72 | 0.70 |
| FN posterior, on θ (reference) | 0.87 | 0.23 | 0.99 | 0.85 | 0.83 | 1.00 |

Within the posterior, CKM + masses explain ≤ 12% of PMNS variance (except the neutrino mass ratio at 32% with loose errors), and less than on prior samples. The PMNS variation is carried by the parameters left free by the ql data (the 5 Y_d phases and s for MN; the F/G entries for FN). The gradient-boosted R² of 0.08-0.10 (linear and quadratic: ≈ 0) is small, may be inflated by mode structure that is shared across walkers, and is not evidence of a relation. **No low-dimensional PMNS-CKM relation found.**

## Parameter counts (honest)
- MN: k = 17 dimensionless (19 with y_b and r_R) + type I + NO (2 bits) vs N = 17 (DR set: 18 with k = 19). Not compressive.
- FN: k = 16 dimensionless (18 absolute) + type I + NO + CP-source/reality choice (~3 bits) vs N = 17. Not compressive.

## Verdict
- Pipeline sanity passes: the published 1306.4468 MN fit is reproduced in our code (their point gives χ² 1.111, our optimizer finds 1.097 in the same mode), and FN likewise.
- Exp 011's SO(10) conclusion stands, with a corrected reason. With current quark-mass precision (2% floor) minimal SO(10) type I reaches only χ² = 130 (80 without δ_ℓ). The gap to 1306.4468 comes from their 30-40% light-quark errors, not from search failure.
- Held-out PMNS prediction from quark + charged-lepton data fails for MN under both error budgets: lepton mixing is accommodated by tuning, not predicted. FN fits all 17 observables (χ² 0.3) but predicts neither δ_ℓ nor the PMNS angles from ql data (its posterior is unconverged and prior-dominated).
- No PMNS-CKM relation in the posteriors. No candidate.

## Limitations
- No RG running inside the fit (fits at M_GUT against SM-run data, as DR "no RGE"), no RH-neutrino thresholds, no intermediate scales, no Higgs-mass constraint. All of these are larger than the 2% mass floor, so the MN χ² = 130 is a statement about that idealised setting.
- PT budget: 25k iterations per ladder (5 ladders) per case. The 100k-iteration runs in `run_all.sh` were stopped at 5k iterations (the shared machine gave 31 ms/iteration) and replaced by `run_rest.sh`. MN tail weights are uncertain by ~2×; FN is not converged.
- Posterior conclusions depend on the uniform prior box (exp 011's, data-centred quark windows). Under DR errors the quark windows (±0.5 dex) truncate m_d at ~2.6σ.
- The minima catalogues (`fn_delta.json`, counts in `fit_*.json`) are optimizer hits, not posterior weights.
- No `src/` bugs found. Reminder from exp 011: `src.rg.observables` takes the ordering from `state["ordering"]`. Typo in 1306.4468 Eq. (27): the FN-IH r_R does not reproduce Δm² (angles do).

## Files / rerun
```
cd experiments/20261118-014-so10-fits ; PY=../../.venv/Scripts/python
$PY data_gut.py ; $PY dr_point.py ; $PY check_reparam.py
bash run_all.sh   # as run: dr_point, check_reparam, MN DR/ours/drrel fits (PT part superseded)
bash run_rest.sh  # PT MN, FN ours fit, PT FN, MN nodl fit, PT MN drrel, analyses
$PY global_fit.py FN ours 3000 7 5 FN_ours_nodl nodl ; PYTHONIOENCODING=utf-8 $PY fn_delta.py
$PY analyze_heldout.py MN MN_ql_1,...,MN_ql_5 heldout_MN   # (and _drrel, FN) ; $PY plot_pulls.py
```
Large outputs (fit minima, PT samples) are in gitignored `artifacts/so10/`. Seeds are in the commands. To extend: rerun `pt_sample.py` with more iterations (e.g. `pt_sample.py FN ql 200000 <seed> FN_ql_<seed> 32 24`), ~20-30 ms/iteration on a loaded core. FN needs a narrower physically motivated prior before its held-out test means anything.

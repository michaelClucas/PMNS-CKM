# 016: Unified quark-lepton modular-flavour models: literature survey, reproduction, refit

## Hypothesis

Modular flavour symmetry with one modulus τ shared by quarks and leptons is the main literature class that claims fewer parameters than observables for both sectors. If one of these models fits our data with honest counting, it is a known baseline (criterion 5 fails), and its structure tells the generator (exp 017) where to search. If none fits, the survey says why.

## Method

1. **Survey** (`survey_raw.csv` from a literature sub-agent, corrected and ranked by `make_survey.py` into `survey.csv`). 39 papers, all arXiv IDs checked against arXiv metadata, 37 read from the PDF. Counts for the top rows were rechecked by me against the LaTeX sources (2409.15823, 2307.14926, 2506.19822). Our counting for each unified model:
   - N_dimless: the dimensionless observables a full SM-flavour fit can score. These are PMNS s12², s13², s23², δ_ℓ, Δm²21/Δm²31, me/mμ, mμ/mτ, CKM 4, mu/mc, mc/mt, md/ms, ms/mb, which gives 15. Absolute masses are dropped together with the overall scale each one fixes.
   - k_cont: the claimed real parameters minus the overall dimensionful scales.
   - bits_est: ceil(log2 of the number of alternative assignments the paper or its series scanned), plus 3 bits for choosing the finite modular group (about 8 groups were tried across the literature), plus 1 bit for gCP or not, plus 1 bit for holomorphic vs non-holomorphic where that applies. These are estimates.
2. **Pick**: the 2O model (Ding-Liu-Lu-Weng arXiv:2307.14926; joint analysis by Ding-Lisi-Marrone-Petcov arXiv:2409.15823). It is the most compressive holomorphic model, and the only one with an independent published joint quark+lepton χ² analysis. The non-holomorphic T' model (arXiv:2506.19822) has one parameter fewer (13 with gCP), but it needs polyharmonic Maaß forms of weights 0-4 (incomplete Gamma-function terms). Its published joint fit already has χ² = 22.65 for 19 observables and 13 parameters (p ≈ 9e-4) under its own loose errors. It was therefore scored at its published point against our data (`tprime_check.py`), not reimplemented.
3. **Modular forms** (`modforms_2o.py`): q-expansions of Y₂⁽²⁾, Y₃⁽²⁾, Y₄̂⁽³⁾, Y₁⁽⁴⁾, Y₃'⁽⁴⁾, Y₂̂⁽⁵⁾, Y₂̂'⁽⁵⁾, Y₄̂⁽⁵⁾, Y₁⁽⁶⁾, Y₁'⁽⁶⁾, Y₂⁽⁶⁾, transcribed from the LaTeX source of 2409.15823 App. A to O(q⁶). The truncation error is below 1e-12 for Im τ ≥ 1. Check: Y(−1/τ) = (−τ)^k ρ(S) Y(τ) holds to ≤ 2e-10 relative for every multiplet, using the paper's ρ(S) matrices (`run.log`).
4. **Model** (`model_2o.py`): M_E, M_D, M_N, M_u, M_d exactly as in 2409.15823 Eqs. (2.3), (2.8). The convention is F^c M f (left rotation from M†M), with Takagi diagonalisation of M_ν = −M_Dᵀ M_N⁻¹ M_D and PMNS = U_e† U_ν. All couplings are real (gCP).
5. **Reproduction** (`reproduce_2o.py`, `fit_pub_inputs.py`): the printed points, and a refit to the paper's own inputs (their Tables 1-2: GUT scale, MSSM tanβ = 5, Bari-2021 lepton data with sin²θ23 set to 0.5).
6. **Refit to our data** (`fit_ours.py`):
   - Model at M_GUT = 2e16 GeV, MSSM tanβ = 5, M_SUSY = 10 TeV (the paper's setting). Our data are run up with `src.rg`.
   - Mixing predictions are brought to M_Z by the data's own running shift and scored by `src.evaluate` with `use = (s12sq, s13sq, s23sq, dcp_deg, ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad)`, NO.
   - The 6 in-sector mass ratios and Δm²21/Δm²31 are scored at M_GUT with exp 014's noise model (each mass gets max(PDG, 2%), combined in quadrature).
   - N = 15, k_cont = 10: Re τ, Im τ, gE2/gE1, gE3/gE1, gu2/gu1, gu3/gu1, gd2..5/gd1, all real with either sign allowed. The four overall scales cancel in every scored observable.
   - Search: a lepton profile χ²(τ) on 758 grid points (3 starts each), a quark profile at the lepton-allowed τ, then a joint least-squares polish that also starts from the published point.
   - Variants: "ours" (2% floor), and "loose", a diagnostic that uses 2409.15823's relative errors on r_uc, r_ct, r_ds, r_sb (31%, 4.3%, 12%, 5.5%).
   - `lepton_only_ours.py` fits the lepton sector alone (k = 4, N = 7).

## Results

### 1. Survey: top 10 unified models by N − k (our counting; full table in `survey.csv`)

| arXiv | group / framework | k claimed | k_cont (dimless) | N_dimless | N − k_cont | bits est. | N − k_total est. | published fit |
|---|---|---|---|---|---|---|---|---|
| 2506.19822 | T' = Γ'₃, non-holomorphic, gCP, type I (2 RH ν) | 13 (14 no gCP) | 9 | 15 | 6 | ~12 | ≈ −6 | χ² 22.65 (19 obs, 13 par, p ≈ 9e-4); 13.33 without gCP (p 0.02); their md/ms error ±53% |
| 2409.15823 | 2O, holomorphic, gCP, type I | 14 | 10 | 15 | 5 | ~15 | ≈ −10 | χ²_comb 20.3 (18 obs, 14 par, p ≈ 5e-4); s12 +3.2σ, ms/mb +2.7σ |
| 2307.14926 | 2O (same model, original paper) | 14 | 10 | 15 | 5 | ~15 | ≈ −10 | χ² 32.6 (lepton 20.0 + quark 12.6) |
| 2006.10722 | S'₄ = Γ'₄, gCP | 15 | 11 | 15 | 4 | ~16 | ≈ −12 | "within 3σ", χ² not quoted |
| 2012.13390 | A₄ = Γ₃, gCP | 16 | 12 | 15 | 3 | ~25 | ≈ −22 | within 3σ; "thousands" of 16-parameter combinations work |
| 1912.07573 | T', texture zeros | 16 | 12 | 15 | 3 | ~10 | ≈ −7 | only within 5σ |
| 2311.10136 | 2D₃ (binary dihedral) | 16 | 12 | 15 | 3 | ~17 | ≈ −14 | near central values, χ² not quoted |
| 2101.02266 | S₄ adjoint SU(5) | 17 | ~14 | 16 | 2 | ~9 | ≈ −7 | χ² 1.6 (NO) |
| 2101.12724 | A₄ SU(5) | 18 (24 for good fits) | 15 | 16 | 1 | ~13 | ≈ −12 | 18-parameter model has s23 and δ_q > 3σ |
| 2011.03501 | A'₅ = Γ'₅ | 19 | 15 | 15 | 0 | ~24 | ≈ −24 | Σmν ≈ 247 meV (excluded by cosmology) |

Not ranked:
- 2404.06520 (Pati-Salam A₄): k 19, N − k_cont ≈ 0.
- 2207.10677 (string-derived eclectic): k about 20.
- SO(10) A₄ (2108.09655): k 27.
- Flipped SU(5) (2209.08796): χ² ≈ 282 with one τ.
- Multiple-moduli and fixed-point models (2103.16311, 2103.02633, 2312.09010): the moduli sit at fixed points, and the counts are unclear or k ≥ N.
- Quark-only modular fits (2212.13336, 2306.05730, 2307.14410) report that one τ with spontaneous CP violation fails for the quarks. They need a second modulus or explicit CP breaking.

No published full quark+lepton fit has fewer than 13 real parameters (9 dimensionless).

### 2. Reproduction of 2409.15823 in our code: passes

| check | ours | published |
|---|---|---|
| printed combined point: PMNS s12², s23², s13² | 0.3444, 0.5082, 0.02314 | 0.344, 0.508, 0.0231 |
| δm², Δm² (eV²) | 7.24e-5, 2.467e-3 | 7.28e-5, 2.49e-3 |
| me/mμ, mμ/mτ, mτ | 0.00482, 0.05858, 1.292 GeV | 0.00474, 0.0586, 1.29 GeV |
| δ_ℓ, m1 | −2.005 rad, 15.00 meV | −2.01 rad, 15.05 meV |
| quark masses mt, mb, ratios | ≤ 0.2% off (r_ds, r_sb ~4%) | |
| CKM θ12, θ23, δ_q at the printed point | 0.217, 0.0410, 1.180 | 0.227, 0.0395, 1.23 |
| refit with their inputs: χ²_comb | **20.17** at τ = −0.2018 + 1.0708i, pulls s12 +3.21, r_sb +2.64 | **20.3** at τ = −0.202 + 1.071i, s12 +3.2σ, r_sb +2.7σ |
| lepton-only / quark-only refit | 4.67 / 0.000 | 4.74 / 0.0025 |

- The printed quark couplings have 3 significant figures. That is not enough to reproduce the CKM at the 4% level: the d-sector cancels between g2d Y₁' and g3d Y₂. The printed combined point therefore gives χ² 255 under their inputs, but our refit from it recovers their minimum.
- Two typos in 2409.15823 Table 4. First, g1E v_d is in GeV, not MeV (MeV gives mτ = 1.3 MeV). Second, the lepton-sector sin²θ23 is fixed at 0.5 by hand, not taken from data.

### 3. Refit with our data (`fit_ours_ours.json`)

| | value |
|---|---|
| best fit τ | −0.2052 + 1.0691 i (≈ the published one) |
| χ² (= χ²_global, NO) | **68.8** (src.evaluate part 48.3: PMNS ≈ 40.7, CKM ≈ 7.6; mass ratios + Δm² ratio 20.5) |
| N, k | 15, k_cont = 10 (+ ≈ 15 bits est. for group/assignment/gCP choices), dof 5 |
| p = p_global | **1.8e-13** |
| max pull | 5.24 (sin²θ12 = 0.346 at M_Z vs 0.308) |
| pulls | s13² +2.42, s23² +1.61, δ_ℓ +1.49 (δ_ℓ = 246°), s23^q −2.46, s13^q −1.10, r_sb +4.01, r_uc +1.94, Δm² ratio −0.81, rest < 0.6 |
| compression_ok | False with honest bits (k ≈ 25 > N − 2 = 13); True only if every discrete choice is ignored (10 ≤ 13) |
| fit_ok | False |
| lepton sector alone (`lepton_only_ours.json`) | χ² 14.4, N 7, k 4, p 0.0024. sin²θ12 = 0.325 (+2.33), s13² −1.38, s23² +1.58, δ_ℓ = 241° +1.35. The PMNS part is 13.9; the lepton mass ratios and the Δm² ratio fit. The lepton sector fails criterion 3 on its own. |

"Loose" diagnostic (`fit_ours_loose.json`: our central values, 2409.15823's relative errors on r_uc, r_ct, r_ds, r_sb):
- χ² = 41.4 for N 15, k_cont 10, p = 7.7e-8, at τ = −0.2003 + 1.0757i.
- Pulls: s12² +4.16, r_sb +3.81 (even with a 5.5% error), s23² +1.60, δ_ℓ +1.44, s23^q −1.39; the rest are < 0.6.
- The looser quark errors remove the s23^q and r_uc tension. The lepton θ12 and ms/mb remain. Our GUT ms/mb (0.01786) is slightly below theirs (0.0182), which costs about 0.5σ more.
- The joint polish had max_nfev 4000 here and 20000 in "ours" (both converged to the same τ island).

### 4. T' non-holomorphic (2506.19822) at its published point vs our data (`tprime_check.json`, no refit)

| | χ² (N 15) | dominant pulls |
|---|---|---|
| gCP, k 13 | 170 | s23^q −8.0, ms/mb −5.8, md/ms +5.3, s13^q −4.7, s12² −2.7 |
| no gCP, k 14 | 776 | md/ms +26, s23^q −5.2, s13^q −4.7, ms/mb −3.9 |

Its published quark inputs are older GUT values with md/ms ± 53%, ms/mb ± 23%, mu/mc ± 42%, θ13^q ± 9%. At the published point the model misses our SM GUT-scale CKM (θ13^q 0.0037 vs 0.0043, θ23^q 0.0374 vs 0.0475) by 5-8σ. A refit could move τ, but there are only 4 quark couplings for 8 quark observables, and the published fit already sat at its quark optimum under looser errors. Indicative only.

## Parameter count (2O model, honest)

- Continuous: 10 dimensionless (τ: 2; leptons gE2/gE1, gE3/gE1; quarks gu2/gu1, gu3/gu1, gd2..5/gd1), plus 4 overall scales that are not counted because no absolute mass is scored.
- Discrete, selected by data:
  - the group 2O among ~8 finite modular groups tried in the series (3 bits);
  - one rep/weight assignment out of "thousands" classified in 2307.14926 (≈ 11 bits);
  - gCP on (1 bit);
  - type-I seesaw with three RH neutrinos (part of the assignment).
  - NO is predicted by the model in the lepton-allowed τ region, so there is no ordering bit.
- Total k ≈ 10 + 15 bits ≈ 25 against N = 15. Criterion 2 fails even before the fit.

## Verdict

- No published unified modular model passes criteria 2 and 3 with our data and counting.
- The most compressive holomorphic model (2O, 14 parameters) is reproduced exactly (χ² 20.17 vs 20.3 with their inputs). Even with their inputs it has p ≈ 5e-4 (18 observables, 14 parameters).
- With our data it reaches χ² 68.8 for N = 15, k_cont = 10 (p 2e-13). The failure has two sources:
  - The lepton sector predicts sin²θ12 ≥ 0.32 wherever θ13, θ23 and the mass ratios fit. NuFIT 6.1 has 0.308 ± 0.012, and JUNO will sharpen this.
  - With PDG 2026 light-quark masses and a 2% floor, ms/mb (+4σ) and θ23^q (−2.5σ) cannot be fitted at the same τ. The published fits used 5-53% light-quark errors, as in exp 014. With those error sizes and our central values, χ² is still 41.4 (p 8e-8).
- The T' non-holomorphic model (13 parameters) fails worse at its published point (χ² 170).
- Known baselines, no candidate. The only quark-lepton link in these models is two real numbers (τ). The CKM is never predicted from leptons: the quark sector keeps n_quark_obs − 2 couplings.

## Leads for the generator (exp 017)

1. **τ-only neutrino sector (2O lepton structure).** With N^c ~ 3, M_D ∝ a constant permutation matrix (weight cancellation k_L + k_Nc = 0) and M_N from one weight-2 multiplet, the light ν mass matrix is fixed by τ up to one scale. PMNS and Δm²21/Δm²31 then depend only on τ plus the two charged-lepton coupling ratios, which the two lepton mass ratios fix. This gives 5 lepton mixing predictions from 2 real numbers (Re τ, Im τ).
   - The generator should scan this exact structure over Γ₃ (A₄), Γ₄ (S₄), Γ'₄, Γ₅, Γ'₅ and 2O, using every rep for N^c N^c Y^(2,4) and every charged-lepton assignment.
   - 2O lands at sin²θ12 ≈ 0.32-0.35. Look for groups or weights whose τ-locus passes through s12² ≈ 0.31 together with s13² ≈ 0.022.
2. **Quark textures with ≤ 4 couplings (T' non-holomorphic).** Weight-0 and weight-1 polyharmonic Maaß forms allow 4 coupling ratios for 8 quark observables. That is the most compressive quark sector in the literature, but its published point misses θ23^q and θ13^q by 5-8σ. Holomorphic 2O needs 6 ratios.
3. **Use the 2% light-quark floor from the start.** Every published "good fit" uses 12-53% errors on md/ms, mu/mc and ms/mb. With PDG 2026 (m_d ±1.5%, m_s ±0.75%), ms/mb and θ23^q are the binding constraints at a common τ.
4. **The τ region is universal.** Every viable quark+lepton fit sits near Re τ ≈ ±0.2, Im τ ≈ 1.05-1.1 (2O), or near τ ≈ i (T', A₄ gCP). The lepton-allowed island of the 2O model has width ≲ 0.04 in τ. An emulator over τ should resolve that scale.
5. **Counting.** These models carry ≈ 10-15 bits of assignment selection, so k ≤ N − 2 needs N − k_cont ≥ 12-17. No single-τ model reaches that. A generator that fixes the assignment in advance (no data selection), or adds observables that the same τ predicts (Majorana phases, m1, 0νββ, Σmν), would be needed to reach criterion 2.

## src gaps

- `src.evaluate` has no theory-floor option for masses (m_e is scored at ppb). It has no in-sector quark ratio keys (mu/mc, mc/mt, md/ms, ms/mb), no lepton mass ratios, and no Δm² or Δm² ratio. Each of the four quark-ratio keys it does have mixes flavours, so at most two can be used together. Mass ratios here were scored by our own code with exp 014's convention.
- `src.evaluate` cannot score GUT-scale predictions. Here, PMNS/CKM GUT predictions were shifted by the data's own running (exact at linear order for small deviations, MSSM tanβ = 5, one-loop Yukawas).
- `src.rg.pmns_at` gives the Δm² ratio at m_lightest = 0. The 2O model predicts m1 ≈ 15 meV, which changes the running of the ratio slightly (tanβ = 5, well below the 2.6% error).

## Files / rerun

```
cd experiments/20261118-016-modular-literature ; PY=../../.venv/Scripts/python
$PY modforms_2o.py            # S-transformation check of the transcribed q-expansions
$PY reproduce_2o.py           # printed points vs Table 3 -> reproduce_2o.json
$PY fit_pub_inputs.py         # refit with 2409.15823 inputs -> fit_pub_inputs.json (~35 s)
$PY fit_ours.py ours 2        # our data, 2 processes (~50 min) -> fit_ours_ours.json, profile_ours.json
$PY fit_ours.py loose 2       # diagnostic error budget -> fit_ours_loose.json
$PY lepton_only_ours.py       # lepton-only refit -> lepton_only_ours.json
$PY tprime_check.py           # T' published point vs our data (reads exp 014 obs_gut.json)
$PY make_survey.py            # survey_raw.csv -> survey.csv with our counting
```
Published parameters and published inputs are hardcoded in `reproduce_2o.py` and `fit_pub_inputs.py` as the papers' numbers (not measured data). All measured values come from `data/` via `src`.

# 20261118-006 Official baseline scores

**Hypothesis:** N/A (scoring infrastructure). Question answered: with one shared χ² (src/evaluate.py), how well does every catalogued known relation (exp 003) and the lead L1 fit NuFIT 6.1 + PDG 2026 data?

**Method:**
- `src/evaluate.py`: `evaluate(pred, ordering, analysis, k, use, inputs, k_discrete)` on top of `src.data` + `src.stats.Objective` (`pmns_mode="joint"`): all predicted PMNS parameters go into one `pmns_delta_chi2` call, so the 2D (s23², δ) table is used whenever both are predicted; other PMNS parameters add 1D profiles. CKM and mass observables are Gaussian with asymmetric errors. `fit_and_evaluate` minimizes χ² over continuous parameters (32 scrambled-Sobol starts, Nelder-Mead, polish; phases optimized unbounded and wrapped).
- `src/baselines.py`: port of exp 003 `baselines.py` (exp 003 copy unchanged), angle extraction through `src.mixing`, plus a registry `BASELINES` (37 entries) declaring for each relation: predicted observables, data inputs, fitted parameters (incl. nuisances), k_continuous, k_discrete, N.
- `score_baselines.py`: every registry entry × {NO, IO} × {IC23_woSKatm, IC24_wSKatm} = 148 fits. **No RG running**: everything is compared at low scale as given. Input sensitivity: for relations that plug in measured inputs, refit with each input at ±1σ (default analysis) and report max |Δχ²_min| (`input_dchi2`).

Rerun (repo root, ~7 min on 2 processes):
```
.venv/Scripts/python experiments/20261118-006-baseline-scores/score_baselines.py
```
Outputs: `results.csv` (one row per baseline/ordering/analysis with best-fit params, predictions, pulls), `run.log` (tables + L1 detail).

## Conventions (decided here, documented in `src/evaluate.py`)

- **N** = number of observables scored. **Inputs** taken from data at central value (θ_C, full V_CKM, m_e, m_μ) are not scored and not counted in N; an observable cannot be both input and prediction (`ValueError`). This is only used when the input's error is small next to the prediction's; otherwise the input is a **nuisance**: fitted parameter *and* scored observable (N+1, k+1, dof unchanged). Used for: θ13 in self-complementarity (A2), m_d and m_s in GST (C1), and all PMNS angles that sum-rule relations take as input (TM1/TM2, solar sum rules, KM sum rule are fitted as full models whose free angles play this role).
- **k** = k_c + k_d (CLAUDE.md strict count, k_d from exp 003 `baselines.md`). `p` uses dof = N − k (nan if ≤ 0). `p_fit` uses dof = N − k_c: the fit quality of the specific relation ignoring discrete choices; informational only, the breakthrough criterion uses `p`.
- **Ordering offset:** `chi2` is within-ordering (relative to that ordering's own minimum). `chi2_global` = `chi2` + offset (0 for NO; 1.486 IC23, 5.908 IC24 for IO), added exactly once and only if a PMNS parameter is predicted. Same dof for both. Picking the better ordering after seeing scores is one more k_d, not added in this table.
- PMNS "pulls" are sign·√(1D Δχ²) per parameter (informational); χ² is the joint value, so Σpull² ≠ χ² when s23² and δ are both predicted. δ pulls are signed on the wrapped difference.
- At the NuFIT best-fit point χ² = 0.012 (NO) / 0.014 (IO), not 0: published values are rounded and tables are gridded. Bilinear interpolation also makes fitted minima land on grid nodes (e.g. identical θ12e, θ23ν across B1 patterns).

## Result (IC23_woSKatm default analysis; IC24 in run.log/results.csv)

χ² within-ordering; χ²_g = global (IO offset 1.49). p uses dof = N − k; `-` = dof ≤ 0. p_fit in parentheses where p is undefined.

| id | relation | k (c+d) | N | NO χ² | NO p (p_fit) | NO max\|pull\| | IO χ² | IO χ²_g | IO p (p_fit) | IO max\|pull\| | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1s | QLC θ12 + θ_C = π/4 | 0 | 1 | 19.2 | 1.2e-5 | 4.4 | 19.2 | 20.7 | 1.2e-5 | 4.4 | excluded |
| A1− | QLC + θ23 = π/4 − θ_cb | 1 (0+1) | 2 | 19.9 | 8.2e-6 | 4.4 | 22.2 | 23.7 | 2.5e-6 | 4.4 | excluded |
| A1+ | QLC + θ23 = π/4 + θ_cb | 1 (0+1) | 2 | 20.6 | 5.8e-6 | 4.4 | 19.9 | 21.4 | 8.2e-6 | 4.4 | excluded |
| A2 | self-compl. θ12 + θ13 = π/4 | 2 (1+1) | 2 | 33.4 | - (7e-9) | 5.7 | 33.0 | 34.4 | - (9e-9) | 5.6 | excluded |
| A3 | TBC s13 = s_C/√2, TBM 12/23 | 1 (0+1) | 3 | 39.7 | 2.3e-9 | 5.0 | 37.5 | 39.0 | 7.1e-9 | 4.6 | excluded |
| A3k | KMS θ13 only | 1 (0+1) | 1 | 25.2 | - (5e-7) | 5.0 | 21.5 | 23.0 | - (4e-6) | 4.6 | excluded |
| A4a-GR2 | V† Φ X, best X (GR2) | 4 (2+2) | 4 | 6.87 | - (0.032) | 1.8 | 6.17 | 7.66 | - (0.046) | 1.8 | disfavoured; TBM/GR1/HEX similar (6.0-7.7), BM 21/40 |
| A4c-GR2 | V* Φ X, best X | 4 (2+2) | 4 | 16.1 | - (3e-4) | 3.6 | 13.1 | 14.6 | - (1e-3) | 3.3 | excluded (all X) |
| **A4b-BM (L1)** | V Φ U_BM V† | 4 (2+2) | 4 | **5.89** | - (**0.053**) | 1.96 | **4.17** | 5.65 | - (**0.125**) | 1.55 | lead; fails k ≤ N−2; other X ≥ 42 |
| B-TBM..HEX | LO patterns (θ13 = 0) | 1 (0+1) | 3 | 1863-2118 | 0 | 43 | same | | 0 | 43 | excluded |
| B1-GR2 | solar sum rule, GR2 (best NO) | 4 (3+1) | 4 | 3.17 | - (0.075) | 1.8 | 0.92 | 2.41 | - (0.34) | 0.95 | viable fit, no compression |
| B1-TBM / GR1 / HEX | solar sum rule | 4 (3+1) | 4 | 3.45 / 3.97 / 3.83 | - (0.06/0.05/0.05) | ≤2.0 | 0.59 / 0.04 / 0.60 | 2.1/1.5/2.1 | - (0.44/0.84/0.44) | ≤0.8 | viable fit, no compression |
| B1-BM | solar sum rule, BM | 4 (3+1) | 4 | 21.9 | - (3e-6) | 4.0 | 41.9 | 43.4 | - | 4.2 | excluded |
| B2-TM1 | TM1 | 3 (2+1) | 4 | 5.12 | 0.024 (0.078) | 1.8 | 1.82 | 3.31 | 0.18 (0.40) | 1.35 | NO disfavoured, IO fine |
| B2-TM2 | TM2 | 3 (2+1) | 4 | 24.6 | 6.9e-7 | 4.6 | 22.9 | 24.4 | 1.7e-6 | 4.6 | excluded |
| C1 | GST \|V_us\| = √(m_d/m_s) | 2 (2 nuis.) | 3 | 0.015 | 0.90 | 0.10 | same | | | | survives (within-sector) |
| C2 | leptonic GST θ13 | 1 (0+1) | 1 | 1440 | - (0) | 38 | 1456 | | - | 38 | excluded |
| C3 | Koide m_τ | 1 (0+1) | 1 | 0.185 | - (0.67) | 0.43 | same | | | | survives (within-sector) |
| E1 | δ_KM^q + δ_KM^l = 0 | 4 (3+1) | 4 | 3.82 | - (0.051) | 1.95 | 0.32 | 1.80 | - (0.57) | 0.56 | NO marginal, IO fine; no compression |

IC24_wSKatm changes nothing qualitatively: L1 χ² 5.13 (NO) / 3.68 (IO, χ²_g 9.59); TM1 4.19 / 1.96; best B1 2.24 (NO GR2) / 0.23 (IO TBM); E1 2.81 / 0.06. The IO offset (5.9) makes every IO χ²_g worse than NO in IC24.

Input sensitivity (default analysis): `input_dchi2` is 0.54 (NO) / 0.50 (IO) for L1, ≤ 0.98 for A4a, and up to 2.1 for A4c and the other A4b patterns (all with χ² ≥ 13); A1 0.87; A3/A3k 2.6 (NO) on χ² of 25-40. No verdict changes. m_e, m_μ inputs: < 4e-4.

### Lead L1: U_PMNS = V_CKM Φ U_BM V_CKM†, α, β fitted, δ via the 2D table

| analysis, ordering | α, β (rad) | χ² | χ²_g | p_fit (dof 2) | predictions s12², s13², s23², δ | 1D pulls |
|---|---|---|---|---|---|---|
| IC23 NO | 1.428, 0.032 | 5.89 | 5.89 | 0.053 | 0.3135, 0.0232, 0.460, 274° | +0.71, +1.24, −0.67, +1.96 |
| IC23 IO | 1.423, 0.150 | 4.17 | 5.65 | 0.125 | 0.3140, 0.0232, 0.465, 274° | +0.77, +1.09, −1.55, −0.34 |
| IC24 NO | 1.425, 0.029 | 5.13 | 5.13 | 0.077 | 0.3140, 0.0232, 0.460, 274° | +0.76, +1.25, −0.71, +1.69 |
| IC24 IO | 1.433, 0.192 | 3.68 | 9.59 | 0.159 | 0.3125, 0.0233, 0.466, 274° | +0.55, +1.19, −1.38, −0.07 |

With the full count k = 4 = N, p is undefined (dof 0). The angles-only optimum from exp 003 (δ = 82°) is a different branch: there the angles give χ² = 0.23 but the full 4-observable χ² is 23.5 (δ pull −3.4). With δ in the fit, the minimizer moves to the δ ≈ 274° branch, trading ~1σ on θ13 and θ12 for δ.

## Where these numbers differ from exp 003 `quick_status.log`, and why

1. **Single-observable pulls → joint χ² with profiles.** quick_status used Gaussian pulls on θ in degrees; here the NuFIT Δχ² profiles in sin² are used. E.g. QLC solar −4.32σ → χ² 19.2 (1D pull 4.38). Small differences everywhere for that reason.
2. **δ now included.** quick_status scored A4 constructions on the 3 angles only (L1 Σpull² 0.20 NO, 12.5-14.5 IO; V†ΦX 4.4-4.9 NO). Including δ through the 2D (s23², δ) table: L1 5.89 NO / 4.17 IO, V†ΦX 6.9-7.4 NO. IO got better for L1 because the joint fit moves to δ ≈ 274°, near the IO best fit (δ ≈ 283°), which the angles-only grid did not reward.
3. **Sum rules scored as full models.** quick_status printed cos δ predictions for TM1/TM2/solar sum rules with angles fixed at central values; here each is a fit with the input angles as nuisances, so the input errors and the θ23-δ correlation enter.
4. **LO patterns.** quick_status scored only s12² (and s23² = 1/2); here N = 3 includes θ13 = 0, which alone gives Δχ² ≈ 1850.
5. **GST** quick_status compared √(m_d/m_s) to |V_us| with the mass error quoted separately (+0.2σ); here m_d, m_s are nuisances, χ² = 0.015 for dof 1.
6. quick_status's text header says IC24 is the default; the program default is IC23_woSKatm. Both are in this table.

## Not scored (no low-scale target or needs other machinery)

C3' Koide-type neutrino (predicts unmeasured m1), C4 SU(5)/GJ ratios (need GUT-scale Yukawas, waits for src/rg.py), C5 Haba-Yamada (seesaw model fit), D (group models; covered by the B/B2 patterns they produce), E2 zero 1-3 hypothesis (per-paper sum rules not implemented), F anarchy (null model).

## Issues found in other modules (not patched)

1. `src/stats.py` `Objective.__call__`: the PMNS pull sign is `np.sign(v - pmns_best[p])` without wrapping δ, so δ = −150° vs best 207° gets the wrong sign (−150 ≡ 210 is above). Proposed fix:
   ```python
   d = v - self.pmns_best[p]
   if p == "dcp_deg":
       d = (d + 180.0) % 360.0 - 180.0
   sgn = np.sign(d) if p in self.pmns_best else 1.0
   ```
   `src.evaluate` works around it by mapping δ into best ± 180° before calling Objective.
2. `src/stats.py` `Objective` calls the PMNS callable as `pmns(ordering, **params)`, so passing `src.data.pmns_delta_chi2` directly silently uses the default analysis. Proposed: an `pmns_kwargs: dict` argument forwarded to every call, or a docstring note to pass `functools.partial(pmns_delta_chi2, analysis=...)` (what `src.evaluate` does).

No bugs found in `src/data.py` or `src/mixing.py` during this work.

## Verdict

Infrastructure done: `src/evaluate.py` is the single scoring path, `src/baselines.py` the registry. Official table above. No known relation passes k ≤ N − 2 with an acceptable p. L1 has the smallest NO χ² among CKM-PMNS constructions (5.89, p_fit 0.053 with 2 continuous parameters, p undefined with the full count) and is not a candidate. Best NO fits overall are the B1 solar sum rules (3.2-4.0) and E1 (3.8), all with k = N. GST and Koide survive but are within-sector.

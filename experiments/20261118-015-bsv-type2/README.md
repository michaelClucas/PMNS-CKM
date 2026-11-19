# 015: BSV type-II SO(10) (b-τ unification → large θ23), scored as a baseline

## Hypothesis

Bajc, Senjanović, Vissani, "b-τ unification and large atmospheric mixing: a case for non-canonical see-saw", arXiv:hep-ph/0210207 (ID, title and text checked; Eqs. cited below are from v2). In renormalizable SO(10) with 10_H + 126bar_H and type-II dominance, M_ν ∝ Y_126 ∝ M_d − M_e (their Eq. 13). In the 2-3 sector the 33 entry of M_ν is m_b − m_τ, so a small CKM angle turns into a large θ23 only if b-τ unify at M_GUT (their Eq. 24: tan 2θ_l = sin 2θ_q / (2 sin²θ_q − ε), ε = (y_b − y_τ)/y_b). Known mechanism, scored here as a baseline against PDG 2026 + NuFIT 6.1 with SM and MSSM running.

## Model and basis (common.py)

Weyl mass matrices f^c M f (convention of exp 011, checked against `src.rg`: here `consistency_gut_pmns` ≤ 1e-15 between our PMNS and `src.rg.observables` on the same GUT state).

    M_u = diag(u)                                   basis: SO(10) flavour rotation makes M_u real diagonal
    M_d = (1+η) U* diag(d_i e^{iφ_i}) U†,           U = diag(1, e^{i a2}, e^{i a3}) V_CKM(M_GUT)
    M_e = [(3+x) M_d − 4y M_u] / (x − 1)            BSV Eq. (17) (= 1306.4468 Eq. 4 with x = s, y = 1/r)
    M_ν = c (M_d − M_e)                             BSV Eq. (13), type-II dominance

M_e is not free: the "unknown rotation" between the d and e bases is the one forced by the SO(10) relation (17), set by x, y and the phases. PMNS = U_e† U_ν with U_e, U_ν from M†M, the ordering label fixed per fit (NO: ascending; IO: (m3, m1, m2)).

Variants:

| variant | free model parameters | discrete | meaning |
|---|---|---|---|
| R0 | x, y real (2) | φ1, φ2 ∈ {0, π}: 4 patterns = 2 bits | BSV's own choice ("we take X orthogonal", no phases beyond the CKM's δ), extended to 3 generations |
| R0eta | x, y, η (3) | 2 bits | R0 + a free universal down-type factor at M_GUT (stands in for tanβ-enhanced SUSY threshold corrections that `src.rg` lacks) |
| C | Re x, Im x, y, a2, a3, φ1, φ2, φ3 (8) | 0 | general minimal 10 + 126bar type II. y real WLOG (an overall phase of M_d plus the phase of y is a rephasing of M_e, M_ν). Same model as exp 011 `so10_II` |

Not counted: c (sets only the absolute neutrino scale; Δm²31 itself is not scored). y is entered as ŷ = y u3/d3.

## Parameter and observable count

Scored observables, N = 18:
- official (`src.evaluate.evaluate`, `use=` m_u, m_d, m_s, m_c, m_b, m_t_direct, ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, s12sq, s13sq, s23sq, dcp_deg; `inputs=` m_e, m_mu, m_tau): N_official = 14. Quark masses and CKM are nuisance parameters (fitted and scored: +10 to N and k, dof unchanged).
- supplementary (the official scorer has no key for them): the three GUT charged-lepton eigenvalues against data run up with `src.rg`, with a 2% theory floor (one-loop running, no thresholds; same floor as exp 011), and Δm²21/|Δm²3ℓ| (Gaussian, NuFIT 6.1 errors).

k = model params + 10 nuisances + discrete bits:

| variant | k_cont | k_d | k (fixed tanβ, fixed ordering) | + ordering bit | + tanβ chosen from 7 by data (3 bits) | N − 2 |
|---|---|---|---|---|---|---|
| R0 | 12 | 2 | 14 | 15 | 18 | 16 |
| R0eta | 13 | 2 | 15 | 16 | 19 | 16 |
| C | 18 | 0 | 18 | 19 | 22 | 16 |

dof = N − k_cont (convention of the hardened `src/evaluate.py`): R0 6, R0eta 5, C 0.

## Method

1. `rg_scan.py`: PDG/NuFIT data at M_Z (`src.rg.build_mz_state`) run to M_GUT = 2×10^16 GeV in the SM and in the MSSM for tanβ ∈ {5, 10, 20, 30, 40, 50, 55}, M_SUSY = 3 TeV (default), 10 TeV and 1 TeV. Records GUT m_b/m_τ, mass ratios, CKM, and a log Jacobian d ln y_{t,b,τ}(GUT)/d ln y_{t,b}(M_Z) used to map the low-scale nuisances to GUT inputs (light-quark and CKM maps are multiplicative). → `rg_scan.csv`.
2. `bsv_analytic.py`: BSV Eq. (24) in the 2-3 limit for every scenario. → `bsv_analytic.csv`.
3. `fit.py`: per scenario, variant, ordering and sign pattern: stage 0 (charged-lepton relations only), stage 1 (all terms, nuisances at data centres) multistart least squares, stage 2 polish of the 2 best with the nuisances free. PMNS compared at M_GUT in this stage. → `fits_*.jsonl`.
4. `finalize.py`: best sign pattern per (scenario, variant, ordering). The fitted GUT matrices (Y = M†, κ ∝ M_ν) are run down to M_Z with `src.rg.run` (same model, tanβ, M_SUSY); the M_Z − M_GUT PMNS shift is fed back and the fit redone until the shift is stable; the M_Z PMNS values are scored. MSSM M_SUSY = 3 TeV fits of R0eta and C are also refitted with θ13^q, θ23^q(M_GUT) scaled by 0.98 (the exp 008 one-loop MSSM systematic). → `results.csv`, `best_fits.json`.

Convention: all p-values use the hardened `src/evaluate.py` (exp 013): dof = N − k_continuous, k_discrete in integer bits. Both `finalize.py` runs happened after that change (`dof_official` = 14 − 12 = 2 for R0 in `results_R0.csv`).

## Results

### m_b/m_τ at M_GUT = 2×10^16 GeV (`rg_scan.csv`, one-loop Yukawas, no SUSY threshold corrections)

| scenario | M_SUSY 1 TeV | 3 TeV (default) | 10 TeV |
|---|---|---|---|
| SM | | 0.629 | |
| tanβ 5 | 0.789 | 0.786 | 0.781 |
| 10 | 0.785 | 0.782 | 0.778 |
| 20 | 0.786 | 0.783 | 0.777 |
| 30 | 0.789 | 0.784 | 0.777 |
| 40 | 0.795 | 0.786 | 0.776 |
| 50 | 0.802 | 0.788 | 0.774 |
| 55 | 0.807 | 0.788 | 0.771 |

Also at 3 TeV: m_s/m_μ = 0.235, m_d/m_e = 2.51 (SM 0.227, 2.42); θ13^q 0.00354 → 0.00329 and θ23^q 0.0394 → 0.0367 from tanβ 5 to 55 (SM 0.00427, 0.0475; these MSSM values are ~2% high, exp 008); θ12^q 0.2271, δ_q 1.154.

So ε = (y_b − y_τ)/y_b ≈ −0.27 in the MSSM at every tanβ and M_SUSY tried, and −0.59 in the SM. b-τ unification does not happen in our running without tanβ-enhanced threshold corrections. Those corrections (gluino/higgsino loops, ∝ tanβ) are ~10-50% at large tanβ, and their size and sign depend on the sparticle spectrum, so in the MSSM the b-τ condition is really a condition on the spectrum.

### BSV Eq. (24), 2-3 limit (`bsv_analytic.csv`)

At the data, sin²θ23^ℓ = 0.006 (SM) and 0.015-0.020 (MSSM). Reaching the measured 0.470 needs ε = −0.0016, i.e. m_b/m_τ(GUT) = 0.998: unification to 0.2%, against 0.78 delivered.

### Fits (`results_R0.csv`, `results_R0eta.csv`, `run.log`)

N = 18, NO, best sign pattern per row. χ²_total = official (14 keys) + charged-lepton GUT (3) + Δm² ratio (1). "inf" means a PMNS value outside the NuFIT tables after the M_Z refit (the shift refit wandered off the fit-stage minimum). The fit-stage GUT χ² (fallback penalty outside the tables) is then the representative number.

| scenario | R0 χ²_tot (fit stage) | R0: s12², s13², s23², δ | R0eta χ²_tot (fit stage) | R0eta: η, m_b(1+η)/m_τ | R0eta: s12², s13², s23², δ |
|---|---|---|---|---|---|
| SM | 4605 (4607) | 0.296, 0.0112, 0.613, 23° | 3689 (3689) | 0.52, 1.04 | 0.209, 0.0120, 0.466, 5° |
| tanβ 5 | inf (3551) | 0.269, 0.0160, 0.179, 2° | inf (3859) | - | - |
| 10 | 3753 (3753) | 0.270, 0.0082, 0.470, 3° | 3541 (3543) | 0.38, 0.95 | 0.260, 0.0080, 0.464, 1° |
| 20 | inf (3565) | 0.278, 0.0139, 0.250, 4° | 3381 (3380) | 0.68, 1.05 | 0.262, 0.0140, 0.508, 17° |
| 30 | 3488 (3529) | 0.264, 0.0137, 0.259, 3° | 3616 (3620) | 0.24, 1.04 | 0.210, 0.0118, 0.466, 5° |
| 40 | 3663 (3656) | 0.275, 0.0086, 0.469, 4° | inf (2839) | 0.12, 0.84 | 0.265, 0.0165, 0.173, 2° |
| 50 | inf (2714) | 0.280, 0.0170, 0.174, 2° | 3619 (3638) | 0.23, 1.04 | 0.210, 0.0119, 0.464, 5° |
| 55 | 3534 (3512) | 0.282, 0.0092, 0.468, 4° | 3228 (3252) | 0.69, 1.05 | 0.264, 0.0149, 0.500, 18° |
| tanβ 10, M_SUSY 10 TeV | - | - | 3465 (3515) | 0.14, 0.91 | 0.260, 0.0133, 0.269, 3° |
| tanβ 50, M_SUSY 10 TeV | - | - | inf (3855) | - | - |

Measured (NO, NuFIT 6.1): 0.309, 0.0225, 0.470, 207°. The R0 fit-stage numbers for inf rows come from `fits_R0.jsonl` (best sign).

Counts: R0 k = 12 + 2 bits = 14 (dof 6), R0eta k = 13 + 2 = 15 (dof 5). Both are ≤ N − 2 = 16 at a fixed, pre-declared tanβ, and fail it once tanβ is picked from the 7 by data (+3 bits) and the ordering bit is added (R0 18, R0eta 19). p_total, p_official and p_global are 0 to machine precision in every row. The CKM-systematic refits (θ13^q, θ23^q × 0.98) change χ² by < 1% (`results_R0eta.csv`, rows with ckm_sys = 0.98).

Typical pulls (R0eta, tanβ 10, NO): m_d −38σ, m_s +5σ, δ_CKM +7σ, s13² −25σ, s12² −7.7σ, δ_ℓ +2.3σ, charged leptons e +12σ, μ −19σ, τ +6σ (2% floor), Δm² ratio −2.2σ. Every row has max|pull| ≥ 35. Full pulls are in the CSVs (`pull_*`).

**IO**: every IO fit is far worse (lepton χ² 8×10^3 to 4×10^4, Δm² ratio up to 10^6): with these parameters, M_d − M_e is hierarchical and cannot give an inverted spectrum.

**Mechanism check.** Whenever the fit reaches large θ23 (s23² ≈ 0.46-0.61), it does it the BSV way. In R0eta, η takes m_b(1+η)/m_τ to 0.95-1.05. In R0 (η = 0), m_d, m_s, m_b and the charged-lepton relations are pulled instead. When b-τ is not reached, s23² stays at 0.17-0.27. The PMNS side is not the main failure. R0 and R0eta cannot reproduce the first-generation charged-lepton mass even without neutrinos: a charged-lepton-only R0eta fit at the SM point gives χ² = 4332 on 3 residuals. With x ≈ 0 the relation gives M_e ≈ −3 M_d for the light entries, so m_μ ≈ 3 m_s (data m_μ/m_s = 4.3) but also m_e ≈ 3 m_d, against data m_e ≈ m_d/2.5. Fixing that takes the complex phases. BSV avoided it by restricting to the 2-3 sector ("the first family ... may suffer from higher dimensional operators"). On top of this, the real model gives δ_ℓ ≈ 0-20°, against 207°, and s13² ≈ 0.008-0.017.

**C (general complex minimal 10 + 126bar type II).** Not converged here. Our local multistart least squares reached χ²_gut = 10564 at the SM, NO (`fits_C.jsonl`), worse than R0 although R0 is a subspace of C. The 18-dimensional landscape is too rugged for local optimisation. I stopped it for the compute budget (one record took ~4 min under load). The program's best number for this model remains exp 011's DE-MC full fit: χ² = 462 (SM, type II, NO, 17 GUT observables), with m_τ/m_b pulled −10σ and sin²θ23 at −16σ. C has k_cont = 18 = N, so it is not compressive at any tanβ and p is undefined.

## Literature status (IDs and abstracts checked on arXiv)

- BSV, hep-ph/0210207 (2002): the mechanism, 2-3 sector, real rotations; Eq. (24).
- Goh, Mohapatra, Ng, hep-ph/0303055 (2003): minimal SUSY SO(10), type II + b-τ unification gives large θ23 and θ12 and predicts U_e3 ≈ 0.16 (the measured sin θ13 = 0.150 is close).
- Bertolini, Frigerio, Malinsky, hep-ph/0406117 (2004): the minimal type-II model is in tension with lepton mixing data; a 120_H is added to rescue it.
- Babu, Macesanu, hep-ph/0505200 (2005): type II with b-τ unification gives large μ-τ mixing but puts the CKM phase in the second quadrant, in conflict with experiment. Consistent fits only in type I or mixed I + II.
- Bertolini, Schwetz, Malinsky, hep-ph/0605006 (2006): the minimal SUSY SO(10) 10 + 126bar is not consistent with data.
- Joshipura, Patel, arXiv:1102.5148 (2011): 10 + 126bar(+120) models fit well with type I; many fail with type-II dominance, attributed to the lack of b-τ unification at M_GUT.
- Dev, Mohapatra, Severson, arXiv:1107.2378 (2011): non-minimal type II (two 10_H, S4-motivated TBM neutrino sector) predicted θ13 ≈ 4-6°, below the measured 8.6°.
- I did not find a post-2012 paper that fits pure minimal type II against current θ13 and δ_CP (several searches). The non-SUSY fits (1306.4468, 1612.04329) use type I or extended Higgs content.

Agreement: consistent with the literature. Pure type-II minimal SO(10) does not fit, and its large θ23 needs b-τ unification at the 0.2% level in the 2-3 limit. Our MSSM running gives m_b/m_τ ≈ 0.78 without thresholds, which matches the standard statement that MSSM b-τ unification needs large tanβ-enhanced threshold corrections. Our CKM-phase tension (δ_CKM pulled up to +22σ in R0/R0eta) is in line with Babu-Macesanu.

## Verdict

**Known baseline: fails fit quality.** Not a candidate.
- Compressive? R0 (k = 14) and R0eta (k = 15) are compressive at a fixed tanβ (N − 2 = 16), but not with tanβ selected by data plus the ordering bit (18, 19). C (k = 18 = N) is not compressive at any tanβ.
- Fits? No. χ² ≥ 2714 (fit stage) or ≥ 3228 (after M_Z running) for N = 18 at every tanβ, M_SUSY, and ordering; p = 0; max|pull| ≥ 35. IO is worse by orders of magnitude.
- Parameter count: x, y (R0) or x, y, η (R0eta) or 8 (C), plus 10 quark/CKM nuisances, plus 2 sign bits (R0, R0eta), + 1 ordering bit, + 3 bits if tanβ is data-selected. The neutrino scale c is not counted (Δm²31 not scored).
- The BSV correlation itself is reproduced: in the fits, large θ23 appears only when m_b(1+η)/m_τ(GUT) ≈ 1 (0.95-1.05), and Eq. (24) needs 0.998.

## src gaps (not edited)
- `src.evaluate` has no key for Δm²21/|Δm²3ℓ| (or the splittings) and cannot score GUT-scale relations. Both were handled here as supplementary χ² terms, outside the official scorer.
- `src.rg` has no tanβ-enhanced SUSY threshold corrections, so GUT m_b/m_τ in the MSSM is spectrum-independent (≈ 0.78). That is the decisive quantity for this model; R0eta's η stands in for it.
- No bugs found in the `src.rg` / `src.evaluate` paths used (GUT PMNS from our matrices vs `src.rg.observables`: agreement to 1e-15).

## Limitations
- Local optimisation only, with 5 starts per sign pattern (R0, R0eta). The C fit did not converge and was aborted. R0eta was run before stage 0 was added to `fit.py` (R0 used it). Several finalize refits with the running shift left the NuFIT table range (rows marked inf). Their fit-stage χ² is quoted, which does not change any conclusion (all χ² > 2700).
- 2% theory floor on the GUT charged-lepton relation. One-loop Yukawa running; the MSSM θ13^q, θ23^q systematic changes χ² by < 1%. No seesaw thresholds, no triplet running.
- R0 uses the PDG phase convention for V_CKM with no other phases. That is a basis-dependent reading of BSV's "X orthogonal".

## Rerun
```
cd experiments/20261118-015-bsv-type2 ; PY=../../.venv/Scripts/python
$PY rg_scan.py ; PYTHONPATH=../.. $PY bsv_analytic.py
$PY fit.py R0eta fits_R0eta.jsonl 5 SM,5,10,20,30,40,50,55,10k10,10k50   # ~9 min; original run had STAGE0 off
$PY fit.py R0 fits_R0.jsonl 5 SM,5,10,20,30,40,50,55                     # ~9 min
$PY finalize.py fits_R0eta.jsonl ; $PY finalize.py fits_R0.jsonl        # results_*.csv, best_fits_*.json
```

# Exp 040: verified scout for the remaining compressive unified quark + lepton models

## Hypothesis
STATE.md next task 1: some published unified quark + lepton model, not yet tested here, shares
continuous parameters across sectors by construction and has honest k ≤ N − 2. If one also fits
PDG 2026 / NuFIT 6.1 data, it is a candidate in which the quark sector constrains PMNS.

## Method
1. Literature: arXiv API keyword listings (abstract searches on "quarks and leptons" + "free
   parameters" + "observables"; "texture" + "quark and lepton" + "unified", newest first), web search
   for 2020-2026 unified fits, and the citation trail of the hits. 26 papers screened (M1-M16 in
   `models.md`). Matrices were read from the rendered PDF pages for 1710.01741, hep-ph/0502161 and
   hep-ph/0502152 (the three models that reached a fit), and from arXiv HTML text for the rest
   (models.md marks which). The PDFs were kept in the session scratchpad, not committed.
2. Cheap filters before any fit, each recorded with its number:
   - (a) Clebsch double ratio R = (m_e/m_μ)/(m_d/m_s), data 0.0938, σ 0.0031 (`R_clebsch`,
     floor √2·2%, exp 022 convention). `bound_R.py` / `bound_R_bpr.py` minimise R's pull over each
     model's d/e parameters with m_e/m_μ held at data and loose bands on m_μ/m_τ, m_s/m_b.
   - (b)/(c) Published parameter-independent predictions or bounds (θ13, θ12, R), scored one key at a
     time with the official scorer (`pub_filters.py`).
   - Counting: honest k = fitted continuous parameters + bits for data-selected discrete choices,
     including the model-version choice.
3. Survivors: official scorer `fit_and_evaluate(method="de")` (popsize 20, maxiter 800),
   `mass_floor=0.02`, PDG 2026 masses, NuFIT 6.1 default tables. Scale: 2e16 GeV, MSSM, tanβ fixed
   before fitting (stated per model), M_SUSY 3 TeV, no thresholds. NO declared (hierarchical models,
   no ordering bit). Program N = CKM 4 + PMNS 4 + dm_ratio + 7 same-scale mass ratios = 16 (15 when a
   model does not predict m_b/m_τ).
4. Held-out for the one model that passed compression with the real-M_R count (BPR): fit
   CKM + mass ratios, freeze, fit only the M_R parameters to PMNS + dm_ratio (`heldout.py`).

## Parameter counts
Per model in the ranked table and in `models.md`. Program N = 16, N − 2 = 14, unless stated. Bits:
- AB: 1 (model version, δ_N = 0, chosen among the published variants).
- BPR: 1 (ζ22^u = 0 and real M_R, chosen among the variants).
- UTZ: 4 (generous count for the Table III Z_5-type phase choices; a strict count of 4 phases × 5 values
  is 10 bits), plus 1 for the H.O. choice ψ_d = π.

## Result

### Reproductions
- UTZ (1710.01741): Table V mass ratios at L.O. and H.O. reproduced exactly (`bound_R.log`). The full L.O.
  benchmark reproduces Table IV, CKM and PMNS (`reproduce_utz.log`): sin θ12^ℓ 0.5537 (paper .554),
  sin θ23^ℓ 0.7783 (.778), sin θ13^ℓ 0.1524 (.152), sin δ_ℓ -0.905 (-.905), dm ratio 0.0209 (.0213),
  sin δ_q 0.561 (.561). The match needs the paper's Majorana relabelling (a → y, b → x) and m_ν passed to
  `src.yukawa` as conj(M M_R⁻¹ Mᵀ). The same convention is used for AB.
- AB (hep-ph/0502161 Eq. 3) benchmark charged sector: m_e/m_μ 0.00526, R 0.114 (+6.5σ). The paper gives no
  printed charged-lepton outputs to compare against.
- BPR fit A (hep-ph/0502152 Eq. 6) as printed does not reproduce: m_e/m_μ 0.0012 against data 0.0047
  (`bound_R_bpr.log`). The matrices themselves (Eq. 1) were checked against the PDF page.

### Filter results (`pub_filters.log`, `bound_R.log`, `bound_R_bpr.log`, `ab_theta13.log`)
- R = 1/9 (Georgi-Jarlskog 1-2 block) has pull +5.6. It holds only at leading order. The numerical bounds
  show that R = 0.0938 is reachable at pull 0 in AB, UTZ-LO, UTZ-HO and BPR once the full matrices are used
  (m_e/m_μ at data; m_μ/m_τ, m_s/m_b inside the bands). So R kills none of the models that were checked
  numerically. The published fit values of CM-T' (R 0.122, +9.2σ) and AKS (0.119, +8.3σ) are their own
  best points, not bounds.
- Published θ13 values (single-key χ², NO): DR and DMM sin θ13 ≈ 0.05: -37.7σ. CM-T' (2007)
  |U_e3| 0.055: -36.4σ. YANG sin θ13 0.0576: -35.6σ. KN23 |U13|² ≤ m_e/m_μ: ≤ -32σ (case ii -42.7σ;
  nonsymmetric -43σ). MP (TBM, θ13 = 0): -43σ. BPW-1998 solar angle 0.06 rad (SMA): outside the tables.
  JLM central sin²2θ13 0.074: -6.1σ, but its quoted range 0.055-0.110 contains the data, so this is not a kill.
- AB θ13 with δ_N = δ'_N = 0 at the paper's benchmark charged sector: random M_R scan, 11 of 50,000 samples
  pass the s12, s23, dm windows, max s13² 0.0083. This matches the paper's own "≲ 1e-2". It is not a bound
  for other charged-sector points: the full fit below reaches the PMNS data exactly.

### Fits of the survivors (official scorer)
All fits: N, k from the scorer (k includes the bits stated). One DE seed per row. The AB and BPR rows
were rerun with a second seed at a second tanβ.

| model | scale | N | k (cont + bits) | χ² | p_global | worst pulls | compression | fit |
|---|---|---|---|---|---|---|---|---|
| AB, δ_N = 0 | MSSM tanβ 5 | 16 | 12 + 1 | 246.7 | 3.4e-52 | mr_b_tau +10.9, mr_s_b +6.9, mr_e_mu +4.9, ckm_s23 +4.3, mr_d_s -3.9 | ok | fail |
| AB, seed 2 | MSSM tanβ 30 | 16 | 12 + 1 | 269.9 | 3.3e-57 | mr_b_tau +11.1, mr_s_b +7.0, mr_e_mu +5.1, ckm_s23 +5.0 | ok | fail |
| BPR, real M_R | MSSM tanβ 10 | 16 | 13 + 1 | ∞ (s12sq 0.0044, outside the table) | 0 | s13sq -36.0 (pred 0.0032), mr_b_tau +9.2, mr_s_b +6.6, ckm_s13 -6.0 | ok (k = N - 2) | fail |
| BPR, seed 2 | MSSM tanβ 30 | 16 | 13 + 1 | ∞ (s12sq 0.0079, outside the table) | 0 | s13sq -35.5, mr_b_tau +10.7, ckm_s13 -10.3, mr_u_c +4.1 | ok | fail |
| UTZ L.O. (Table III phases) | MSSM tanβ 30 | 15 | 10 + 4 | 1395.0 | 1.7e-299 | ckm_s23 -24.9, ckm_delta -23.5, s13sq +9.2, mr_s_b +5.9 | fail (14 > 13) | fail |
| UTZ H.O. (+ d_d, ψ_d = π) | MSSM tanβ 30 | 15 | 11 + 5 | 385.6 | 3.6e-82 | ckm_s23 -8.1, mr_mu_tau +8.0, s23sq +7.7, mr_d_s -6.5, ckm_s13 -6.0, s12sq -5.9 | fail | fail |

- AB: the six M_R parameters (a, b, c complex) put all five neutrino observables at their data values
  (pulls ≤ 0.04 in both runs). The PMNS is accommodated, not predicted: the neutrino-only parameter count
  (6) exceeds the neutrino observables (5). All of the misfit is in the shared charged sector (the D and L
  matrices, which hold σ, ε, δ, δ', φ). b-τ unification is part of the misfit, but without mr_b_tau the
  χ² is still 125.9 (tanβ 5) / 141.5 (tanβ 30) for N 15. JLM (hep-ph/0510353) uses the same D and L, so
  its m_e/m_μ, m_d/m_s, m_s/m_b and m_b/m_τ face the same structure (not fitted separately).
- BPR with real hierarchical M_R falls back to the small-angle solar solution (s12sq ≈ 0.004, s13sq ≈ 0.003),
  as in the original BPW paper. Held-out (`heldout_BPR.log`): stage 1 (CKM + 7 mass ratios, N 11, k_cont 10)
  gives χ² 174.5 (mr_b_tau +9.0, ckm_s13 -7.2, mr_s_b +4.6, ckm_s23 +3.4). Stage 2 (x, y, z only, fitted to
  PMNS + dm_ratio) gives s12sq 0.0031, s13sq 0.0027, s23sq 0.506, dm_ratio 0.030, with χ² infinite
  (s12sq outside the table). Complex x, y, z would give k_cont 16 > 14 and fail compression.
- UTZ at L.O. reproduces the paper's own remark that V_cb comes out low (here ckm_s23 -24.9σ). The H.O.
  operator lifts V_cb but leaves χ² 386, and with its extra parameter and bit k exceeds N - 2 anyway.


### Ranked table
Ranked by how far each model got. k is in program units (dimensionless, N = 16 unless stated). "pub" means
the paper's own count.

| # | model | arXiv | k_cont + bits | N | killed by | χ² / p (if fitted) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | BPR SO(10)/G224 (BPW + phases), real M_R | hep-ph/0502152 (Eq. 1, 4) | 13 + 1 = 14 | 16 | fit: s12sq ≈ 0.004 (SMA), s13sq -36σ, b-τ +9σ | ∞ / 0; held-out stage 1 174.5 (N 11) | excluded (fit quality) |
| 2 | Albright-Barr lopsided SO(10), δ_N = 0 | hep-ph/0002155, hep-ph/0502161 (Eq. 2-5) | 12 + 1 = 13 | 16 | fit: charged sector (b-τ +11, m_s/m_b +7, m_e/m_μ +5, V_cb +4-5) | 246.7 / 3e-52 (tanβ 5); 269.9 / 3e-57 (tanβ 30) | excluded (fit quality); PMNS absorbed by M_R |
| 3 | UTZ Δ(27) universal texture zero | 1710.01741 (Eq. V.1, II.8) | L.O. 10 + 4 = 14; H.O. 11 + 5 = 16 | 15 | compression (honest bits) and fit (V_cb -25σ at L.O.) | 1395 (L.O.); 385.6 (H.O.) | excluded (fit quality); passes compression only with the paper's count (9) |
| 4 | Ji-Li-Mohapatra lopsided SO(10) | hep-ph/0510353 | ≈ 11-12 + 1 | 16 | same D, L as AB (charged-sector misfit); θ13 central -6.1σ, but the range 0.055-0.110 contains the data | not fitted | no evidence; inherits AB's D/L structure (indicative) |
| 5 | UTZT SO(10) 10+126+120 Hermitian | 2606.24571, 2410.18743 | 14 + 4 = 18 | 16 | compression (sign bits) | not fitted (paper: χ² < 4 on 9 obs, with masses as inputs) | fails compression |
| 6 | Pati-Salam A4×Z5 "A to Z" (CSD4) | 1406.7005 | ≥ 16 (quark) + 1 + 3 | 16 | compression | not fitted; s23sq predicted ≥ 0.49 (+1.1σ) | fails compression |
| 7 | Δ(27)×SO(10) | 1512.00850 | 18 (pub) | 18 (pub) | compression | pub χ² 2.7-3.3 | fails compression |
| 8 | S4×SO(10) 10+126 (Cai-Yu) | hep-ph/0608022 | ≈ 20 | 16 | compression; minimal-SO(10) class | not fitted | fails compression |
| 9 | A4×SU(5) CSD2 (Antusch-King-Spinrath) | 1301.6764 | 10 + 2 + bits | ≈ 18 | none applied numerically; R at their own best fit 0.119 (+8.3σ), m_s/m_d 25.3 | not fitted | open (SU(5)+CSD2 class; exp 027 killed 1808.09364 with R) |
| 10 | SUSY SU(5)×T' (2007; 2013 large-θ13 update) | 0705.0714, 1307.7711 | 9 (pub) / 10 | ≈ 16 | 2007: θ13 (U_e3 0.055), -36.4σ. 2013: R = 1/9 at LO only (fit value 0.122, +9.2σ), not a bound | not fitted | 2007 excluded; 2013 open (indicative R tension) |
| 11 | SO(10)×D3 (Dermisek-Raby) | hep-ph/0507045 | 14 (pub) | 20 (pub) | θ13 (sin θ13 ≈ 0.05): -37.7σ | | excluded (published prediction) |
| 12 | SO(10)×S4 rank-one type II (Dutta-Mimura-Mohapatra) | 0911.2242 | n/a | | θ13 (≈ θ_C/(3√2)): -37.7σ | | excluded (published prediction) |
| 13 | SU(3) family gauge (W.-M. Yang) | 1011.4573 | 6 (pub) | 25 (pub) | θ13 (sin θ13 0.0576): -35.6σ | | excluded (published prediction) |
| 14 | SO(10) S3×P (Morisi-Picariello) | hep-ph/0505113 | ≈ 5 per matrix | | θ13 = 0 (TBM): -43σ; GJ m_e ≈ m_d/3 | | excluded |
| 15 | Universal 2-3 symmetric textures (Koide-Nishiura-Matsuda) | hep-ph/0209333, hep-ph/0403008, hep-ph/0506192 | per sector | | θ13: U13² ≤ m_e/m_μ (≤ -32σ); no shared continuous parameter | | excluded |
| 16 | BPW original (real, no ζ22) | hep-ph/9812538 | 9 | | solar angle 0.06 rad (SMA): outside the tables | | excluded (superseded by #1) |
| - | screened out by construction | hep-ph/9503313 (no ν sector); 1501.07261, 1601.05062 (3-3-1, no sharing); 1411.2913, 1403.2525 (SU(5), separate quark/lepton coefficients); 2209.00021 (minimal SO(10), tested); modular unified models (tested class) | | | criteria (i)/(ii), or already tested | | |

## Verdict
No candidate. 26 papers were screened, and 16 model entries got a filter or a count. Three models passed
compression with an honest (or near-honest) count and reached the official scorer: BPR (k 14 = N - 2), AB (k 13)
and UTZ (k 14 vs N - 2 = 13 at L.O.; only the paper's own count of 9 passes). All three fail fit quality by ≥ 9σ in
at least one observable, at two tanβ values and two seeds (AB, BPR). Per protocol item 7 this is "excluded"
(fit quality), not "no evidence".

- The R filter, applied as a numerical bound and not as the leading-order formula, kills none of the
  numerically checked models. The R = 1/9 of Georgi-Jarlskog-type 1-2 blocks is a leading-order statement.
  Off-diagonal and complex entries move R to the data value without pushing the other ratios outside loose
  bands. Exp 021's filter is weaker than assumed for models with complex 1-3 or 2-2 entries. The
  parameter-independent kills here come from published θ13 values (five model families at -36 to -43σ).
- The STATE.md pattern holds in a new form. Where the lepton sector fits (AB), neutrino-only parameters do the
  fitting: 6 M_R parameters for 5 PMNS observables, so PMNS is not predicted. The shared quark/charged-lepton
  matrices then fail at 2% light-quark precision plus b-τ unification. Where the neutrino sector is
  restricted (BPR with real hierarchical M_R), the model cannot reach large solar mixing: in the held-out
  test the quark sector fixes s12sq near 0.003.
- Left open: AKS (1301.6764) and the 2013 SU(5)×T' update. They were checked only against their own fit values
  (R +8 to +9σ), which is not a bound. Both are GUT-Clebsch charged-lepton-correction models (F2-like, free
  CSD/TBM neutrino sector), where the CKM link is a small charged-lepton rotation.
- Held-out test (brief step 3): run for BPR, which passed compression. It does not predict PMNS (s12sq 0.003).
  Nulls were not run because no model passed fit quality.

## Caveats
- One DE run per (model, tanβ). AB and BPR were rerun with a second seed at a second tanβ. UTZ L.O. and
  H.O. had one run each. The UTZ discrete phases were fixed at the Table III values and not re-optimised.
  If other Z_5 choices fit better, that costs more bits, which only worsens compression.
- No SUSY threshold corrections. For AB and BPR they would mostly fix mr_b_tau (+9 to +11σ), at the price
  of at least 1 counted parameter (η̄_b). Without mr_b_tau, AB still has χ² 126-142 for N 15. BPR's failure
  is in PMNS and does not depend on thresholds.
- BPR's printed fit A does not reproduce m_e/m_μ with Eq. (1) as written (0.0012 vs 0.0047), so our reading
  could not be confirmed against a benchmark. The matrices were checked against the PDF page. Complex x, y, z
  (k_cont 16 > 14) were not fitted.
- The AB neutrino convention (m_ν passed to `src.yukawa` as conj(-M_N M_R⁻¹ M_Nᵀ)) was fixed by reproducing
  the UTZ paper, not an AB benchmark. The fitted PMNS matches data with this convention, and the
  charged-sector failure does not depend on it.
- `ab_theta13` is random sampling (8-11 kept of 50,000 samples). It is a sanity check that agrees with the
  paper's own "≲ 1e-2" at the benchmark charged sector, not a bound.
- Models 5-15 were read from arXiv HTML text extraction or abstracts (marked in `models.md`), not the rendered
  PDF. Kills 11-15 rest on each paper's own published θ13 number, and 5-8 on each paper's own parameter count.
- Literature coverage: arXiv API keyword listings plus web search; no INSPIRE citation-graph pass. Recent
  SU(5)/Pati-Salam models with CSD neutrino sectors are under-sampled.
- Some short one-off checks early in the run (convention tests, timing, `reproduce_utz.py`) were launched
  without `tools/pin.py`. All fits and filters ran pinned, 1 thread each, at most 2 at a time.

## Files
- `models.md`: matrices, equation numbers and counts per model.
- `lib40.py`: builders (AB, UTZ, BPR).
- `bound_R.py`, `bound_R_bpr.py` → `.log`/`.json`: R filter bounds.
- `pub_filters.py` → `.log`/`.json`: published-prediction pulls.
- `ab_theta13.py` → `.log`/`.json`: AB θ13 scan.
- `reproduce_utz.py` → `.log`: UTZ benchmark reproduction.
- `fit_more.py` → `fit_AB_*`, `fit_UTZ_*`, `fit_UTZHO_*` (`.log`/`.json`): AB and UTZ fits.
- `fit_bpr.py` → `fit_bpr_full_*`: BPR fits. `fit_bpr_qcl_tb10_s1.log` is a failed launch (singular default
  M_R, since fixed in the script).
- `heldout.py` → `heldout_BPR.log`, `heldout_BPR_tb10_s1.json`.
- `run.log`: job exits.

Rerun from this folder with `../../.venv/Scripts/python ../../tools/pin.py ../../.venv/Scripts/python <script> [args]`
(fits ≈ 12-20 min each on 1 core; filters ≈ 5 min).

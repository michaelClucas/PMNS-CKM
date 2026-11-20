# Exp 043: why compressive quark-lepton models fail (structural claims, tested on random shared textures)

Diagnostic, not a candidate search (STATE "Next planned tasks" item 2).

## Hypothesis
Every compressive quark-lepton model tested so far fails fit quality, and every model that fits has
k ≥ N. If that is structural, it should be expressible as "structures sharing property X cannot fit
observables Y", and a random sample of shared-parameter textures with X should essentially never fit
while the same textures with X broken should fit at a measurable rate.

## Step 1: the compressive models (from committed result files)

| model (exp) | k (cont + bits) | N | χ² | observables carrying the χ² | mechanism the experiment identified | class |
|---|---|---|---|---|---|---|
| BSV type II, R0eta (015) | 13 + 2 | 18 | ≥ 3228 after running (`results_R0eta.csv`); 1976-2194 DE rerun (exp 031) | m_d -38σ, s13² -25σ, charged leptons e +12, μ -19 | M_e is a fixed combination of M_d, M_u: m_e ≈ 3 m_d (data m_d/2.5); large θ23 needs m_b/m_τ(GUT) = 0.998 | M (+ L) |
| common-τ A4 catalog (017) | 13 + 23 | 15 | 446.5 (`results.csv`) | quark sector (best quark-only 255) | one shared τ cannot give the quark textures; leptons fit with their own τ | Q |
| S4 lepton τ fixed, quarks (020) | 6 + 10 | 8 | 3167.5 (`results_fixed_tau_top.csv`) | δ_q -46σ, s23^q -28σ | lepton-selected τ is worse than 98/100 random τ for quarks | Q |
| 2O modular (016) | 10 (+bits) | 15 | 68.8 (README; PMNS ≈ 40.7, CKM 7.6, masses + dm 20.5) | lepton s12², CKM, mass ratios | shared τ; good published fits need 12-53% light-quark errors | Q + M |
| yukawaon (038) | 6 + 4 | 13 | 1027 (`fit_full_s1.log`) | mr_u_c +22.9, mr_s_b +20.2 | Φ0 built from charged-lepton masses fixes m_u/m_c ≈ 0.79 m_e/m_μ and m_s/m_b ≥ 0.0225; held-out s13² ≈ 0.003 | M (+ L) |
| FZ4U universal four-zero (039) | 11 + 2 | 15 | ∞ (Gaussian 1159, `summary.json`) | s23² 0.003-0.013 | shared shape a and φ2 ≈ 0 that V_cb needs cancel θ23^ℓ the same way | L |
| DEMU universal democracy (039) | 9-10 + 1 | 15 | ∞ (Gaussian 5074) | V_ub -20σ, δ_q = 0 (-46σ), near-diagonal PMNS | shared democratic shape | L + Q |
| BPR SO(10) (040) | 13 + 1 | 16 | ∞ (`fit_bpr_full_tb10_s1.json`) | s12² ≈ 0.004 (outside table), s13² -36σ, m_b/m_τ +9 | quark-like real hierarchical M_R gives small solar angle; held-out s12² 0.003 | L |
| Albright-Barr lopsided (040) | 12 + 1 | 16 | 246.7 (`fit_AB_tb5_s1.json`) | m_b/m_τ +10.9, m_s/m_b +6.9, m_e/m_μ +4.9, V_cb +4.3 | shared D/L charged matrices miss the mass ratios; PMNS fitted by 6 M_R-only parameters | M |
| Δ(27) UTZ (040) | 10 + 4 | 15 | 1395 LO / 385.6 HO (`fit_UTZ_tb30_s1.json`) | V_cb -24.9σ, δ_q -23.5σ | universal texture zero forces small V_cb | Q |
| exp 041 grammar G8 | 8 + 5 | 15 | ∞ (Gaussian 4137, `summary.json`) | s12² ≈ 0.001, s13² -32σ, m_μ/m_τ | quark fit needs near-rank-one 2-3 block; fixed Clebsches give m_μ/m_τ 0.019 or 0.40-0.77 | M + L |
| SU(5) + CSD3 (022) | 9 + 1 | 14 | 22.40, p_global 4.4e-4 (README) | mr_d_s -2.6 | R bound (GJ Clebsch pair) plus pure CSD3 leptons; no CKM → PMNS link | M (weak: ≈ 3.5σ) |

Classes: **M** = mass-ratio rigidity (lepton mass ratios fixed by quark parameters through fixed
factors, or the reverse); **L** = mixing inheritance (lepton left rotations inherit the quark
hierarchy, so a PMNS angle comes out quark-sized); **Q** = the shared parameter cannot even fit the
quark sector. SU(5)+CSD3 is the one compressive model whose failure is not ≥ 10σ.

## Step 2: pre-registered claims
`claims.md`, SHA-256 `8fc6f81c…5018719` logged in `run.log` (19:50 UTC) before any step-3 fit. Generator
G43: complex symmetric textures sharing a zero pattern Z ⊂ {11, 12, 13, 22, 23} (+33) in all sectors, free
u and d parameters (3|Z|), charged leptons Y_e = C_e∘Y_d, neutrinos from the quark matrices with Clebsch
matrices from CSET = {±1/2, ±1, ±3/2, ±2, ±3, ±9/2, ±6}, or with lepton-only parameters added.
- **C1** charged-lepton Clebsch rigidity: Y_e = C∘Y_d with C in CSET cannot fit Y1 (CKM + six mass ratios, N 10)
  for any quark-viable Z and any C.
- **C2** shared neutrino matrix: m_ν a fixed Clebsch image of Y_u or Y_d (Dirac, type I with M_R ∝ 1 or
  ∝ C'∘Y_d, type II ∝ C∘Y_d) cannot fit Y2 (Y1 + PMNS 4 + dm_ratio, N 15), even with free charged-lepton Clebsches.
- **C3** fits need ≥ 6 lepton-only continuous parameters (l); falsified by any pass with l ≤ 5.
Survival rule: X group 0 passes with CP95 upper < 2%, designated broken group CP95 lower > 1%.

## Step 3: method
- Stage 0 (`stage0.py`, `bank.json`): quark-only fits for all 15 patterns, 30 starts. 8 are quark-viable
  (official max |pull| < 2, all at χ² ≈ 0): {11,13,23}, {12,22,23}, {13,22,23} and all five 4-entry patterns.
- C1 profiling (`screen.py`, `screen.json`): all 19,208 (|Z| = 3) or 268,912 (|Z| = 4) Clebsch matrices with
  c33 > 0 screened on (m_e/m_μ, m_μ/m_τ) at the stage-0 quark points; the 150 best per pattern plus 30 random
  screened-out ones refitted jointly (quark parameters free).
- Random structure groups (`main.py`, seed 4302), pre-registered sizes, 4 (Y1) / 6 (Y2) starts each:
  scipy least_squares on a proxy residual, decision by the official scorer (`src.evaluate.Scorer`, NO,
  IC23_woSKatm, 2% floor, M_Z). Pass = official χ² finite and max |pull| < 2.
- Fast observable path (`lib43.fast_observables`) checked against `src.yukawa.observables_batch`
  (`check_fast.py`): identical mass ratios and angles; δ differs only where an angle is 0 or π/2 (δ
  undefined). Every official score uses the exact engine.
- Proxy validation (2512 optima, `summary.json`): median relative χ² difference 1.5e-4, Spearman 0.979,
  pass/fail agreement 2393/2407 for unpolished fits; 105 near-passes polished on the official χ².
  The proxy treats δ_ℓ as Gaussian (±20-23°) and is stricter than the NuFIT table for δ_ℓ far from 207°,
  so it can only miss passes, not create them.
- Deviations: a pilot run at 0.4× sizes and fewer starts was stopped after 36 fits because it ran faster
  than estimated (kept as `fits_pilot_scale0.4.jsonl`, not used); the full run used the pre-registered sizes.
  Stage 0 used 30 starts (no number was pre-registered). Wall time 43 min on 3 pinned workers.

## Result (`summary.json`, `summarize.log`, `posthoc.log`)

| group | property | l (lepton-only cont.) | k_cont | n | pass | CP95 |
|---|---|---|---|---|---|---|
| C1-S, Y1 (profiled C_e) | X1 | 0 | 9-12 | 1440 | 1052 | [0.707, 0.753] |
| ... |Z| = 3, top-screened C | X1 | 0 | 9 (N 10) | 450 | 217 | [0.435, 0.529] |
| ... |Z| = 3, random C | X1 | 0 | 9 | 90 | 0 | [0, 0.040] |
| ... |Z| = 4, random C | X1 | 0 | 12 | 150 | 85 | [0.483, 0.647] |
| C1-P1, Y1 | 1 free c_e | 1 | 10-13 | 150 | 75 | [0.417, 0.583] |
| C1-P2, Y1 | 2 free c_e | 2 | 11-14 | 150 | 134 | [0.833, 0.938] |
| C1-F, Y1 | all c_e free | 3-4 | 12-16 | 16 | 16 | [0.794, 1] |
| C2-S, Y2 | X2 (+ F_e) | 3-4 | 12-16 | 332 | 2 | [0.001, 0.022] |
| ... |Z| = 3 | X2 | 3 | 12 | 133 | 0 | [0, 0.027] |
| ... C = all ones (Pati-Salam minimal) | X2 | 3-4 | 12-16 | 32 | 0 | [0, 0.109] |
| C2-F, Y2 (free m_ν) | broken | 9-12 | 18-24 | 24 | 24 | [0.858, 1] |
| C3-P1, Y2 (X2 + 1 free m_ν entry) | | 5-6 | 14-18 | 150 | 18 | [0.073, 0.183] |
| C3-P2, Y2 (X2 + 2 free entries) | | 7-8 | 16-20 | 150 | 68 | [0.372, 0.537] |
| C3-SS, Y2 (fully shared) | X1 + X2 | 0 | 9-12 | 100 | 0 | [0, 0.036] |

Pass rate against k_cont − N (all Y2 structures):

| k_cont − N | -6 | -3 | -1 | +1 | +3 | +5 | +9 |
|---|---|---|---|---|---|---|---|
| pass / n | 0/31 | 0/202 | 1/55 | 14/251 | 26/104 | 56/98 | 15/15 |

Y2 by l: l 0: 0/100; 3: 0/133; 4: 2/199; 5: 1/55; 6: 17/95; 7: 12/52; 8: 56/98; 9: 9/9; 12: 15/15.

Other facts:
- Every viable pattern has a quark Jacobian of rank 8 with 3|Z| parameters (`posthoc.log`), so |Z| = 3 leaves
  one quark-flat direction (a phase that is unphysical for quarks but not for Y_e = C∘Y_d once C breaks the
  rephasing symmetry), and |Z| = 4 leaves four. Those directions act as lepton parameters.
- The two C2-S passes are both |Z| = 4 (k_cont 16 > N 15): a Dirac structure (χ² 4.46, δ_ℓ +2.0σ) and a
  type-II m_ν ∝ C∘Y_d structure with C = (1, -9/2, -9/2, 6, 1/2) (χ² 4.38, δ_ℓ 342°, +1.9σ).
- The one pass with k_cont < N: C3-P1, Z = {12, 22, 23}, type I with M_R ∝ C'∘Y_d plus one free m_ν(12) entry,
  k_cont 14, N 15, χ² 0.08 (`posthoc.json`). Its Clebsch matrices were drawn at random, not selected.

## Verdict per claim (pre-registered rule)
- **C1: falsified.** Fixed rational Clebsches reproduce the charged-lepton ratios together with the quarks in
  all 8 viable patterns. What remains true: for |Z| = 3 (k_cont 9 for N 10) no random Clebsch matrix fits
  (0/90, CP95 ≤ 4%) and at least 36, 129 and 52 of the 19,208 do in the three patterns (only the top 150
were refitted, so these are lower bounds); the fit is
  bought by choosing C, which costs log2(19,208) ≈ 14 bits, so honest k ≈ 23 ≫ N. The analytic argument in
  claims.md was wrong because complex 1-3 / 1-1 entries and the quark-flat phase move both ratios (same
  finding as exp 040's R bounds). For |Z| = 4 the quark sector is under-determined and most C fit.
- **C2: falsified as stated** (2/332 passes, CP95 upper 2.2% > 2%), but both passes have k_cont = 16 > N.
  With k_cont < N (|Z| = 3) 0/133. The pre-registered mechanism (hierarchical m_ν from the quark matrices)
  holds in the bulk: median proxy χ² 3410; I1 and IR never fit; all-ones Clebsches never fit (0/32,
  ckm_s23 / m_s/m_b / s23² at 17-27σ in the best one).
- **C3: falsified** (3 passes with l ≤ 5: two at l = 4, one at l = 5). The parameter count that matters is
  the total, not the lepton-only part: the Y2 pass rate is 0/233 for k_cont ≤ N − 3 (CP95 upper 1.6%),
  1/55 at N − 1, and rises to 6%, 25%, 57%, 100% at N + 1, + 3, + 5, + 9.
- **Post hoc (not pre-registered; to be tested before use):** within G43, no structure with
  k_cont ≤ N − 2 fit (0/233 at N − 3 or below, and no structure with k_cont = N − 2 was sampled); with the
  Clebsch choices counted as bits, no structure that fit has k ≤ N − 2, and the fully shared compressive
  group (k_cont 9-12, the analogue of exps 039/041) is 0/100. So the failure of compressive models is not
  tied to one sharing property (mass rigidity, mixing inheritance): any fixed sharing works for some
  discrete choice once the continuous parameters roughly equal N, and nothing works below that.

## Reading of the 11 models
The models in step 1 fail by ≥ 10σ, but this experiment does not support a single structural cause. The
random textures show that sharing with fixed Clebsches is not by itself fatal (C1, C2 have counterexamples);
what is fatal, in this generator, is having fewer continuous parameters than observables, regardless of
which property is shared. In the published models the misfit lands on whichever observables the shared
parameters fix (mass ratios in BSV/yukawaon/AB, θ23^ℓ in FZ4U, θ12^ℓ in BPR), which is what one expects if
the cause is counting at 2% precision rather than a specific mechanism. Caveat: G43 is one generator
(symmetric, shared zero pattern, M_Z, no running) and Y2 Clebsch matrices were sampled, not profiled.

## The remaining loophole
A quark-lepton relation that is compressive and fits would need one of:
1. a quark sector that predicts some of its own 8 observables (k_q ≤ 6), leaving room for ≤ 5 lepton
   parameters. This is the cheapest target.
2. a discrete choice (Clebsch matrix) selected with few bits. With k_cont ≈ 12-14 the budget is 0-1 bit, so
   any such relation must be fixed by theory in advance (a single SO(10)/SU(5) assignment, not a scan).
3. structures outside G43: non-symmetric (lopsided, SU(5) L ↔ d^c) textures, different zero patterns per
   sector, or running/thresholds at the GUT scale.
Concrete next search: enumerate quark textures with ≤ 6 continuous parameters (zero patterns + fixed ratios,
including non-symmetric) against the 8 quark observables at the 2% floor; only survivors get a lepton
sector, with every discrete choice counted. G43 has no such quark sector by construction (3|Z| ≥ 9 quark
parameters; the 3-entry patterns are the smallest and 7 of 10 fail stage 0), so this needs a new generator.

## Files
- `claims.md` (pre-registration), `run.log` (hash, launches, progress; local paths replaced by `<PMNS-CKM>`).
- `lib43.py` generator, fast observables, proxy, fitter, official scoring; `check_fast.py`; `probe.py`.
- `stage0.py` → `bank.json`; `screen.py` → `screen.json`; `main.py` → `fits.jsonl` (slim: spec, k, χ², max pull,
  pass; parameters and pulls kept for Y2 passes). Full records (parameters, pulls, predictions for every
  structure) in gitignored `artifacts/exp043/fits_full.jsonl`; regenerate with `main.py 3 1.0` (≈ 45 min).
- `summarize.py` → `summary.json`, `summarize.log`; `posthoc.py` → `posthoc.json`, `posthoc.log` (post hoc).
- `fits_pilot_scale0.4.jsonl`: stopped pilot, not used.

Rerun from this folder: `../../.venv/Scripts/python ../../tools/pin.py ../../.venv/Scripts/python <script>`.

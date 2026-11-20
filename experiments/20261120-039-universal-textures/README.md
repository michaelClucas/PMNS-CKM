# Exp 039: universal textures (democratic, four-zero, GEO grammar)

## Hypothesis
STATE.md open question: can a family in which quark and lepton sectors share a small number of
continuous parameters by construction reach k ≤ N − 2 while fitting PDG-precision data? Test the
most compressive "universal texture" classes: one texture pattern and one shared shape/phase set
for u, d, e, ν, sectors differing only through their own mass ratios (and, in post-hoc variants,
fixed sign factors).

## Method
- Structures read from the papers (arXiv IDs, equation numbers): `model.md`. Classes:
  - (a) **DEMU** universal democracy (hep-ph/9912358 eqs (4.44), (4.47) with ε = δ, (4.52)) in
    all four sectors, one shared phase φ; **DEMU2** the same with φ_u = φ_ν, φ_d = φ_e;
    **DEMFX/DEMFX2** literature lepton structure (review eqs (5.4)-(5.6); hep-ph/9808272 eq (5)):
    democratic charged leptons, near-degenerate neutrinos, quarks as DEMU (one or two phases).
  - (b) **FZ4U** universal four-zero Hermitian texture (hep-ph/0212195 eqs (4)-(9)): shared shape
    a (A_f = (1−a)λ3 + a max(0, λ1, λ2)), shared phases (φ1, φ2) on d and e, universal eigenvalue
    sign η. Post-hoc variants: **FZ4S** fixed −1 factors on the lepton (1,2) and/or (2,3) phases
    (φ_k → φ_k + π, 2 bits); **FZ4E** a_e fixed at its endpoint, a shared by u, d, ν (2 bits for
    the sector choice); **FZ4F** diagnostic with a per sector (not universal).
  - (c) **GEO grammar**: M_f[ij] = c_ij sqrt(μ_i μ_j) e^{i s_f φ_ij}, c_ij ∈ {0,1}: 64 patterns
    × {Dirac, seesaw} = 128 entries (7 bits). MLP emulator trained on 1.34M synthetic theory-space
    draws (no measured number), inverted per entry against a Gaussian surrogate of the data, then
    the emulator's best entries re-fitted with the exact engine and the official scorer
    (planned; not completed, see Result).
- Neutrinos: Dirac (same Hermitian matrix) or type-I seesaw with M_D of the pattern and M_R ∝ 1
  (1 bit). Phase sharing d ↔ e, u ↔ ν declared before fitting (no bit).
- Scoring: `src.evaluate` (official scorer), NuFIT 6.1 default analysis, PDG 2026 masses with the
  2% floor (`mass_floor=0.02`). N = 15: CKM 4 (s12, s13, s23, δ), PMNS 4, six same-scale mass
  ratios (mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau), dm_ratio. mr_b_tau is not scored: the
  sector scales c_f are free and cancel, so the model makes no b/τ prediction. NO declared for
  hierarchical textures (no ordering bit); DEMFX scored in both orderings.
- Scale: primary M_Z (no running, declared before fitting). Robustness: GUT (2e16 GeV, SM, via
  `scale=`) for FZ4U, DEMU, DEMU2, DEMFX. Running changes no verdict (table below).
- Fits: scipy DE (popsize 10, maxiter 300) on `Scorer.chi2_soft(include_finite=True)`, L-BFGS-B
  polish, official `evaluate` at the optimum (`fit.py`). FZ4U: 2 seeds per configuration
  (identical optima). Where a PMNS prediction is outside the NuFIT tables the official χ² is inf;
  `chi2_soft_gauss` (scorer's Gaussian fallback) says how far.
- Held-out (`heldout.py`): fit everything except the two neutrino mass parameters to quarks +
  charged leptons (N 10), freeze, fit only the neutrino parameters to PMNS + dm_ratio (N 5).
- Honest k: fitted continuous parameters + bits for η, neutrino option, sign factors, pattern.

## Parameter count
| class | k_cont | k_d (bits) | k | N | N−2 |
|---|---|---|---|---|---|
| FZ4U | 11 (8 mass + a + φ1 + φ2) | 2 (η, ν option) | 13 | 15 | 13 |
| FZ4S | 11 | 4 (+ 2 sign bits) | 15 | 15 | 13 |
| FZ4E | 11 | 4 (+ 2 sector-choice bits) | 15 | 15 | 13 |
| FZ4F (diagnostic) | 14 | 2 | 16 | 15 | 13 |
| DEMU | 9 (8 + φ) | 1 | 10 | 15 | 13 |
| DEMU2 | 10 | 1 | 11 | 15 | 13 |
| DEMFX / DEMFX2 | 9 / 10 | 0 (+1 ordering) | 9-11 | 15 | 13 |
| GEO entry | 8 + n_off (≤ 3) | 7 | 15-18 | 15 | 13 |

Only FZ4U, DEMU, DEMU2 and DEMFX pass compression by count. The GEO grammar fails compression for
every entry by count (7 bits).

## Result
Best fit per class (official scorer; full table with pulls in `summary.json`, every run in
`fits.jsonl`). "gauss" = scorer Gaussian fallback when a PMNS value is outside the NuFIT tables
(official χ² = inf there).

| class (best options) | scale | N | k | χ² | p_global | worst pulls | compression | fit |
|---|---|---|---|---|---|---|---|---|
| FZ4U (η +1, seesaw) | M_Z | 15 | 13 | inf (gauss 1159) | 0 | s23sq outside table (pred 0.003), mr_d_s −4.8, δ_q +3.1 | ok | fail |
| FZ4U (η +1, seesaw) | GUT | 15 | 13 | inf (gauss 1156) | 0 | s23sq outside table, mr_d_s −4.7 | ok | fail |
| FZ4U other 3 options | M_Z | 15 | 13 | inf (gauss 1208-1745) | 0 | s23sq pred 0.003-0.013 | ok | fail |
| DEMU (Dirac) | M_Z | 15 | 10 | inf (gauss 5074) | 0 | s23sq 0.046, δ_q −46σ (pred 0), ckm_s13 −20 | ok | fail |
| DEMU2 (Dirac / seesaw) | M_Z | 15 | 11 | inf (gauss 2395 / 6923) | 0 | s23sq 0.05-0.14, s13sq −35, ckm_s13 −21 | ok | fail |
| DEMFX2 (NO) | M_Z | 15 | 10 | 2233 | 0 | s13sq −32, ckm_s13 −21, s12sq +16 (pred 0.500) | ok | fail |
| DEMFX (NO / IO) | M_Z, GUT | 15 | 9 | 4441-5117 | 0 | δ_q −46, s13sq −32, ckm_s13 −20 | ok | fail |
| FZ4S (η +1, seesaw, φ2 + π) | M_Z | 15 | 15 | 41.4 | 2.3e-8 | mr_d_s −4.6, δ_q +2.4, δ_ℓ +2.0 | fail | fail |
| FZ4S other 8 runs | M_Z | 15 | 15 | 180-626 or inf | ≤ 1e-37 | | fail | fail |
| FZ4F (diagnostic, a per sector) | M_Z | 15 | 16 | 0.20 | 0.66 | δ_ℓ −0.4 | fail (k 16 > N) | ok |

Mechanism, (b). In the four-zero texture |V_23| ≈ |sqrt((m3−A)/m3)_up − e^{iφ2} sqrt((m3−A)/m3)_down|
(hep-ph/0212195 eq (10c)). With a universal shape both terms are ≈ sqrt(a) in every sector, and the
quark fit needs φ2 ≈ 0 (here 6.18-6.20 rad) for the cancellation that makes V_cb small. The same
a and φ2 then cancel the lepton 2-3 rotation too: sin²θ23 ≤ 0.013 in every FZ4U run against 0.47.
The diagnostic FZ4F fits everything (χ² 0.20, k 16) with a_u ≈ 0.17, a_d ≈ 0.18 (the quarks are
close to universal) but a_e ≈ 0.88-1.00 and a_ν ≈ 0.24-0.53: large θ23 needs a charged-lepton
shape far from the quarks'. A fixed −1 on the lepton φ2 (FZ4S, post hoc, 2 bits) lifts the
cancellation (s23sq enters the table) but the shared a then misfits mr_d_s and δ: best χ² 41.4
(N 15, dof 4).
FZ4E (a_e fixed at its endpoint, a shared by u, d, ν; k 15) was defined from the FZ4F result but
not run: no CPU budget after the CPU-rule change. By count it fails compression anyway.
FZ4S shift (1,1) ran for one of four (η, ν) options only (χ² 209); the batch was stopped by the
CPU-rule change.

Mechanism, (a). The democratic quark texture with ε = δ predicts |V_ub| ≈ 0.0018-0.0021
(−20σ against 0.00376; the |V_ub/V_cb| ≈ sqrt(m_u/m_c) problem that hep-ph/0212195 describes),
and one universal φ gives δ_CKM = 0 (−46σ). Universal democracy in the lepton sector gives
U_PMNS ≈ 1 + small (charged leptons and neutrinos are both diagonalised by U_0 at leading order):
s13sq ≈ 0.002-0.005, s23sq 0.05-0.14. The Fritzsch-Xing lepton structure (DEMFX) reproduces its
published prediction (s12sq = 0.500, s23sq = 0.70, i.e. sin²2θ_sun = 1, sin²2θ_atm = 0.84;
hep-ph/9808272 Table 1: 0.99 and 0.84-0.94), which is excluded (s12sq +16σ), and its leptons share
no continuous parameter with the quarks anyway.

Scale. GUT-scale scoring moves the FZ4U best Gaussian χ² from 1159 to 1156 and DEMFX from 4441 to
4955; the failing pulls are 10-50σ, far above any running or 2-loop effect. No verdict depends on
the scale.

Held-out (quarks + charged leptons → PMNS, `heldout.json`). FZ4U stage 1 (N 10, k_cont 9):
χ² 27.2 for both η, driven by mr_d_s −4.0σ, a = 0.18-0.19. Stage 2 (only the two neutrino mass
parameters free, N 5): PMNS outside the NuFIT tables for all four options; over a 60×60 grid of
neutrino parameters with dm_ratio within ±30% of data, s23sq ∈ [0.0016, 0.0088] (Dirac, η +1) /
[0.0018, 0.0057] (Dirac, η −1). The quark side does fix the lepton 2-3 mixing, at the wrong value.
DEMU stage 1 already fails (χ² 2861, N 10).

(c) GEO grammar: NOT COMPLETED. Synthetic theory-space data were generated (1.5M draws, 89.6%
finite, seed 39, `artifacts/exp039/geo_train.npz`; the batched generator agrees with the scalar
`model.geo_model` on 1173 of 1176 compared values, the 3 mismatches at degenerate spectra). The
emulator was not trained to completion: the first run (h = 512) took 222 s per epoch; after the
CPU-rule change the second (h = 256) finished one epoch (val MAE 0.82 in feature units, i.e. far
from usable) and then got ≈ 0.4 core; the third did not finish an epoch in 7 min and was stopped at
the 3 h wall budget. So no emulator ranking, no search.json and no exact verification exist.
What stands without it: by count every GEO entry has k = 8 + n_off + 7 ≥ 15 > N − 2 = 13, so no
grammar hit could pass compression; and the one grammar entry closest to (b) (pattern 46, the
four-zero with unit coefficients) is a special case of FZ4U's structure with the shape fixed, so
it inherits FZ4U's θ23 cancellation. Whether some other zero pattern fits at all is untested.
`grammar.py` holds the last attempted settings (h 256, 2 epochs, 128 starts × 200 steps, top 3
verified with DE popsize 8, maxiter 100).

## Verdict
No candidate. Every universal class that passes compression by count fails fit quality by a wide
margin: FZ4U (k 13 = N − 2) predicts sin²θ23 ≤ 0.013 (the shared shape and the φ2 ≈ 0 that V_cb
needs cancel the lepton 2-3 rotation exactly as they cancel V_cb); DEMU/DEMU2 (k 10-11) give
|V_ub| at −20σ and near-diagonal PMNS; DEMFX reproduces its published sin²θ12 = 0.5, excluded at
+16σ. These are fit-quality failures at > 10σ under both scales, so per protocol item 7 the
universal four-zero (shared a, φ1, φ2) and universal democratic (ε = δ, shared φ) classes are
**excluded** at PDG/NuFIT precision, not just "no evidence". Making the texture fit needs
sector-specific freedom (FZ4F: a_e ≈ 0.9-1.0 vs a_q ≈ 0.17, χ² 0.20) that costs k 16 > N; the
cheapest fixed-factor repair tried (FZ4S, −1 on lepton φ2, k 15) still fails (χ² 41.4, p 2e-8).
The held-out direction works mechanically (quark side fixes θ23^ℓ) but predicts the wrong value.
This is the pattern STATE.md records for BSV and common τ: sharing parameters forces a quark-like
lepton sector. Criterion-4 protocol not run (no class passed compression and fit). Part (c) is
incomplete (above). Not tested here: universal textures with 2-3 symmetric shapes (Koide et al.,
hep-ph/0209333) and non-Hermitian universal textures.

## Files
- `model.py` structures; `model.md` paper equations and how the classes differ from GEN.
- `fit.py` exact fits (`fits.jsonl`, all runs incl. both seeds); `summarize.py` → `summary.json`.
- `heldout.py` → `heldout.json`.
- `grammar.py` part (c): `gen`, `train`, `search`, `verify` (only `gen` completed). Training data and checkpoint in gitignored `artifacts/exp039/`
  (regenerate: `python grammar.py gen`, seed 39).
- `run.log` (fits), `run_grammar.log` (part c), `run_rest.sh` (jobs after the CPU-rule change,
  pinned with `tools/pin.py`, 2 workers). The logs contain failed launches (interpreter path,
  a stale chain whose search/verify ran before the emulator existed and errored), kept as run.

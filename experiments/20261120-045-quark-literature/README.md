# Exp 045: published predictive quark-sector structures at PDG 2026 precision

## Hypothesis
STATE.md task 1 (exp 043 loophole): a compressive quark-lepton model needs a quark sector that predicts some
of its own 8 observables (k_cont ≤ 6, or 7 with one relation). The literature already proposes such
structures. Catalogue them, fit each to the 8 quark observables Q8 = (s12, s13, s23, δ, m_u/m_c, m_c/m_t,
m_d/m_s, m_s/m_b) with the official scorer at the paper's stated scale, and report which survive. Exp 044
(another agent) enumerates a texture grammar from scratch; this experiment covers what papers propose.

## Method
1. Literature (models.md has matrices, equation numbers, and what was read for each): arXiv TeX source of
   Ramond-Roberts-Ross (hep-ph/9303320); rendered PDF text of RRRV (hep-ph/0104088), Tanimoto-Yanagida
   (1601.04459), Ludl-Grimus (1501.04942), Belfatto-Berezhiani (2305.00069), Harrison-Scott (2501.18508);
   ar5iv HTML of Giraldo-Rojas (1511.08858, 1811.05068), Verma (1708.08993), Fritzsch-Xing (hep-ph/9707215);
   abstracts of Hall-Rasin (hep-ph/9303303), Fritzsch-Xing (hep-ph/9807551), Blum-Hagedorn-Lindner
   (0709.3450), Rodejohann-Zhang (1101.5525). Papers kept in the session scratchpad, not committed.
   23 structures in total: 16 textures (incl. one non-predictive reference), 6 closed-form relations,
   Harrison-Scott; 4 more non-predictive or unscorable entries listed in models.md.
2. Counting. k_cont = Jacobian rank of the 8 observables with respect to all moduli and phases the paper
   writes (`lib45.reduce_params`, `ranks.log`); rephasing-redundant phases are pinned before fitting. The
   ranks equal the papers' own counts wherever the paper states one (RRR: "Solutions 1,3,4 and 5 have 8
   parameters ... Solution 2 has an additional phase", which is 6 and 7 in Q8 units since the 2 sector
   scales drop out). Bits: sign of J where a relation fixes only |δ| or cos α (1); tanβ choice at GUT (1);
   Belfatto-Berezhiani's choice among its 3 benchmark scales (2). η̄_b, and the fitted scale of
   Harrison-Scott, count as +1 continuous.
3. Fits: `src.evaluate.fit_and_evaluate`, PDG 2026, `mass_floor=0.02`, use = Q8 (N 8), NO (no PMNS keys).
   Scales: M_Z; GUT 2e16 GeV MSSM tanβ 10 and 30 (M_SUSY 3 TeV); SM 1e3-1e16 GeV for the two papers that
   state SM scales. Optimiser: DE (`fit45.py`, 1-2 seeds), then L-BFGS-B multistart (`refine45.py`, 32
   Sobol points plus every optimum found for the same structure at any scale, three rounds for the
   textures). The DE seeds disagreed for 6 textures (e.g. LG_S10 1882 vs 78, RRR5 at GUT 431 vs 34,
   Harrison-Scott at 1e7 GeV 20.5 vs 0.78 after warm-starting from the paper's parameters); the refits
   reproduced their own minima in all three rounds, and all numbers below are the minima over all runs.
   The "seeds spread" column of `summarize.log` includes the bad DE runs.
4. η̄_b (eta_q = 0; only (1+η_q)/(1+η_b) enters Q8) at GUT tanβ 30: local refits from the best G30 point on
   a grid (−0.45 ... +0.45 step 0.15; step 0.05 in [−0.4, 0] for RRR3, RRR5) (`post45.py eta`, `eta.log`).
5. Dominant-miss profile (`post45.py miss`, `miss.log`): drop the key with the largest pull, refit the other
   7, and score the dropped key at that optimum. For a relation this is its parameter-independent miss; for
   a texture it is the miss when the other 7 observables are fitted.

## Result

### Ranked table (primary row = the best option the paper allows; `summarize.log` has every scale)
| # | structure | arXiv / source | k_cont + bits | predicts (N − k_cont) | scale used | χ² | p | max pull | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GST \|V_us\| = √(m_d/m_s) | Gatto-Sartori-Tonin 1968; RRRV Eq. (2) | 7 + 0 | 1 (V_us) | M_Z | 0.01 | 0.94 | 0.1 | fits |
| 2 | Belfatto-Berezhiani asymmetric Fritzsch, x_u = 1 | 2305.00069 Eq. (3.2) | 7 + 2 | 1 | SM 1e3 GeV (paper benchmark) | 0.16 | 0.69 | 0.2 | fits (also M_Z 0.00, 1e6 GeV 0.96; 1e16 GeV 5.82, p 0.016) |
| 3 | Harrison-Scott "geometric" texture | 2501.18508 Eq. (2.1)-(2.3) | 6 + 0 (5 + scale) | 2 | SM 1e7 GeV (fitted; paper 1e4 TeV) | 0.78 | 0.68 | 0.6 | fits (M_Z 16.8; 1e12 GeV 9.0) |
| 4 | right unitarity triangle α = 90° | Fritzsch-Xing hep-ph/9807551; Harrison-Scott Eq. (1.2) | 7 + 1 | 1 (δ) | M_Z | 0.42 | 0.52 | 0.6 | fits |
| 5 | RRR solution 5 (= Giraldo-Rojas II, Verma IV) | hep-ph/9303320 Table 1 | 7 + 1 (6 + η̄_b) | 1 | GUT tanβ 30, η̄_b ≈ −0.25 | 0.95 | 0.33 | 0.6 | fits only with η̄_b; η̄_b = 0: χ² 34.2 (G30), 36.3 (G10); M_Z (GR, Verma scale) 20.8, p 3e-5, excluded |
| 6 | RRR solution 3 (= Verma I) | hep-ph/9303320 Table 1 | 7 + 1 | 1 | GUT tanβ 30, η̄_b ≈ −0.30 | 4.95 | 0.026 | 2.0 | excluded (marginal; η̄_b = 0: 132; M_Z 98) |
| 7 | Giraldo-Rojas case I (= 1811.05068 pattern) | 1511.08858 Eq. (16) | 7 + 0 | 1 | M_Z | 5.54 | 0.019 | 1.6 | excluded at its stated scale (marginal); at GUT tanβ 10/30 χ² 0.14/0.00 (not the paper's scale) |
| 8 | D7 \|V_us\| = cos(3π/7) | 0709.3450 | 7 + ≥3 | 1 | M_Z | 15.16 | 1e-4 | 3.9 | excluded (V_us −3.9σ, parameter-independent) |
| 9 | \|V_td/V_ts\| = √(m_d/m_s) | Hall-Rasin hep-ph/9303303; RRRV Eq. (1) | 7 + 1 | 1 | M_Z | 15.95 | 7e-5 | 3.1 | excluded (m_d/m_s −5.1σ when the others are fitted) |
| 10 | Ludl-Grimus S15 | 1501.04942 Table 3 | 7 + 0 | 1 | M_Z | 19.83 | 8e-6 | 3.4 | excluded (V_ub −5.8σ at the 7-key fit) |
| 11 | Ludl-Grimus S10 ("most predictive") | 1501.04942 Eq. (11) | 7 + 0 | 1 | M_Z | 77.7 | 1e-18 | 7.0 | excluded |
| 12 | RRR solution 2 | hep-ph/9303320 | 7 + 1 | 1 | GUT tanβ 30 | 254.9 | 2e-57 | 13.4 | excluded; known-excluded (RRRV hep-ph/0104088 Sec. 2.2, > 99% CL) |
| 13 | RRR solution 1 | hep-ph/9303320 | 7 + 1 (6 + η̄_b) | 1 | GUT tanβ 30 | 300.0 | 3e-67 | 14.6 | excluded; known-excluded (RRRV Sec. 2.2) |
| 14 | RRR solution 4 | hep-ph/9303320 | 7 + 1 | 1 | GUT tanβ 30 | 305.7 | 2e-68 | 14.6 | excluded; known-excluded (RRRV Sec. 2.2) |
| 15 | Ludl-Grimus S9 | 1501.04942 Table 3 | 7 + 0 | 1 | M_Z | 309.4 | 3e-69 | 14.3 | excluded |
| 16 | Dimopoulos-Hall-Raby / HRR (RRR sol. 2, E' = 0) | PRD 45 (1992) 4192; RRR footnote | 6 + 1 (5 + η̄_b) | 2 | GUT tanβ 30 | 338.2 | 4e-74 | 15.2 | excluded; known-excluded (RRR footnote: V_cb = √(m_c/m_t) too large) |
| 17 | Hall-Rasin \|V_ub/V_cb\| = √(m_u/m_c) | hep-ph/9303303 | 7 + 0 | 1 | M_Z | 373.1 | 4e-83 | 16.1 | excluded; known-excluded (RRRV Sec. 2.2) |
| 18 | Fritzsch-Xing: tan θ_u = √(m_u/m_c), tan θ_d = √(m_d/m_s), φ = 90° | hep-ph/9707215 Eq. (8); hep-ph/9807551 | 5 + 1 | 3 | M_Z | 442.3 | 2e-95 | 15.8 | excluded (contains Hall-Rasin) |
| 19 | Fritzsch six-zero | PLB 73B (1978) 317 | 7 + 1 (6 + η̄_b) | 1-2 | best: GUT tanβ 30, η̄_b −0.45 (grid edge) | 460.6 | 4e-102 | 13.5 | excluded (M_Z 613.5); known-excluded (Belfatto-Berezhiani 2305.00069 Sec. 4.2.1; Verma 1708.08993) |
| - | Ludl-Grimus S8, S12 | 1501.04942 | 8 | 0 | M_Z | 230.1, 184.1 | n/a | 12.3, 11.1 | not predictive in Q8, and fails anyway |
| - | Ludl-Grimus S13; Fritzsch four-zero (reference) | 1501.04942; hep-ph/0212195 | 8 | 0 | M_Z | 0.00, 0.00 | n/a | 0 | not predictive in Q8 (k = N) |

Not fitted (models.md): Tanimoto-Yanagida 1601.04459 (k 8), RRRV texture (k 10), Belfatto-Berezhiani with
x_u free (k 10), Verma four-zero cases (k ≥ 8), Koide-type quark relations (need an inter-sector mass
ratio that is not in Q8).

### Dominant miss
In 10 of the 14 excluded entries the largest pull is V_ub low (ckm_s13 −3.4 to −16σ). Dropping it and fitting
the other 7 puts V_ub at −5.8σ (LG_S15), −7.1σ (RRR5, η̄_b = 0), −13 to −23σ (all others) (`miss.log`).
This is the |V_ub/V_cb| ≈ √(m_u/m_c) prediction of exact (1,1) and (1,3) zeros: √(m_u/m_c) = 0.044 at
M_Z, data |V_ub/V_cb| = 0.090 (Hall-Rasin row, −16σ). Structures survive only when the up sector has a
non-zero (1,3) entry (RRR3, RRR5, GR1, Harrison-Scott's complex 12 and 23 entries) or the down 2-3 block
is asymmetric (Belfatto-Berezhiani). The other four: D7 (V_us −3.9σ), |V_td/V_ts| = √(m_d/m_s) (m_d/m_s −3.1σ), and the two marginal entries RRR3 and GR1 (m_d/m_s +2.0 and +1.6).

### Survivors (p > 0.05): lepton extensions in the source papers (not fitted)
- GST: a single relation, no lepton sector in the sources; the usual GUT partner (m_e/m_μ from a GJ factor) is the exp 021-024 R_clebsch filter, already tested.
- Belfatto-Berezhiani: SU(5) × SU(3)_H; Y_e has "analogous form, with asymmetry parameter x_e", with the d-e degeneracy broken by an SU(5)-breaking 24 VEV (Sec. 2). Extension: ≥ 1 continuous (x_e) plus the unspecified 24-VEV contribution; no neutrino sector in the paper.
- Harrison-Scott: no lepton extension (Sec. 5 only remarks on m2/m3 = tan²(π/8) from earlier work). 0 published.
- α = 90°: a relation; no lepton extension in the sources read.
- RRR5 with η̄_b: RRR take Y_e = Y_d with the (2,2) entry × 3 (Georgi-Jarlskog), so charged leptons cost 0 new continuous parameters (the h_b = h_τ condition then sits against η̄_b ≠ 0, which needs η̄_ℓ, +1); no neutrino sector in the paper (needs its own count). Of the five survivors this is the only one with a fixed Clebsch pattern for leptons.

## Verdict
Five published structures fit the 8 quark observables at PDG 2026 precision with the 2% floor and p > 0.05.
They are candidates for the quark half of a compressive model only, and none predicts more than 2 of its
8 observables:
- 1-relation (k_cont 7): GST, α = 90°, Belfatto-Berezhiani (x_u = 1), RRR solution 5 (only with η̄_b ≈ −0.25
  at tanβ 30, k_cont 6 + η̄_b).
- 2-relation (k_cont 6): Harrison-Scott (5 texture parameters + fitted scale ≈ 1e7 GeV, SM running),
  χ² 0.78, p 0.68. It is the only survivor with k_cont ≤ 6.
Excluded (fit quality): 14 entries, including all 6-zero structures (Fritzsch, DHR), RRR 1, 2, 4, the
Hall-Rasin and Fritzsch-Xing mass-mixing relations, D7, |V_td/V_ts| = √(m_d/m_s), and Ludl-Grimus S9, S10,
S15. Marginal (p 0.02-0.03): RRR3 with η̄_b, Giraldo-Rojas case I at M_Z. Six of the excluded structures
were already excluded in the literature (RRRV 2001, Belfatto-Berezhiani 2023, RRR's own footnote).

For the exp 043 loophole: the quark sectors that survive buy at most 2 predictions, so a lepton extension
can spend at most 1-2 parameters more than its lepton observables before the joint model stops being
compressive (k ≤ N − 2). Only RRR5
comes with a fixed lepton Clebsch pattern (GJ), and it needs η̄_b, which costs one more parameter.

## Caveats
- Optimiser: DE alone was unreliable here (seed disagreements up to ×20 in χ²). The final minima come from
  three L-BFGS-B multistart rounds that agree, but excluded verdicts are still upper bounds on the best
  χ². The marginal entries (RRR3 p 0.026, GR1 p 0.019) are the ones a better optimiser could move.
- η̄_b: only at tanβ 30, eta_q = 0, on a grid with local refits (step 0.05 for RRR3 and RRR5, 0.15
  otherwise; the minima for F6, RRR1, RRR4 sit at the −0.45 grid edge, χ² ≥ 300 there). RRR5's η̄_b ≈ −0.25 at
  tanβ 30 is a fitted threshold, not a prediction.
- Harrison-Scott: the scale is fitted on a 6-point grid (1e3 ... 1e16 GeV), counted as 1 continuous parameter.
  The matrix reading (diagonal entries as |ε|^n) is fixed by reproducing the paper's Table 1 exactly
  (`ranks.log`); the other reading gives m_u/m_c 0.0127.
- RRR textures were fitted with the observables run by `src.rg` in the MSSM, not with RRR's own RG and
  threshold treatment (they had m_t 145-185 GeV and h_b = h_τ).
- Ludl-Grimus textures are complex symmetric; we used their Table 3 parametrisation (1 = real positive,
  2 = complex) and did not test whether more phases would be physical.
- Relations are fitted at M_Z. GST and α = 90° are RG-stable at one loop; V_ub/V_cb and m_u/m_c are too.
- D7 bits (choice of n, j, m_u, m_d among the dihedral options) are not counted exactly; it fails at
  −3.9σ regardless.
- Some literature entries were read from abstracts or HTML only (marked in models.md); the five survivors
  were read from PDF (BB, HS), TeX (RRR) or are one-line relations.

## Files
- `models.md`: catalogue, matrices, equation numbers, counts, what was read.
- `lib45.py`: texture builder, observables, Jacobian-rank counting. `structures.py`: the catalogue in code.
- `ranks.py` → `ranks.log`, `ranks.json`: parameter counts and the Harrison-Scott reproduction.
- `fit45.py` (DE), `refine45.py` (L-BFGS-B multistart) → `fits.jsonl`; `post45.py` → `eta.jsonl`, `eta.log`,
  `miss.jsonl`, `miss.log`; `summarize.py` → `summarize.log`, `summary.json`. `probe_data.py`: data targets.
- `run.log`: job log.
Rerun from this folder with `../../.venv/Scripts/python ../../tools/pin.py ../../.venv/Scripts/python <script> [args]`
(fit45 ≈ 40 min on 2 processes; refine45 ≈ 10 min per round; post45 ≈ 10 min).

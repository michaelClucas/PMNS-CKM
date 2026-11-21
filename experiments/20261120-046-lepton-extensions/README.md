# Exp 046: lepton extensions of the exp 045 quark survivors

## Hypothesis
Exp 045 left three published quark structures with a theory-motivated route to leptons: Ramond-Roberts-Ross
solution 5 (RRR5, with η̄_b ≈ −0.25 at GUT tan β 30), Belfatto-Berezhiani 2305.00069 (BB, x_u = 1) and
Harrison-Scott 2501.18508 (HS). If one of them can be extended to charged leptons and neutrinos with ≤ 2-3
new parameters, the joint model could reach k ≤ N − 2 with N = 15 and predict PMNS from the quark sector.

## Method
- Extensions fixed in `extensions.md` before any lepton fit; SHA-256
  `a870d768…3d28` logged in `run.log` (2026-11-20T23:12:30Z). Summary of what was fixed:
  - RRR5: Georgi-Jarlskog Y_e (Y_d with (2,2) × −3, RRR's pattern), η̄_b and η̄_ℓ fitted (MSSM, 2e16 GeV,
    tan β 30, M_SUSY 3 TeV).
  - BB: Y_e = same A, B, C as Y_d with its own asymmetry x_e (paper Sec. 2, after Eq. 2.14). The
    SU(5)-breaking term is the 24-plet VEV in the heavy vector-like fermion mass (g S + f Σ + h Φ) F F̄
    (Eq. 2.14), which splits X_L from X_D, i.e. x_e ≠ x_d and a different sector normalisation; no other
    SU(5)-breaking contribution. SM at the paper's benchmarks 1e3, 1e6, 1e16 GeV.
  - HS: the paper has no lepton sector and none of the 7 citing papers on INSPIRE adds one. Minimal SU(5)
    (Y_e = Y_d^T) and Pati-Salam (Y_e = Y_d, Y_ν = Y_u) as separate options; SM at 1e7 GeV (exp 045's
    fitted scale; scale counted as 1 continuous).
  - Neutrinos, every model: type-I seesaw with Y_ν = Y_u or Y_d (PS: Y_u only) and M_R = 1 (ν1) or
    M_R = diag(1, x, y) (ν2, 2 parameters).
- Scoring: `src.evaluate`, NuFIT 6.1 (IC23_woSKatm), PDG 2026, mass_floor 0.02, ordering NO (declared in
  advance), N = 15 = Q8 (CKM 4 + 4 quark mass ratios) + m_e/m_μ + m_μ/m_τ + 4 PMNS + Δm²₂₁/|Δm²₃₁|. All
  observables are predicted at the model scale and translated with `scale=` (src.rg running, with the η̄
  thresholds for RRR5).
- Counting (all listed options count): RRR5 k = (6 + η̄_b + η̄_ℓ + {0, 2}) + 3 bits (tan β 1, neutrino
  options 2); BB k = (7 + x_e + {0, 2}) + 4 bits (scale 2, neutrino 2); HS k = (5 + scale + {0, 2}) + 3 bits
  (6 lepton options). N − 2 = 13. RRR5-ν2 and HS-ν2 are compressive by count (13, 11); BB-ν2 is not (14).
- Tests, in order of strength:
  1. Parameter-independent bounds (`bounds46.py` → `bounds.log`, `bounds.json`).
  2. Charged-lepton stage (`charged46.py` → `charged.log`, `charged.json`): fit Q8 + m_e/m_μ + m_μ/m_τ (N 10)
     with every quark parameter (and x_e) free. The neutrino sector does not enter these 10 observables, so
     this χ² bounds every N 15 option of the model from below.
  3. Full N 15 fits (`fit46.py` → `fits.jsonl`, summary `summarize46.py` → `summary.json`, `summarize.log`):
     L-BFGS-B multistart (`fit_and_evaluate(method="lbfgsb")`), 6 Sobol + 6 random warm starts from exp 045's
     quark optimum + the best points of a grid over the lepton-only parameters (x_e: 17 values; ln x, ln y:
     35 × 35 over [−34, 34]); 3 rounds (seeds 1-3) per cell. RRR5 at η̄_b = −0.25 (exp 045 optimum),
     η̄_ℓ ∈ {−0.15, 0, +0.15}.
  4. Conditional PMNS test (`cond46.py` → `cond.json`, `cond.log`): quark parameters refitted on Q8 alone,
     then only ln x, ln y free on a 273 × 273 grid (step 0.25) plus L-BFGS-B polish, scoring the PMNS block
     (4 PMNS + dm_ratio). This is the held-out test in its sharpest form: once quarks are fitted, Y_ν and
     U_e are fixed and 2 parameters act on 5 observables.

## Result

### 1. Parameter-independent bounds (`bounds.log`)
- RRR5 + Georgi-Jarlskog: the block-diagonal Y_d makes m_e/m_μ an exact function of m_d/m_s
  (m_e m_μ = m_d m_s, m_μ − m_e = 3 (m_s − m_d)), R = (m_e/m_μ)/(m_d/m_s) ≥ 0.1199 vs data 0.094. The
  (mr_d_s, mr_e_mu) block alone gives χ² ≥ 51.4 (pulls m_d/m_s −6.3σ, m_e/m_μ +3.5σ), the same for
  η̄_b = 0, −0.15, −0.25, −0.35 (R is threshold-invariant). With dof ≤ 5 this is p < 1e-9 for every RRR5
  option. This is the known GJ failure (exp 042 found the same R value for Chen-Mahanthappa).
- HS minimal SU(5) and Pati-Salam: charged-lepton ratios equal the down ratios at the model scale (R = 1).
  The 4-ratio block gives χ² ≥ 2069-2114 at 1e5-1e9 GeV (m_d/m_s −34.6σ, m_μ/m_τ −28σ).
- M_R = 1 (every model): m_ν ∝ Y_ν* Y_ν†, so dm_ratio ≈ (y₂/y₃)⁴: 1.9e-10 (Y_u) or 1.2e-7 (Y_d), still
  ≤ 3e-7 with both ratios raised by 10 %; data 0.0299; outside the NuFIT table (Δχ² = ∞).

### 2. Charged-lepton stage, N 10 = Q8 + m_e/m_μ + m_μ/m_τ, all quark parameters free (`charged.log`)
Three seeds agree to the printed digits in every cell.
| model | cell | χ² (N 10) | largest pulls |
|---|---|---|---|
| BB (x_e) | SM 1e3 / 1e6 / 1e16 GeV | 5422 / 5275 / 13010 | m_μ/m_τ −49.9σ, m_e/m_μ +38σ (1e16: V_us −78σ) |
| RRR5 (GJ) | G30, η̄_b −0.25, η̄_ℓ −0.15 / 0 / +0.15 | 564 / 392 / 273 | m_μ/m_τ −17.7 to −8.9σ, m_e/m_μ +10.5σ |
| RRR5 (GJ) | η̄_ℓ +0.30 / +0.45 / +0.60 (seed 1, `charged_eta.log`) | 198 / 159 / 147 | m_e/m_μ +10.6σ at every η̄_ℓ |
| HS SU5, HS PS | SM 1e7 GeV | 2173 | m_d/m_s −34.6σ, m_μ/m_τ −29.6σ |

BB mechanism: Y_e = [[0,A,0],[A,0,B x_e],[0,B/x_e,C]] has det = −A²C and a 2-3 block with det −B², both
independent of x_e, so with A, B, C fixed by the quarks m_μ m_τ ≈ m_s m_b and m_e m_μ m_τ = m_d m_s m_b in
the same normalisation; x_e only moves the spectrum when B x_e ~ C, which destroys the hierarchy. The
paper's single SU(5)-breaking parameter cannot produce m_μ/m_τ ≈ 3 m_s/m_b and m_e/m_μ ≈ m_d/(3 m_s).

### 3. Full N 15 fits (`summarize.log`, `summary.json`; 90 jobs, 3 seeds per cell)
Every option is outside the NuFIT tables (official χ² = ∞, p_global = 0). "soft" is the minimiser's
landscape (1e4 + Gaussian fallback for PMNS + finite terms), given only to rank failures.

| option (Y_e; Y_ν; M_R) | k_cont + bits = k | compression_ok | χ² | soft (seeds) | fit_ok |
|---|---|---|---|---|---|
| RRR5 GJ; Y_u; 1 | 8 + 3 = 11 | yes | ∞ | 19163 (×3) | no |
| RRR5 GJ; Y_u; diag(1,x,y) | 10 + 3 = 13 | yes | ∞ | 17764 / 17794 / 17803 | no |
| RRR5 GJ; Y_d; 1 | 8 + 3 = 11 | yes | ∞ | 18112 (×3) | no |
| RRR5 GJ; Y_d; diag(1,x,y) | 10 + 3 = 13 | yes | ∞ | 13074 / 13451 / 13451 | no |
| BB x_e; Y_u; 1 | 8 + 4 = 12 | yes | ∞ | 24010 / 31189 / 31189 | no |
| BB x_e; Y_u; diag(1,x,y) | 10 + 4 = 14 | no | ∞ | 26728 / 26821 / 35198 | no |
| BB x_e; Y_d; 1 | 8 + 4 = 12 | yes | ∞ | 23679 / 23984 / 31685 | no |
| BB x_e; Y_d; diag(1,x,y) | 10 + 4 = 14 | no | ∞ | 24465 / 27510 / 27510 | no |
| HS SU5 (Y_d^T); Y_u; 1 | 6 + 3 = 9 | yes | ∞ | 20530 (×3) | no |
| HS SU5; Y_u; diag(1,x,y) | 8 + 3 = 11 | yes | ∞ | 15004 / 15012 / 18235 | no |
| HS SU5; Y_d; 1 | 6 + 3 = 9 | yes | ∞ | 21226 (×3) | no |
| HS SU5; Y_d; diag(1,x,y) | 8 + 3 = 11 | yes | ∞ | 15960 / 15962 / 15962 | no |
| HS PS (Y_d); Y_u; 1 | 6 + 3 = 9 | yes | ∞ | 20530 (×3) | no |
| HS PS; Y_u; diag(1,x,y) | 8 + 3 = 11 | yes | ∞ | 15004 / 15012 / 18235 | no |

N = 15 and p_global = 0 throughout; max pull ∞ (PMNS outside the tables; the largest finite pulls are
s13² ≈ −43σ, m_μ/m_τ, m_e/m_μ, m_d/m_s, see `summarize.log`). Soft values are minima over cells (scales;
η̄_ℓ ∈ {−0.15, 0, +0.15} for RRR5, best at +0.15). HS SU5 and HS PS with Y_ν = Y_u give identical numbers:
Y_d is Hermitian, so Y_d^T = Y_d* has the same singular values and conjugate left rotations. The seeds do
not agree to Δχ² < 1 on the soft landscape for several options (spread up to 30 %), but every seed of every
option is outside the tables, and sections 1-2 bound the χ² from below independently of the optimiser.

### 4. Conditional PMNS test (quarks fitted, only M_R's x, y free; `cond.json`)
Run for 3 of the 11 ν2 cells before the time limit (stopped; the traceback in `cond.log` is from the stop):

| cell | Q8 χ² | grid points inside NuFIT tables | best PMNS prediction |
|---|---|---|---|
| RRR5 GJ, Y_ν = Y_u, G30 | 0.95 | 0 / 74,529 | s12² 0.31, s13² 1e-4, s23² 0.015, dm_ratio ≈ 0 |
| RRR5 GJ, Y_ν = Y_d, G30 | 0.95 | 0 / 74,529 | s12² 0.00, s13² 0.022, s23² 1.0 |
| BB, Y_ν = Y_u, 1e3 GeV | 0.16 | 0 / 74,529 | s12² 0.96, s13² 0.002, s23² 1e-4 |

With the quark sector fitted, M_R = diag(1, x, y) does not bring the PMNS angles into the NuFIT 6.1 ranges
anywhere on the grid: the hierarchical Y_u or Y_d leaves at most one large angle.

### Held-out test
Not run: no option has fit_ok. The conditional test shows the PMNS block is not reached even with the 2
neutrino parameters fitted to it, so PMNS is not predicted.

### Diagnostic only: PMNS-only fits (`fits_PMNS.jsonl`, `run_pmns.log`; stopped after 22 of 45 jobs)
Fitting only the 4 PMNS + dm_ratio over all parameters reaches χ² 3.3 (RRR5, Y_ν = Y_u) and 0.0 (BB,
Y_ν = Y_u), but only with anarchic quark parameters (e.g. sin θ_23^q = 1, m_s/m_b ≈ 0). The failure is the
joint constraint, not the seesaw form alone. These numbers are neither bounds nor fits of the model.

## Verdict
All 14 lepton extensions (RRR5 × 4, BB × 4, HS × 6) are excluded (fit quality). Each fails at the
charged-lepton stage before neutrinos enter: RRR5 by the Georgi-Jarlskog R bound (χ² ≥ 51 from two mass
ratios, ≥ 147 for Q8 + charged leptons at any η̄_ℓ tried), BB because x_e cannot separate the lepton from the
down spectrum (N 10 χ² ≥ 5275), HS because minimal SU(5) / Pati-Salam give R = 1 (χ² ≥ 2069). No N 15 fit
enters the NuFIT tables, and with the quark sector fitted the ν2 option tested does not reach the PMNS
ranges. The compressive-by-count options (RRR5-ν2 k 13, HS-ν2 k 11, all ν1) all fail fit quality; BB-ν2 also
fails compression (k 14 > 13). No candidate; PMNS is not predicted by any of these extensions.

## Caveats
- Ordering NO only (declared in advance). IO would cost 1 bit and does not touch the charged-lepton failures.
- RRR5's η̄_b fixed at −0.25 (exp 045 optimum) in the N 15 fits; η̄_ℓ on a 3-point grid there and a 6-point
  grid in the N 10 stage. The R bound is η-independent at leading order (checked at 4 η̄_b values).
- RRR5 run with src.rg (1-loop Yukawas, MSSM single threshold), not RRR's own running.
- The refinement round (`refine46.py`) was written but not run: the machine slept ~14.5 h mid-run and the
  overseer set a deadline. The conditional test covers 3 of 11 cells. Both are moot for the verdict, which
  rests on sections 1-2 (optimiser-independent, or seed-stable to the printed digits).
- BB's SU(5)-breaking enters only through x_e and the sector normalisation, as the paper describes; a
  10-plet contribution y_d T F̄ H (paper p. 9) would add parameters and was not included.
- BB at 1e6 GeV: exp 045 had no quark optimum there; round-1 N 15 jobs at that scale had no quark warm start
  (the other two scales did), `cond46`/`charged46` start from the nearest scale refitted at 1e6 GeV.
- Not counted in k: the selection of these 3 structures from exp 045's 23, and NO.
- Process note: one early unpinned interpreter check (`import pypdf`, < 1 s) and one stray `python3` call
  (hung, no work, stopped) ran outside tools/pin.py.

## Files
- `extensions.md` (SHA-256 in `run.log`), `lib46.py` (matrices → observables), `bounds46.py` → `bounds.log`,
  `bounds.json`; `charged46.py`, `charged_eta.py` → `charged.log`, `charged.json`, `charged_eta.log`,
  `charged_eta.json`; `fit46.py` → `fits.jsonl` (N 15), `fits_PMNS.jsonl`; `summarize46.py` → `summarize.log`,
  `summary.json`; `cond46.py` → `cond.json`; `refine46.py` (not run); `run.log`, `run_pmns.log`.
- Rerun from this folder with `../../.venv/Scripts/python ../../tools/pin.py ../../.venv/Scripts/python <script>`:
  `bounds46.py` (seconds), `charged46.py 1` (≈ 40 min), `fit46.py all 1,2,3 2 6 6 N15` (≈ 1 h on 2
  processes), `summarize46.py`, `cond46.py 1` (≈ 4 min per cell).

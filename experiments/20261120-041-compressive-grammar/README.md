# Exp 041: compressive Clebsch grammar (exp 039 part (c) redesigned)

## Hypothesis
Exp 039's zero-pattern grammar had k ≥ 15 > N − 2 = 13 for every entry, so it could not produce a
compressive hit. Build a grammar that is compressive by construction (shared continuous parameters,
fixed rational sector factors, a small pattern grammar, honest k ≤ 13 for N = 15) and measure what
fraction of it fits PDG/NuFIT data at all.

## Design: the bit budget fixes the grammar size
N = 15 (CKM s12, s13, s23, δ; PMNS s12², s13², s23², δ; mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu,
mr_mu_tau; dm_ratio), so k_cont + ceil(log2 G) ≤ 13.

1. Up quarks cannot share parameters with down quarks through a short Clebsch list. For any
   symmetric texture with Y11 = Y13 = 0 and Y33 = 1, |det Y| = |Y12|², so a sector factor β on the
   (1,2) entry gives (m_u m_c m_t)/(m_d m_s m_b) = (β_u/β_d)² up to (m3_u/m3_d)³ = O(1). Data (M_Z):
   (m_u/m_c)(m_c/m_t)² / ((m_d/m_s)(m_s/m_b)²) = 1.54e-3, i.e. β_u/β_d ≈ 0.039. Nothing in
   {±1, ±3, ±1/3, ±2} comes close (nearest 1/3 is off by 9x in the ratio, ~80x in m_u/m_c). So u and d
   need their own parameters.
2. The quark sector then needs ~8 parameters for its 8 observables: the four-zero texture (Y_u real,
   Y_d with 2 phases) is the most economical that fits (below). That leaves 5 bits for the leptons.
3. The leptons get no parameters at all: Pati-Salam / SO(10) quark-lepton relations per texture entry.

Result: a grammar of 32 entries (G32) plus a 2-bit phase-placement choice (below) gives k = 15, and
the compressive sub-grammar G8 (8 entries) gives k = 13 = N − 2. This is as large as a compressive
grammar can be once the quark sector is fitted; a larger one would need the quark sector to predict
some of its own observables, which no texture tried in the program does at the 2% floor.

## Method
- Grammar (`grammar.py`, `grammar.json`). Symmetric four-zero Yukawas, (3,3) = 1 (sector scales not
  fitted, ratios only):
  Y_u = [[0, zu, 0], [zu, wu, xu], [0, xu, 1]] (real), Y_d = [[0, zd e^{ip1}, 0], [zd e^{ip1}, wd e^{ip2}, xd], [0, xd, 1]].
  8 real parameters (fit coordinates: log z, log x, signed log of w − x², p1, p2).
  Each texture position P ∈ {12, 22, 23} couples through a (1,2,2) ("10", h) or (15,2,2) ("126", H)
  Higgs: Y_e[P] = λ Y_d[P], Y_ν[P] = λ Y_u[P], λ = 1 or −3; (3,3) is always 10. 8 patterns.
  Neutrino options: Dirac; type-I with M_R ∝ 1; type-II m_ν ∝ F_d (126 part of Y_d, SO(10) 126
  dominance); type-II m_ν ∝ F_u. 8 × 4 = 32 entries. Entry name: rep letters for (12, 22, 23) then ν.
  2 entries (hhh-IIFd, hhh-IIFu) are empty by construction (no 126 position, m_ν ≡ 0); they stay in G.
- Phase placement (disclosed, counted). The first implementation put the d phases on (z, x): the
  quark-only fit failed (χ² 310, |V_ub| −14σ). Phases on (z, w) fit the quarks exactly (χ² 0.0);
  (w, x) gave 2.5; three phases were also tried. Data-selected choice among 4 tried: 2 bits.
- G8 (compressive sub-grammar), declared after the quark-only fits and before any lepton prediction
  was computed: Georgi-Jarlskog placement ((2,2) is 126), (1,2) and (2,3) free, ν ∈ {I1, IIFd}.
- Scoring: `src.evaluate` official scorer, NuFIT 6.1 default analysis, PDG 2026 masses with
  `mass_floor=0.02`, M_Z (no running), ordering NO declared (hierarchical textures; no bit).
- Step 2 filter (`filter.py`, `filter.json`), rule fixed before lepton numbers were looked at:
  fit_ok needs χ² ≤ 14.07 (p 0.05, dof 7) and |pull| < 2, so a passing point has χ²_quark ≤ 14.07
  and s23² in the NuFIT NO 1D 2σ range [0.4436, 0.5746]. Quark-only DE (4 seeds, all χ² 0.0 at the
  same point), Laplace clouds at 1, 4, 16 × covariance, 6263 quark-allowed points; kill an entry if
  no point reaches the 2σ range. Sampling-limited; checked by exact fits of every entry.
- Emulator (`emu.py`): 2,000,000 synthetic draws (uniform entry, uniform parameters in the fit box,
  `src.yukawa.observables_batch`; 1,874,597 finite; seed 41; theory-space only, no measured number),
  MLP 17 → 4×256 SiLU → 17 features, 90/10 split, 13 epochs (1518 s, 2 threads). Search: 512 Adam
  starts per entry on a diagonal Gaussian surrogate χ² (measured values as target only), 304 s.
- Exact verification (`fit41.py`): every entry (the grammar is smaller than the planned top-200 +
  random-50 sample, so all 32 are fitted): DE (popsize 12, maxiter 200, sobol) seeds 1 and 2 with
  L-BFGS-B polish, plus L-BFGS-B + Nelder-Mead polish from the emulator optimum. 89 fit runs.
- Stage-1 diagnostic (`fit_qe.py`): quarks + charged leptons only (N 10, k_cont 8) per rep pattern.
- Post-hoc diagnostic (`posthoc_clebsch.py`, not part of the grammar): at the quark-only best fit,
  scan Y_e Clebsches over a 14-value SU(5)-type list for the two charged-lepton ratios.

## Parameter count
| grammar | G | bits (pattern + phase placement) | k_cont | k | N | N−2 | compression by count |
|---|---|---|---|---|---|---|---|
| G8 (compressive) | 8 | 3 + 2 | 8 | 13 | 15 | 13 | ok |
| G32 (full, diagnostic) | 32 | 5 + 2 | 8 | 15 | 15 | 13 | fails |
k distribution: 8 entries at k 13 (G8), 24 at k 15. Trials factor: 5 bits (G8) / 7 bits (G32).

## Result
Filter: 31 of 32 entries killed (all 8 of G8). Survivor HhH-IIFd: its m_ν is near-degenerate (dm_ratio = 1 at the
quark point), so s23² sweeps [0, 1] over the cloud; its exact fit is χ² 1.4e5. No killed entry fits (exact fits).

Quark-only best fit (`quarkpoint.json`): Y_d 2-3 block x = 0.450, w = 0.197, i.e. w ≈ x², nearly
rank one (the four-zero "a ≈ 0.17" regime of exp 039). Lepton predictions there are parameter-free:
m_μ/m_τ = 0.0187 (hh on 22, 23: Y_e = Y_d in that block), 0.40 (hH), 0.63 (Hh), 0.77 (HH) against
0.0589; any −3 on (2,2) or (2,3) destroys the w ≈ x² cancellation. Dirac/I1 give s23² ≤ 0.033;
type-II gives s23² anywhere from 0.02 to 1.0 by pattern, with dm_ratio 0.38-1 or 0 (data 0.030).

Stage 1, quarks + charged leptons (N 10, dof 2): best χ² 201 (hHh), then hhH 268, hHH 303,
HhH 1050, HHH 1091, HHh 1181, hhh 14709 (Y_e = Y_d), Hhh 36096. Forcing m_μ/m_τ pushes the fit to small
x, where |V_ub| fails (−12 to −15σ), the same failure as the (z, x) phase placement. The charged
leptons alone exclude every rep pattern before any neutrino observable.

Full fits (official scorer, `summary.json`; "gauss" = scorer Gaussian fallback when PMNS is outside
the NuFIT tables, official χ² = inf there):

| family | best entry | k | official χ² | gauss χ² | p_global | worst |
|---|---|---|---|---|---|---|
| ν IIFu | hHH-IIFu | 15 | inf | 3868 | 0 | s12² 0.0014 (outside table), s13² −35σ, s23² 0.73 |
| ν IIFd (G8) | hHH-IIFd | 13 | inf | 4137 | 0 | s12² 0.0009, s23² 0.83 (outside), s13² −32σ |
| ν Dirac | hHH-dirac | 15 | inf | 5094 | 0 | s12², s23² outside, s13² −43σ |
| ν I1 | hhH-I1 | 15 | inf | 7101 | 0 | s23², dm_ratio outside, s13² −43σ |
| G8 best I1 | hHh-I1 | 13 | inf | 7124 | 0 | as above |
| only finite χ² | HhH-IIFu | 15 | 29015 | | 0 | δ_q −77σ, dm_ratio +76σ, mr_u_c +75σ |
Per rep pattern best (gauss χ²): hHH 3868, HHH 4428, hHh 5104, hhH 5779, HHh 6065, HhH 6105,
hhh 23049, Hhh 43722. Fraction of the grammar that fits: 0/32 (0/8 for G8); compression_ok and fit_ok
together: 0. In the type-II entries the fit trades the quarks for the charged leptons and dm_ratio
(all near data) but then s12² ≈ 0.001: no solar angle.

Emulator (`emulator_train.json`, `search.json`): held-out MAE in log10 units 0.014-0.04 for mass
ratios and CKM angles (3-10%, coarser than the 2% floor), 0.17 for log s13², 0.13 for log dm_ratio,
0.18-0.19 for cos/sin δ_ℓ. Ranking against exact best χ² over the 30 valid entries: Spearman 0.78,
top-5 overlap 3/5; the emulator start beat DE in 5/30 entries. It ranked the two empty entries
(m_ν ≡ 0) 6th and 8th: it cannot know an entry is degenerate. Useful as a coarse filter only (as in
exp 017); every number above is from exact fits.

Post hoc (`posthoc_clebsch.json`): at the quark point, none of 2744 Clebsch triples from
{±1/2, ±1, ±3/2, ±2, ±3, ±9/2, ±6} reproduces the two charged-lepton ratios (best χ² 62 for
(3/2, 3, 3/2)). Richer lepton Clebsches would not rescue this quark solution.

Not done: GUT-scale scoring (pulls of 10-75σ and outside-table PMNS are far beyond running effects;
exp 039 found scale changes nothing at this size); held-out test (no entry passed); nulls (not
applicable).

## Verdict
No candidate. The compressive grammar (G8, k 13 = N − 2, 5 bits) and the full 32-entry diagnostic
grammar (k 15) have 0 entries that fit: best χ² is outside the NuFIT tables everywhere (Gaussian
fallback ≥ 3868 for N 15). Per protocol item 7 this is a fit-quality failure by > 10σ, so the class
"four-zero symmetric Yukawas with Pati-Salam/SO(10) per-entry 10/126 lepton Clebsches (λ ∈ {1, −3}),
type-I (M_R ∝ 1), type-II (∝ F_d or F_u) or Dirac neutrinos" is **excluded** at PDG/NuFIT precision.
The mechanism is exp 039's in a new form: the quark fit needs a nearly rank-one 2-3 block (w ≈ x²,
x ≈ 0.45), and fixed lepton Clebsches either keep it (m_μ/m_τ = m_s/m_b, s23² tiny) or break it far
too much (m_μ/m_τ 0.4-0.8); fitting the charged leptons instead breaks |V_ub|. Bit budget: once the
quark sector takes 8 parameters, a compressive grammar has at most 32 entries (5 bits), and any
grammar in which up quarks share parameters with down quarks through a short rational list fails the
determinant relation above. This is a structural limit on part (c)-type searches at N = 15.

## Files
- `grammar.py` structures, enumeration, k accounting (`python grammar.py`); `grammar.json`.
- `filter.py` → `filter.json`, `run_filter.log`; `quarkpoint.py` → `quarkpoint.json`.
- `emu.py` gen/train/search → `emulator_train.json`, `search.json`, `run_emu.log` (draws and
  checkpoint in gitignored `artifacts/041/`; regenerate with `python emu.py gen 2000000 41 r1`,
  `python emu.py train 13 r1 45`, `python emu.py search 512`). The first training launch (30-epoch
  schedule) was stopped after epoch 0 and relaunched with 13 epochs (logged).
- `fit41.py` exact fits → `fits.jsonl` (DE seeds 1, 2), `fits_emu.jsonl` (emulator starts), `run.log`,
  `run_emufit.log`. `fit_qe.py` → `fits_qe.jsonl`, `run_qe.log`. `posthoc_clebsch.py` → `posthoc_clebsch.json`.
- `summarize.py` → `summary.json` (best fit per entry with pulls and predictions, per family, emulator stats).
All commands were run through `tools/pin.py` with ≤ 3 threads total (a few one-off print/JSON commands
early in the session were not; the overseer re-pinned them).

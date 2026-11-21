# Exp 047: skeptic audit of exp 044 survivors and of DE-only "excluded" fits

STATE.md iteration 16, next task 2. Skeptic role: look for errors, not confirmation. No experiment READMEs'
conclusions were read before the checks below; only their code and result files.

## (a) Texture A6842 vs arXiv:2110.05065

### Hypothesis under test
Exp 044 reports A6842 (Hermitian; Y_u nonzero at 11, 13, 23, 33; Y_d at 11, 12, 13, 33; fixed phases d12 = +π/2,
d13 = 0, diagonal signs +; 6 free moduli; (3,3) = 1 in both sectors) fitting Q8 (CKM s12, s13, s23, δ;
mr_u_c, mr_c_t, mr_d_s, mr_s_b; PDG 2026, 2% floor, M_Z) with χ² 0.47. Bagai et al., arXiv:2110.05065, say only
Fritzsch-like Hermitian four-zero textures are viable. One of the two must be wrong, or they must assume different
things.

### Method
- `textures.py`: own implementation from the texture definitions (entries transcribed by hand from
  `survivors.json`; parameter order u23, u13, u11, d13, d12, d11 as log10 moduli, then free phases). Own
  observable map: V = U_u† U_d from Y Y†, PDG angles, δ from the rephasing invariant
  V_us V_cb V_ub* V_cs* = s12 s23 s13 c13² (c12 c23 e^{iδ} - s12 s23 s13) (lib44 uses a cos δ / J formula
  instead). Cross-check against the official engine `src.yukawa.observables` at 800 random points: max relative
  deviation 1.6e-10. Score: official `src.evaluate.Scorer(Q8, "NO", mass_floor=0.02, scale=...)`.
- Refit: L-BFGS-B multistart (the exp 044 point, 20 perturbations σ = 0.3 dex, and 300 starts taken as the
  best 300 of 15,000 log-uniform box points, each pre-fitted with trust-region least squares on the residuals),
  Nelder-Mead polish of the best 5. Phases boxed in [-2π, 2π] (a first run with [-π, π] clipped A6624's stored
  phase 6.06 rad and looked like a failed reproduction; that was my bug, fixed).
- `scratch_fit.py`: fully independent from-scratch fits (no warm start): 400 random starts, mass-ratio-only
  least squares, then all residuals, then L-BFGS-B.
- `rerun044_fitter.py`: exp 044's own production fitter (lib44 `sector_solutions` + `fit_from_sectors`,
  n_start 8, iters 50, as in `main.py`) rerun with 20 seeds per survivor.
- `equivalence.py`: where A6842 sits in the paper's classification, and the diagonalising matrices at the
  best fit.
- Read the full paper (22 pages).

### Results

| texture | scale | χ² exp 044 | χ² recomputed (own code) | refit χ² (warm) | from scratch, 400 starts | exp 044 fitter, 20 seeds |
|---|---|---|---|---|---|---|
| A6842 | M_Z | 0.4681 | 0.4681 (pred. agree to 5e-13) | 0.4681, p 0.79, max pull 0.61 | best 6.58 (fit fails), 0/400 reach the basin | fit_ok 1/20, median χ² 318 |
| A6624 | GUT10 | 0.2025 | 0.2025 | 0.1975, p 0.91, max pull 0.21 | best 0.62 (fit_ok), 2/400 below 5 | fit_ok 11/20 |
| A47977 | M_Z | 0.6685 | 0.6685 | 0.6684, p 0.72, max pull 0.63 | best 0.77 (fit_ok), 22/400 below 5 | fit_ok 18/20 |

All three survivors are real: the best-fit points reproduce exactly with independent code and the official scorer,
and a local refit only lowers χ² slightly. A6842 with both d-sector phases free (k 8 = N) reaches χ² 1e-9 at
phases (-0.005, 1.581), i.e. the fit sits next to the fixed values (0, π/2).

### Why 2110.05065 excludes it
- Classification (`equivalence.json`). In the paper's Table 1 notation A6842's u pattern is S6 = class I_f and its
  d pattern is S3 = class I_c: both are Fritzsch-class matrices, but different permutations of it. No common
  permutation turns the pair into a same-structure (I_x I_x, "Fritzsch-like") pair. In Table 2 the combination
  I_f I_c is in the p_k = p2 row.
- The paper excludes every pair with different permutations (30 of 36 in class I-I, likewise in categories 2-4)
  analytically on p. 12, from the leading-order diagonalising matrices Eqs. (24)-(25), which have U^a ≈ 1. Those
  expansions assume the (3,3) element C of the I_a form is the heavy one, C ≈ m3. The exact Eq. (23) only needs
  m1 < C < m3. Only the I_aI_a, II_aII_a, I_aII_a, II_aI_a combinations get a numerical scan.
- At the A6842 best fit, the u sector mapped to the I_a form has C/m_t = 0.0115 and D/m_t = 0.985: the other
  branch, with the heavy entry on (2,2). Its U^a is close to a permutation (largest off-diagonal |U^a_ij| = 0.99),
  not to the identity. U_U^a† p2 U_D^a is then close to the identity and the CKM matrix has the usual form. The d
  sector is on the paper's branch (C/m_b = 0.97).
- So the paper's "only Fritzsch-like is viable" holds only on the C ≈ m3 branch of each permuted texture. It is an
  approximation artifact, not a data exclusion. Their inputs (PDG 2020, Huang-Zhou masses at M_Z with 7-17% light
  quark errors) are looser than ours (our same-scale ratio errors with the 2% floor: mr_u_c 3.9%, mr_c_t 2.5%,
  mr_d_s 2.6%, mr_s_b 2.2%), so data differences cannot explain it. A6842 fits PDG 2026 at M_Z with
  tighter errors.
- Is A6842 equivalent to Fritzsch I_aI_a? Not by a common permutation (checked). Relabelled so both heavy entries
  sit on (3,3), A6842 is (u: zeros at 11, 12; d: zeros at 11, 13 = Fritzsch). The light u quark mixes with the
  third generation, not the second. A general unitary weak-basis map between the two was not checked. By Branco et
  al. (hep-ph/9911418) three zeros are always reachable and the fourth is one physical constraint, which differs
  between the two textures in general. So I'd call A6842 "a Fritzsch-class texture on the branch 2110.05065
  dropped", which fits the literature-check wording in grammar.md: a known matrix class, not a new one.

### Verdict (a)
Exp 044 is right that A6842 fits. 2110.05065's claim is wrong for this pair: it rests on an assumption about
which diagonal entry is heavy, applied when excluding all permuted combinations analytically. Two
qualifications for STATE.md:
1. Exp 044's fitter finds A6842 in 1 of 20 seeds (A6624 11/20, A47977 18/20), and my generic multistart never
   reaches A6842's basin from scratch in 400 starts. Exp 044 ran each of the 64,438 sub-grammar-A entries once
   with this fitter (main.py, n_start 8). The survivor list is therefore a lower bound: other entries were
   probably missed, so a "fraction of G44 that fits" or "only these survive" reading is search-limited. This does
   not affect the survivors themselves.
2. A6842 is a permuted Fritzsch-class pair, not a structurally new texture. It belongs in the literature-check
   column as "Fritzsch class I_f/I_c, branch not scanned by 2110.05065".

## (b) Optimizer audit of DE-only "excluded" fits

### Method
`refit_lbfgsb.py`: for each fit, the experiment's own model function, bounds, periodic list, observables, inputs,
floor, scale and k (imported from `fit_more.py`, `fit.py`, `fit42.py`), through `src.evaluate.fit_and_evaluate`
with `method="lbfgsb"`. Starts: 50 scrambled-Sobol points over the box (seed 4702), every stored DE optimum, and 16
Gaussian perturbations of the best DE point (σ = 5% or 2% of each box width); L-BFGS-B maxfun 300·dim; the best
start gets the Nelder-Mead polish. Exp 042 results are rescored with fit42.py's honest k (k_cont + extra + bits +
ordering bit). Stored DE χ² recomputed with the current code first: all reproduce to 1e-3 or better.

### Results

| fit | N | k | old χ² (DE) | new χ² (L-BFGS-B multistart) | best Sobol start | max pull (new) | fit_ok | compression_ok | verdict change |
|---|---|---|---|---|---|---|---|---|---|
| 040 Albright-Barr, tanβ 5 | 16 | 13 | 246.68 | 246.68 | 283.2 | 10.9 (mr_b_tau) | no | yes | none |
| 040 Albright-Barr, tanβ 30 | 16 | 13 | 269.93 | 269.93 | 472.4 | 11.1 (mr_b_tau) | no | yes | none |
| 040 UTZ L.O., tanβ 30 | 15 | 14 | 1395.00 | 1395.00 | 1418.3 | 24.9 (ckm_s23) | no | no | none |
| 040 UTZ H.O., tanβ 30 | 15 | 16 | 385.59 | 385.59 | 636.4 | 8.1 (ckm_s23) | no | no | none |
| 038 yukawaon full (floor 2%) | 13 | 10 | 1027.05 (s2), 1027.18 (s1) | 1027.05 | 26501 | 22.9 (mr_u_c) | no | yes | none |
| 042 AKS NO | 16 | 14 | 160.96 | 160.96 | 506.5 | 7.5 (s23sq) | no | yes | none |
| 042 CM NO | 16 | 9 | 764.24 (s1), 764.30 (s3), 1271.48 (s2) | 729.74 | 729.74 | 15.2 (ckm_delta_rad) | no | yes | none (DE was 34.5 high) |
| 042 CM IO | 16 | 9 | 855.62 | 855.62 | 1183.9 | 15.2 (ckm_delta_rad) | no | yes | none |

### Verdict (b)
No verdict changes. In every case the best point is at or below the DE value, and the best fresh (Sobol) L-BFGS-B
start is no better than DE except for CM NO, where a Sobol start found 729.7 against DE's 764.2 (one of three DE
seeds stopped at 1271.5). All χ² stay above 160 with |pulls| ≥ 7.5, far outside any optimizer slack: the "excluded
(fit quality)" verdicts for AB and UTZ now rest on DE plus an independent 67-69-start L-BFGS-B multistart that
agree. That is still a search, not a parameter-independent bound: a narrow basin missed by both cannot be ruled
out (part (a) shows such basins exist, see A6842). Given the χ² sizes this is unlikely to matter.

## Free parameters
This audit adds no model and fits nothing new to data beyond the existing models; k values are those of the
audited experiments.

## Files
- `textures.py` → `textures.log`, `textures.json` (independent implementation, claimed-point check, refits)
- `scratch_fit.py` → `scratch_fit.log`, `scratch_fit.json` (from-scratch fits)
- `rerun044_fitter.py` → `rerun044_fitter.log`, `rerun044_fitter.json` (exp 044 fitter repeatability)
- `equivalence.py` → `equivalence.log`, `equivalence.json` (classification, branch diagnostics)
- `refit_lbfgsb.py`, `run_refits.sh` → `refit_<config>.json`, `run.log` (part b; AB30 wall time 53,000 s includes an overnight machine sleep; two streams appended to the same
  log, so some lines are interleaved; the JSON files are authoritative)
- Rerun: `.venv/Scripts/python tools/pin.py .venv/Scripts/python experiments/20261120-047-skeptic-audit/<script>`;
  `sh experiments/20261120-047-skeptic-audit/run_refits.sh AB5 AB30 UTZ UTZHO YK AKS CMNO CMIO`.

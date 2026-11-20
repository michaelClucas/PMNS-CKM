# Exp 044: predictive quark textures (k_cont <= 6 for the 8 quark observables)

STATE "Next planned tasks" items 1-2 (the exp 043 loophole).

## Hypothesis
Exp 043: every quark sector used so far spends >= 9 parameters on the 8 quark observables Q8 (CKM s12, s13,
s23, delta; m_u/m_c, m_c/m_t, m_d/m_s, m_s/m_b), so nothing is left to predict leptons. If some quark
texture with <= 6 continuous parameters fits Q8 at PDG 2026 precision with the 2% floor, it predicts >= 2 of
its own observables and could carry a compressive quark-lepton model.

## Method
- Pre-registration `grammar.md`, SHA-256 `920108fe…f30d9a` logged in `run.log` (21:16 UTC) before any fit.
- Grammar G44 (`lib44.py`): Y_u, Y_d both Hermitian (H) or both complex symmetric (S); zero pattern per
  sector = (3,3) + any subset of {11, 12, 13, 22, 23}; (3,3) = 1 (ratios only). Canonicalisation: pattern
  pairs identified under the simultaneous 1 <-> 2 swap; weak-basis rephasing removed by greedy gauge fixing
  (H: P Y P^dag; S: P Y P plus one overall phase per sector); every remaining phase slot is free or fixed in
  {0, pi/2, pi, 2pi/3}, H diagonals carry a sign, and a global CP flag negates the fixed phases.
  Sub-grammar B adds exactly one tie |Y_q| = r |Y_p| (within a sector, to (3,3), or u-d at the same
  position), r in {1, 2, 3, 1/2, 1/3}. k_cont = free moduli + free phases <= 6.
  General complex textures are not fitted: with <= 4 entries per sector 0/965 canonical pairs are
  structurally viable (some |V_ij| = 0), and with >= 5 entries k_cont >= 8 without ties.
- Grammar size: 242 (H) and 243 (S) structurally viable pattern pairs of 544. A (no ties) = 64,438 entries
  (16.0 bits); B (one tie) = 20,337,940; G44 = 20,402,378 (24.3 bits). Upper bounds: residual discrete
  equivalences are not removed in the count; survivors are deduplicated by a numeric fingerprint.
- Filters: (1) CP: J = 0 at 64 random points; (2) per-sector mass reachability (marginal 2x2 mass chi^2 is
  a lower bound on the total chi^2; kill if above chi2_0.95(8 - k_cont) at every scale).
- Fits: official scorer `src.evaluate.Scorer(Q8, "NO", mass_floor=0.02, scale=...)` at MZ, GUT10
  (MSSM tan beta 10, mu 2e16) and GUT30 (tan beta 30); no eta thresholds. Fast path: per-sector mass
  solutions as starts, 8-start batched Levenberg-Marquardt on a vectorised proxy whose residuals equal the
  official ones, official score at the optimum, Nelder-Mead polish on the official chi^2 when max |pull| < 3.
  Thorough refit (`fit_thorough`, 64 starts + scipy least_squares + polish) of every fit with chi^2 < 20.
  Pass = official fit_ok (max |pull| < 2 and p > 0.05, dof = 8 - k_cont).
- A fitted exhaustively at MZ (all 58,477 filter survivors). B: uniform random sample of 3000 entries
  (seed 4401, drawn with replacement), MZ only.
- Lepton step (pre-registered in grammar.md): Y_e = C o Y_d with Georgi-Jarlskog C (C_22 = -3), Y_nu = Y_u,
  type I with M_R = 1 or M_R = diag(1, x, y); N 15 fit and held-out PMNS test, at each survivor's best scale.
- Scripts: `main.py` (enum, filter, fit), `validate.py`, `gut_subset.py`, `summarize.py`, `posthoc.py`,
  `lepton.py`. Bulk records in gitignored `artifacts/044/`.

Deviations (logged):
- GUT10/GUT30 fits ran only on A entries with MZ chi^2 < 200 (901) plus a random check sample of 600 others,
  not on all filter survivors (wall time). Check sample: 0/600 pass at GUT10 (min chi^2 95.2), 0/582 at GUT30
  (min 32.6). B was fitted at MZ only.
- One early python command (a text edit of `lib44.py`) ran without `tools/pin.py`; all compute ran pinned
  with <= 3 processes.

## Validation
- Proxy vs official chi^2 at the same point: max relative difference 6.8e-11 over 237 points (`posthoc.json`).
- Fast fitter is not globally reliable: 150 random A failures refitted thoroughly: 53 improved chi^2 by > 10%,
  median ratio 0.98, 0 became passes (min thorough chi^2 36.1). Near refits (chi^2 < 20): MZ 8/95 pass,
  GUT10 17/73, GUT30 12/69, B 1/2; the survivor list is the union of fast and thorough passes.
- Filters killed little: CP 5,849/64,438 (9.1%); sector filter 0 further entries at all three scales
  (MZ alone 112, GUT30 1,608). In B: CP 132/3000, alive at some scale 1,776/3000 (fixed ties to (3,3) fail masses).

## Result (`summary.json`, `posthoc.json`, `survivors.json`)

| | fitted | pass | pass fraction (CP95) |
|---|---|---|---|
| A, MZ | 58,477 | 9 fast (+ thorough) | 1.5e-4 [7e-5, 2.9e-4] (fast) |
| A, GUT10 (subset) | 1,501 | 16 fast, 17 thorough | |
| A, GUT10/30 check of MZ chi^2 >= 200 | 600 / 582 | 0 / 0 | |
| A, GUT30 (subset) | 1,463 | 11 fast, 12 thorough | |
| B sample, MZ | 1,752 of 3000 (rest killed by filters) | 1 | 1/3000 of the sample |

20 entries pass at some scale (19 after fingerprint deduplication: A1197 = A1199, a u11 sign flip). Every
survivor has k_cont = 6: none with k_cont <= 5 passes. Each predicts 2 of the 8 quark observables.

| id | type: u pattern, d pattern (non-33 entries) | fixed phases / ties | scales passing | best chi^2 | p (dof 2) | max pull |
|---|---|---|---|---|---|---|
| A6624 | H: 11 13 23, 12 13 | d12 -pi/2, d13 free, u11 + | GUT10 | 0.20 | 0.90 | 0.24 |
| A6842 | H: 11 13 23, 11 12 13 | d12 +pi/2, d13 0, signs + + | MZ | 0.47 | 0.79 | 0.61 |
| A47977 | S: 11 13 23, 11 12 23 | d11 pi, d12 -pi/2, d23 -2pi/3, u11 pi | MZ, GUT10, GUT30 | 0.67 | 0.72 | 0.63 |
| A59097 | S: 11 12 13 23, 11 12 | d11 pi, d12 -2pi/3, u11 -2pi/3, u12 0 | MZ, GUT10, GUT30 | 0.71 | 0.70 | 0.65 |
| A11250 | H: 11 13 22 23, 12 13 | d12 +pi/2, d13 0, u11 +, u22 + | MZ, GUT10 | 0.99 | 0.61 | 0.83 |
| A47963 | S: 11 13 23, 11 12 23 | d11 -pi/2, d12 0, d23 -2pi/3, u11 pi | MZ, GUT10, GUT30 | 1.03 | 0.60 | 0.70 |
| A7096 | H: 11 13 23, 11 12 23 | d12 -pi/2, d23 -2pi/3, u11 - | MZ, GUT10 | 1.22 | 0.54 | 0.71 |
| A52778 | S: 12 13 23, 11 12 13 | d11 +pi/2, d12 0, d13 0, u12 +pi/2 | GUT10, GUT30 | 1.92 | 0.38 | 1.04 |
| A48143 | S: 11 13 23, 11 12 23 | d11 0, d12 -2pi/3, d23 0, u11 -2pi/3 | MZ, GUT30 | 1.93 | 0.38 | 1.25 |
| A29333 | S: 11 12 13, 11 12 23 | d11 pi, d12 -2pi/3, d23 pi, u11 -2pi/3 | MZ, GUT10 | 2.78 | 0.25 | 1.31 |
| A20058 | S: 11 23, 11 12 13 23 | d11 pi, d12 -2pi/3, d13 -2pi/3, d23 -pi/2 | GUT10, GUT30 | 3.71 | 0.16 | 1.86 |
| A1197/A1199 | H: 11 23, 11 12 13 23 | d11 -, d12 -pi/2, d23 -2pi/3 | GUT10, GUT30 | 3.79 | 0.15 | 1.88 |
| A20128 | S: 11 23, 11 12 13 23 | d11 0, d12 0, d13 -pi/2, d23 -2pi/3 | GUT10, GUT30 | 4.06 | 0.13 | 1.95 |
| A20114 | S: 11 23, 11 12 13 23 | d11 pi, d12 -pi/2, d13 pi, d23 -2pi/3 | GUT10, GUT30 | 4.06 | 0.13 | 1.95 |
| A46880 | S: 11 13 23, 11 12 13 | d11 +pi/2, d12 0, d13 0, u11 pi | MZ, GUT10, GUT30 | 4.39 | 0.11 | 1.66 |
| A47857 | S: 11 13 23, 11 12 23 | d11 pi, d12 -pi/2, d23 -2pi/3, u11 0 | GUT10 | 4.94 | 0.085 | 1.48 |
| A47843 | S: 11 13 23, 11 12 23 | d11 -pi/2, d12 0, d23 -2pi/3, u11 0 | GUT10 | 4.94 | 0.085 | 1.57 |
| A7094 | H: 11 13 23, 11 12 23 | d12 -pi/2, d23 -2pi/3, u11 + | GUT10, GUT30 | 4.98 | 0.083 | 1.57 |
| B41 | H: 11 12 13, 11 12 13 23 | tie |d11| = 3 |d23|; d12 +pi/2, d13 +pi/2, d23 pi, d11 - | MZ | 5.87 | 0.053 | 1.72 |

Best-fit points, all pulls and predictions per scale: `posthoc.json` ("survivors") and `survivors.json`
(parameter order: free log10 moduli in `lib44.layout` entry order, then free phases). Example A6842 (MZ):
pulls s12 -0.05, s13 -0.16, s23 0.05, delta 0.61, m_u/m_c 0.15, m_c/m_t 0.05, m_d/m_s 0.17, m_s/m_b 0.09.

Parameter counting. k_cont = 6, N = 8 for every survivor, so 2 quark observables are predicted. Honest k
includes the data-selected texture and scale: k = 6 + 16.0 (A) + 2 = 24 for the A survivors and
6 + 24.3 + 2 = 32 if G44 is the search space. As a quark-only claim, compression fails by a wide margin
(k >> N - 2 = 6). Survivor rate in A is ~3e-4 per entry, so with 2^16 entries ~ 20 passes is what the trials
factor allows by itself; no null was run (not authorised for this task).

## Literature check (survivor classes)
- Survivors are all four- or five-zero non-parallel textures. A6842 is, after the 1 <-> 2 relabelling,
  u = {13, 22, 23}, d = {12, 22, 23}: a Fritzsch-like d with a non-Fritzsch-like u ("non-parallel texture
  4-zero", Ahuja-Gupta et al.). Bagai, Vashisht, Awasthi, Ahuja, Gupta (arXiv:2110.05065) report that only
  Fritzsch-like texture 4-zero Hermitian combinations remain viable; A6842 (non-parallel 4-zero) and the other H survivors
  pass here at PDG 2026 with the 2% floor and fixed (pi/2) phases. The difference may come from their mass
  inputs or criteria; not resolved here, flagged for the skeptic.
- A6624 relabels to u = {13, 22, 23}, d = {12, 23} (Fritzsch-type d): a five-zero non-parallel texture of the
  Ramond-Roberts-Ross type with one free and one maximal (-pi/2) phase. Not checked line by line against
  RRR's five solutions (hep-ph/9303320); five-zero Hermitian textures were also studied in arXiv:1303.1338.
- The complex-symmetric survivors (S, phases 2pi/3 and pi/2 on diagonal and off-diagonal entries) are zero
  patterns covered by Ludl-Grimus's complete survey of quark texture zeros (arXiv:1406.3546); the specific
  fixed-phase choices were not found in a quick search. Fritzsch six-zero (both sectors {12, 23}) and GST-only
  structures do not survive, as known.
- Verdict per class: zero patterns known; fixed-phase specialisations are data-selected out of 2^16 and are
  baselines at most.

## Lepton extension (`lepton.json`, `posthoc.json` "lepton")
Every one of the 19 survivor classes was extended with the pre-registered Georgi-Jarlskog Y_e and Y_nu = Y_u
(M_R = 1, and M_R = diag(1, x, y)), at its best scale (local fit from the quark point, 6 starts).
- N 15 fit: official chi^2 = inf for all 38 fits (PMNS outside the NuFIT tables); Gaussian fallback
  >= 8192. Typical predictions: s12^2 0.01-0.07, s13^2 <= 2e-4, s23^2 <= 0.006, dm_ratio ~ 0
  (M_R = 1 gives m_nu ~ Y_u Y_u^T, so dm_ratio ~ (m_c/m_t)^4).
- The charged leptons fail before PMNS: the held-out stage fit (Q8 + m_e/m_mu + m_mu/m_tau, N 10) has
  chi^2 >= 2901 for every survivor (GJ gives m_mu/m_tau ~ 0.008-0.03 against 0.0589).
- Held-out PMNS: not scoreable (outside tables) for every survivor.
- M_R = 1 is excluded structurally (dm_ratio). For M_R = diag(1, x, y) the search was local (6 starts), so
  "no fit found", but the charged-lepton failure (chi^2 >= 2901, independent of M_R) already excludes every
  extension under the pre-registered GJ Clebsch.

## Verdict
- Quark sector: 19 distinct textures with k_cont = 6 fit Q8 at PDG 2026 with the 2% floor (best chi^2 0.20,
  p 0.90); none with k_cont <= 5. So quark sectors that predict 2 of their own 8 observables exist. They are
  non-parallel four/five-zero Hermitian or complex-symmetric textures with phases fixed to pi/2 or 2pi/3.
  None is a candidate by itself: the selection costs 16-24 bits (k >= 24 vs N 8), and the zero patterns are
  in the literature. A6842-type non-Fritzsch-like 4-zero passes conflict with arXiv:2110.05065 and should be
  checked by the skeptic.
- Quark-lepton extension (GJ, Y_nu = Y_u): every survivor fails fit quality by > 50 sigma at the charged
  lepton stage (N 10 chi^2 >= 2901) and none enters the NuFIT tables: excluded under the pre-registered
  Clebsch and neutrino choices. Exp 043's loophole 1 is open at the quark level only; with 16+ bits spent on
  choosing the texture, a compressive N 15 model would need k_cont + lepton parameters <= 13 - 18 < 0, so this
  grammar cannot give a compressive quark-lepton model unless the texture is fixed by theory in advance.

Not done: B fitted at GUT scales; full GUT fits of A (subset + check sample instead); nulls; a broader
lepton Clebsch scan (would cost more bits).

## Files
`grammar.md`, `run.log`, `lib44.py`, `main.py`, `validate.py`, `gut_subset.py`, `summarize.py` ->
`summary.json`, `survivors.json`, `summarize.log`; `posthoc.py` -> `posthoc.json`, `posthoc.log`;
`lepton.py` -> `lepton.json`. Gitignored `artifacts/044/`: `A.jsonl`, `B_sample.jsonl`, `filter_*.jsonl`,
`sectors_*.json`, `fits_*.jsonl`, `thorough_*.jsonl`, `lepton_*.json`. Regenerate from this folder with
`../../.venv/Scripts/python ../../tools/pin.py ../../.venv/Scripts/python main.py enum`, then
`main.py filter A 3`, `main.py fit A MZ 3` (~57 min), `gut_subset.py A 200 600`, `main.py fit A GUT10 3
gut_ids_A.json` (same for GUT30), `main.py filter B 3`, `main.py fit B MZ 3`, `validate.py near A MZ 20 3`
(and GUT10, GUT30, B), `validate.py miss A MZ 150 3`, `summarize.py`, `lepton.py 19,4,5,37,8,21 a` (and b, c
index lists as in run.log), `posthoc.py`.

# Exp 044 pre-registration: grammar G44 of predictive quark textures

Written before any fit of a G44 texture. SHA-256 logged in `run.log`. Code: `lib44.py`.

## Target
The 8 quark observables Q8 = (ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, mr_u_c, mr_c_t, mr_d_s, mr_s_b),
official scorer `src.evaluate.Scorer(Q8, "NO", mass_floor=0.02, scale=...)`, PDG 2026. Sector scales are not
fitted (ratios only): the (3,3) entry of Y_u and of Y_d is 1.

## Grammar
- Type t in {H (Hermitian), S (complex symmetric)}, the same for Y_u and Y_d. LR convention, observables
  from Y Y^dag (`src.yukawa`).
- Zero pattern per sector: (3,3) plus any subset of {11, 12, 13, 22, 23} (symmetric positions).
- Pattern pairs are identified under the simultaneous 1 <-> 2 index swap (the only weak-basis permutation
  that keeps (3,3)); canonical representative = lexicographic minimum.
- Structural viability (no data): each sector generically rank 3 with distinct singular values, and the
  generic CKM has all nine |V_ij| > 0 (two random complex points).
- Moduli: every nonzero non-(3,3) entry has a free modulus, except an entry fixed by a tie.
- Ties (sub-grammar B only, exactly one tie): within a sector |Y_q| = r |Y_p| (q after p in the canonical
  entry order 33, 23, 22, 13, 12, 11) or |Y_q| = r (tie to the (3,3) entry); across sectors
  |Y_d,P| = r |Y_u,P| at the same position P != 33. r in {1, 2, 3, 1/2, 1/3}.
- Phases. Canonicalisation (weak-basis rephasing): H: Y -> P Y P^dag with common diagonal P; S: Y -> P Y P
  with common P plus a free overall phase per sector. Greedy gauge fixing in canonical entry order
  (u entries first): an entry whose rephasing vector is independent of those already chosen gets phase 0.
  Every remaining entry is a phase slot: free, or fixed to a value in {0, pi/2, pi, 2pi/3}. H diagonal
  entries are real: a sign slot, fixed to + or - (2 options; (3,3) is +). A global CP flag negates every
  fixed phase; it doubles an entry only if some slot holds pi/2 or 2pi/3 (otherwise the conjugate is the
  same entry).
- k_cont = free moduli + free phases <= 6.
- Sizes (computed by `lib44`, no data): H 242 viable pattern pairs (of 544), S 243 (of 544).
  Sub-grammar A (no ties): H 11,354 + S 53,084 = 64,438 entries (16.0 bits).
  Sub-grammar B (one tie): H 2,724,435 + S 17,613,505 = 20,337,940 entries.
  G44 = A + B = 20,402,378 entries (24.3 bits). These are upper bounds on distinct models: residual
  discrete equivalences (sign flips P = diag(+-1), shifts by pi) are not removed in the count; survivors are
  deduplicated by a numeric fingerprint (observables at 3 shared random parameter points, rounded 1e-8).
- General complex (G) textures: with <= 4 nonzero entries per sector no pair is structurally viable
  (0/965 canonical pairs: one mixing plane per sector forces some |V_ij| = 0); with >= 5 entries per sector
  and no ties k_cont >= 8. G is therefore not fitted. Y Y^dag of any general Y equals that of a Hermitian
  matrix, so G adds no new observable maps beyond Hermitian ones with other zero sets.

## Scales (data-selected choice, counted: 3 options = 2 bits)
MZ (no running), GUT10 = {mu 2e16, MSSM, tan beta 10}, GUT30 = {mu 2e16, MSSM, tan beta 30}; no eta thresholds.

## Filters (applied in this order; kill fractions reported)
1. CP: an entry is killed if J = 0 at 64 random points (e.g. all slots real and no free phase).
2. Sector mass reachability: for each sector alone, minimise the 2-ratio mass chi^2 (marginal 2x2 block of
   the official mass covariance with the floor, at the given scale) over that sector's free moduli and
   phases (multi-start least squares). The marginal chi^2 is a lower bound on the total chi^2, so an entry
   is killed at a scale if either sector's minimum exceeds chi2_0.95(dof = 8 - k_cont). An entry is killed
   if it is killed at all three scales.
3. Full fit (survivors, each scale where step 2 passed): 2048 log-uniform random points in the box
   (moduli log10 in [-6, 1], phases in [-pi, pi]), least squares (proxy = official residuals, vectorised)
   from the best 6, official scorer at the best point; Nelder-Mead polish on the official chi^2 if
   max |pull| < 3 and not passed. The proxy is validated against the official scorer on every reported fit.
Sub-grammar A is fitted exhaustively. Sub-grammar B: a uniform random sample of 3000 entries (seed 4401),
same pipeline; its pass fraction is reported with a Clopper-Pearson interval.

## Pass rule and reporting
Pass at a scale: official chi^2 finite, max |pull| < 2, p > 0.05 with dof = 8 - k_cont (fit_ok).
Report every passing entry (best scale), its best-fit point, pulls, k_cont, the number of quark
observables it predicts (8 - k_cont), and honest k = k_cont + bits (grammar bits + 2 scale bits).
Literature check of each survivor class (Fritzsch, GST, Ramond-Roberts-Ross, Branco et al., Xing,
Roberts-Romanino-Ross-Velasco-Sevilla): known ones are baselines.

## Lepton extension (only if a survivor exists), fixed now
- Y_e = C o Y_d with the Georgi-Jarlskog pattern: C = 1 everywhere except C_22 = -3 (chosen before fitting).
- Neutrinos: type-I with Y_nu = Y_u and (i) M_R = 1 (0 parameters) or (ii) M_R = diag(1, x, y), x, y > 0
  (2 parameters). Both are tried (1 bit).
- Fit N 15 = Q8 + (mr_e_mu, mr_mu_tau) + (s12sq, s13sq, s23sq, dcp_deg) + dm_ratio, NO, default NuFIT
  analysis, 2% floor, official scorer, at the survivor's scale. Held-out test: fit Q8 + 2 charged-lepton
  ratios, predict the 5 PMNS-side observables, score them.
Wording: "candidate" only; "excluded" only for fit-quality failure.

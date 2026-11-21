# Exp 046: lepton extensions of the exp 045 quark survivors (fixed before any lepton fit)

Conventions: LR Yukawas as in `src.yukawa` (rows = doublet index), V_CKM = U_u† U_d, U_PMNS = U_e† U_ν,
type-I seesaw m_ν = −Y_ν* M_R⁻¹ Y_ν† (overall scale unphysical here: only dm_ratio is scored).
Quark matrices are exactly those of exp 045 (`structures.py`, `models.md`), with the same pinned phases.

Scored observables, all three models: N = 15 =
Q8 (ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, mr_u_c, mr_c_t, mr_d_s, mr_s_b)
+ charged-lepton ratios (mr_e_mu, mr_mu_tau)
+ PMNS (s12sq, s13sq, s23sq, dcp_deg) + dm_ratio.
Official scorer `src.evaluate` (NuFIT 6.1 default analysis IC23_woSKatm, PDG 2026, mass_floor 0.02),
ordering NO, declared here before fitting (IO is not fitted; using it would add 1 bit).
The predictions are made at the model scale and translated with `scale=` (stated per model).
mr_b_tau (b-τ unification) is not among the 15 and is not scored; each model's Y_e normalisation is free.

## Neutrino options (same for every model)
- ν1: M_R = 1 (0 parameters).
- ν2: M_R = diag(1, x, y), x, y real positive (2 parameters, fitted as ln x, ln y ∈ [−35, 35]).
- Dirac neutrino Yukawa: Y_ν = Y_u or Y_ν = Y_d (the quark matrices as written in the LR basis of the fit,
  same phases), unless a model's assignment fixes it (below).

## RRR5 (Ramond-Roberts-Ross hep-ph/9303320 Table 1 solution 5)
Y_u = [[0,0,C],[0,B,B'],[C*,B'*,A]] (Hermitian), Y_d = [[0,F,0],[F*,E,0],[0,0,D]] (Hermitian), exp 045 pins.
Y_e = Georgi-Jarlskog, RRR's pattern: Y_e = Y_d with the (2,2) entry multiplied by −3:
      Y_e = [[0,F,0],[F*,−3E,0],[0,0,D]]. 0 new texture parameters.
Scale: GUT 2e16 GeV, MSSM, tan β 30, M_SUSY 3 TeV (exp 045's surviving option), η̄_q = 0,
η̄_b fitted (exp 045), η̄_ℓ fitted (always included, so "if needed" costs no bit).
Y_ν ∈ {Y_u, Y_d}; M_R ∈ {ν1, ν2}.
Count: k_cont = 6 (texture) + η̄_b + η̄_ℓ + {0, 2} = 8 or 10.
Bits: tan β choice 1 (exp 045) + 4 neutrino options 2 = 3.  k = 11 (ν1) or 13 (ν2); N − 2 = 13.
η̄_b, η̄_ℓ are fitted on a grid with local refits of the other parameters (each η value needs its own
scorer). Parameter-independent check: GJ fixes m_e m_μ = m_d m_s and m_μ − m_e = 3 (m_s − m_d) in the 2×2
block, so R = (m_e/m_μ)/(m_d/m_s) is a function of m_d/m_s alone; R is RG- and threshold-invariant for
the first two generations at leading order, so the (mr_d_s, mr_e_mu) block gives a bound for all η.

## BB (Belfatto-Berezhiani 2305.00069, x_u = 1)
Y_u = Fritzsch Hermitian with 2 phases, Y_d = [[0,A,0],[A,0,B x_d],[0,B/x_d,C]] real (exp 045 form).
Charged leptons (paper Sec. 2, after Eq. 2.14: "the Yukawa matrix of charged leptons will present
analogous form, with asymmetry parameter x_e, while the degeneracy between the down quark and lepton
masses is removed by the contribution of the VEV [of the 24-plet] which breaks SU(5)"):
      Y_e = [[0,A,0],[A,0,B x_e],[0,B/x_e,C]] × (free overall factor),
same A, B, C as Y_d, real, same basis as Y_d. New parameter: x_e (fitted as ln x_e ∈ [−6, 6]).
SU(5)-breaking term, stated explicitly: the 24-plet VEV Φ = v_24 diag(2,2,2,−3,−3) enters the heavy
vector-like fermion mass (g_D S + f_D Σ + h_D Φ) F F̄ (Eq. 2.14), so the doublet (L) and triplet (D) members
of F get different octet-to-singlet ratios X_L ≠ X_D and different overall masses M_L ≠ M_D. Through
Y = λ M⁻¹ y (Eq. 2.11) this changes only x (x_e ≠ x_d) and the sector normalisation (unscored). No other
SU(5)-breaking contribution (e.g. the 10-plet term y_d T F̄ H, paper p. 9) is included; that would add
parameters. Transposition (SU(5) Y_e ~ Y_d^T) maps the form to itself with x → 1/x, so it is absorbed by
the free x_e.
Scale: SM, μ ∈ {1e3, 1e6, 1e16} GeV (the paper's three benchmarks, 2 bits as in exp 045).
Y_ν ∈ {Y_u, Y_d}; M_R ∈ {ν1, ν2}.
Count: k_cont = 7 + x_e + {0, 2} = 8 or 10. Bits: scale 2 + neutrino 2 = 4. k = 12 (ν1) or 14 (ν2).
So BB-ν2 fails compression by count (14 > 13) even before fitting; it is fitted to report fit quality.

## HS (Harrison-Scott 2501.18508)
Quark matrices: Eq. (2.1)-(2.3) as in exp 045 (5 parameters + fitted SM scale).
Lepton sector in the paper: none (Sec. 5 has only a remark on neutrino masses). The 7 papers citing it on
INSPIRE (checked 2026-11-20) contain no lepton extension by the authors or others.
Two assignments, as the brief specifies:
- HS-SU5 (minimal SU(5)): Y_e = Y_d^T; Y_ν ∈ {Y_u, Y_d}; M_R ∈ {ν1, ν2}.
- HS-PS (Pati-Salam): Y_e = Y_d, Y_ν = Y_u; M_R ∈ {ν1, ν2}.
Scale: SM, μ fitted on exp 045's grid {1e5, 1e7, 1e9} GeV (counted as 1 continuous, as in exp 045).
Count: k_cont = 6 + {0, 2} = 6 or 8. Bits: (2 + 1) × 2 = 6 options = 3. k = 9 (ν1) or 11 (ν2).
Parameter-independent check: both assignments give charged-lepton ratios equal to the down ratios at μ
(singular values of Y_d^T, Y_d are those of Y_d), i.e. R = 1 and m_μ/m_τ = m_s/m_b at μ.

## Options and bits (all listed options count)
| model | options | bits | k_cont (ν1 / ν2) | k (ν1 / ν2) | N − 2 |
|---|---|---|---|---|---|
| RRR5 | tan β (2) × Y_ν (2) × M_R (2) | 3 | 8 / 10 | 11 / 13 | 13 |
| BB | scale (3) × Y_ν (2) × M_R (2) | 4 | 8 / 10 | 12 / 14 | 13 |
| HS | {SU5 × Y_ν (2), PS} × M_R (2) | 3 | 6 / 8 | 9 / 11 | 13 |
Not counted: the selection of these 3 structures out of exp 045's 23 (and of exp 045 itself from the
program's search history); the choice of NO.

## Tests
1. Parameter-independent bounds first (RRR5: GJ R; HS: R = 1; ν1: dm_ratio = function of quark or
   lepton mass ratios only).
2. Full N 15 fits: `src.evaluate.fit_and_evaluate(method="lbfgsb")` multistart (Sobol + warm starts from
   exp 045's quark optimum with random neutrino parameters), 3 rounds with different seeds; minimum over
   rounds reported, rounds must agree to Δχ² < 1 (or the spread is reported).
3. Held-out (only for options with fit_ok): fit Q8 + mr_e_mu + mr_mu_tau + dm_ratio, predict the 4 PMNS
   parameters at that optimum (ν2's x, y are then fixed by dm_ratio and the quark/charged-lepton fit only
   through the profile; reported as the PMNS χ² at the N 11 optimum).

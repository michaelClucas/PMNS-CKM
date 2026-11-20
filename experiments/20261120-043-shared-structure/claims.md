# Exp 043 pre-registration: structural claims about shared quark-lepton Yukawa structures

Written before any structure in step 3 is fitted. Its SHA-256 is logged in `run.log` before the
first fit. Only `probe.py` (scorer timing and data values, no fit) ran before this file.

## Generator G43 (theory space; no measured number enters it)

All four Dirac-type matrices are complex symmetric (SO(10)/Pati-Salam-like: left and right fields in
one multiplet) and share a zero pattern Z, a subset of P = {11, 12, 13, 22, 23}; entry 33 is always
present and set to 1 in Y_u and Y_d (ratios only; sector scales are not observables here).
Patterns: all Z with |Z| = 3 or 4 (10 + 5 = 15 patterns).

- Quark parameters (always free, never shared between u and d; exp 041 showed a short rational list
  cannot tie u to d): Y_u[ij] = Y_u[ji] = u_ij real, Y_d[ij] = Y_d[ji] = d_ij exp(i phi_ij), ij in Z.
  3|Z| real parameters (u: signed log; d: log modulus + phase). Declared count; may include
  one unphysical phase.
- Clebsch set CSET = {±1/2, ±1, ±3/2, ±2, ±3, ±9/2, ±6} (14 values, the exp 041 post-hoc list).
- Charged leptons, Y_e[ij] = c^e_ij Y_d[ij] for ij in Z ∪ {33}:
  - S_e: every c^e in CSET (fixed; no lepton-only continuous parameter; l_e = 0).
  - P_e^m: m of the |Z|+1 Clebsches continuous (signed log, |c| in [0.01, 10]), the rest in CSET
    (l_e = m; m = 1, 2; positions drawn uniformly).
  - F_e: all c^e continuous, c^e_33 = 1 (l_e = |Z|).
- Neutrinos, built from the quark matrices with Clebsch matrices C in CSET (l_nu = 0):
  - D:  Dirac, Y_nu = C∘Y_u.
  - I1: type I, Y_nu = C∘Y_u, M_R ∝ 1.
  - IR: type I, Y_nu = C∘Y_u, M_R ∝ C'∘Y_d (126-type right-handed mass).
  - II: type II, m_nu ∝ C∘Y_d (126 dominance; BSV's M_d - M_e is the case C = 1 - C_e).
  - P_nu^J: one of D/I1/IR/II plus J lepton-only complex symmetric entries added to m_nu (normalised
    to its largest singular value) at J random positions of {11,12,13,22,23,33} (l_nu = 2J; J = 1, 2).
  - F_nu: m_nu complex symmetric on Z ∪ {33}, m_33 = 1, all other entries free (l_nu = 2|Z|).
- Scored at M_Z, no running (as exps 039, 041), NO declared (no ordering bit), mass_floor 0.02.
  Observable sets: Y1 = CKM (s12, s13, s23, delta) + six mass ratios (mr_u_c, mr_c_t, mr_d_s,
  mr_s_b, mr_e_mu, mr_mu_tau), N 10. Y2 = Y1 + s12sq, s13sq, s23sq, dcp_deg, dm_ratio, N 15.
  N_l (lepton observables in Y2) = 7.

## Quark-viability conditioning
Stage 0 fits the quark sector alone (8 quark observables of Y1) for each Z. Only patterns whose best
quark fit has official max |pull| < 2 enter the populations below ("quark-viable"). This conditions on
quark data only; the claims are about what the shared structure then does to the leptons.

## Claims

**C1 (charged-lepton Clebsch rigidity).** If Y_e = C∘Y_d entrywise with C in CSET^(Z ∪ 33)
(property S_e) and Y_d is a symmetric texture of G43, then Y1 (CKM + four down-type/charged-lepton
mass ratios + the up ratios) cannot all be within 2 sigma at PDG 2026 precision with a 2% floor, for
any quark-viable Z and any C.
Analytic argument (2-3 block, Y_d = [[w, x], [x, 1]]): m_s/m_b ≈ |w - x²|/(1+|x|²) and
m_mu/m_tau ≈ |c22 c33 w - c23² x²| / (c33² + c23² |x|²). Either |x|² is of order m_s/m_b (then
m_mu/m_tau ≈ |c22/c33| m_s/m_b, which needs |c22/c33| ≈ 3.1: Georgi-Jarlskog, and then in four-zero
textures R = (m_e/m_mu)/(m_d/m_s) = c12²/c22², which CSET puts at 1/9 (+5.6 sigma) or 1/16 (-10 sigma),
never 0.0938), or |x|² >> m_s/m_b (the w ≈ x² cancellation the quarks use in exps 039/041), and then
m_mu/m_tau is O(|x|²) unless c22 c33 = c23² to the level of m_s/m_b: a fine-tuning a finite set hits
only by coincidence.
Broken groups: P_e^1, P_e^2, F_e (1, 2, |Z| continuous charged-lepton Clebsches).

**C2 (shared neutrino matrix).** If the light-neutrino mass matrix is a fixed Clebsch image of the
quark matrices (options D, I1, IR, II with C, C' in CSET; property S_nu), then even with all
charged-lepton Clebsches continuous (F_e), Y2 cannot be within 2 sigma.
Analytic argument: the quark matrices are hierarchical (m_c/m_t ≈ 0.004, m_s/m_b ≈ 0.019). D and I1
inherit the up hierarchy (dm_ratio ~ (m_c/m_t)² or its square, vs 0.030; exp 035(b) found 1e-5 for
Y_nu = Y_u), II and IR inherit the down hierarchy (dm_ratio ~ (m_s/m_b)² ≈ 4e-4 before Clebsch
factors, which change it by at most (12)² for |c| in [1/2, 6]), and a hierarchical m_nu has small
2-3 and 1-2 rotations, so the large angles must come from U_e, which then also has to give
m_e/m_mu and m_mu/m_tau with only |Z| moduli.
Broken groups: P_nu^1, P_nu^2, F_nu (with F_e).

**C3 (lepton observables are accommodated, not predicted).** In G43, a structure fits Y2 within
2 sigma only if its number of lepton-only continuous parameters l = l_e + l_nu is at least
N_l - 1 = 6. Equivalently, with the quark sector taking ~8 parameters for 8 quark observables, no
structure with k <= N - 3 fits.
Test: pass rates in the groups (F_e + S_nu: l = |Z|), (F_e + P_nu^1: l = |Z| + 2),
(F_e + P_nu^2: l = |Z| + 4), (F_e + F_nu: l = 3|Z|), and the fully shared S_e + S_nu (l = 0) if time
allows. C3 is falsified by any pass with l <= 5.

## Fitting and pass rule
- Fitter: scipy least_squares (trf, bounds) on a proxy residual vector: CKM Gaussian (PDG errors,
  delta wrapped), the six mass ratios in log space whitened with the official log covariance plus the
  2% floor, PMNS asymmetric Gaussian with NuFIT 6.1 1-sigma errors (dcp wrapped), dm_ratio Gaussian in
  log. Starts: quark parameters from a bank of stage-0 quark solutions for the same Z (random draw),
  lepton parameters random in their boxes. Starts per structure: 4 (Y1), 6 (Y2).
- Every reported pass/fail is from the official scorer (`src.evaluate.Scorer`, NO, IC23_woSKatm,
  mass_floor 0.02) at the best proxy point. Structures whose proxy optimum has max |residual| < 3 get a
  Nelder-Mead polish on the official chi2 before the decision.
- Pass ("fits within 2 sigma") = official chi2 finite and official max |pull| < 2. fit_ok (adds
  p_global > 0.05) is also recorded where dof > 0.
- Proxy validation: proxy chi2 vs official chi2 on >= 20 optima spanning good and bad fits; pass/fail
  agreement reported.
- C1 profiling over C (S_e): for each quark-viable Z, every C in CSET^(|Z|+1) is screened at the
  stage-0 quark solutions (charged-lepton ratio chi2, 2 terms, quark parameters fixed); the 150 best
  C per Z (all with screen chi2 < 200 if fewer) get a joint Y1 refit with quark parameters free.
  Screen false-negative check: 30 random screened-out C get the same joint refit.
  C1 is decided on the profiled result (any Z, any C passing falsifies it).
- Random-sample targets (scaled down uniformly if the 3 h budget requires; every launched structure
  is reported): Y1: P_e^1 200, P_e^2 200, F_e all Z × 2 seeds. Y2: F_e+S_nu 300 (C random; plus
  C = all ones for every Z × option), F_e+P_nu^1 150, F_e+P_nu^2 150, F_e+F_nu all Z × 3 seeds,
  S_e+S_nu 100 (optional).
- Intervals: Clopper-Pearson 95% (two-sided) on each pass fraction.

## Survival rule (per claim)
A claim survives if its property-X group has 0 passes with CP95 upper limit < 2% (for C1: no pass in
the profiled search) AND its designated broken group (C1: F_e; C2: F_e + F_nu; C3: the l >= 7 groups)
has a pass fraction whose CP95 lower limit is > 1%. If the X group never fits but the broken group
does not fit either, the claim is reported as "not structural": the failure is then not shown to come
from property X.

# Exp 042 models (read from the rendered arXiv PDFs; equation numbers as printed)

## AKS: Antusch, King, Spinrath, "Spontaneous CP violation in A4 x SU(5) with Constrained Sequential Dominance 2", arXiv:1301.6764 (v2)

Conventions (App. A, Eq. A.1-A.6): LR Yukawas, -L = Y_ij psibar_L^i psi_R^j H; L_nu = -1/2 Lbar M_nu L^c;
U_CKM = U_u U_d^dag, U_PMNS = U_e U_nu^dag with U_f diagonalising Y_f Y_f^dag (and M_nu M_nu^dag).
In src.yukawa terms: Y_f as printed, m_nu = conj(M_nu) (fixed by reproducing delta ~ 130 deg; the other choice
gives delta ~ 207 deg and theta23 ~ 35 deg).

- Neutrino sector, Eq. (3.6), (2.1): two right-handed neutrinos, Y_nu = (A, B), A = (0, a, -a)^T, B = (b, 0, 2b)^T,
  M_R = diag(M_A, M_B), M_B phase -pi/3. Seesaw:
  M_nu = m_a [[eps e^{i alpha}, 0, 2 eps e^{i alpha}], [0, 1, -1], [2 eps e^{i alpha}, -1, 1 + 4 eps e^{i alpha}]],
  eps = |m_b|/|m_a|, alpha = pi/3 (fixed by the vacuum alignment; footnote 5: "motivated by the current data which
  favours theta23 in the first octant"). m_1 = 0, NO by construction.
  Footnote 3: B = (b, 2b, 0)^T is the other CSD2 option, "not considered further".
- Down / charged-lepton sector, Eq. (3.8) (LR), omega-bar = e^{4 pi i/3}:
  Y_d = [[0, i eps2, 0], [omegabar eps102, eps23, 2 omegabar eps102 - eps23], [0, 0, eps3]]
  Y_e = [[0, -3/2 omegabar eps102, 0], [-3/2 i eps2, 9/2 eps23, 0], [0, -3 omegabar eps102 - 9/2 eps23, eps3]]
  Clebsch factors -3/2 (eps2, eps102) and 9/2 (eps23), chosen over the GJ factor 3 "because recent lattice results
  suggest a much smaller error for the strange quark mass" (p. 9). b-tau unification at the GUT scale (Clebsch 1).
  Phases fixed by the discrete vacuum alignment: theta2 phase pi/2, theta102 phase 4 pi/3 (p. 8, 18).
  (Exp 040's models.md listed the Y_d entries correctly but not the Y_e (3,2) entry; corrected here.)
- Up sector, Eq. (3.10): Y_u = [[a_u, b_u, 0], [b_u, c_u, d_u], [0, d_u, e_u]], real symmetric.
- Paper's fit (Sec. 6, Table 4): 10 parameters (a_u..e_u, eps2, eps102, eps23, eps3, tan beta) fitted to 13
  charged observables (9 Yukawas, 4 CKM) with REAP MSSM running, M_SUSY = 750 GeV, chi^2 2.05 / 3 dof.
  tan beta = 1.49 (small, b-tau unification by top RG effects, no SUSY thresholds). Neutrino sector not fitted:
  random scan over (m_a, eps), Fig. 2 and Eq. (7.2): theta13 8-9 deg, theta23 ~ 38.5, theta12 ~ 33, delta ~ 130,
  alpha2 ~ 315. Footnote 6: their fit gives y_s/y_d = 25.3 against ~ 19 from QCD.
- Parameters in our units (dimensionless observables, overall scales cancel): up 4 (a_u..d_u / e_u),
  d/e 3 (eps2, eps102, eps23 / eps3), neutrino 1 (eps), tan beta 1 (paper-fitted) = 9 continuous.
- Discrete data-selected choices: alpha (Z_n vacuum phase, 6 options, 3 bits); theta2 (Z_4, 2 bits) and theta102
  (Z_3 with sign, 6 options, 3 bits) phases; 2-2 Clebsch 9/2 vs 3 vs 6 (2 bits); CSD2 B direction (1 bit).
  Strict 11 bits. Generous 5 bits (alpha plus one effective charged-lepton phase delta_12^e from Z_4).

## CM: Chen, Huang, Mahanthappa, Wijangco, "Large theta13 in a SUSY SU(5) x T' Model", arXiv:1307.7711 (JHEP 1310:112)

Update of Chen-Mahanthappa arXiv:0705.0714 (exp 040: killed by its own theta13 = theta_c/(3 sqrt2), -36 sigma).
Adds one singlet flavon eta'' ~ 1'' coupling to the neutrinos only. Charged sector unchanged.

Conventions: Eq. (9) -L = U_R M_u Q_L + D_R M_d Q_L + E_R M_e l_L (RL), so the src.yukawa LR Yukawa is M^T;
the reproduction also needs the complex conjugate (Y_LR = M^dag): it gives |U_e,L| of Eq. (43), delta 197.8 in the
Eq. (65) range and delta_q = +46.1 deg (Eq. 20: 45.6). Without the conjugate delta_q = -46 deg and delta = 162.

- Eq. (13): M_u / (y_t v_u) = [[i g, (1-i)/2 g, 0], [(1-i)/2 g, g + h, k], [0, k, 1]].
- Eq. (14): M_d / (y_b v_d phi0 zeta0) = [[0, (1+i) b, 0], [-(1-i) b, c, 0], [b, b, 1]], M_e^T the same with
  c -> -3c (Eq. 12; GJ factor from Delta_45). b-tau unification exact at the GUT scale.
- Complex coefficients come only from T' Clebsch-Gordan coefficients (Yukawa couplings real): no free phase.
- Eq. (15) definitions; Eq. (16) benchmark b = 0.00304, c = -0.0172, k = -0.0266, h = 0.00426, g = 1.45e-5;
  Eq. (18) tan beta = 10 "taken"; outputs quoted at the EW scale after RG: Eq. (19) |V|, Eq. (20)
  beta 23.6, alpha 110, gamma = delta_q 45.6 deg, Eq. (21) lambda .227, A .798, rhobar .299, etabar .306.
- Neutrinos, p. 7-8: M_RR = s0 Lambda [[1,0,0],[0,0,1],[0,1,0]];
  M_D = zeta0 zeta0' v_u [[2 xi0 + eta0, -xi0, -xi0 + eta0''], [-xi0, 2 xi0 + eta0'', -xi0 + eta0],
  [-xi0 + eta0'', -xi0 + eta0, 2 xi0]] (lambda_1..4 = 1 "without loss of generality");
  Eq. (23) M_nu = -M_D M_RR^-1 M_D^T. eta0'' = 0 gives TBM (2007 model).
  Eq. (38) U_nu = U_TBM U_phi Q; Eq. (43) U_e,L fixed by the quark sector; Eq. (51)-(53): all three lepton angles
  (and delta, Eq. 63-65) are functions of the single angle phi. Eq. (58) benchmark xi0 .185, eta0 -.865,
  eta0'' .200 -> Eq. (59) theta12 33.351, theta23 49.162, theta13 9.189; Eq. (60) masses .00398, .00959, -.0505 eV;
  Eq. (65) delta 195.9-200.1 (NH).
- Paper's count: 10 real parameters (7 charged: b, c, k, h, g, y_t, y_b; 3 neutrino: xi0, eta0, eta0'');
  tan beta and the neutrino mass scale outside. Both NH and IH allowed (Fig. 1).
- Parameters in our units: charged 5 (b, c, k, h, g; y_t, y_b cancel in ratios), neutrino 2 (the direction of
  (xi0, eta0, eta0''); the overall scale cancels) = 7 continuous. tan beta 10 fixed by hand (not counted).
- Discrete: model version (2013 eta'' variant chosen after the theta13 measurement, 1 bit); ordering (both allowed;
  1 bit); strict count adds 1 bit for the eta'' singlet type (1'' vs 1').

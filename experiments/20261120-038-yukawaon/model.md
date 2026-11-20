# Koide-Nishiura yukawaon + democratic seesaw model, as written in the paper

Source: Y. Koide and H. Nishiura, "Quark and Lepton Mass Matrix Model with Only Six
Family-Independent Parameters", arXiv:1510.05370 (Phys. Rev. D 92, 111301(R) (2015)).
Theory companion with the same numerical model: arXiv:1512.08386 (Table 1 quoted from 1510.05370).
Equation numbers below are those of arXiv:1510.05370v2 unless marked.

## Why this paper

arXiv API listing of Koide & Nishiura (2008-2019): the unified quark + lepton papers are
1011.1312, 1202.5815, 1209.1275, 1308.2129, 1405.0069, 1503.04900, 1510.05370, 1512.08386,
1701.06287. 1510.05370 is the most compressive with a full numerical fit (6 family-independent
parameters; the earlier 1405.0069-era models needed 10, per 1510.05370's introduction).
1512.08386 rederives the same VEV relations from superpotentials and keeps Table 1 unchanged.
1701.06287 changes the flavon VEV scales ("drastically changed ... keeping previous
phenomenological success") but its mass matrices are the same form, M_f = Phi_f S_f^-1 Phibar_f
(its Eq. 1.4-1.6), phases only in the up sector, and its Appendix A reviews the 1510.05370 fit.
1805.07334 is quark-only; 1911.03411 is neutrino-only. So the 1510.05370 model is the
representative of the class; no later version is more compressive.

## Structure

Charged-lepton input (family-dependent, the only one):
- Eq. (5): Phi_0 = v_0 diag(z1, z2, z3), z_i proportional to sqrt(m_ei), sum z_i^2 = 1, masses at mu = M_Z (Sec. 4).

Democratic flavons, Eq. (6)-(7):
- S_f = v_Sf (1 + b_f e^{i beta_f} X3), X3 = (1/3) [all ones].
- b_e = b_nu = 0 (from superpotential Eq. (10)); b_u real (beta_u = 0), b_d e^{i beta_d} complex.

Phase flavon, Eq. (4): P_u = diag(e^{i phi1}, e^{i phi2}, e^{i phi3}), P_d = P_nu = P_e = 1. Only
(phi1 - phi3, phi2 - phi3) are observable (remark (d)); the paper fits them as (phi1, phi2) with phi3 = 0.

Mass matrices, Eq. (18)-(19) (Sec. 4, dimensionless):
- Y_u = Phi_0 Pbar_u (1 + b_u X3)^-1 P_u Phi_0
- Y_d = Phi_0 (1 + b_d e^{i beta_d} X3)^-1 Phi_0
- Y_e = Y_nu = Phi_0 Phi_0 (diagonal: the charged-lepton basis is the flavour basis)
- Y_R = Phi_0 Y_u + Y_u^T Phi_0 + xi_R Y_e^T Y_e (RH-neutrino Majorana matrix; xi_R term added for the Dm^2 ratio, Eq. (9))
- M_nu = Y_nu Y_R^-1 Y_nu^T (type-I seesaw, Eq. (8))

Origin of the inverse (Eq. 12-17): a 6x6 universal seesaw with heavy F, M_F = -S_f, block-diagonalised;
two "assumptions" (Eqs. 15-16) are needed to drop the direct Yukawa term. The inverse is
(1 + b X3)^-1 = 1 - b/(1+b) X3; b_u near -1 makes the top heavy (democratic seesaw, ref [9] = Koide-Fusaoka).

Parameters (paper's count, "six family-independent"): b_u, b_d, beta_d, (phi1, phi2), xi_R.

## CKM -> PMNS link

Y_u enters the RH Majorana matrix Y_R, so (b_u, phi1, phi2) are shared by the up-quark sector
and the neutrino sector. The down sector (b_d, beta_d) does not enter PMNS. Paper's own logic
(Sec. 4): b_u from m_c/m_t; b_d, beta_d from down-quark ratios; (phi1, phi2) from the CKM;
xi_R from the Dm^2 ratio; then the four PMNS parameters are predictions with no freedom.
This is a genuine "quark sector fixes PMNS" structure, not F1 (no shared rotation of a fixed
pattern), not F2 (no Clebsch on a charged-lepton rotation: U_e = 1 here), not SO(10) or
modular. Closest earlier test: exp 035(b) Dirac Y_nu = Y_u (here Y_u enters M_R, not Y_nu).

## Conventions fixed by the benchmark (reproduce.py, reproduce.log)

Our engine: src.yukawa (LR, V = U_u^dag U_d, U_PMNS = U_e^dag U_nu, Takagi m_nu). Of 16
variants (Y_u symmetric/hermitian, overall conjugation, M_nu conjugation, sign of phi), exactly one
reproduces every published number:
- Y_u hermitian, as printed in Eq. (18) (Pbar_u on the left, P_u on the right);
- Pbar_u = diag(e^{-i phi1}, e^{-i phi2}, 1), i.e. Pbar = P^* with P_u from Eq. (4) as stated in
  1512.08386 Sec. 3. Sec. 4 of 1510.05370 prints "Pbar_u = diag(e^{i phi1}, e^{i phi2}, 1)"; with
  that sign |V_cb| = 0.0985 instead of the published 0.0400, so the printed Sec. 4 sign is a typo
  or a different CKM convention;
- M_nu conjugated relative to the engine's Majorana convention (same as exps 022, 023).

## Things the paper's Table 1 does not show

- The paper quotes sin^2 2theta12 = 0.8254 (octant-blind). With m1 < m2 (Dm^2_21 > 0) the
  benchmark's actual |U_e2|^2/(|U_e1|^2+|U_e2|^2) = 0.712 (theta12 = 57.6 deg, the "dark side"),
  not 0.291. Same for theta23: s23^2 = 0.529.
- Its "observed" quark ratios are Xing et al. 2008 / Fusaoka-Koide 1998; it admits sqrt(m_u/m_c)
  = 0.061 vs 0.045 and m_s/m_b = 0.027 vs 0.019 (Xing).

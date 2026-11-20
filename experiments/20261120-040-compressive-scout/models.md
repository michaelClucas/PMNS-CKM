# Exp 040 models: matrices, equation numbers, counts

Source reading: full PDF pages (rendered) for 1710.01741, hep-ph/0502161, hep-ph/0502152; arXiv HTML
(ar5iv) text for the others, as marked. "ar5iv" entries were read through a text extractor, not the
rendered PDF; where a kill rests on an equation, that equation is either a published number in the
abstract or was checked numerically here. Convention unless stated: rows = left-handed fields (LR),
the `src.yukawa` convention. Program N (no absolute scales): CKM 4 + PMNS 4 + dm_ratio + 7 same-scale
mass ratios (mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau) = 16, N - 2 = 14, unless
the model leaves some of these unpredicted.

## M1. Universal texture zero, Δ(27) (de Medeiros Varzielas, Ross, Talbert, arXiv:1710.01741), "UTZ"
- Eq. (II.1): M_a^D ≈ m3 [[0, ε³, ε³], [ε³, r ε², r ε²], [ε³, r ε², 1]], r_{u,d} = 1/3, r_e = -1
  (Georgi-Jarlskog r_e/r_d = -3, κ = 0 case of Sec. II.A).
- Eq. (V.1) (fit form), i ∈ {u, d, e, ν}, α = β = 0:
  M_i/c = a e^{iγ} [[0,1,1],[1,2,0],[1,0,-2]] + b e^{iδ} [[0,0,0],[0,1,1],[0,1,1]] + E33,
  with a ↔ θ123 θ23 and b ↔ r θ23 θ23 Σ (Eq. II.3); so a_d = a_e, b_d = -b_e/3.
- Higher order (Eq. II.8, Table III "H.O."): d' e^{iψ} (θ23 θ3 + θ3 θ23) = d' e^{iψ} [[0,0,0],[0,0,1],[0,1,2]],
  carries Σ, so d'_e = -3 d'_d.
- Neutrinos: Dirac (a, b)_ν, γ_ν, δ_ν; Majorana same form with (x, y, ρ, φ) and scale M (Eq. III.3, Table III).
- Reproduced here: Table V L.O. and H.O. mass ratios exactly (md/mb .00090/.0010, ms/mb .020/.020,
  me/mτ .00031/.00026, mμ/mτ .055/.049, mu/mt 7.16e-6/7.89e-6, mc/mt .0027/.0025).
- Sharing: d and e share a, b, γ, δ (and d'); u and ν have their own (a, b). PMNS sees the quark
  sector only through U_e = U_d-like rotation.
- Count (paper): 9 at LO ("18 measurable quantities determined by nine parameters"); Table III lists
  (a,b)_e, (a,b)_u, (a,b)_ν, (x,y), (γ,δ)_e = 10 continuous, with (γ,δ)_u = 0, (γ,δ)_ν = (2π/5, 0),
  (ρ, φ) = (0, -2π/5) chosen discrete. HO adds d_d (and ψ_d = π). Honest: k_cont 10 (LO) / 11 (HO)
  + bits for the discrete phase choices (≥ 4 values chosen among a Z_5-like set: ≥ 4 bits) + r_e/r_d
  case (κ = 0 vs 2: 1 bit). k ≈ 15-16 vs program N 16 (N - 2 = 14). Fails compression with honest
  bits; passes with the paper's count.

## M2. Albright-Barr lopsided SO(10) (hep-ph/0002155, hep-ph/0003251; as summarised in hep-ph/0502161), "AB"
- hep-ph/0502161 Eq. (2): M_U = [[η, δ_N, δ'_N],[δ_N, 0, -ε/3],[δ'_N, ε/3, 1]] m_U,
  M_D = [[0, δ, δ' e^{iφ}],[δ, 0, -ε/3],[δ' e^{iφ}, σ+ε/3, 1]] m_D,
  M_N = [[η, δ_N, δ'_N],[δ_N, 0, ε],[δ'_N, -ε, 1]] m_U, M_L = [[0, δ, δ' e^{iφ}],[δ, 0, σ+ε],[δ' e^{iφ}, -ε, 1]] m_D.
- Eq. (3) benchmark: σ 1.83, ε 0.147, δ 0.00946, δ' 0.00827, φ 119.4°, η 6e-6.
- Eq. (4): M_R = [[c²η², -bεη, aη],[-bεη, ε², -ε],[aη, -ε, 1]] Λ_R; Eq. (5) M_ν = -M_N M_R^-1 M_N^T.
- Sharing: ε, η shared by all four Dirac sectors and M_R; σ, δ, δ', φ by D and L. Strongly unified.
- Count: 6 (σ, ε, δ, δ', φ, η) + a, b, c complex (6) = 12; +2 with δ_N, δ'_N. Program N 16:
  k 12 ≤ 14 passes compression by count (no data-chosen bits beyond version choice, ≥ 1 bit).
- Paper's own scan (abstract, hep-ph/0502161): 1e-5 ≲ sin²θ13 ≲ 1e-2 for general complex a, b, c.

## M3. Ji-Li-Mohapatra modified lopsided SO(10) (arXiv:hep-ph/0510353), "JLM" (ar5iv)
- Eq. (1)/(3): D and L identical to M2; U, N modified with κ, ρ, ω entries; M_R = diag(a, b, 1) Λ_R.
- Eq. (9): 12 fitted values (σ 1.83, ε 0.1446, δ 0.01, δ' 0.014, φ 27.9°, η 1.02e-5, ω, ρ, κ, scales).
- Eq. (4): m_e/m_μ ≃ (1/9) t_L t_R, m_d/m_s ≃ t_L t_R (so R ≈ 1/9 at leading order).
- Predictions: sin²2θ13 = 0.074 (0.055-0.110), δ_ℓ ≈ π (all couplings real except φ).
- Count: paper says 10, 12 and 13 in different places; k ≈ 11-12 dimensionless + version bit.

## M4. Babu-Pati-Wilczek / Babu-Pati-Rastogi SO(10)/G224 (hep-ph/9812538; hep-ph/0502152), "BPW", "BPR"
- hep-ph/0502152 Eq. (1) (PDF page 4, verified): M_u = [[0, ε', 0],[-ε', ζ22u, σ+ε],[0, σ-ε, 1]],
  M_d = [[0, η'+ε', 0],[η'-ε', ζ22d, η+ε],[0, η-ε, 1]], M_ν^D = [[0,-3ε',0],[3ε', ζ22u, σ-3ε],[0, σ+3ε, 1]],
  M_l = [[0, η'-3ε', 0],[η'+3ε', ζ22d, η-3ε],[0, η+3ε, 1]]; Eq. (4) M_R = [[x,0,z],[0,0,y],[z,y,1]] M_R.
- Eq. (6) fit A: σ = 0.109-0.012i, η = 0.122-0.0464i, ε = -0.103, η' = 2.4e-3, ε' = 2.35e-4 e^{i69°},
  ζ22d = 9.8e-3 e^{-i149°}, ζ22u = 0. Not reproduced: with these printed values m_e/m_μ = 0.0012
  (data 0.0047); CKM comes out reasonable (s12 0.2206, s13 0.0033, s23 0.0355, δ 1.17 rad).
- Original BPW (hep-ph/9812538, ar5iv, Eq. 28/35, real parameters, no ζ22): solar angle θ_eμ ≈ 0.06 (SMA).
- Count (BPR): σ, η, ε', ζ22d complex, ε, η' real = 10, + x, y, z (≥ 3) = 13 vs N 16. Passes by count
  if x, y, z real (16 if complex: fails).

## M5. Universal two-zero texture in SO(10) (Chen, Jiang, Zhou arXiv:2606.24571; earlier 2410.18743), "UTZT" (arXiv HTML)
- Eq. (1): (1,1) and (1,3) zeros, Hermitian Y_f = ζ_f [[0, C_f, 0],[C_f*, B̃_f, B_f],[0, B_f*, A_f]].
- Eq. (5): Y_u = H + r2 F + i r3 G, Y_d = r1 (H + F + iG), Y_ν = H - 3 r2 F + i c_ν G,
  Y_e = r1 (H - 3F + i c_e G); Eq. (6) type I+II, Eq. (7) M_R ∝ F.
- Free: 7 (A_u, A_d, φ1, φ2, m_L, m_R, c_ν) with 9 charged-fermion eigenvalues as inputs, plus
  sign choices η_u, η_d, η_e, ζ_d (4 bits). Honest: each sector's other 3 texture entries are fixed by
  its 3 masses, so in program units k_cont = 14 (16 minus 2 overall scales) for N 16, + 4 bits = 18 > 14.
  Fails compression (bits). Same 10+126+120 Hermitian class as exp 014's FN with texture zeros added.

## M6. SUSY SU(5)×T' (Chen-Mahanthappa arXiv:0705.0714; large-θ13 update Chen-Huang-Mahanthappa-Wijangco 1307.7711), "CM-T'" (ar5iv)
- 0705.0714: M_d and M_e share b (off-diagonal 1-2, ±(1±i)b) and c; (2,2) = c in M_d, -3c in M_e (GJ).
  Eq. (37): θ12^e ≃ sqrt(m_e/m_μ) ≃ (1/3) sqrt(m_d/m_s). Eq. (41)-(42) ratios: m_d/m_s 0.050, m_e/m_μ 0.0061.
- 9 parameters (7 charged + 2 neutrino). θ13 ≃ θ_C/(3√2), |U_e3| 0.055 (0705.0714).
- 1307.7711 (abstract): adds one neutrino flavon; θ13 8-10°, δ 195-200°. Charged sector unchanged.

## M7. A4×SU(5) with CSD2 (Antusch, King, Spinrath arXiv:1301.6764), "AKS" (ar5iv)
- Eq. (3.8): Y_d entries (iε2, ω̄ε102, ε23, 2ω̄ε102 - ε23, ε3); Y_e = same ε's with Clebsch -3/2 (1-2, 2-1)
  and 9/2 (2-2, 2-3); "m_μ/m_s = 9/2 at the GUT scale".
- 10 parameters (9 Yukawa + tanβ) fitted to 13 charged observables, χ² 2.05; PMNS from CSD2 (M_A, M_B).
- Table 5: y_e/y_μ = 4.74e-3, y_d/y_s = 11.7/295 → m_s/m_d = 25.3 (paper notes tension with ≈ 19).

## M8. Pati-Salam A4×Z5 "A to Z" (King arXiv:1406.7005), "AtoZ" (ar5iv)
- Eq. (32): Y^u = Y^ν = [[0,b,0],[a,4b,0],[a,2b,c]] (CSD4); Eq. (38) Y^d; Eq. (107)/(115) Y^e with
  Clebsch -3 and a flavon-mixing-modified -4.5 (Eq. 116: y_μ = 4.1 y_s, y_e = y_d/2.6, i.e. R tuned to 1/10.7).
- Paper: quark sector 16 free parameters; leptons m_a, m_b, m_c. Predictions (Eq. 120): θ12 31-34°,
  θ13 8.4-9.7°, θ23 44.4-46.4°, δ 256-266°.

## M9. Δ(27)×SO(10) (Björkeroth, de Anda, de Medeiros Varzielas, King arXiv:1512.00850), "D27SO10" (ar5iv)
- Eq. (2.4): m^f = μ_a^f e^{2iρ_atm} A + μ_s^f e^{2iρ_sol} S + μ_d^f e^{2iρ_dec} D; Y_d, Y_e same flavons, independent coefficients.
- Fit: 18 parameters (12 quark + 6 lepton: y^e_{1,2,3}, μ_{a,b,c}) for N = 18, χ² 2.7-3.3.

## M10. SO(10)×D3 (Dermisek-Raby arXiv:hep-ph/0507045), "DR" (abstract)
- 14 parameters for 20 observables; predicts sin θ13 ≈ 0.05.

## M11. SO(10)×S4 rank-one + type II (Dutta-Mimura-Mohapatra arXiv:0911.2242), "DMM" (abstract)
- m_μ ≃ 3 m_s, m_b ≃ m_τ; U_e3 ≃ θ_C/(3√2) ≈ 0.05.

## M12. S4×SO(10) 10+126 (Cai-Yu arXiv:hep-ph/0608022), "CY" (ar5iv)
- Eq. (2)-(6): M_u = M_u^(10) + M_u^(126), M_d = M_d^(10) + M_d^(126), M_l = M_d^(10) - 3 M_d^(126),
  M_ν^D = M_u^(10) - 3 M_u^(126); Eq. (7)-(10) S4 forms. 18 complex parameters, reduced by basis choice;
  d and e masses: 6 real parameters (b_0-2, e_0-2) for 6 masses (no prediction); tanβ fixed by hand.
- Minimal-SO(10) (10+126) class, exps 011/014.

## M13. SO(10) S3×P (Morisi-Picariello arXiv:hep-ph/0505113), "MP" (ar5iv)
- Eq. (15a)-(15e): M^d = k^d I + Δ^d + (...)A, M^l = k^d I - 3Δ^d + (...)A; GUT relations m_e ≈ m_d/3,
  m_μ ≈ 3 m_s (Eq. 24); PMNS tri-bimaximal (θ13 ≈ 0).

## M14. Universal 2↔3-symmetric textures (Koide, Nishiura, Matsuda et al. hep-ph/0209333, hep-ph/0403008, hep-ph/0506192), "KN23" (ar5iv / abstract)
- hep-ph/0403008 Eq. (2)-(3): M_f = P_f† M̂_f P_f†, M̂_f = [[0, A, A],[A, B, C],[A, C, B]], separate A, B, C per sector
  fixed by its masses (Eq. 7, 12). Eq. (29)/(45): |U13|² = (m_e/(m_τ+m_e)) sin²(..) (case ii) or ≃ (m_e/m_μ) sin²(..) (case i).
- hep-ph/0506192 (nonsymmetric version, abstract): |U13|² ≈ 1.3e-10.
- No continuous parameter shared between quark and lepton sectors (only the texture).

## M15. SU(3) family gauge model (W.-M. Yang arXiv:1011.4573; 2504.13213), "YANG" (ar5iv)
- Eq. (58), (65): Y_u, Y_d, Y_D, Y_e built from shared flavon matrices F̃_k with one shared ε0; y_q, y_l, tanβ,
  M_R scale. 6 parameters for "25 values". Prediction: sin θ13 = 0.0576 (sin²θ13 ≈ 0.0033).

## M16. Screened out without a filter (fail criterion (i) or (ii) by construction)
- Babu-Shafi predictive SO(10) (hep-ph/9503313): quark + charged lepton only, no neutrino sector.
- 3-3-1 models (T7: 1501.07261; Δ(27): 1601.05062): quark and lepton Yukawas independent.
- SU(5) with Z_N flavour (Cárcamo Hernández et al. 1411.2913; Campos et al. 1403.2525, abstract only):
  quark and lepton coefficients fitted separately (15 fitted for 18 in 1411.2913), only λ shared.
- Fu-King-Marsili-Pascoli-Turner-Zhou non-SUSY SO(10) (2209.00021): minimal 10+126 class (exps 011, 014).
- Modular unified models (2101.02266, 1912.07573, 2409.15823 and the exp 016 survey): common-τ class (exps 016-017).

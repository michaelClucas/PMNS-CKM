# Exp 045 catalogue: published predictive quark structures

Conventions. Y_u, Y_d in the LR convention of `src.yukawa` (rows = left-handed). `c` marks a complex entry
(modulus and phase), `r` a real entry of either sign, `1` the per-sector normalisation, `*` the complex
conjugate of the mirror entry (Hermitian). Every structure is scored on the program's 8 quark observables
Q8 = (s12, s13, s23, δ, m_u/m_c, m_c/m_t, m_d/m_s, m_s/m_b), PDG 2026, 2% mass floor.

**k_cont** is the number of independent continuous directions of the 8 observables: the Jacobian rank at
random points (`lib45.reduce_params`, central differences, threshold 1e-7 relative; `ranks.log`). Phases
that rephasing removes are pinned to 0 before fitting. The overall scale of each sector drops out of Q8,
so k_cont is two less than the paper's count of real parameters plus phases whenever the paper counts the
6 masses. That count matches the paper's own in every case where the paper gives one (noted below).

"Read from" says what was read: PDF = rendered PDF text (pdftotext), TeX = the arXiv source, HTML = ar5iv
HTML, abs = the abstract only. The PDFs and source were kept in the session scratchpad, not committed.

## Textures

| id | source (read from) | Y_u | Y_d | paper count | k_cont (rank) | stated scale |
|---|---|---|---|---|---|---|
| F6 | Fritzsch, PLB 73B (1978) 317; 85B (1979) 81 (no arXiv). Form = RRR eq. `f5` (TeX) | [[0,c,0],[*,0,c],[0,*,1]] | same | 6 masses + 2 phases | 6 | not fixed (we try M_Z and GUT) |
| DHR | Dimopoulos-Hall-Raby PRD 45 (1992) 4192, Harvey-Ramond-Reiss NPB 199 (1982) 223; RRR footnote to Table 1: "Solution 2 with E'=0" (TeX) | Fritzsch | [[0,F,0],[F*,E,0],[0,0,D]] | 6 + 1 phase | 5 | GUT (MSSM) |
| RRR1 | Ramond-Roberts-Ross hep-ph/9303320, Table 1 sol. 1 (TeX) | [[0,C,0],[C,B,0],[0,0,A]] | [[0,F,0],[F*,E,E'],[0,E',D]] | 7 real + 1 phase ("Solutions 1,3,4 and 5 have 8 parameters") | 6 | GUT (MSSM, h_b = h_τ) |
| RRR2 | same, sol. 2 | Fritzsch [[0,C,0],[C,0,B],[0,B,A]] | [[0,F,0],[F*,E,E'],[0,E'*,D]] | 7 real + 2 phases ("an additional phase") | 7 | GUT |
| RRR3 | same, sol. 3 (= Giudice ansatz; = Verma 1708.08993 case I) | [[0,0,C],[0,B,0],[C,0,A]] | as sol. 1 | 8 | 6 | GUT |
| RRR4 | same, sol. 4 | [[0,C,0],[C,B,B'],[0,B',A]] | [[0,F,0],[F*,E,0],[0,0,D]] | 8 | 6 | GUT |
| RRR5 | same, sol. 5 (= Giraldo-Rojas 1511.08858 case II, Eq. 21; = Verma 1708.08993 case IV) | [[0,0,C],[0,B,B'],[C,B',A]] | [[0,F,0],[F*,E,0],[0,0,D]] | 8 (GR: 7 real + 1 phase) | 6 | GUT (RRR); M_Z (GR, Verma) |
| GR1 | Giraldo-Rojas 1511.08858 case I, Eq. 16 (HTML); same zero pattern as Giraldo-Rojas 1811.05068 Eq. 12a,b (HTML) | [[0,0,ξ],[0,α,β],[ξ*,β*,γ]] | Fritzsch | 7 real + 2 phases | 7 | M_Z |
| BB | Belfatto-Berezhiani 2305.00069 Eq. (3.2), (3.4), scenario x_u = 1 (Sec. 4.2.3) (PDF) | Fritzsch, symmetric | [[0,A,0],[A,0,B x_d],[0,B/x_d,C]] real; 2 phases in F = F_u F_d* | 7 real + 2 phases | 7 | 1e3, 1e6, 1e16 GeV (SM) |
| LG_S8 | Ludl-Grimus 1501.04942 Table 3 (PDF) | complex symmetric [001;011;110] (u) | [001;022;122] (d) | 1 = real, 2 = complex | 8 | M_Z |
| LG_S9 | same | [001;010;102] | [001;011;112] | | 7 | M_Z |
| LG_S10 | same, Eq. (11) "most predictive" | [010;120;001] | [001;011;112] | | 7 | M_Z |
| LG_S12 | same | [001;011;112] | [001;022;120] | | 8 | M_Z |
| LG_S13 | same | [010;111;012] | [001;022;120] | | 8 | M_Z |
| LG_S15 | same | [010;111;012] | [001;010;102] | | 7 | M_Z |
| F4 (reference) | Fritzsch-Xing hep-ph/0212195 four-zero (abs; standard form) | [[0,c,0],[*,r,c],[0,*,1]] | same | 8 masses-equivalent + 2 phases | 8 (not predictive) | M_Z |

Ludl-Grimus matrices are written row by row, 0 = zero, 1 = real positive, 2 = complex (their Table 3
convention; matrices are complex symmetric, M = M^T, not Hermitian). The entry used as normalisation is
in `structures.py`. LG_S8, S12, S13 have k_cont = 8 = N: they predict nothing among the 8 observables
(Ludl-Grimus call them "predictive for m_d" by a Δχ² criterion on absolute masses with 5-40% errors).

Harrison-Scott 2501.18508 (PDF), Eq. (2.1)-(2.3):

    M_q / n_q = [[c_q |ε_q|^4, b ε_q^3, 0], [*, b |ε_q|^2, A0 ε_q^2], [0, *, 1]],   q = u, d (Hermitian)
    ε_u / ε_d = -i tan(π/8)  (Eq. 2.3, fixed),   |ε_d + ε_u| = λ0  (Eq. 2.2)

Five free parameters (A0, b, c_u, c_d, λ0) plus the fitted scale (paper footnote 8 counts the scale:
χ²/dof ≈ 1.0/2). The paper's fit is at μ ≈ 1e7 GeV (1e4 TeV), SM running; its fit at μ = m_t is
"excluded by its χ²/dof (≈ 100/3)". The PDF text does not show which powers are moduli (bold vs light
font is lost). Diagonal entries as moduli and off-diagonal entries as complex powers reproduces Table 1's
fitted column exactly at the published parameters (`ranks.log`: m_u/m_c 2.002e-3, m_d/m_s 4.976e-2,
m_c/m_t 3.455e-3, m_s/m_b 1.968e-2, λ 0.22499, A 0.8758, ρ̄ 0.1519, η̄ 0.3477, α 91.3°; paper 2.00e-3,
4.97e-2, 3.46e-3, 1.968e-2, 0.22499, 0.876, 0.1519, 0.3477, 91.30°). The all-complex reading gives
m_u/m_c 0.0127 and is wrong. k_cont = 5 (texture) + 1 (scale) = 6. Discrete: the paper keeps one of two
near-degenerate mass-sign solutions (no χ² difference; not counted).

Non-predictive published textures (not fitted beyond F4, listed for completeness):
- Tanimoto-Yanagida 1601.04459 "Occam's razor" (PDF abstract and Sec. 2): Y_u diagonal, Y_d with three
  zeros, "six real parameters and one CP violating phase" for Y_d. In Q8 units: 2 (up ratios) + 5 + 1 = 8 = N.
  Their "prediction" of the Cabibbo angle uses sin 2β, θ13, θ23 and masses as inputs.
- Roberts-Romanino-Ross-Velasco-Sevilla hep-ph/0104088 Eq. (24)-(25) (PDF): symmetric
  U = [[0, bε³, cε⁴], [bε³, ε², aε²], [cε⁴, aε², 1]], D the same with barred symbols, two phases (Eq. 29).
  4 + 4 moduli + 2 phases = 10 > 8. Its Sec. 2.2 is the published exclusion of the exact (1,1), (1,3)
  zeros in both sectors: the relations |V_ub/V_cb| = √(m_u/m_c), |V_td/V_ts| = √(m_d/m_s) were "consistent
  only at greater than the 99% CL" already in 2001.
- Fritzsch four-zero (F4 above), and Belfatto-Berezhiani with x_u free (10 parameters for 10 observables,
  their Sec. 4.2.3).
- Verma 1708.08993 (HTML) cases II, VII, VIII (four zeros) and Giraldo-Rojas 1811.05068 (two phases in Y_u,
  9 parameters with masses): not predictive in Q8. Verma cases I and IV are RRR3 and RRR5.

## Relations (closed form)

| id | relation | source (read from) | free parameters | k_cont + bits |
|---|---|---|---|---|
| GST | \|V_us\| = √(m_d/m_s) | Gatto-Sartori-Tonin, PLB 28B (1968) 128 (no arXiv); RRRV hep-ph/0104088 Eq. (2) without the up term; Verma 1708.08993 uses √(m_d/(m_d+m_s)) | 4 ratios, s13, s23, δ | 7 + 0 |
| HR | \|V_ub/V_cb\| = √(m_u/m_c) | Hall-Rasin hep-ph/9303303 (abs); RRRV Eq. (1) (PDF); Fritzsch-Xing hep-ph/9707215 Eq. (14) (HTML) with tan θ_u = √(m_u/m_c) | 4 ratios, s12, s23, δ | 7 + 0 |
| TDTS | \|V_td/V_ts\| = √(m_d/m_s) | Hall-Rasin; RRRV Eq. (1); Giraldo-Rojas 1511.08858 LO relation | 4 ratios, s12, s13, s23 (δ solved in (0, π)) | 7 + 1 (sign of J) |
| ALPHA90 | α = 90° | Fritzsch-Xing hep-ph/9807551 (abs: "close to or identical to a rectangular triangle"); Antusch-King-Malinsky-Ross 0910.5127 (not read; cited for the phase sum rule); Harrison-Scott 2501.18508 Eq. (1.2) | 4 ratios, s12, s13, s23 | 7 + 1 |
| FX | V = Eq. (8) of hep-ph/9707215 with tan θ_u = √(m_u/m_c), tan θ_d = √(m_d/m_s), φ = 90° | Fritzsch-Xing hep-ph/9707215 Eq. (8), (14) (HTML); mass relations and φ = 90° from hep-ph/9807551 (abs and secondary citations) | 4 ratios, θ | 5 + 1 |
| D7 | \|V_us\| = \|cos(3π/7)\| = 0.2225 | Blum-Hagedorn-Lindner 0709.3450 (abs): \|V_us\| = \|cos(π(m_u − m_d) j / n)\|, n = 7, j = 1, m_u − m_d = 3 | 4 ratios, s13, s23, δ | 7 + bits for (n, j, m_u, m_d) (≥ 3, not needed: see result) |

Not scored: Koide-type quark relations. Rodejohann-Zhang 1101.5525 (abs) propose them only for the
cross-sector triplets (u, d, s) and (c, b, t), and state the relation fails within (u, d). Cross-sector
triplets need an inter-sector mass ratio (e.g. m_c/m_b), which is not among the program's 8 quark
observables; the within-sector version K_u = K_d = 2/3 is not proposed by any paper read here.

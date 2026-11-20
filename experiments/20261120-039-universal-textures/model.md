# Exp 039 model structures and sources

All matrices were read from the papers' own text (PDFs fetched from arXiv, text extracted; equation
numbers below are the papers' numbers). Code: `model.py`.

## (a) Democratic textures

**Fritzsch-Xing, "Mass and flavor mixing schemes of quarks and leptons", Prog. Part. Nucl. Phys. 45
(2000) 1, arXiv:hep-ph/9912358**, section 4.5 (quarks) and 5.2 (leptons).

- Eq. (4.44): symmetry limit M_q^(0) = (c_q/3) J, J the all-ones matrix (S(3)_L x S(3)_R flavour
  democracy), c_u = m_t, c_d = m_b. Eq. (4.45): U_0 (the matrix that maps the democratic to the
  hierarchical basis).
- Eq. (4.47): first breaking ΔM_q^(1) = (c_q/3) [[0,0,ε_q],[0,0,ε_q],[ε_q,ε_q,δ_q]]; case (b),
  ε_q = δ_q, eq. (4.51), "more favored by current data on V_cb".
- Eq. (4.52): second, complex breaking ΔM_q^(2) = (c_q/3) σ_q [cos φ_q K1 + i sin φ_q K2],
  K1 = [[1,0,-1],[0,-1,1],[-1,1,0]], K2 = [[0,1,-1],[-1,0,1],[1,-1,0]] (Hermitian).
- Eq. (4.53): with ε = δ the hierarchical-basis matrices have four texture zeros.
- Leptons, eqs. (5.4)-(5.6): M_l^(0) = (c_l/3) J, M_ν^(0) = c_ν I; breaking (c_l/3) diag(0,0,ε_l) +
  (c_l/3) diag(-iδ_l, iδ_l, 0) for charged leptons and c_ν diag(-δ_ν, δ_ν, ε_ν) for neutrinos
  (Majorana phases dropped, as in the paper). Leading-order mixing = U_0 (eqs. (5.7), (5.8)).

**Fritzsch-Xing, "Large leptonic flavor mixing and the mass spectrum of leptons", Phys. Lett. B440
(1998) 313, arXiv:hep-ph/9808272**: eq. (3) symmetry limits, eq. (4) first step, eq. (5)
M_l = (c_l/3)[[1-δ_l,1,1],[1,1+δ_l,1],[1,1,1+ε_l]], M_ν = c_ν diag(1-δ_ν, 1+δ_ν, 1+ε_ν) (CP
conserving), eq. (11)-(12): sin²2θ_sun ≈ 1 - (8/3)ξ², sin²2θ_atm ≈ (8/9)(1 ± ...). Earlier:
hep-ph/9509389 (PLB 372 (1996) 265), same symmetry limit.

Implemented:
- **DEMU** (universal democracy, the brief's class (a)): all four sectors use the quark structure
  (4.44)+(4.47, ε=δ)+(4.52), M_f = J + M1(δ_f) + σ_f(cos φ K1 + i sin φ K2) (times c_f/3, which
  cancels). (δ_f, σ_f) per sector are fitted (they set the sector's two mass ratios); ONE phase φ is
  shared by all sectors (the universal parameter). Neutrinos: Dirac with the same matrix, or type-I
  seesaw with M_D of that form and M_R ∝ 1 (1 bit). k_cont = 8 + 1 = 9, k_d = 1.
- **DEMFX** (literature lepton structure): quarks as DEMU; charged leptons and neutrinos as review
  eqs. (5.4)-(5.6). The leptons share no continuous parameter with the quarks, so this is a
  baseline (it cannot carry CKM information by construction). k_cont = 9; scored in NO and IO
  (ordering bit).

## (b) Four-zero Hermitian texture

**Fritzsch-Xing, "Four-zero texture of Hermitian quark mass matrices and current experimental tests",
Phys. Lett. B555 (2003) 63, arXiv:hep-ph/0212195.**
- Eq. (4): M_q = [[0, C_q, 0],[C_q*, B̃_q, B_q],[0, B_q*, A_q]]; eq. (5): M_q = P_q M̄_q P_q†, M̄ real
  with |C|, |B|; P_q = diag(1, e^{iφ_C}, e^{i(φ_B+φ_C)}).
- Eq. (6)-(7): eigenvalues λ_i with |λ_i| = m_i, λ3 > 0, A > 0, λ1λ2 < 0;
  B̃ = λ1+λ2+λ3-A (7a), |B| = sqrt((A-λ1)(A-λ2)(λ3-A)/A) (7b), |C| = sqrt(-λ1λ2λ3/A) (7c);
  η = λ2/m2 = ±1.
- Eq. (9): V = O_u^T P_u† P_d O_d with φ1 = φ_Cu - φ_Cd, φ2 = φ_Bu - φ_Bd.
- Eq. (10c), (11a): |V_cb| ≈ |sqrt((m_t-A_u)/m_t) - e^{iφ2} sqrt((m_b-A_d)/m_b)|; the paper notes
  φ2 ≈ 0 is needed for the cancellation that makes V_cb small.
- "Universal texture" literature for leptons with the same four-zero form: Fritzsch-Xing
  hep-ph/0601104 (PLB 634 (2006) 514; abstract read, equations not used); Ahuja-Gupta et al.
  (e.g. hep-ph/0703005, review arXiv:1206.3844; not read, cited as context only). Those lepton fits
  use sector-specific (2,2) elements and phases, i.e. they are not universal.

Implemented:
- **FZ4U** (universal): all four sectors M̄_f of eq. (5) with λ_f = (-η m1, η m2, m3) (η universal,
  1 bit), the sector's own m1/m2, m2/m3 fitted (2 per sector), and ONE shared shape
  a ∈ (0,1): A_f = (1-a) λ3 + a max(0, λ1, λ2) (so (λ3-A_f)/λ3 ≈ a in every sector). Phases: M_u and
  M_ν real, M_d and M_e carry the same P = diag(1, e^{iφ1}, e^{i(φ1+φ2)}) (declared d↔e, u↔ν
  correspondence; no bit). Neutrinos Dirac (same Hermitian form) or type-I with M_D of this form and
  M_R ∝ 1 (1 bit). k_cont = 8 + 3 = 11, k_d = 2, k = 13 = N - 2 for N = 15.
- **FZ4F** (diagnostic, not universal): a_f per sector, k_cont 14.
- A literal "geometric" variant (|B|² = B̃ A in every sector, FX's "roughly geometric" remark) was
  checked analytically and dropped: with c_ij ∈ {0,1} scaling it forces |λ1| ≈ |λ2| (it is GEO
  pattern 0b111010 below, so it is covered by part (c)).

## (c) GEO grammar

M_f[ij] = c_ij sqrt(μ_i^f μ_j^f) e^{i s_f φ_ij}, c_ij ∈ {0,1} on (11, 22, 33, 12, 13, 23):
64 patterns; μ^f = (μ1, μ2, 1) fitted per sector (8); shared phases φ_ij on the nonzero
off-diagonals, applied to d and e only (≤ 3); neutrino Dirac | seesaw. 128 grammar entries: 7 bits.
k = 8 + n_off + 7 ≥ 15 > N - 2 = 13 for every entry: the grammar fails compression by count;
part (c) only tests whether ANY universal pattern fits.

## How these differ from GEN (src.theoryspace)

GEN draws K = 2-3 shared complex operator matrices with Froggatt-Nielsen suppression and random O(1)
complex coefficients, GUT Clebsch factors and complex vev ratios; it has O(20-50) continuous
parameters per world and was only used as a training family for information tests, never fitted.
Here the matrices are Hermitian with a fixed texture (zeros, democracy or unit coefficients), the
only sector-specific inputs are each sector's own two mass parameters, the shape (a or the unit
coefficients) and the phases are common to all sectors, and every structure is fitted to data with
the official scorer.

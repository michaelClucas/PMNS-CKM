# Model-family menu for synthetic theory-space data

Each family is a generator `theta -> (CKM observables, PMNS observables[, mass ratios])` that can be sampled without limit. Measured data enters only as the target. Conventions and helper functions: `baselines.md`, `baselines.py` (PDG standard parametrization, `angles_from_matrix` extracts θ12, θ13, θ23, δ, J from any unitary).

Common output vector (all families): `[θ12^q, θ13^q, θ23^q, δ_q, J_q, θ12^l, θ13^l, θ23^l, δ_l, J_l]`, plus optional `[m_d/m_s, m_s/m_b, m_u/m_c, m_c/m_t, m_e/m_μ, m_μ/m_τ, Δm²21/Δm²31, m_lightest]`. Families with a GUT-scale definition output at M_GUT; the RG step (`rg_running.md`) maps to M_Z, or data is run up instead.

Parameter-count notation: **c** = continuous parameters, **d** = discrete choices. For a final candidate every fitted c and every data-selected d counts toward k (CLAUDE.md).

---

## F1. Shared rotation: U_PMNS = A(V_CKM) · Φ · X

**Why plausible.** In SU(5)/SO(10)/Pati-Salam the charged-lepton left rotation is tied to a quark rotation (U_e ~ V_d or V_d^*), so if the neutrino sector is diagonalized by a symmetry-fixed X, U_PMNS = (quark rotation)† · X. This is the mechanism behind QLC, KMS and the Smirnov hidden-sector models (baselines A1, A3, A4).

**Parameterization.**
- A(V) ∈ {V†, V*, V^T, V, V·(·)·V†} (5 orderings; Qin-Ma's sandwich is the last).
- Φ = diag(1, e^{iα}, e^{iβ}), α, β ∈ [0, 2π).
- X from either (a) a finite catalog: BM, TBM, GR1, GR2, HEX, the TM1/TM2 one-parameter families, columns/permutations of these (row and column permutations: 36 per pattern), representation-theory mixing matrices of small groups (A4, S4, A5, Δ(27), Δ(6n²) residual-symmetry matrices); or (b) low-complexity SU(3): X = exp(i Σ_k a_k λ_k), λ_k Gell-Mann, a_k ∈ {rational multiples of π with small denominators} or continuous with an L1 / description-length penalty.
- Optional power: V → V^p via matrix logarithm (p ∈ {1/2, 1, 2}), to test "lepton mixing = quark mixing raised to a power" ideas. Speculative; tag as such.

**Counts.** c = 2 (α, β) for catalog X; c = 2 + n_active for continuous X. d = 1 (ordering) + 1 (X choice) + 1 (permutation) ≈ log2(5 · |catalog| · 36) bits for look-elsewhere.

**Priors / sampling.** V_CKM: sample (θ's, δ_q) from the measured posterior (split-normal from `data/ckm.json`, or `src/haar.shifted_data_universe`), so CKM uncertainty propagates. α, β uniform. Catalog X uniform over entries. For fake universes: replace V_CKM by Haar-random (`src/haar.py`) and replace the PMNS target likewise.

**Outputs.** PMNS θ's, δ, J (CKM is input, not output). Cost: one 3×3 product, ~μs; no NN needed for the forward map, the NN is useful as the learned-X search (method 4) and as a classifier of "which X and ordering" (amortized inference).

**Sanity checks inside the family.** α = β = 0, A = V†, X = BM reproduces QLC-like numbers; A = V†, X = TBM with Φ chosen gives TBC; quick_status shows V Φ BM V† fits the three angles (Σpull² = 0.2, NO) but misses δ.

---

## F2. Charged-lepton Clebsch corrections on a symmetry-fixed U_ν

**Why plausible.** GUT Clebsch factors (1, 3, 1/3, 9/2, 6, …; Georgi-Jarlskog, Antusch-Spinrath 0902.4644) relate the charged-lepton mixing angles to the down-quark angles, so lepton mixing = (fixed pattern) corrected by CKM-sized rotations with a group-theory coefficient. TBC (θ13 = θ_C/√2), the solar sum rules and Cabibbo haze are all special cases.

**Parameterization.**
`U_PMNS = R23(c23 θ_cb, φ23)† · R12(c12 θ_C, φ12)† · U_ν(θ_ν, pattern)`, or with R13 included.
- c12, c23 ∈ Clebsch set C = {0, 1/3, 1/2, 1/√2, 1, √2, 2, 3, 9/2, 6} (discrete), optionally c continuous.
- φ12, φ23 phases.
- U_ν ∈ LO patterns (5) or group + residual + CP families (TM1/TM2 with angle θ_ν and CP-fixed phase; Feruglio-Hagedorn-Ziegler 1211.5560, King-Neder 1403.1758).

**Counts.** c = 2 (φ's) + 0-1 (θ_ν), d = 2 (Clebsch choices) + 1 (pattern) + 1 (ordering of R23/R12).

**Priors.** θ_C, θ_cb from data posterior (or from GUT-scale values after RG, see rg_running.md); φ uniform; θ_ν uniform on [0, π/2) for TM families; discrete choices uniform.

**Outputs.** PMNS θ's, δ, J; closed form exists for all outputs (sum rules), so symbolic regression should recover exact formulas. This is the best testbed for method 3 (neural-guided symbolic search) because the ground truth formulas are known for subcases.

---

## F3. GUT Yukawa textures with seesaw (SU(5), SO(10), Pati-Salam)

**Why plausible.** The only established framework where quark and lepton Yukawas are the same objects. Relations: SU(5) Y_e = Y_d^T × Clebsch; SO(10) 10 + 126 + 120: Y_u, Y_d, Y_e, Y_ν linear combinations of 2-3 symmetric/antisymmetric matrices (Babu-Mohapatra hep-ph/9209215, Bajc-Senjanović-Vissani hep-ph/0210207); type-II seesaw M_ν ∝ M_d − M_e; Haba-Yamada Y_D ∝ Y_d V_CKM^T (1801.02904).

**Parameterization (minimal SO(10) 10_H + 126_H, symmetric):**
- H (10) and F (126) complex symmetric 3×3: in the basis where H is real diagonal, 3 + 12 = 15 real parameters, plus vev ratios r, s (complex): Y_u = H + r F, Y_d = H + F, Y_e = H − 3F, Y_ν = H − 3 r F (standard 10 + 126 Clebsches; signs/normalization to be fixed against Babu-Mohapatra before use).
- Seesaw: type-I M_ν = −v² Y_ν M_R^{-1} Y_ν^T with M_R ∝ F (one scale), or type-II M_ν ∝ F.
- SU(5) light version: Y_d (complex 3×3, 18 real), Y_e = Clebsch-weighted Y_d^T, Y_u symmetric, M_ν from Weinberg operator with random κ (anarchic) or from texture.

**Counts.** c ≈ 15-20 (minimal SO(10)), up to ~40 (SU(5) generic). d: Higgs content, type-I vs II, texture zeros. Not compressive by itself; the payoff is a data-constrained posterior from which correlations (e.g. a CKM-PMNS ridge) can be extracted and distilled.

**Priors.** Log-uniform eigenvalue hierarchies (or Froggatt-Nielsen ε^n scaling) with Haar-random unitary factors; complex phases uniform. Masses must be fit too, so include mass ratios in the output.

**Outputs.** Full CKM + PMNS + mass ratios + Δm² ratio, at M_GUT. Forward cost: a few SVDs (~10-50 μs), plus RG if included (expensive: this is where a NN emulator pays off).

**NN method fit.** Methods 1 (emulator) and 2 (SBI / normalizing flows over ~20-dim θ). Expect multi-modal posteriors; use NPE/NLE with flows.

---

## F4. Modular flavour symmetry (Γ_N with one modulus τ)

**Why plausible.** One complex modulus τ fixes all Yukawas as modular forms, so a unified quark-lepton model has few continuous parameters (τ plus a handful of coefficients). Feruglio, "Are neutrino masses modular forms?", arXiv:1706.08749. Caveat: Kähler-potential terms not fixed by the symmetry weaken predictivity, Chen, Ramos-Sánchez, Ratz, arXiv:1909.06910.

**Parameterization.** τ in the fundamental domain, group N ∈ {2, 3, 4, 5} (S3, A4, S4, A5), weight and representation assignments (discrete), 3-8 real coefficients.

**Counts.** c = 2 (τ) + 3-8 coefficients; d = group + weights + reps (large discrete space, must be counted for look-elsewhere).

**Priors.** τ uniform in fundamental domain (or near fixed points ω, i, i∞); coefficients log-uniform O(1).

**Outputs.** Both sectors from the same τ; mass ratios too.

**NN method fit.** Emulator over τ is smooth and low-dimensional; good SBI target. Implementation of modular forms (Dedekind eta / Eisenstein series) is a day of work; medium priority.

---

## F5. Froggatt-Nielsen with SU(5)-correlated charges

**Why plausible.** In SU(5), 10 = (Q, u^c, e^c) and 5̄ = (d^c, L) share U(1)_FN charges, so the hierarchy of CKM and the charged-lepton/neutrino mixing pattern come from the same integer charges. Barger, arXiv:2512.13630 is a recent single-flavon example.

**Parameterization.** Charges q_10 ∈ Z³, q_5̄ ∈ Z³, q_1 ∈ Z³ (small integers 0-4), ε ∈ (0.1, 0.3), O(1) complex coefficients c_ij drawn log-normal(0, 0.3) with uniform phases. Y_ij = c_ij ε^{q_i + q_j}.

**Counts.** c = 1 (ε) plus all c_ij if fitted (≈ 40), d = charge assignment (~5^9). Predictions are distributions, not points; compression is weak unless the c_ij are marginalized (then k = 1 + charges).

**NN method fit.** Classification of charge assignments from observables (amortized). Useful mainly as a broad "hierarchy-only" null model that is more realistic than Haar anarchy.

---

## F6. Texture zeros + seesaw with Y_ν tied to a quark Yukawa

**Why plausible.** Two-zero textures of M_ν (Frampton-Glashow-Marfatia hep-ph/0201008) and quark textures (Ramond-Roberts-Ross hep-ph/9303320) each give GST-like relations; tying Y_ν = Y_u (SO(10)) or Y_D ∝ Y_d V^T (Haba-Yamada) makes PMNS depend on CKM through M_R.

**Parameterization.** Choose zero positions (d), nonzero entries complex with hierarchical priors, M_R diagonal (3 masses) or 2-zero textured.

**Counts.** c = 4-10, d = texture choice (7 viable two-zero M_ν textures per FGM, more for Yukawa pairs).

**NN method fit.** Enumerate textures, emulator per texture, SBI for M_R. Medium payoff.

---

## F0. Null families (needed for criterion 4)

- Haar-random U(3) for CKM and PMNS independently (`src/haar.py`).
- Data-shifted universes (`src/haar.shifted_data_universe`).
- Hierarchical null: F5 with random charges (realistic CKM hierarchy, no CKM-PMNS link).

---

## Ranking by expected payoff

| Rank | Family | Dim (c + d) | Forward cost | Why |
|---|---|---|---|---|
| 1 | F1 shared rotation | 2 + ~12 bits | μs | Directly tests the cleanest CKM-PMNS hypothesis; method 4 (learned X on SU(3) with complexity penalty) and exhaustive enumeration both feasible; known baselines (QLC, TBC, Smirnov) are inside it for sanity checks; look-elsewhere is countable. |
| 2 | F2 Clebsch corrections | 2-3 + ~8 bits | μs | Closed-form outputs, ideal for symbolic regression (method 3); GUT-motivated coefficients make k small; recovers TBC, TM1, solar sum rules as pipeline tests. |
| 3 | F3 GUT textures + seesaw | 15-40 | 10-50 μs (+RG) | Most physical; where emulators and SBI (methods 1, 2) are actually needed; can reveal correlations a human would not guess, but compression is hard to achieve. |
| 4 | F4 modular | 4-10 + large d | ~ms | Low continuous dimension, strong predictivity if Kähler issue ignored; implementation cost. |
| 5 | F6 texture zeros + seesaw | 4-10 + texture | 10 μs | Well explored in the literature; moderate novelty. |
| 6 | F5 FN charges | 1 + 9 integers (+ O(1) coeffs) | 10 μs | Good null/background model, weak for point predictions. |

## Recommendation for the first experiments

Start with **F1 as an exhaustive-plus-learned search with fake-universe calibration from day one**: enumerate A(V) × catalog X × permutations × (α, β) grid, score with the evaluator's χ² on (θ12, θ13, θ23, δ) for NO and IO separately, then train a small network to parameterize X ∈ SU(3) (Gell-Mann exponential) with an explicit description-length penalty, and run the identical pipeline on ≥ 100 Haar universes so the hit rate is known before anyone looks at the best real-data hit (c = 2, d ≈ 12 bits; the quick scan already shows V Φ BM V† matching the angles but missing δ, a concrete thing to check). In parallel, **F2 as the symbolic-regression testbed**: generate (θ_C, θ_cb, Clebsch, φ, θ_ν) → PMNS data, fit an emulator, distill with PySR, and require that it rediscovers TBC (s13 = s_C/√2), TM1 and the solar sum rule before trusting any new formula (c = 2-3, d ≈ 8 bits). Third, once the RG module exists, **F3 minimal SO(10) / SU(5) with NPE** (normalizing-flow posterior over ~20 parameters, masses included) to look for data-favoured ridges that tie δ_l or θ23 to CKM quantities, which then feed back into F1/F2 as new closed-form hypotheses.

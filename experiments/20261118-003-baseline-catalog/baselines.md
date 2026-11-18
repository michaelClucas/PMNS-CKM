# Baseline catalog: known CKM-PMNS and within-sector relations

Iteration 1 literature survey. Everything here is **known**; a search that rediscovers one of these is a pipeline sanity check, not a candidate (OVERSEER_PROMPT.md, breakthrough criterion 5).

## Conventions

- PDG standard parametrization for both matrices: `U = R23(θ23) · U13(θ13, δ) · R12(θ12)`, `U_e3 = s13 e^{-iδ}`, `J = Im(U_e2 U_μ3 U_e3* U_μ2*) = c12 s12 c23 s23 c13² s13 sin δ`. Majorana phases are not observables here.
- CKM angles: θ_C ≡ θ12^q (|V_us| = s12 c13), θ_cb ≡ θ23^q, θ_ub ≡ θ13^q, δ_q. PMNS angles: θ12, θ13, θ23, δ (no superscript).
- θ12^ν, θ23^ν: angles of the neutrino-sector rotation U_ν; θ12^e: charged-lepton correction, `U_PMNS = U_e† U_ν`.
- `baselines.py` implements every relation below (function names given in each entry) in this convention, with self-tests (`python baselines.py`).

## Parameter counting

Each entry gives:
- **N**: observables the relation predicts (inputs from the other sector are not counted as predictions).
- **k_c**: fitted continuous constants or unfixed coefficients.
- **k_d**: discrete choices that were selected by looking at data (pattern choice, sign, which angle maps to which, ordering of matrix factors). Counted 1 each when the alternative was live at the time of choosing, 0 when fixed by the model's structure before data. Where it is ambiguous I count it (the repo's strict rule).
- **k = k_c + k_d.** Breakthrough needs k ≤ N − 2.

## Rough status vs current data

The "pull" numbers are from `quick_status.py` (log: `quick_status.log`) using `data/` (NuFIT 6.1, IC24 with SK-atm unless stated; PDG 2026 CKM global fit; PDG masses), **no RG running**, one observable at a time, asymmetric 1σ, no correlations. They are a screening tool; the evaluator's χ² supersedes them. "NO"/"IO" = normal/inverted ordering.

Inputs as printed by the script: θ_C = 13.013°, θ_cb = 2.401°, θ_ub = 0.2156°, δ_q = 1.154 rad. NuFIT 6.1 IC24 NO: θ12 = 33.76 (+0.42 −0.41)°, θ13 = 8.62 ± 0.11°, θ23 = 43.29 (+0.96 −0.79)°, δ = 212 (+26 −36)°; IO: θ23 = 47.90°, θ13 = 8.65°, δ = 274°.

---

## A. Quark-lepton complementarity (QLC) family

### A1. QLC (solar and atmospheric)
- **Formula:** θ12 + θ_C = π/4; θ23 + θ_cb = π/4. Motivation: U_PMNS ≈ V_CKM† U_BM, so bimaximal minus Cabibbo-size rotations. (`qlc`)
- **N = 2** (θ12, θ23). **k_c = 0, k_d = 1** (the sign choice θ23 = π/4 − θ_cb vs π/4 + θ_cb is not fixed; the θ23 octant was unknown when proposed). k = 1. Fails k ≤ N − 2 (needs k ≤ 0).
- **Scale:** proposed as a high-scale (GUT) relation; Minakata-Smirnov discuss RG corrections.
- **Refs:** Raidal, "Prediction θ_C + θ_sol = π/4 from flavor physics", arXiv:hep-ph/0404046 (PRL 93, 161801). Minakata, Smirnov, "Neutrino mixing and quark-lepton complementarity", arXiv:hep-ph/0405088 (PRD 70, 073009). Parametrization dependence: Zheng, arXiv:1002.0919 (PRD 81, 073009). Systematic texture scan of extended QLC: Plentinger, Seidl, Winter, arXiv:hep-ph/0612169.
- **Status:** θ12 pred 31.99° vs 33.76° → pull −4.3σ (NuFIT 6.1 shrank the θ12 error to 0.42°). θ23 = 45° − θ_cb = 42.60°: −0.9σ (NO), −5.8σ (IO); θ23 = 45° + θ_cb = 47.40°: +4.3σ (NO), −0.5σ (IO). So the atmospheric relation picks a sign per ordering (one extra data-selected bit). The solar relation is excluded at low scale; running of θ_C (~0.9%, see rg_running.md) cannot recover 1.8°.
- **Historical note:** Chauhan, Picariello, Pulido, Torrente-Lujan used QLC + data to predict θ13 = 9 (+1 −2)°, arXiv:hep-ph/0605032, before θ13 was measured.

### A2. Self-complementarity (SC)
- **Formula:** θ12 + θ13 = π/4 (standard parametrization; other forms θ12^ν + θ13^ν = π/4). (`self_complementarity`)
- **N = 1, k_c = 0, k_d = 1** (which pair of angles and which of 9 parametrizations; 1305.4916 abstract says it works best in a non-standard parametrization, a data-selected choice). k = 1.
- **Refs:** Zheng, Ma, arXiv:1106.4040 (EPJ Plus 127, 7). Zhang, Ma, arXiv:1202.4258 (PLB 710, 630). Zhang, Zhang, Ma, "Quark-lepton complementarity and self-complementarity in different schemes", arXiv:1211.3198 (PRD 86, 093019). Qu, Ma, "New mixing pattern for neutrinos", arXiv:1305.4916 (PRD 88, 037301).
- **Status:** pred θ12 = 36.38° vs 33.76° → +6.2σ in the standard parametrization. Excluded in that form.

### A3. Tri-bimaximal-Cabibbo (TBC) / King-Mohapatra-Smirnov (KMS) relation
- **Formula:** sin θ13 = sin θ_C / √2, s12 = 1/√3, s23 = 1/√2 (King); KMS usually means just the θ13 part, |U_e3| ≈ sin θ_C/√2. Arises from U_PMNS = R12(θ_C)† U_TBM (charged-lepton Cabibbo rotation), giving θ13 = θ_C/√2 exactly for the s13 part. (`tbc_king`)
- **N = 3** (TBC) or **N = 1** (KMS θ13 only). **k_c = 0, k_d = 1** (choice of θ_C rather than another small angle like √(m_e/m_μ) or λ²) . k = 1.
- **Scale:** GUT scale in the Pati-Salam realisations King discusses.
- **Refs:** King, "Tri-bimaximal-Cabibbo mixing", arXiv:1205.0506. Mohapatra and Smirnov conjectured it earlier (per Ahluwalia, arXiv:1206.4779); **original Mohapatra-Smirnov reference not identified**. Rederived in a seesaw ansatz: Albergaria, Branco, Bastos, Silva-Marcos, arXiv:2304.06747. Same θ13 relation in an SU(5)×S4 model: Zhao, Zhang, arXiv:1402.5834.
- **Status:** pred θ13 = 9.16° vs 8.62 ± 0.11° → +4.9σ (NO), +4.7σ (IO), no running. With SM running θ_C grows ~0.9% to the GUT scale (Antusch-Hinze-Saad, arXiv:2510.01312), which moves the prediction the wrong way by ~0.08°. TBC's s12 = 1/√3 is the TBM value: +3.7σ. Low-scale TBC is excluded; the θ13 part survives only with extra O(λ²) corrections (extra parameters).

### A4. "U_PMNS = V_CKM† · X" constructions
- **Formula:** U_PMNS = V_CKM† · Φ · X, Φ = diag(1, e^{iα}, e^{iβ}), X ∈ {BM, TBM, GR, …} (Li-Ma; Ludl-Smirnov; Smirnov-Xu; Chu-Smirnov). Variant U_PMNS = V_CKM* · T* (Giarnetti-Marciano-Meloni, KMS form in 2304.06747). Variant V_CKM† U_PMNS V_CKM = U_TBM, i.e. U_PMNS = V_CKM U_TBM V_CKM† (Qin-Ma). (`ckm_dagger_x`, orders `Vdag_X`, `Vconj_X`, `V_X_Vdag`)
- **N = 4** (θ12, θ13, θ23, δ). **k_c = 2** (α, β; zero only if a model fixes them) , **k_d = 2** (choice of X among ~5 patterns, ordering of factors among ~3). k = 4, or k = 2 when the phases are fixed by the model.
- **Scale:** GUT scale in SO(10) + hidden-sector realisations (Ludl-Smirnov, Smirnov-Xu); low scale in empirical uses.
- **Refs:** Li, Ma, "Relations between quark and lepton mixing angles and matrices", arXiv:hep-ph/0504161. Qin, Ma, "A new relation between quark and lepton mixing matrices", arXiv:1106.3284. Ludl, Smirnov, "Lepton mixing from the hidden sector", arXiv:1507.03494. Chu, Smirnov, arXiv:1604.03977. Smirnov, Xu, arXiv:1803.07933 (PRD 97, 095030): only a few X viable, BM among them, S4 realisation with a δ range. Smirnov, "Neutrino mixing via the neutrino portal", arXiv:1905.00838. Giarnetti, Marciano, Meloni, "On quark-lepton mixing and the leptonic CP violation", arXiv:2407.02487. Koide, Nishiura, arXiv:0808.0370. Harada (QLC at NLO, bimaximal product), arXiv:1304.4526 (from API search listing).
- **Status (quick_status, angles only, best over (α, β) on a 5° grid):**
  - V_CKM† Φ X, any X except BM: Σpull² ≈ 4.5-4.9 (NO), 27-29 (IO); θ13 lands at ~8.83° (+1.9σ) for every X because θ13 ≈ θ_C · |X_μ3| = θ_C/√2-ish corrected by θ_ub.
  - V_CKM Φ X V_CKM† with X = BM: Σpull² = 0.20 (NO), 12.5 (IO), but predicts δ ≈ 82-86° vs 212° (NO). With δ included, it is ~3σ off in the NO δ direction (δ errors are non-Gaussian, evaluator must use the χ² profile). Flagged for the evaluator, not a candidate: k = 2 + 2 for N = 4.
  - Literature: Smirnov-Xu find BM-type X viable with δ in a predicted range; 2407.02487 says leading corrections are needed to fit data.

---

## B. Leading-order fixed patterns and their corrections

All have θ13 = 0, θ23 = π/4 at leading order; they differ in θ12. (`lo_pattern(name)`, `lo_matrix(name)`)

| Pattern | θ12 rule | sin²θ12 | Ref | Pull of s12² (NuFIT 6.1) |
|---|---|---|---|---|
| TBM | sin²θ12 = 1/3 | 0.3333 | Harrison, Perkins, Scott, arXiv:hep-ph/0202074 | +3.7σ |
| BM | θ12 = π/4 | 0.5 | Vissani, arXiv:hep-ph/9708483; Barger, Pakvasa, Weiler, Whisnant, arXiv:hep-ph/9806387 | +28.5σ |
| GR1 | cot θ12 = φ | 0.2764 | Kajiyama, Raidal, Strumia, arXiv:0705.4559; A5: Everett, Stuart, arXiv:0812.1057 | −4.9σ |
| GR2 | cos θ12 = φ/2 (θ12 = π/5) | 0.3455 | Rodejohann, arXiv:0810.5239; D10: Adulpravitchai, Blum, Rodejohann, arXiv:0903.0531 | +5.5σ |
| HEX | θ12 = π/6 | 0.25 | Albright, Dueck, Rodejohann, arXiv:1004.2798 | −8.9σ |

- **N = 3** (angles; δ undefined), **k_c = 0, k_d = 1** (pattern choice). All are excluded at leading order by θ13 ≠ 0 (θ13 = 8.6° measured). They survive only with corrections, below.
- Reviews: Altarelli, Feruglio, arXiv:1002.0211; King, Luhn, arXiv:1301.1340; Feruglio, Romanino, arXiv:1912.06028; Xing, arXiv:1909.09610. GR with A5: Ding, Everett, Stuart, arXiv:1110.1688.

### B1. Charged-lepton corrections, "Cabibbo haze", solar sum rules
- **Formula:** U_PMNS = R12(θ12^e, φ)† Φ U_ν with U_ν = R23(θ23^ν) R12(θ12^ν) (θ13^ν = θ13^e = 0). Exact sum rule (tau row untouched): `|U_τ1|² = sin²θ12^ν (1 − c13² c23²)`, hence
  `cos δ = [s12² s23² + c12² c23² s13² − sin²θ12^ν (1 − c13² c23²)] / (2 s12 s23 c12 c23 s13)`.
  Leading order (Antusch-King): θ12 ≈ θ12^ν + θ13 cos δ (our sign convention), and s13 = sin θ12^e sin θ23^ν. (`charged_lepton_corrected`, `solar_sum_rule_cosdelta`, `antusch_king_linear`)
- **N = 1** (δ, given the three measured angles) for a fixed θ12^ν. **k_d = 1** (θ12^ν pattern). k = 1. If θ12^e is set to θ_C ("Cabibbo haze"), it also predicts θ13 (= TBC, A3).
- **Refs:** Frampton, Petcov, Rodejohann, arXiv:hep-ph/0401206. Antusch, King, arXiv:hep-ph/0508044. Datta, Everett, Ramond, "Cabibbo haze in lepton mixing", arXiv:hep-ph/0503222 (PLB 620, 42); Everett, arXiv:hep-ph/0510256. Petcov, arXiv:1405.6006. Girardi, Petcov, Titov, arXiv:1410.8056 (EPJC 75, 345). Ballett, King, Luhn, Pascoli, Schmidt, "Testing solar lepton mixing sum rules", arXiv:1410.7573 (JHEP 12 (2014) 122).
- **Status (cos δ predicted from NuFIT 6.1 angles):** TBM −0.105 (NO) / −0.143 (IO); BM −1.27 / −1.49 (unphysical: BM + 1-2 charged-lepton correction cannot reproduce the angles); GR1 +0.29; GR2 −0.19; HEX +0.48. Data: NO δ = 212° (cos −0.85), IO 274° (cos +0.07). NO prefers none strongly given the wide δ errors; the evaluator must use the δ χ² profile. Girardi-Petcov-Titov reach the same qualitative ranking with older data.

### B2. Trimaximal TM1 / TM2 (atmospheric sum rules)
- **Formula:** TM1 keeps the first TBM column: `s12² = (1 − 3 s13²)/(3 c13²)` and `|U_μ1|² = 1/6` →
  `cos δ = (1/6 − s12² c23² − c12² s13² s23²)/(2 s12 s13 s23 c12 c23)`.
  TM2 keeps the second column: `s12² = 1/(3 c13²)`, `|U_μ2|² = 1/3` →
  `cos δ = (c12² c23² + s12² s23² s13² − 1/3)/(2 c12 c23 s12 s23 s13)`. (`tm1`, `tm2`)
- **N = 2** (θ12 and δ given θ13, θ23), **k_c = 0, k_d = 1** (TM1 vs TM2). Fails k ≤ N − 2 only by one unit; a model fixing θ13 too (e.g. Δ(6n²) with generalised CP) raises N.
- **Refs:** Albright, Rodejohann, arXiv:0812.0436; Xing, arXiv:hep-ph/0204049; Ballett, King, Luhn, Pascoli, Schmidt, "Testing atmospheric mixing sum rules", arXiv:1308.4314 (PRD 89, 016016). Realisations: A4/S4 semi-direct models (King-Luhn review 1301.1340); Δ(6n²) + CP: King, Neder, arXiv:1403.1758; discrete + CP: Feruglio, Hagedorn, Ziegler, arXiv:1211.5560.
- **Status:** TM1 s12² = 0.318 → +1.4σ; TM2 0.341 → +4.8σ (excluded by θ12). TM1 cos δ = −0.13 (NO; data cos δ = −0.85 at best fit) / +0.22 (IO; data +0.07). TM1 is the best-surviving LO-symmetry baseline; its δ prediction in NO (δ ≈ 97° or 263°) puts the 263° branch ~2σ above the NO best fit (212 +26°), taken naively; the δ profile is non-Gaussian. Evaluator to compute.

---

## C. Mass-mixing relations

### C1. Gatto-Sartori-Tonin (GST)
- **Formula:** |V_us| = √(m_d/m_s); refinement |V_us| = |√(m_d/m_s) − e^{iφ}√(m_u/m_c)| (Fritzsch-type texture). (`gst`)
- **N = 1, k_c = 0 (or 1 with φ), k_d = 0.** Holds for textures with a (1,1) zero.
- **Scale:** mass ratio is approximately scale independent (same scheme, same scale), so the relation can be tested at 2 GeV.
- **Refs:** Gatto, Sartori, Tonin, Phys. Lett. B 28 (1968) 128 (pre-arXiv). Texture context: Ramond, Roberts, Ross, arXiv:hep-ph/9303320.
- **Status:** √(m_d/m_s) = 0.2249 (PDG m_d, m_s at 2 GeV in data/) vs |V_us| = 0.2252; mass-ratio error ±0.0019, so consistent at ~0.2σ. Classic within-sector success, not a CKM-PMNS link.

### C2. Leptonic GST analogue
- **Formula:** θ12^e = arcsin √(m_e/m_μ) applied to a LO neutrino pattern; gives s13 = √(m_e/m_μ)/√2 for θ23^ν = π/4. (`lepton_gst`)
- **N = 1, k = 1** (k_d: pattern + choice of mass ratio).
- **Refs:** standard charged-lepton-correction literature (hep-ph/0401206, hep-ph/0508044); no single canonical paper identified, **attribution unverified**.
- **Status:** θ13 = 2.82° vs 8.62° → −53σ. Excluded; that failure is why Cabibbo-size (θ_C) rather than √(m_e/m_μ) corrections are used in A3.

### C3. Koide and Koide-type
- **Formula:** Q = (m_e + m_μ + m_τ)/(√m_e + √m_μ + √m_τ)² = 2/3. Koide-type neutrino version: same with neutrino masses, allowing sign flips of √m_i (Rodejohann-Zhang). (`koide_Q`, `koide_predict_m3`, `koide_neutrino_m1`)
- **N = 1** (m_τ from m_e, m_μ), **k_c = 0, k_d = 0** for leptons (the 2/3 was not fitted, though chosen by data in 1981-82; strict count: k_d = 1). For neutrinos: N = 1 (m_lightest), k_d = 1 (sign choice).
- **Scale:** holds for pole masses; it is *not* RG invariant (running masses break it), a standard criticism.
- **Refs:** Koide, Lett. Nuovo Cim. 34 (1982) 201; Phys. Lett. B 120 (1983) 161; Phys. Rev. D 28 (1983) 252 (pre-arXiv; refs as cited in Koide's review arXiv:0706.2534); Koide, arXiv:hep-ph/0506247 (review). Rodejohann, Zhang, "Extended empirical fermion mass relation", arXiv:1101.5525 (PLB 698, 152).
- **Status:** m_τ pred 1776.969 MeV vs 1776.932 (+0.085) MeV → +0.4σ. Neutrino version (NO): no solution with all + signs; with sign of √m1 flipped, m1 = 0.37 meV. Not a CKM-PMNS relation; included because the overseer's list requires it and because a mass-including search will rediscover it.

### C4. SU(5) and SO(10) Yukawa relations
- **Formulas:**
  - Minimal SU(5): Y_e = Y_d^T at M_GUT → y_e/y_d = y_μ/y_s = y_τ/y_b = 1. Implies the left-handed charged-lepton rotation equals the *right-handed* down rotation, which is unobservable in the SM, so no direct U_e = V_CKM statement. (`gut_yukawa_ratios("minimal_SU5")`)
  - Georgi-Jarlskog (45-dim Higgs): y_e/y_d = 1/3, y_μ/y_s = 3, y_τ/y_b = 1 → double ratio (y_μ/y_s)(y_d/y_e) = 9. (`gut_yukawa_ratios("GJ")`)
  - Alternative Clebsch ratios y_μ/y_s ∈ {9/2, 6}, y_τ/y_b ∈ {3/2, …} from higher-dim operators: Antusch, Spinrath.
  - SO(10) minimal (10 + 126): Y_ν related to Y_u; type-II seesaw gives M_ν ∝ M_d − M_e, so b-τ unification forces large θ23 (Bajc-Senjanović-Vissani).
- **N:** 3 mass ratios (SU(5), GJ); more in full SO(10) fits. **k_d = 1** (which Clebsch pattern) at minimum; full SO(10) fits have ≥ 10 continuous parameters, so k ≥ N there (not compressive).
- **Scale:** M_GUT (≈ 2 × 10^16 GeV MSSM, ≈ 10^15-10^16 SM). Strongly dependent on SUSY threshold corrections and tan β.
- **Refs:** Georgi, Jarlskog, Phys. Lett. B 86 (1979) 297 (pre-arXiv). Babu, Mohapatra, arXiv:hep-ph/9209215. Bajc, Senjanović, Vissani, arXiv:hep-ph/0210207. Antusch, Spinrath, arXiv:0902.4644 (PRD 79, 095004). GUT-scale running inputs: Antusch, Maurer, arXiv:1306.6879; Antusch, Hinze, Saad, arXiv:2510.01312.
- **Status:** minimal SU(5) y_μ/y_s = 1 is excluded by the observed lepton/quark mass ratios at any scale (textbook); GJ and the 9/2, 6 alternatives depend on SUSY thresholds and tan β per 0902.4644. Evaluator needs GUT-scale Yukawas from 2510.01312 tables before scoring.

### C5. Seesaw with Y_ν tied to quarks
- **Formula:** Y_D ∝ diag(y_d, y_s, y_b) V_CKM^T (Haba-Yamada), type-I seesaw; Y_ν = Y_u in SO(10).
- **N = 3-5** (δ, Majorana phases, m_lightest), k_c ≥ 3 (M_R entries). Not compressive alone.
- **Refs:** Haba, Yamada, arXiv:1801.02904 (PRD 97, 055018): only one relation survives, NO only.
- **Status:** per the paper, viable in NO only. Evaluator to recompute.

---

## D. Flavour-group model predictions (A4, S4, Δ(27), A5, Δ(6n²))

These are not single formulas but the groups that produce the patterns in B. For the catalog the testable output is the leading-order pattern plus its sum rule:

| Group | Typical LO pattern | Typical correction / sum rule | Refs |
|---|---|---|---|
| A4 | TBM (direct), TM2 (semi-direct) | atmospheric sum rule TM2 | Ma, Rajasekaran, arXiv:hep-ph/0106291; Altarelli, Feruglio, arXiv:hep-ph/0504165, arXiv:hep-ph/0512103; He, Keum, Volkas (quarks + leptons), arXiv:hep-ph/0601001 |
| S4 | TBM, BM, TM1 (semi-direct) | TM1 sum rule | Lam, arXiv:0804.2622; King-Luhn review arXiv:1301.1340 |
| Δ(27) | TBM / trimaximal | TM-type | de Medeiros Varzielas, King, Ross, arXiv:hep-ph/0607045 |
| A5 | GR1 | solar sum rule with GR1 | Everett, Stuart, arXiv:0812.1057; Ding, Everett, Stuart, arXiv:1110.1688 |
| D10 | GR2 | | Adulpravitchai, Blum, Rodejohann, arXiv:0903.0531 |
| Δ(6n²) + gCP | TM1 with discrete θ13 set, δ and Majorana phases fixed | | King, Neder, arXiv:1403.1758 |
| any finite G + CP | all angles + δ fixed up to one parameter | | Feruglio, Hagedorn, Ziegler, arXiv:1211.5560 |

Parameter count for a "group + residual symmetry + CP" model: k_c = 1 (one free angle θ), k_d = 1-3 (group, residual subgroups, CP choice), N = 4 (3 angles + δ). Group catalogs: Ishimori et al., arXiv:1003.3552.

---

## E. CKM-PMNS relations from the last ~5 years (arXiv API search, 2020-2026)

| Relation | Formula / claim | N, k (my count) | Ref | Status |
|---|---|---|---|---|
| KM-phase sum rule | δ_KM^q + δ_KM^l ≈ 0 in the original Kobayashi-Maskawa parametrization (`km_phase`) | N = 1, k_d = 1 (choice of KM parametrization among many) | Pan, Sun, Ma, He, arXiv:2003.09921 | quick_status: δ_KM^q = 88.9°; sum = −57° (NO, IC24), +5° (IO, IC24). With σ(δ) ≈ 25-35°, NO is ~2σ off, IO consistent. Parametrization-dependent, so a data-selected choice. |
| Zero 1-3 mixing hypothesis | θ13^e = θ13^ν = 0 (and analogues for quarks) in the flavour basis → sum rules predicting θ23 and δ from θ13 in GUT / Pati-Salam classes | N = 2, k_d ≥ 1 | Antusch, Hinze, Saad, arXiv:2205.11531 | per paper, specific θ23/δ predictions; evaluator to compute |
| U_PMNS = V_CKM* T* (seesaw) | T from a seesaw-like mass matrix of BM/TBM/GR type plus leading corrections | N = 4, k ≥ 3 | Giarnetti, Marciano, Meloni, arXiv:2407.02487 | per paper, fits with corrections |
| KMS from type-I seesaw ansatz | \|U_e3\| ≈ sin θ_C/√2 | N = 1, k_c ≥ 2 | Albergaria, Branco, Bastos, Silva-Marcos, arXiv:2304.06747 | see A3: −/+ 4.9σ at low scale |
| SO(2)_L × SO(2)_R flavour | common chiral SO(2) structure for quark and (Dirac) lepton mass matrices | qualitative | Xu, Zhang, arXiv:2304.05017 | no closed-form relation |
| Asymptotic safety | quark and lepton mixing structures from a UV completion with gravity | qualitative | Eichhorn, Gyftopoulos, Held, arXiv:2507.18304 | no closed-form relation |
| Single flavon ε fixed by lepton mass ratios | \|V_us\|, \|V_cb\|, \|V_ub\|, θ13^PMNS as powers of one ε | N ≈ 4, k_d large (integer charges chosen per element) | Barger, arXiv:2512.13630 (PRD 113, 013002); companion arXiv:2603.06934, arXiv:2603.11341 (from API listing only) | Froggatt-Nielsen powers have O(1) coefficients; strict k ≥ number of charges. Low compressive value. |
| Exponent pattern in mass ratios and mixings | CKM and PMNS as correlated outputs of one exponent structure | unclear | Baron, arXiv:2608.04266 (API listing only, **content not read**) | unknown |
| E8 claim that CKM and PMNS are not independent | qualitative | — | Wilson, arXiv:2407.18279 (API listing only) | no testable formula found in abstract |

Older but directly relevant: Raidal 2004 (A1), Li-Ma 2005 and Qin-Ma 2011 (A4), King 2012 (A3), Ludl-Smirnov 2015 and Smirnov 2019 (A4), Ke-Li "equalities among CKM and PMNS elements", arXiv:1506.02324 and arXiv:1412.0116 (from API listing; no explicit formula in abstract); Sharma, Chauhan et al. QLC-correlation-matrix predictions for θ23 and δ, arXiv:1511.02143, arXiv:1711.08796 (API listing).

---

## F. Null baseline: anarchy

- **Formula:** U_PMNS (and/or V_CKM) Haar-random in U(3). (`haar_unitary`; also `src/haar.py`.)
- **Refs:** Hall, Murayama, Weiner, arXiv:hep-ph/9911341 (PRL 84, 2572); de Gouvêa, Murayama, arXiv:1204.1249 (anarchy still consistent after θ13).
- **Role:** the "fake universe" generator of breakthrough criterion 4. Any CKM-PMNS relation must beat Haar-random pairs.

---

## Summary table

| ID | Relation | N | k_c | k_d | k | k ≤ N−2? | Quick status (NuFIT 6.1, no running) |
|---|---|---|---|---|---|---|---|
| A1 | QLC θ12 + θ_C = π/4, θ23 ± θ_cb = π/4 | 2 | 0 | 1 | 1 | no | θ12 −4.3σ |
| A2 | Self-complementarity θ12 + θ13 = π/4 | 1 | 0 | 1 | 1 | no | +6.2σ |
| A3 | TBC / KMS s13 = s_C/√2 | 3 (1) | 0 | 1 | 1 | yes (TBC) / no | θ13 +4.9σ, s12² +3.7σ |
| A4a | V_CKM† Φ X | 4 | 2 | 2 | 4 | no | angles Σpull² ≈ 4.5 (NO) |
| A4b | V_CKM Φ X V_CKM† (X = BM) | 4 | 2 | 2 | 4 | no | angles Σpull² 0.2 (NO), δ off by ~130° |
| B-TBM/BM/GR1/GR2/HEX | LO patterns | 3 | 0 | 1 | 1 | yes | all excluded (θ13 = 0) |
| B1 | Solar sum rule, fixed θ12^ν | 1 | 0 | 1 | 1 | no | cos δ predictions in text |
| B2-TM1 | TM1 | 2 | 0 | 1 | 1 | no | s12² +1.4σ; δ ~1-2σ |
| B2-TM2 | TM2 | 2 | 0 | 1 | 1 | no | s12² +4.8σ |
| C1 | GST | 1 | 0 | 0 | 0 | no (N too small) | +0.2σ |
| C2 | Leptonic GST | 1 | 0 | 1 | 1 | no | −53σ |
| C3 | Koide (leptons) | 1 | 0 | 0-1 | 0-1 | no | +0.4σ |
| C3' | Koide-type neutrino | 1 | 0 | 1 | 1 | no | m1 = 0.37 meV (sign-flipped) |
| C4 | SU(5) / GJ / Antusch-Spinrath ratios | 3 | 0 | 1 | 1 | yes | needs GUT-scale masses |
| C5 | Haba-Yamada Y_D ∝ Y_d V_CKM^T | 3-5 | ≥3 | 1 | ≥4 | no | NO only (paper) |
| D | Group + residual + CP | 4 | 1 | 1-3 | 2-4 | borderline | pattern-dependent |
| E1 | δ_KM^q + δ_KM^l = 0 | 1 | 0 | 1 | 1 | no | NO −57° (~2σ), IO +5° |
| E2 | Zero 1-3 hypothesis | 2 | 0 | ≥1 | ≥1 | no | per paper |
| F | Anarchy (null) | — | — | — | — | — | null model |

Count: 19 rows, 26 distinct named relations/patterns (counting each LO pattern and TM1/TM2 separately).

No known relation in this catalog meets k ≤ N − 2 *and* survives current data at low scale. The ones closest to useful as baselines for the evaluator are TM1 (A4/S4/Δ(6n²)), V_CKM† Φ X (Smirnov et al.), and GST.

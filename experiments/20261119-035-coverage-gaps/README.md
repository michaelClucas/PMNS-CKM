# 20261119-035 Coverage gaps: rephasing invariants, Dirac Y_ν = Y_u, Δm² ratio

## Hypothesis

Three relation classes that exp 031 listed as never tested contain a CKM ↔ PMNS (or quark ↔ neutrino-mass) relation that passes fit quality and compression:

- (a) relations between rephasing invariants: |U_ij|² vs |V_kl|, and J_ℓ vs J_q / λ^n;
- (b) Dirac neutrinos with Y_ν = Y_u up to GUT Clebsch factors (SO(10) 10 + 126bar structure, Dirac instead of seesaw);
- (c) the Δm² ratio r = Δm²21 / |Δm²3ℓ| as a target of quark and lepton mass ratios, and whether any `src/theoryspace.py` family predicts r from quark data.

## Method

All measured values come from `src.data` / `src.rg` loaders. All fits end in the official scorer (`src.evaluate`) with explicit `use=`, so the alias rules are enforced. Measured quantities that a relation consumes are fitted as nuisance parameters and scored, not passed as `inputs`, because their errors are not small next to the target's. Each such quantity adds 1 to N and 1 to k_cont.

### (a) Invariants (`invariants.py`, `invariants_null.py`)

The grammar was fixed before any scoring.

- **G1 moduli:** |U_ij|² = c |V_kl|^p.
  - (ij): all 9 PMNS elements.
  - (kl): the 6 off-diagonal CKM elements.
  - p ∈ {1/2, 1, 2}.
  - c ∈ C = {1, 2, 3, 1/2, 1/3, 2/3, 3/2}.
  - Size: 1134.
- **G2 Jarlskog:** J_ℓ = ± c J_q^p / λ^n.
  - p ∈ {1, 1/2, 1/3}.
  - n ∈ {0..6}.
  - c ∈ C.
  - Size: 294.
- **Total:** M = 1428 relations, 11 bits.

Each relation fixes one PMNS combination. It is imposed by eliminating one PMNS parameter.
- δ-free elements (|U_e1|², |U_e2|², |U_e3|², |U_μ3|², |U_τ3|²) eliminate one angle.
- The other four elements fix cos δ, and J_ℓ fixes sin δ. Both δ branches are fitted and the better one is kept, which costs +1 bit.

The other PMNS parameters and the CKM quantities (|V_kl|, or J_q and λ) are fitted.
- N = 4 + n_ckm (5 or 6).
- k_cont = 3 + n_ckm.
- dof = 1.
- k = k_cont + 11 or 12 bits.

There are two stages.
1. **Screen:** profile χ² over a 4D PMNS grid. The `src.data.pmns_delta_chi2` joint χ² is separable as T12 + T13 + T23-DCP; the grid matches `src.data` to 3e-14.
2. **Official refit:** `fit_and_evaluate(method="lbfgsb")` for every relation with χ²_screen < 9. A relation is kept if max |pull| < 2.

Descriptive look-elsewhere check: the same screen with 300 N1-type hierarchical fake CKMs per ordering, counting relations with χ²_screen < 4.

### (b) Dirac Y_ν = Y_u (`dirac.py`, `run_dirac.sh`)

The structure was read from arXiv:2511.16196 (Chen, Fang, Zhou), model M1, appendix eq. (12): renormalizable SO(10) with complex 10_H + 126bar_H, in the basis where Y_126 is real diagonal.

    Y_u = S + D,  Y_d = r2 S + e^{iω} r1 D,  Y_e = r2 S − 3 e^{iω} r1 D,  Y_ν = S − 3 D

S is complex symmetric (from 10_H) and D is real diagonal (from 126bar_H). In the paper the neutrinos are Majorana (M_R = v_R D, type-I seesaw); their eq. (4) has the same −3 Clebsch on the 126 and 120_15 terms. Here the seesaw is replaced by Dirac neutrinos, m_ν = Y_ν v_ν. The absolute scale needs a separate tiny vev and is not scored. The two Clebsch factors are generalised:

    Y_e = r2 S + c_e e^{iω} r1 D,   Y_ν = S + c_ν D,   c_e, c_ν ∈ {1, −3, 1/3, 9/2, 4/9, 6}

That is 36 pairs (6 bits). (−3, −3) is the paper's structure, and c = 1 gives Y_e = Y_d or Y_ν = Y_u.

- **k_cont = 17:**
  - D = diag(d1, d2, 1): 2. The overall scale cancels in every scored ratio. d_i ≥ 0 without loss of generality, because their signs rephase into S.
  - S: 6 moduli + 6 phases.
  - r1, r2, ω: 3.
- **N = 16:** CKM 4 (s12, s13, s23, δ), PMNS 4, mass ratios 7 (mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau), dm_ratio.
- **k = 17 + 6 bits = 23.**
- **Engine:** `src.yukawa.observables`. The Dirac U_ν is the left rotation of Y_ν. It is passed as the symmetric matrix U_ν* diag(y) U_ν†, whose Takagi factor is U_ν, so the PMNS and the singular values are the Dirac ones.
- **Running:** predictions are made at 2e16 GeV and scored with `scale={"mu": 2e16, "model": "SM"}` (src.rg translation). PMNS translation uses the src.rg Majorana κ running at m_lightest = 0. That running is negligible in the SM for a hierarchical spectrum, and the Dirac running (y_τ-driven) is also negligible. The quark and charged-lepton ratio running is what matters here.
- **Exact limit** (`dirac.py exact`): Y_ν = Y_u or Y_u^T, and Y_e = Y_d or Y_d^T, with any common factor. Y^T has the same singular values as Y, so every exact variant predicts:
  - mr_e_mu = mr_d_s, mr_mu_tau = mr_s_b, mr_b_tau = 1;
  - r = (y_c² − y_u²)/(y_t² − y_u²) ≈ (m_c/m_t)²;
  - irrespective of right-handed rotations.

  It is fitted with the four quark ratios free (k_cont 4, N 8).

### (c) Δm² ratio (`dmratio.py`)

The grammar was fixed before scoring.
- **Bases:** X = {mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, λ}.
- **G1:** r = c x^p, with p ∈ {1/2, 1, 3/2, 2} and c ∈ {1, 2, 3, 1/2, 1/3}. Size 140.
- **G2:** r = x^a y^b, with a, b ∈ {1/2, 1}. Size 84.
- **Two scales:** mass ratios at M_Z, or at 2e16 GeV (SM). At 2e16 GeV, r is scored through `scale=`.
- **Total:** M = 448 relations, 9 bits.

Scoring: use = dm_ratio plus the bases as fitted nuisances, so N = 1 + n_b, k_cont = n_b, dof = 1. Both orderings are scored.

Look-elsewhere is checked two ways:
- Šidák over M, with p_loc = P(a log-uniform target on [0.003, 0.3] lands within |z_best| σ_eff of the best relation);
- 5000 fake targets, log-uniform on the same range with the real relative error. The fake-target rate is the fraction whose best grammar member is at least as close.

Theory space: accepted draws (exp 025 filter, NO, up to 40k per family, max 400 raw draws per kept draw, seed 3500) of the `src.theoryspace` families. A gradient-boosted regressor maps quark-only features (log CKM sines, cos and sin δ_q, log of 4 quark mass ratios) to log10 r. Reported: held-out R², the conditional σ against the marginal σ (as Gaussian bits), and the conditional prediction at the measured quark point.

Cores: at most 4 in use at any time. Running times are in the logs.

## Parameter count

| class | N | k_cont | discrete bits | k | k ≤ N − 2? |
|---|---|---|---|---|---|
| (a) G1 moduli | 5 | 4 | 11 (+1 δ branch for the 4 δ-dependent elements) | 15-16 | no (needs ≤ 3) |
| (a) G2 J, n = 0 | 5 | 4 | 12 | 16 | no |
| (a) G2 J, n ≥ 1 | 6 | 5 | 12 | 17 | no (needs ≤ 4) |
| (b) Dirac SO(10) | 16 | 17 | 6 (+1 if ordering chosen) | 23 | no (k_cont > N, dof −1) |
| (b) exact Y_ν = Y_u | 8 | 4 | 0-2 (transpose choices) | 4-6 | yes, but fails fit (below) |
| (c) single base | 2 | 1 | 9 | 10 | no |
| (c) two bases | 3 | 2 | 9 | 11 | no |

A single relation among measured numbers can never be compressive here. It fixes one combination, and every other parameter it touches is fitted. Even a set of relations that fixes all four PMNS parameters (k_cont 0 + nuisances) would cost about log2 C(1428, 4) ≈ 38 bits.

## Result

### (a) Rephasing invariants (`run_invariants.log`, `invariants.json`, `run_invariants_null.log`)

| | NO | IO |
|---|---|---|
| screen χ² < 9 (of 1428) | 160 | 108 (109 refits incl. the sanity relation) |
| official refits with all pulls < 2σ | 141 (96 moduli, 45 J) | 91 (82 moduli, 9 J) |
| … and p_global > 0.05 (`fit_ok`) | 123 | 49 |
| `compression_ok` | 0 | 0 |
| kept in both orderings | 87 | |

Best official refits:

| relation | ord | χ² | p_global | max pull | N | k |
|---|---|---|---|---|---|---|
| J_ℓ = −½ J_q^{1/3} | NO | 0.005 | 0.95 | 0.07 | 5 | 16 |
| \|U_e1\|² = 3 \|V_us\| (c12² c13² = 3 sin θ_C) | NO | 0.008 | 0.93 | 0.09 | 5 | 15 |
| \|U_e1\|² = 3 \|V_us\| | IO | 0.008 (global 1.49) | 0.22 | 0.09 | 5 | 15 |
| \|U_μ1\|² = ½ \|V_cd\| | NO | 0.063 | 0.80 | 0.21 | 5 | 16 |
| J_ℓ = −J_q / λ⁴ | NO | 0.085 | 0.77 | 0.29 | 6 | 17 |
| J_ℓ = −J_q / λ⁴ | IO | 6.55 | 0.005 | 2.56 | 6 | 17 |
| \|U_τ2\|² = 2 \|V_ts\|^{1/2} | IO | 0.008 (global 1.49) | 0.22 | 0.09 | 5 | 16 |
| sanity: \|U_e3\|² = ½ \|V_us\|² (θ13 = θ_C/√2) | NO / IO | 23.55 / 20.26 | ≈ 0 | s13² +4.68 / +4.36 | 5 | 15 |

The sanity relation reproduces the known TBC/θ_C/√2 pull: +4.7σ, against exp 003's rough +4.9σ.

Two kinds of relation pass, both for structural reasons.
- **J relations.** δ is poorly measured, so J_ℓ ranges over roughly [−0.033, +0.01] at 2σ. Any J_ℓ target of magnitude up to ~0.02 passes, including tiny ones (CP-conserving, δ ≈ 180°).
- **Moduli relations.** 9 elements × 6 CKM moduli × 3 powers × 7 coefficients cover [0, 1] densely, with spacing comparable to the PMNS errors.

Descriptive N1 check (`invariants_null.py`, 300 hierarchical fake CKMs per ordering, screen χ² < 4):

| | real CKM | fakes (mean, 5/50/95%) | fakes with ≥ as many passing relations |
|---|---|---|---|
| NO | 116 | 116.9 (60, 122, 158) | 57% |
| IO | 73 | 83.5 (38, 89.5, 121) | 66% |

Every fake universe has passing relations (100%). The real CKM produces exactly as many passing invariant relations as a random hierarchical CKM.

Literature:
- θ13 = θ_C/√2 (|U_e3|² = |V_us|²/2): King, TBC (arXiv:1205.0506); Antusch-Gross-Maurer-Sluka (arXiv:1205.1051); a flavour GUT model (arXiv:1305.6612). Known, and fails here at +4.7σ.
- Rodejohann-Tanimoto-Watanabe (arXiv:1201.4936) note |U_e3|² ~ Δm²21/Δm²31 as an order-of-magnitude coincidence.
- No paper found with a J_ℓ = c J_q / λ^n relation. Related work:
  - Jarlskog (hep-ph/0507212) and arXiv:1102.2823 argue that QLC-type relations should be stated through invariants;
  - arXiv:1901.06146 compares quark and lepton J in a specific setting;
  - Tanimoto-Yamamoto (arXiv:1501.07717) predict δ_PMNS from the unitarity-triangle angle φ2 in an SO(10)×PS model. That is a phase relation through a model, not a J relation.

  No paper was found proposing |U_e1|² = 3|V_us| or the other passing moduli relations. The search covered arXiv/web and was not exhaustive.

### (c) Δm² ratio, grammar (`run_dmratio_grammar.log`, `dmratio_grammar.json`)

Data, from `src.data.load_dm_ratio`:

| ordering | r | relative error |
|---|---|---|
| NO | 0.02990 | 1.6% |
| IO | 0.03015 | 1.6% |

GUT → low factor for r in the SM is 1.000 (NO) and 1.001 (IO).

| | NO | IO |
|---|---|---|
| screen \|z\| < 2 (of 448) | 6 | 5 |
| official refits with all pulls < 2 | 6 | 5 |
| best | r = ½ m_μ/m_τ (M_Z): χ² 0.82, p_global 0.37, max pull 0.91, N 2, k 10 | r = ½ (m_c/m_t)^{1/2} (M_Z): χ² 0.47, p_global 0.16 (IO offset), max pull 0.63, N 2, k 10 |
| next | ½ (m_c/m_t)^{1/2}: χ² 1.61 | (m_d/m_s · m_s/m_b)^{1/2} = (m_d/m_b)^{1/2}: χ² 1.38, N 3, k 11 |
| p_loc of best (log-uniform target on [0.003, 0.3]) | 6.8e-3 | 5.1e-3 |
| Šidák over M = 448 | 0.95 | 0.90 |
| fake-target rate (≥ as close), 5000 fakes | 0.750 | 0.592 |
| fakes with a member at \|z\| < 1 | 0.75 | 0.76 |

Predictions:
- ½ m_μ/m_τ = 0.0294;
- ½ (m_c/m_t)^{1/2} = 0.0304 at M_Z, but 0.0289 when the relation is imposed at 2e16 GeV (NO pull 2.09, so it fails there);
- (m_d/m_b)^{1/2} = 0.0306.

The two good relations prefer opposite orderings. Their scale choice (M_Z or GUT) is a further data-selected bit, already inside the 9.

Literature: no paper found proposing r = m_μ/(2 m_τ) or r = ½ √(m_c/m_t). Related work:
- Lipmanov (hep-ph/0304207, hep-ph/0501248, hep-ph/0511266) relates r to charged-lepton mass-ratio hierarchies (a different functional form);
- Alikhanov (arXiv:2601.18781) proposes (√Δm²31 + √Δm²21)/(√Δm²31 − √Δm²21) = √2, a neutrino-only relation;
- Mohapatra (arXiv:1007.1633) discusses a model in which the neutrino mass ratio is tied to λ (abstract only read).

### (b) Dirac Y_ν = Y_u up to Clebsch (`dirac_exact.json`, `dirac_de_direct.jsonl`, `dirac_staged.jsonl`, `dirac_lower.jsonl`)

**Exact limit** (Y_ν = Y_u or Y_u^T, Y_e = Y_d or Y_d^T): fitted at 2e16 GeV with the four quark ratios free (N 8, k_cont 4).
- The prediction r = (y_c² − y_u²)/(y_t² − y_u²) sits far outside the NuFIT Δm² tables, so χ² = ∞ in both orderings.
  - With the data's GUT-scale m_c/m_t = 0.00328, r = 1.1e-5, against 0.030 measured.
  - Even the best fit, which pulls mr_c_t to +120σ, reaches only r = 8.6e-5.
- The other pulls at that point: mr_b_tau +112σ (prediction 1 at GUT, data 0.63), mr_s_b +202σ, mr_d_s −54σ.
- This holds for every transpose variant, because only singular values enter. With Y_e = Y_d, the PMNS is V_CKM† as well (s13² ~ 1e-5).
- **Dead analytically:** a Dirac Y_ν proportional to Y_u carries the up-quark hierarchy, (m_c/m_t)² ≈ 1e-5, into r.

**Two-operator SO(10) Dirac structure** (36 Clebsch pairs, k_cont 17, N 16, k 23):

| attempt | setup | result |
|---|---|---|
| plain DE through `fit_and_evaluate(method="de")` | default polish | stopped: > 10 min per pair, no output. Its unconditional Nelder-Mead polish allows up to 68k evaluations at 2.3 ms each. |
| capped DE on the official `chi2_soft(include_finite=True)` + official `evaluate` | paper pair (−3, −3), NO, popsize 8, maxiter 400, 103k evaluations, 565 s | never entered the NuFIT tables (χ² = ∞; soft objective 1.5e5; mr_s_b +202σ, ckm_s13 +184σ). Seeds 2 and 3 were lost (concurrent writes to one file corrupted them). |
| staged: quark-only DE (3 seeds; quark χ² 3e-9, 3e-6, 25) → L-BFGS-B on the full χ² for all 36 pairs + (−3, −3) IO | 3 starts per pair, maxfun 3000 | no pair enters the NuFIT tables. Finite part of the χ² at the end: 4.2e5 (best, (4/9, 1)) to 2.4e8; the paper pair NO 2.4e8. |
| lower bound: N = 12 finite subset (CKM 4 + 7 mass ratios + r, no PMNS), DE popsize 10 × maxiter 300 seeded with the quark-exact points | (−3, −3) seeds 1 and 2; (1, 1) | best χ² ≥ 2.7e5 (−3, −3), 1.9e6 (1, 1); largest pulls ckm_s12 −300σ, mr_s_b +200σ, mr_u_c +183σ |

The quark-exact points (quark χ² ≈ 0) carry lepton-ratio and r terms of order 1e5 or more. No search found a point that fits quarks and leptons together.

These are search failures in a hierarchical 17-dimensional landscape (log moduli with cancellations), not a demonstrated minimum. So fit quality is **not determined**: no acceptable point was found, and the best χ² found are upper bounds on the minimum. The verdict does not depend on this. k_cont = 17 > N = 16 (dof −1, p undefined), so the model cannot satisfy criterion 2 at any χ², and its PMNS is not predicted from the quark sector (more parameters than observables). Running: SM at 2e16 GeV via `scale=` throughout.

### (c) Δm² ratio, theory-space families (`run_dmratio_families*.log`, `dmratio_families*.json`)

Measured log10 r = −1.524 (NO). Quark-only features, gradient-boosted regressor, 80/20 split.

| family | accepted | held-out R² | σ marginal → conditional (dex) | bits | log10 r at measured quarks [68%] | measured r inside? | median neighbour distance (σ units) |
|---|---|---|---|---|---|---|---|
| GEN | 40000 | 0.026 | 0.698 → 0.689 | 0.02 | −2.11 [−2.79, −1.42] | yes | 2.8 |
| CTRL (no sharing) | 40000 | −0.001 | 0.737 → 0.738 | 0.00 | −1.73 [−2.59, −0.88] | yes | 2.4 |
| MN (minimal SO(10) type I) | 40000 | 0.232 | 0.552 → 0.484 | 0.19 | −2.48 [−2.82, −2.07] | **no** (too small) | 2.1 |
| FN (real SO(10) 10+126+120) | 10204 (acceptance-capped) | 0.067 | 0.768 → 0.742 | 0.05 | −1.76 [−2.58, −0.81] | yes | 2.9 |
| SU5 (SU(5)+CSD3) | 40000 | 0.000 | 0.401 → 0.401 | 0.00 | −1.93 [−2.39, −1.41] | yes | 3.2 |
| S4SO (S4×SO(10)) | 40000 | 0.000 | 0.582 → 0.582 | 0.00 | −1.95 [−2.64, −1.37] | yes | 1.9 |
| MOD (common-τ modular) | 18042 (acceptance-capped) | 0.143 | 0.559 → 0.518 | 0.11 | −0.82 [−1.22, −0.39] | **no** (too large) | 4.0 |
| F2 (Clebsch corrections) | 40000 | 0.000 | 0.780 → 0.780 | 0.00 | −1.64 [−2.59, −0.81] | yes | 1.8 |

F1 was not run. Its masses are drawn independently of the mixing, so it is zero by construction, as F2 is.

No family predicts r from quark data:
- The largest information is 0.19 bits (MN), and there the conditional prediction misses the measured r.
- In every family the measured quark point lies 1.8-4σ from the bulk of the accepted draws, so the conditional values are extrapolations.
- SU5 and S4SO have r set by lepton-only parameters (CSD3 / seesaw), as exps 022-023 found for PMNS.

## Verdict

No candidate in any of the three classes. The `src.nulls` protocol was not run for any class: it is gated on both fit quality and k ≤ N − 2, and no relation or model in this experiment has k ≤ N − 2.

- **(a) Rephasing invariants: no evidence; fails compression.**
  - 141 (NO) / 91 (IO) of 1428 relations have all pulls < 2σ, but every one has k = 15-17 against N − 2 = 3-4.
  - Hierarchical fake CKMs produce as many passing relations as the real CKM (NO 57%, IO 66% of fakes produce at least as many; every fake produces some).
  - The J relations pass because δ_ℓ is poorly measured. The moduli grammar covers [0, 1] densely.
  - The known θ13 = θ_C/√2 relation is in the grammar and fails at +4.7σ.
  - The class is uninformative at current precision.
- **(b) Dirac Y_ν = Y_u up to Clebsch: fails compression.**
  - Exact proportionality (Y_ν ∝ Y_u or Y_u^T, Y_e ∝ Y_d or Y_d^T) fails fit quality analytically: r ≈ (m_c/m_t)² ~ 1e-5 against 0.030, and m_b/m_τ = 1 at GUT against 0.63.
  - The SO(10) 10 + 126bar Dirac variant (arXiv:2511.16196 structure, 36 Clebsch pairs) has k_cont 17 > N 16, so it is not compressive and does not predict PMNS.
  - Its fit quality is undetermined: every search stayed at χ² ≥ 2.7e5 on the finite subset, which is search-limited.
  - Known model structure (Majorana version published), not a candidate.
- **(c) Δm² ratio: no evidence; fails compression.**
  - 6 (NO) / 5 (IO) of 448 grammar relations are within 2σ. The best are r = ½ m_μ/m_τ (NO, χ² 0.82) and r = ½ √(m_c/m_t) (IO, χ² 0.47).
  - Look-elsewhere removes them: Šidák p 0.95 / 0.90, fake-target rate 0.75 / 0.59.
  - The grammar has a member within 1σ of a random target 75% of the time.
  - Structurally, k = 10-11 for N = 2-3.
  - In theory space: no `src.theoryspace` family (GEN, CTRL, MN, FN, SU5, S4SO, MOD, F2) predicts r from quark data. The most is 0.19 bits (MN), and both families with any information (MN, MOD) miss the measured r.

Wording: "candidate" is not used for anything here, because nothing passes compression. No class is "excluded" except the exact Y_ν ∝ Y_u limit, which fails fit quality.

### src gaps found

- `src.evaluate` has no PMNS keys for |U_ij|² or J_ℓ. Relations among invariants have to be imposed by eliminating a PMNS parameter, as `invariants.py` does. A `pmns_J` / `pmns_Uij2` derived key, profiled from the NuFIT tables, would let them be scored directly.
- `fit_and_evaluate(method="de")` always runs a Nelder-Mead polish with up to 4000 × dim evaluations. For 17-parameter Yukawa models at about 2 ms per evaluation, that alone takes minutes per fit. An option to skip or cap the polish is needed.
- `src.yukawa.observables` has no Dirac-neutrino path. A Dirac Y_ν has to be converted to an equivalent symmetric matrix (U_ν* diag(y) U_ν†), and the Majorana phases it returns are then meaningless.
- `scale=` translation runs PMNS with Majorana κ RGEs. There is no Dirac option (negligible in the SM here, but not general).
- `src.data.pmns_delta_chi2` is scalar-only. A vectorized version would make grid profiles like the one here a single call.

## Files

- `invariants.py` → `run_invariants.log`, `invariants.json`: (a) grammar, grid screen, official refits (4 processes, 561 s).
- `invariants_null.py` → `run_invariants_null.log`, `invariants_null.json`: (a) descriptive N1 fake-CKM count.
- `dirac.py`: (b) model and exact limit. `python dirac.py exact` → `dirac_exact.json`; `python dirac.py <c_e> <c_nu> <ord> <seed> [maxiter]` is the capped direct DE → `dirac_de_direct.jsonl` (via `run_dirac.sh`, `run_dirac.log`).
- `dirac_staged.py`: (b) staged fit (`stage1`, `stage2`) via `run_dirac_staged.sh` → `run_dirac_staged.log`, `dirac_staged.jsonl`; N = 12 lower-bound fit (`lower`) via `run_dirac_lower.sh` → `dirac_lower.jsonl`. Stage-1 starts and per-pair outputs live in gitignored `artifacts/035/`.
- `dmratio.py`: (c) `grammar` → `run_dmratio_grammar.log`, `dmratio_grammar.json`; `families` → `run_dmratio_families.log`, `dmratio_families.json` (GEN … S4SO; the run was stopped after S4SO). MOD and F2 were run as `families MOD` / `families F2` in parallel (same seed) → `run_dmratio_families_MOD.log`, `_F2.log`, `dmratio_families_MOD.json`, `_F2.json`. Lines with a RuntimeWarning that carried a local path were removed from the family logs.

Run from the repo root with `.venv/Scripts/python experiments/20261119-035-coverage-gaps/<script>`, or from this folder with `../../.venv/Scripts/python`. Set `PYTHONIOENCODING=utf-8` on Windows.

Literature consulted (abstracts or HTML; no PDFs committed):
- arXiv:2511.16196, eqs. (4), (5), (12), read for the matrix structure;
- 1205.0506, 1205.1051, 1305.6612, 1201.4936, 1501.07717, 1102.2823, hep-ph/0507212, 1901.06146, 1007.1633, 2601.18781, hep-ph/0304207, hep-ph/0501248, hep-ph/0511266.

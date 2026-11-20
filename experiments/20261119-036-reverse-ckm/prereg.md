# Exp 036 pre-registration (hashed before any real-data evaluation; SHA-256 in run.log)

Question: does the measured PMNS (plus masses) carry information about the measured CKM, within a synthetic theory-space mixture? Reverse of exps 025/028: the CKM is the predicted variable, the PMNS the condition.

## Inputs (all via `src`)
- CKM: `src.data.load_ckm()` records `s12`, `s13`, `s23`, `delta_rad` (PDG 2026 global fit).
- Masses: `src.evaluate.mass_ratio_targets()` keys `src.theoryspace.MR_KEYS` (6 M_Z same-scale ratios), plus the Δm² ratio `src.data.load_dm_ratio("NO")["value"]`.
- PMNS: NuFIT 6.1 NO default analysis, centre `src.evaluate.pmns_best("NO")`, measurement shape `src.nulls.pmns_noise("NO", ·)`.
- NO only.

## Variables
- Target (CKM, 4D): c = (log10 s12, log10 s13, log10 s23, logit(δ_q / 2π)), δ_q in [0, 2π).
- PMNS context (5D): p = (logit s12², logit s13², logit s23², cos δ_ℓ, sin δ_ℓ).
- Mass context (7D): m = log10 of the 6 MR_KEYS ratios and log10 Δm² ratio.

## Oracle power (step 1, before training; no networks, no real-data statistic)
Link family (out of mixture): x = true PMNS (s12², s13², s23², δ_ℓ) ~ U(B), B = the N2 box (`src.nulls.n2_centre`: NuFIT NO 3σ ranges widened ×3 about their midpoints, δ uniform on [0, 360)). Measured CKM in units of its measurement errors: û = a g(x̃) + ε, ε ~ N(0, I4) (ε is the CKM measurement noise; any extra link noise is absorbed into a), x̃ = x mapped linearly to [−1, 1]⁴ (δ via (cos δ, sin δ) features), g = 4 random Fourier maps (8 features each, ω ~ N(0, 1.5²), φ ~ U[0, 2π)), each standardised to mean 0, s.d. 1 over U(B). Measured PMNS centre ŷ = x + o, o from `pmns_noise` (pure translation).
- I_link = I(û; x) with x ~ U(B), by Monte Carlo (prior sample K = 20000 for p(û)); a set by bisection to levels 1, 2, 4, 8, 12, 16 bits. I_meas = I(û; ŷ), same with p(û | ŷ) from 512 posterior points ŷ − o_j.
- Oracle statistic T_or(ŷ') = log mean_j N(û; a g(ŷ' − o_j), I) over 512 fixed offsets o_j (posterior points of the truth given centre ŷ'); world: ŷ, fakes: 2000 N2 centres ŷ_f. k = #(T_or(ŷ_f) ≥ T_or(ŷ)); detected iff `src.nulls.decide(k, 2000)` passes. 50 worlds per level, x ~ U(B).
- Perfect-link ceiling: a deterministic, invertible link known exactly (CKM pins x exactly). Then T_or(ŷ') = −Δχ²(x − ŷ' + t_real)/2 with the NuFIT NO tables (1D s12², s13² profiles + 2D (s23², δ) table, as `src.data.pmns_delta_chi2` joint mode, translated without octant mirror). 1000 worlds with x ~ U(B), plus 200 worlds with x = NuFIT best fit. Also the information ceiling I_max = I(x; ŷ) = log2 vol(B) − h(o) (differential entropy of the measurement shape, from the tables).
- Interpretation: the detection rate of the perfect link is the best any estimator can do under this null and rule at today's PMNS precision, whatever the CKM precision.

## Networks (step 2, kept small; the decision to keep them small is made here, before the oracle numbers are known, because exp 028 measured I(CKM; PMNS | m, f) ≤ 1 bit for every family except MN (3.75 bits), and mutual information is symmetric)
- Training data: exp 025's synthetic NO data `artifacts/025/data.npz` (exp 025 `gen.py` seed 25; `src.theoryspace` reproduces these draws, exp 033), families GEN, CTRL, F1, F2, MN (30,000 each). Excluded: FN, MOD (marginal held-out information in exp 028) and SU5, S4SO (near-delta samples around published points; SU5 gave negative held-out information in exp 028). This is a budget choice, not a data-selected one.
- Split 80/10/10 with a fresh permutation (seed 3601) over the 150,000 rows.
- Flows: zuko NSF, 4 transforms, hidden (96, 96), 8 bins. q1(c | p, m, f), q0(c | m, f), family one-hot appended to the standardised context. Adam 1e-3, batch 1024, ReduceLROnPlateau (0.5, patience 2), early stop after 5 epochs without > 1e-3 improvement in validation NLL, epoch cap 60, wall cap 15 min per flow, best validation state kept. Seeds 0, 1, 2. 1 torch thread per job, at most 3 jobs (3-core budget).
- Ensemble: per seed, mixture density Q(·) = (1/5) Σ_f q(· | ·, f) (equal fixed family weights; no classifier); ensemble log density = mean over seeds of log Q.

## Statistic
T = log2 mean_{i,j} Q1(c_i | p_j, m) − log2 mean_i Q0(c_i | m), with
- c_i, i = 1..16: CKM measurement points: c_0 = centre; c_1..15 = independent split-normal draws of s12, s13, s23, δ_q from their records (rng seed 3603).
- p_j, j = 1..128: PMNS posterior points p(centre − o_j), o_j = `pmns_noise("NO", rng)` with rng seed 3602 (fixed; the same o_j are applied to every fake's centre). s² clipped to (1e−6, 1 − 1e−6), δ wrapped.
- m: measured mass context, fixed.
Secondary (reported, not used): point statistic with c_0 and the centre only.
Note: under the N2 null the Q0 term is the same for the real data and every fake, so the test ranks how well each PMNS centre predicts the measured CKM.

## Null and rule
- N2 (`src.nulls.n2_universe` on `Universe.real("NO")`): real CKM and masses, NO PMNS centre from `n2_centre`; fake i uses rng `default_rng([36, 2, i])`, i = 0..1999. Δm² ratio unchanged.
- k = #(T_fake ≥ T_real). Pass iff `src.nulls.decide(k, 2000)["passed"]` (one-sided CP95 upper < 1%, k ≤ 11).
- Decision: pass → "candidate signal" (to be audited); fail → "no evidence that the PMNS carries CKM information within this mixture", with the oracle ceiling and injection power stated.

## Sanity and power
- Held-out information per family (test split, family known): E[log2 q1(c | p, m, f) − log2 q0(c | m, f)], 3-seed ensemble. CTRL should be ≈ 0.
- Injection (pipeline power): 20 GEN worlds drawn fresh from `src.theoryspace.sample_accepted("GEN", ...)` (rng `default_rng([36, 9, b])` per batch), keeping the first 20 whose true s12², s13², s23² lie inside B (real-like PMNS, so world and fakes are exchangeable without a link). Measurement noise: CKM sines and δ_q with split-normal noise of the real records' relative errors (sines) / absolute error (δ); mass log ratios with `mass_ratio_targets` log covariance (6 keys); Δm² ratio as drawn; PMNS centre = truth + one `pmns_noise` offset. Each world: same T, its own CKM and masses, N2 fakes `default_rng([36, 3, w, j])`, evaluated in blocks of 100, stop once k ≥ 12 (outcome then fixed). Report detection rate with two-sided CP95 interval.
- If time allows (secondary, reported separately): the same with 20 F2 worlds.

## Free parameters
None fitted to measured data. Choices: family set (5 inclusion bits), network size, statistic form, N2 box (fixed by `src.nulls`), oracle link family and levels, injection family and in-box condition.

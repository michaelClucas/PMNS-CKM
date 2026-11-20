# 20261119-032 Power of the exp 028 NN information test

## Hypothesis

Exp 028 found no evidence that the real CKM carries PMNS information beyond (λ, A): the real CKM sits at rank 0.52 among λ-matched fakes. That statement has no stated sensitivity. This experiment measures it: if the world had a CKM → PMNS link of known information content, how often would the unchanged exp 028 pipeline (trained flows, conv T, λ-matched N1 null, CP95 rule) flag it?

Expectation stated before running: the flows only know the CKM → PMNS dependences in their training mixture, so a link outside the mixture should be detected at about the false-positive rate whatever its size; links drawn from the mixture (MN in particular) should be detected sometimes.

## Pre-registration

Fixed before any injection is evaluated. Deviations are listed under "Deviations".

**Pipeline under test (unchanged from exp 028, NO).** Trained networks from `artifacts/028/models/` (NO q1, q0, c1, c0, seeds 0, 1, 2), equal family weights, ensemble = mean over seeds of log q, statistic T_conv = log2 mean_j q1(y_j | CKM, m) − log2 mean_j q0(y_j | m) over 256 PMNS points y_j = (measured best fit − o_j), with o_j the exp 028 offsets (`src.nulls.pmns_noise`, rng seed 2525), Δm² ratio fixed at the world's value. Code: exp 028 `analyze.py` functions (`models`, `all_family_logq`, `statistic`, `ckm_ctx` logic), imported, not copied. No retraining.

**Null and detection rule (exp 028's primary, equal weights).** For each pseudo-real world, 2000 λ-matched fakes: λ and A drawn from split normals centred on the world's measured λ, A with the real `lambda` / `A` record errors (`src.data.load_ckm`), ρ̄, η̄ ~ U[−1, 1] (`src.nulls.N1_PRIOR`), CKM from `src.evaluate.ckm_from_wolfenstein` (identical to `n1_universe(...).load_ckm()`; checked). Masses and PMNS are the world's. k = #(T_fake ≥ T_world). Detected iff `src.nulls.decide(k, 2000)` passes (one-sided CP95 upper < 1%, i.e. k ≤ 11). Fakes are evaluated in blocks of 100 and evaluation stops once k ≥ 12; the rule's outcome is then already fixed, so this is exact. The Dirichlet family-weight scan of exp 028 is not part of this rule (it could only lower the detection rate).
Secondary rule (reported, not used for the headline): the 300-fake rule (k = 0 among the first 300 fakes).

**Measurement noise at today's precision (every world).**
- CKM: truth (λ, A, ρ̄, η̄) plus split-normal noise with the real record errors.
- Mass ratios (6 exp 025 keys): log ratios plus multivariate normal noise with `src.evaluate.mass_ratio_targets` log covariance.
- PMNS: measured best fit = truth + one `src.nulls.pmns_noise("NO")` offset (NuFIT 6.1 likelihood shape), s²'s clipped to (1e−6, 1 − 1e−6), δ wrapped. Δm² ratio not noised (≈1-2% error, flows' scale is decades).

**Injection A: out-of-mixture (OOM) random smooth links.** World i at level I (rng `default_rng([32, code(I), i])`):
- CKM truth: λ, A = real central values; (ρ̄0, η̄0) ~ U[−1, 1]² (the null's prior, so with no link the world and its fakes are exchangeable).
- Masses: real central values. Δm² ratio: real.
- PMNS in exp 028 target space (logit s12², logit s13², logit s23², logit δ/2π): y = y_ref + g(ρ̄0, η̄0) + σ ε, ε ~ N(0, I4), y_ref = NuFIT NO best fit (`src.evaluate.pmns_best`).
- g: random smooth map, per output k: Σ_{m=1..8} w_km cos(ω_km · x + φ_km), w ~ N(0, 1), ω ~ N(0, 1.5² I2), φ ~ U[0, 2π]; each component standardised to mean 0, s.d. 1 over U[−1, 1]², then scaled by a = 0.3 (≈ ±30% in s13², ±0.06 in s12² and s23², ±20° in δ at 1σ). A fresh g per world. This family is not in the exp 025 mixture (no family generates PMNS as a smooth function of ρ̄, η̄ alone at fixed λ, A, masses).
- Information: I_link = I(y; ρ̄, η̄ | λ, A, m) in bits, with (ρ̄, η̄) ~ U[−1, 1]² (the N1λ reference prior). Computed exactly for the Gaussian channel by Monte Carlo over y (1000 draws, fixed common random numbers) and 151 × 151 grid quadrature (checked against 301 × 301: agrees to 0.01 bits up to ~11 bits) over (ρ̄, η̄) for p(y). σ is set by bisection so I_link = level. Also reported: I_meas, the same quantity after adding today's PMNS measurement noise approximated as Gaussian in target space (covariance from 4000 `pmns_noise` draws around the NuFIT best fit).
- Levels: 0 (no link: y = y_ref + g(x') + 0.3 ε with x' independent of the CKM; size check), 0.5, 1, 2, 4, 8, 12 bits. 20 worlds per level. (The 8 and 12 bit levels were added after a one-world smoke test, see Deviations.)
- Oracle reference (not the pipeline): the same rank test with the exact link density as statistic, T_or = log N(ŷ; y_ref + g(ρ̄, η̄), σ² I + Σ_meas), ŷ = measured best fit in target space, same 2000 fakes, same rule. It shows whether the injected information is detectable at all at today's precision.

**Injection B: in-mixture (IM) positive control.** Worlds drawn from the exp 025 generators (`gen.py`, `GENS[f]`, `features` filter, NO only) at their training prior, rng `default_rng([32, code, i])`, taking the first draw that also passes a "real-like" condition. CKM (Wolfenstein via `src.mixing.standard_to_wolfenstein`), masses, PMNS, Δm² ratio from the draw; measurement noise as above. 20 worlds per set:
- **F2**: λ ∈ (0.15, 0.3), (ρ̄, η̄) ∈ [−1, 1]² (inside the null's box), s12² ∈ (0.15, 0.5), s13² ∈ (0.005, 0.06), s23² ∈ (0.25, 0.75).
- **MN**: same conditions (≈ 3 accepted per 200k raw MN draws; the MN prior almost never gives real-like PMNS).
- **MNu**: MN with the CKM conditions only (PMNS unrestricted). Its PMNS is mostly far from the real values (median s12² ~ 6e−4), where NuFIT offsets are meaningless and clip at 0, so for this set the measurement noise and the 256 conv offsets are applied as target-space shifts (Gaussian with the measured covariance for the world's best fit; the exp 028 offsets mapped to target space at the NuFIT best fit for the conv points). This is the closest to "the training distribution", where the networks have the most data and MN has 3.75 bits of held-out CKM information.
The conditioning was chosen after the smoke test showed that unconditioned MN worlds have ρ̄, η̄ far outside the null's box (e.g. (21.7, −12.7)) and PMNS at s² ~ 1e−6 (see Deviations).

**Outputs.**
- Detection rate per level / family with two-sided CP95 intervals (`src.infogain.cp_interval`).
- Minimum detectable I_link at 50% power: linear interpolation of the detection rate in log2(I) between adjacent levels; "> max level" if 0.5 is never reached.
- Oracle detection rate per level.
- Median rank, median (T_world − median T_fake).

**Interpretation rule.** Exp 028's "no evidence" is restated as "no evidence for a link of ≥ X bits at 50% power", with X from injection A (outside the mixture) and the IM detection rates (inside the mixture). Nothing here is a candidate.

**Budget.** 5 single-thread worker processes (torch 1 thread each). Seeds and codes: code(I) = 100 + 10 I for OOM levels, 901 MN, 902 F2, 903 MNu; fake j of world i uses `default_rng([32, code, i, 1, j])`.

## Deviations

Item 0 was made before the pre-registration hash (`run.log`, SHA-256 prefix `86bb4b9ba9fade7d`, 2026-11-19T23:21:33Z; the README as hashed is `artifacts/032/README_prereg.md`). Items 1-4 came after.

0. Before the main run (this pre-registration was edited once, before hashing): a 3-world smoke test (OOM 2 bits world 0, IM MN world 0, OOM 0 bits world 0; results deleted, the main run regenerated these worlds from the same seeds with the final code; the information grid was also refined from 101² to 151² after a quadrature accuracy check) showed the oracle at rank 0.145 for a 2-bit link. A rank < 0.55% among 2000 fakes needs a likelihood ratio that only ~0.5% of the prior beats, i.e. roughly 7-8 bits of evidence for the real world's (ρ̄, η̄), so the originally planned 0.5-4 bit levels could not reach 50% power even with a perfect estimator. The conditional 8-bit level was made unconditional and a 12-bit level added. The same smoke test's MN world had ρ̄, η̄ = (21.7, −12.7) and s² ~ 1e−6 (clipped by the NuFIT offsets, giving NaN fake T values), so the IM sets were redefined as above (conditioned F2 / MN, plus MNu), and NaN T values are counted as ≥ the world's T (against detection). Nothing else changed.
1. **Incomplete OOM levels (deadline stop).** The machine ran at 100% CPU (other agents), so the 5 single-thread workers ran at ~0.5 s per fake context instead of ~0.2 s. Workers were stopped at the overseer's deadline (stop logged in `run.log`) with 173 / 200 worlds done: all 20 at OOM 12, 8, 4, 2 bits and all 60 IM worlds; OOM 1 bit 15 / 20, 0.5 bit 10 / 20, 0 bit (size check) 8 / 20. The missing worlds are at levels where even the oracle has 0 detections, so they cannot move the 50%-power point.
2. **Interim look.** `analyze.py` was run once at 16:52 local on the 75 worlds done then (all 60 IM, 15 OOM at 4-12 bits). It printed IM F2 0/20, MN 0/20, MNu 3/20, OOM 0 detections. Nothing was changed after it; worlds, seeds and the rule were fixed in advance and the run continued unchanged.
3. **Post-hoc diagnostic (not pre-registered).** `diag_rhoeta.py` measures how much PMNS information (ρ̄, η̄) carry inside the F2 and MN generators themselves, to explain the IM results (Result 3). It does not use the networks and changes no detection number.
4. **Rank of stopped worlds.** "Median rank" and "rank ≤ 0.05" use k / n_evaluated with sequential stopping at k = 12, so ranks of non-detected worlds are biased upward (most stop after 100 fakes). Detection itself is exact.

## Method (as run)

- `inject.py meas_cov`: covariance of today's PMNS measurement noise in exp 028 target space from 4000 `pmns_noise` draws (`artifacts/032/meas_cov.json`). s.d. of (logit s12², logit s13², logit s23², logit δ/2π) = 0.032, 0.027, 0.156, 1.10.
- `inject.py run <tasks>`: builds each world, applies measurement noise, evaluates the exp 028 ensemble (functions imported from exp 028 `analyze.py` / `train.py`, networks from `artifacts/028/models`, NO, seeds 0-2, equal weights) on the world and on its λ-matched fakes in blocks of 100, stops at k = 12, writes `artifacts/032/inj/<kind>_<label>_<i>.json`.
- `run_all.py 5 main`: 5 single-thread workers, round-robin over 200 tasks. `analyze.py`: aggregation → `results.json`, `analyze.log`.
- `diag_rhoeta.py 100`: Result 3 → `diag_rhoeta.json`, `diag_rhoeta.log`.
- Measured inputs only through `src` loaders (CKM records and errors, mass-ratio log covariance, NuFIT 6.1 best fit and Δm² values, `pmns_noise`). No retraining.

## Free-parameter count

Nothing is fitted to measured data; this measures the sensitivity of an existing pipeline. Choices that shape the answer, all fixed before the run: the OOM link family (8 random Fourier features, ω s.d. 1.5, amplitude 0.3 in target space, isotropic Gaussian link noise); the information levels; information defined against U[−1, 1]² for (ρ̄, η̄) at fixed λ, A, masses; the IM real-like windows (λ ∈ (0.15, 0.3), (ρ̄, η̄) in the box, s12² ∈ (0.15, 0.5), s13² ∈ (0.005, 0.06), s23² ∈ (0.25, 0.75)); 20 worlds per set. Levels 8 and 12 bits and the IM conditioning were set after the smoke test (Deviation 0).

## Result

### 1. Out-of-mixture links (random smooth PMNS = f(ρ̄, η̄) + noise)

| I_link (bits) | worlds | I_meas (bits, median) | pipeline detected | CP95 | oracle detected | oracle CP95 |
|---|---|---|---|---|---|---|
| 0 (no link) | 8 | 0 | 0 | [0, 0.37] | n/a | |
| 0.5 | 10 | 0.41 | 0 | [0, 0.31] | 0 | [0, 0.31] |
| 1 | 15 | 0.79 | 0 | [0, 0.22] | 0 | [0, 0.22] |
| 2 | 20 | 1.55 | 0 | [0, 0.17] | 1 | [0.001, 0.25] |
| 4 | 20 | 2.89 | 0 | [0, 0.17] | 6 | [0.12, 0.54] |
| 8 | 20 | 5.37 | 0 | [0, 0.17] | 10 | [0.27, 0.73] |
| 12 | 20 | 6.05 | 0 | [0, 0.17] | 13 | [0.41, 0.85] |

- I_link is the link's information before measurement (σ tuned by bisection; achieved to < 1e−6 bits). I_meas is the same after today's PMNS measurement noise (Gaussian approximation). It saturates near 6 bits: s23² and δ are poorly measured, and the link moves s12² and s13² by only 0.3 in logit.
- The 300-fake rule also gives 0 detections at every level.
- World T minus median fake T: −0.07 to +0.20 bits at every level. The networks see nothing.
- **Minimum detectable information at 50% power: pipeline > 12 bits (0 / 113 linked worlds detected). Oracle with the exact link density: 8 bits I_link (≈ 5.4 bits I_meas).** The oracle value is a floor for any estimator under this null and rule at today's precision: passing needs the world's (ρ̄, η̄) to beat ≥ 99.45% of the box, which takes roughly 7-8 bits of evidence about (ρ̄, η̄).

### 2. In-mixture worlds (drawn from the training families)

| set | worlds | detected | CP95 | 300-rule | median T − median T_fake (bits) | rank ≤ 0.05 |
|---|---|---|---|---|---|---|
| MNu (MN prior, CKM in the box, PMNS unrestricted) | 20 | **3** (k = 3, 1, 2 of 2000) | [0.03, 0.38] | 1 | +2.45 | 5 |
| MN, real-like PMNS | 20 | 0 | [0, 0.17] | 0 | −0.15 | 2 |
| F2, real-like PMNS | 20 | 0 | [0, 0.17] | 0 | −0.04 | 1 |

- MNu: 7 of 20 worlds needed more than 100 fakes (k between 1 and 20 at stop). The pipeline does see the MN link where MN has its training data, but only 15% of worlds clear the 1% rule.
- Real-like MN and F2: the world's T sits at the fakes' median.

### 3. Why real-like F2 and MN are not detected (post-hoc, `diag_rhoeta.py`)

For each base draw of the generator (100 per family), (ρ̄, η̄) is scanned over the null's box with λ, A and all other model parameters fixed. Oracle information I(PMNS_meas; ρ̄, η̄) at today's precision:

| family | bases | I_meas q10 / q50 / q90 (bits) | fraction ≥ 7 bits | median s.d. of logit s13² over the box |
|---|---|---|---|---|
| F2, all | 100 | 0.00 / 2.09 / 6.13 | 0.03 | 0.091 |
| F2, real-like PMNS | 27 | 0.00 / 1.64 / 6.07 | 0.07 | 0.077 |
| MN (CKM in box) | 100 | 6.05 / 7.08 / 8.32 | 0.54 | 1.02 |

- **F2.** (ρ̄, η̄) enter only through θ13^CKM ≈ 0.004 and δ_CKM (multiplied by θ13^CKM), times O(1) Clebsches. The PMNS shift is mostly at or below the s13² measurement error: median 1.6 bits for real-like worlds, against the ~8 bits the rule needs even for an oracle. F2's 0/20 is what any estimator would get. It does not show pipeline insensitivity.
- **MN.** (ρ̄, η̄) carry ~7 bits at the MN prior: the CKM enters the quark-anchored Md and, through F = (Mu/r − Md)/(s − 1), the seesaw. An oracle would detect about half of MNu-like worlds; the pipeline gets 15%. For real-like MN worlds the generator accepts ≈ 3 per 200k raw draws, so the training set has on the order of 20 MN samples in that region and the flows cannot have learned the link there. MN's 0/20 there is pipeline insensitivity (no training support), not absence of information. No real-like MN base appeared among the 100 diagnostic draws, so their oracle information was not measured directly.

## Verdict

Exp 028's null result has low power:
- **Outside the mixture:** no detection up to 12 bits of link information (0 / 113 linked worlds; CP95 upper ≤ 0.17 per complete level), while an oracle reaches 50% at 8 bits. Exp 028 says nothing about CKM → PMNS links that are not one of its training families.
- **Inside the mixture:** 15% power (3/20, CP95 [0.03, 0.38]) for typical MN-prior worlds; 0/20 (CP95 upper 0.17) for MN or F2 worlds whose PMNS looks like ours. For F2 that is the information limit at today's precision; for real-like MN it is missing training support.
- **Floor set by the null and rule:** any estimator needs ≈ 8 bits of (ρ̄, η̄) information (≈ 5.4 bits after today's PMNS noise) for 50% power with the λ-matched null and the 1% rule.

Suggested restatement of exp 028: "No evidence that the CKM carries PMNS information beyond (λ, A). Power: < 17% (95% CL) against smooth (ρ̄, η̄) links outside the training families up to 12 bits; ~15% against typical MN-family links; none for real-like F2 or MN worlds. Even a perfect estimator needs ≈ 8 bits of (ρ̄, η̄) information for 50% power under this null and rule." This is a statement about the instrument. Nothing here is a candidate.

## Caveats

- One OOM family (smooth random maps of (ρ̄, η̄) only). Links through λ or A are invisible to the λ-matched null by design and were not injected.
- I_meas uses a Gaussian approximation of the NuFIT noise (the real s23 / δ noise is bimodal).
- Real-like MN worlds come from a region with almost no training data, and their own (ρ̄, η̄) information was not measured.
- The 1, 0.5 and 0-bit levels are incomplete (Deviation 1). The size check has 8 worlds, 0 false detections (CP95 upper 0.37); exp 028's own fakes calibrate the false-positive rate more directly.
- MNu worlds use target-space measurement noise and conv offsets (pre-registered), not the physical NuFIT offsets used for every other set.
- `diag_rhoeta.py` draws λ, A uniformly in (0.15, 0.3) × (0.3, 2) rather than from each family's prior.

## Files

- `README.md` (pre-registration hash in `run.log`), `inject.py`, `run_all.py`, `analyze.py`, `diag_rhoeta.py`.
- `results.json` (per set and per world: T, k, fakes evaluated, detection, oracle rank, true Wolfenstein parameters), `analyze.log`, `diag_rhoeta.json`, `diag_rhoeta.log`, `run.log`.
- Gitignored `artifacts/032/`: `inj/*.json` (per world, with every evaluated fake T and (ρ̄, η̄)), `meas_cov.json`, `logs/main_worker*.log`, `README_prereg.md`.

To rerun from this folder with `../../.venv/Scripts/python`: `inject.py meas_cov`; `run_all.py 5 main` (~2.5 h on 5 loaded cores; resumable, finished worlds are skipped); `analyze.py`; `diag_rhoeta.py 100` (~3 min).

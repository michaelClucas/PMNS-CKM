# 20261119-034 Detectability ceiling for Cabibbo-scale CKM → PMNS links (plain N1 null)

## Hypothesis

Exp 032 measured the power of the exp 028 NN information test under the λ-matched null: information beyond (λ, A) is undetectable at today's PMNS precision below ≈ 7-8 bits, whatever the estimator. Links through λ and A were not tested, because the λ-matched null cannot see them by construction. The plain N1 null (λ, A, ρ̄, η̄ all from the N1 prior) is where a Cabibbo-scale link would show. This experiment measures, under the plain N1 null and exp 028's rule, the minimum link information that (a) the exp 028 pipeline and (b) an oracle using the exact link density can detect, today and at JUNO / DUNE / Hyper-K precision (oracle only).

Expectation stated before running: the oracle's 50%-power point is again several bits (the rule needs the world's CKM to beat ≥ 99.45% of the N1 box), and the pipeline detects out-of-mixture links at about the false-positive rate.

## Pre-registration

Fixed before any injected world is evaluated. The SHA-256 of this section (the text from the line "## Pre-registration" up to, not including, the line "## Deviations", UTF-8, as extracted by `prereg_hash.py`) is logged in `run.log` before the run starts. Deviations go under "Deviations".

**Pipeline under test (exp 028, NO, unchanged).** Networks `artifacts/028/models/` NO q1, q0, c1, c0, seeds 0-2, equal family weights, statistic T_conv over the world's measured PMNS best fit and 256 exp 028 offsets (`src.nulls.pmns_noise`, rng 2525), Δm² ratio fixed at the real value. Code: exp 032 `inject.py` (`Pipe`, `conv_points`, `ctx_from_w`, `measure`, `real_base`), which imports exp 028 `analyze.py` / `train.py`. Imported, not copied. No retraining.

**Null and detection rule (exp 028 secondary null as the primary here).** Plain N1: for each world, 2000 fakes with (λ, A, ρ̄, η̄) ~ U(`src.nulls.N1_PRIOR`) = U(0.05, 0.5) × U(0.3, 2) × U(−1, 1)², drawn in the `n1_universe` order, fake j of world i with `default_rng([34, code, i, 1, j])`; CKM context exact from `src.evaluate.ckm_from_wolfenstein` (as `n1_universe(...).load_ckm()`); masses and PMNS are the world's. k = #(T_fake ≥ T_world), NaN fakes count as ≥. Detected iff `src.nulls.decide(k, 2000)` passes (one-sided CP95 upper of the rank < 1%, i.e. k ≤ 11). Fakes in blocks of 100, stop once k ≥ 12 (the outcome is then fixed; exact).

**Worlds.** World i of set S at level ℓ uses `default_rng([34, code, i])`. Truth u = normalized (λ, A, ρ̄, η̄) ∈ [−1, 1]⁴, drawn uniform (the N1 prior, so without a link world and fakes are exchangeable). Masses: real central values. Measurement noise at today's precision exactly as exp 032 (`measure`): CKM split-normal with the real record errors, mass log ratios with the `mass_ratio_targets` covariance, PMNS with one `pmns_noise("NO")` offset (clipped).

**Link families** (PMNS in exp 028 target space y = (logit s12², logit s13², logit s23², logit δ/2π), y_ref = NuFIT 6.1 NO best fit):
- **L2**: y = y_ref + g(u_λ, u_A) + σ ε, g the exp 032 random smooth map (per output Σ_{m=1..8} w cos(ω·x + φ), w ~ N(0,1), ω ~ N(0, 1.5² I), φ ~ U[0, 2π], each output standardised over the prior box, amplitude 0.3), fresh g per world.
- **L4**: same with g(u_λ, u_A, u_ρ̄, u_η̄).
- **σ** set per world so that I_link = I(y; λ, A, ρ̄, η̄) under the N1 prior equals the level (exact Gaussian channel: Monte Carlo over 500 (u, ε) draws with common random numbers `[34, code, i, 7]`, p(y) by quadrature over the box: 151² grid for L2 (exp 032's), scrambled Sobol 2¹⁶ points (seed 3404) for L4; root found to 0.005 bits). Also reported: I_meas, the same after adding today's PMNS measurement noise as a Gaussian in target space (exp 032 `meas_cov.json`), and after the nominal and optimistic future noise below.
- **Levels**: 0.5, 1, 2, 4, 8 bits (pipeline and oracle), plus 12 bits (oracle only).
- **Named known-physics links** (exact, no link noise; everything not linked stays at y_ref): **K13** sin θ13 = λ/√2 (s13² = λ²/2); **KQLC** K13 plus quark-lepton complementarity θ12 = π/4 − θ_C (θ_C = arcsin λ). Truth from the N1 prior as above. **KR**: K13 and KQLC with truth at the measured λ, A, ρ̄, η̄ (central values via `src`), oracle only, 100 measurement-noise draws each: answers whether such a relation would be detected for our λ.
- **S0** (size check): y = y_ref + g(u') + 0.3 ε with u' independent of the CKM (pipeline only).

**Counts.** Pipeline + oracle: L2 and L4 at 5 levels × 20 worlds (i = 0-19), K13 and KQLC 20 each, S0 10 (250 worlds). Oracle only: i = 20-39 at the 5 levels, i = 0-39 at 12 bits, K13 / KQLC i = 20-59, KR 100 each. So the oracle has 40 worlds per level (60 for K13 / KQLC). Pipeline tasks run i-major (world i of every set before world i + 1); a deadline stop leaves equal counts.

**Oracle.** T_or(w) = log N(ŷ; mean(u(w)), σ² I + Σ) with mean the exact link (y_ref + g, or the named relation), evaluated at the world's measured CKM and at the same 2000 fakes; same k and rule. Four versions: today with the world's physical NuFIT noise (ŷ = target of the measured PMNS, Σ = today's Gaussian covariance; headline oracle), today Gaussian (ŷ = y_true + N(0, Σ_today)), nominal and optimistic future (ŷ = y_true + N(0, Σ_fut)). Σ_fut = D Σ_today D, D = diag(σ_fut / σ_today) in target space, σ_fut from exp 029 `sensitivities.json` linearised at the NuFIT best fit (nominal: s12² 0.0016, s13² current, s23² 0.010, δ 15°; optimistic: 0.0010, current, 0.006, 8°), each ratio capped at 1. CKM and mass noise unchanged (future CKM gains ignored; they only matter for (ρ̄, η̄), not λ, A).

**Outputs.** Detection rate per set and level with two-sided CP95 (`src.infogain.cp_interval`), pipeline and oracle (4 versions). 50%-power information: linear interpolation of the rate in log2(I_link) between adjacent levels ("> 12" / "> 8" if never reached), for the pipeline and each oracle version; median I_meas per level. Median (T_world − median T_fake) for the pipeline. For K13 / KQLC / KR: detection rates and I_meas.

**Interpretation rule.** The "minimal detectable Cabibbo-scale link" today = the oracle (today physical) 50%-power I_link for L2, with I_meas alongside; future = the nominal / optimistic oracle values. Exp 028's plain-N1 rank 0.38 is then read as "no evidence for a (λ, A) link above X bits at 50% power" with X from the pipeline (and the oracle floor). Nothing here is a candidate.

**Budget.** ≤ 5 cores: 2 pipeline workers × 2 torch threads, 1 oracle worker × 1 thread (`run_all.py`). Hard stop 2.5 h wall from the task start; unfinished worlds are reported as missing.

## Deviations

The pre-registration hash is in `run.log` (2026-11-20T02:11:22Z, SHA-256 `00b571ca1de5ea13929c88322de560df976a01bbb22db92d66f28b34cfcde6d8`; `python prereg_hash.py` reproduces it from this README). Item 0 came before the hash, items 1-4 after.

0. **Before hashing (design checks, no setting changed by them):** `ceiling.py qcheck` (`qcheck.log`): information at the tuned σ with 2¹⁸ instead of 2¹⁶ Sobol points (L4) or a 301² instead of 151² grid (L2) agrees to ≤ 0.01 bits at 8 bits and ≤ 0.08 bits at 12 bits. `ceiling.py time`: one L4 world. A pipeline smoke test on two worlds outside the task list (L4 3 bits i = 99, K13 i = 99; 200 fakes; output in a scratch folder, not used).
1. **Stopping check every 25 fakes instead of 100** (`EXP034_BLOCK=25`, `restart_pipe.py`). After 13 min only 9 / 250 pipeline worlds were done (~1 s per fake context on 2 threads), so the 2 pipeline workers were stopped and restarted with the finer check (the oracle-only worker kept running). The detection outcome does not depend on the block size (stop only once k ≥ 12, which already fixes "not detected"); worlds, fakes, seeds and rule are unchanged. The 9 finished worlds keep their 100-fake blocks. Effect: `n_fakes_eval` and the rank of non-detected worlds are coarser (rank = k / n_evaluated, biased upward for stopped worlds, as in exp 032).
2. **Third pipeline worker.** When the oracle-only worker finished (03:17Z) its core was given to a 1-thread pipeline worker running the same task list in reverse (`extra_pipe.py`). Still ≤ 5 cores. All 250 pipeline and 520 oracle-only worlds finished by 04:01Z, inside the 2.5 h budget.
3. **Interim look.** `analyze.py` was run once at 03:17Z (oracle-only worlds complete, pipeline 71 / 250). Nothing was changed after it.
4. **Post-hoc diagnostic** (`diag_weak.py`, `diag_weak.log`, not pre-registered): the oracle detects 5-10% of worlds at 0.5-2 bits, above the ~0.5% size, so its size and its expected rate at weak links were checked by redrawing the measurement 50 times for the 20 oracle-only worlds of four weak sets. Size with ŷ generated at an independent CKM: 0.006-0.009 (correct). With the link: L2 0.5 bits 0.010, L2 2 bits 0.067, L4 1 bit 0.033, L4 2 bits 0.115. So the weak-level detections are real oracle power: the per-world evidence is heavy-tailed (worlds near an extreme of g are identified much better than the average information suggests). It changes no number in the tables.

## Method (as run)

- `ceiling.py`: worlds, information and σ tuning (float32 quadrature, 6 bisection steps then regula falsi to 0.005 bits), oracle in four noise versions, the exp 028 pipeline through exp 032's `inject.py`. One JSON per world in `artifacts/034/pipe/` (pipeline + oracle) and `artifacts/034/orc/` (oracle only), with every evaluated fake T.
- `run_all.py 04:15` (driver, task lists, i-major order), then `restart_pipe.py` and `extra_pipe.py` (Deviations 1, 2). Worker logs in `artifacts/034/logs/`.
- `analyze.py` → `results.json`, `analyze.log`. `diag_weak.py` → `diag_weak.log`.
- Today's PMNS measurement covariance in target space is exp 032's (`artifacts/032/meas_cov.json`, 4000 `pmns_noise` draws; s.d. 0.032, 0.027, 0.156, 1.10 for logit s12², logit s13², logit s23², logit δ/2π). Future: s12² and s23² and δ errors from exp 029 `sensitivities.json` linearised at the NuFIT best fit, s13² unchanged.
- Measured inputs only via `src` (CKM records and errors, mass ratios and covariance, NuFIT 6.1 best fit, Δm² values, `pmns_noise`, `N1_PRIOR`).

## Free-parameter count

Nothing is fitted to measured data; this measures an instrument and an information floor. Choices that shape the answer, all fixed before the run: the random-link family (8 Fourier features, ω s.d. 1.5, amplitude 0.3, isotropic Gaussian link noise), information defined against the N1 prior box, the levels, the two named relations (sin θ13 = λ/√2, θ12 = π/4 − θ_C), the linearised Gaussian future noise, world counts. Not a candidate search, so no look-elsewhere count applies.

## Result

### 1. Random smooth links PMNS = f(λ, A) + noise (L2) and f(λ, A, ρ̄, η̄) + noise (L4), plain N1 null, 1% rule

Pipeline: 20 worlds per level. Oracle: 40 worlds per level (the 20 pipeline worlds plus 20). Two-sided CP95 in brackets. I_meas is the median information left after today's PMNS noise (Gaussian approximation).

| set | I_link (bits) | I_meas today | pipeline detected | median T_w − T_fake (bits) | oracle today (physical noise) | oracle today (Gaussian) |
|---|---|---|---|---|---|---|
| L2 | 0.5 | 0.41 | 0/20 [0, 0.17] | +0.10 | 2/40 [0.01, 0.17] | 1/40 |
| L2 | 1 | 0.78 | 0/20 [0, 0.17] | −0.16 | 4/40 [0.03, 0.24] | 0/40 |
| L2 | 2 | 1.54 | 0/20 [0, 0.17] | −0.09 | 2/40 [0.01, 0.17] | 2/40 |
| L2 | 4 | 2.99 | 1/20 [0.00, 0.25] | −0.09 | 5/40 [0.04, 0.27] | 4/40 |
| L2 | 8 | 5.29 | 0/20 [0, 0.17] | −0.16 | 24/40 [0.43, 0.75] | 20/40 |
| L2 | 12 | 6.15 | (not run) | | 23/40 [0.41, 0.73] | 24/40 |
| L4 | 0.5 | 0.41 | 0/20 [0, 0.17] | +0.00 | 2/40 [0.01, 0.17] | 2/40 |
| L4 | 1 | 0.77 | 0/20 [0, 0.17] | −0.12 | 4/40 [0.03, 0.24] | 5/40 |
| L4 | 2 | 1.48 | 0/20 [0, 0.17] | +0.17 | 6/40 [0.06, 0.30] | 6/40 |
| L4 | 4 | 2.77 | 0/20 [0, 0.17] | +0.29 | 6/40 [0.06, 0.30] | 6/40 |
| L4 | 8 | 5.02 | 0/20 [0, 0.17] | −0.08 | 17/40 [0.27, 0.59] | 17/40 |
| L4 | 12 | 6.56 | (not run) | | 28/40 [0.53, 0.83] | 31/40 |

- **Pipeline: 1 / 200 random-link worlds detected** (pooled CP95 [0.0001, 0.028]); the one hit (L2 4 bits, world 12, k = 2 / 2000) is a world the oracle does not detect (oracle rank 0.010). This is the false-positive rate, not power (the rule's size is 12 / 2001 ≈ 0.6% for a uniform rank; S0 size check 0/10). World T minus median fake T is within ±0.3 bits at every level.
- **Oracle 50%-power I_link** (interpolated in log2 I): L2 **6.9 bits** with today's physical noise (8.0 Gaussian), L4 **8.9 bits** (8.7 Gaussian). At that point I_meas ≈ 5 bits.
- **Saturation.** Today's PMNS precision caps I_meas near 6-6.5 bits (s23² and δ are poorly measured), so oracle power plateaus: L2 is 0.60 at 8 bits and 0.58 at 12 bits. A generic Cabibbo-scale link is detected by a perfect estimator at most ~60-70% of the time today, however strong it is.

### 2. Named links (exact, no link noise)

| set | worlds | I_meas today | pipeline detected | oracle today (physical) | oracle today (Gaussian) | oracle nominal future | oracle optimistic future |
|---|---|---|---|---|---|---|---|
| K13 sin θ13 = λ/√2, λ from the N1 prior | 20 / 40 | 5.20 | 0/20 [0, 0.17] | 17/40 [0.27, 0.59] | 16/40 | 13/40 [0.19, 0.49] | 18/40 [0.29, 0.62] |
| KQLC (K13 + θ12 = π/4 − θ_C) | 20 / 40 | 5.47 | 1/20 [0.00, 0.25] | 12/40 [0.17, 0.47] | 14/40 | 26/40 [0.48, 0.79] | 34/40 [0.70, 0.94] |
| KR-13 (K13 at the measured λ, A) | 100 (oracle) | 5.20 | | 49/100 [0.39, 0.59] | 34/100 | 43/100 [0.33, 0.53] | 40/100 [0.30, 0.50] |
| KR-QLC (KQLC at the measured λ, A) | 100 (oracle) | 5.47 | | 53/100 [0.43, 0.63] | 32/100 | 56/100 [0.46, 0.66] | 71/100 [0.61, 0.80] |

- An exact sin θ13 = λ/√2 relation carries ≈ 5.2 bits about λ after today's s13² error and is detected by the oracle about half the time (median oracle rank 0.007-0.009, against a 0.0055 threshold). No planned experiment improves s13², so its future power does not change.
- Adding exact QLC brings in s12², which JUNO improves by ×4-7: oracle power goes from ~0.3-0.5 today to 0.56-0.65 (nominal) and 0.71-0.85 (optimistic).
- Pipeline: 0/20 for K13; 1/20 for KQLC (world 0, λ = 0.075, k = 3 / 2000, while the oracle gives rank 0.037 for that world). With 2 hits in 240 linked worlds at the rule's size, neither hit is evidence of pipeline power.

### 3. Future precision (oracle only, Gaussian noise; 50%-power I_link)

| set | today (Gaussian) | nominal (JUNO 6 yr, Hyper-K / DUNE) | optimistic (JUNO 20 yr, DUNE + Hyper-K) |
|---|---|---|---|
| L2 f(λ, A) | 8.0 bits | **5.2 bits** (I_meas ≈ 4.5) | 5.6 bits |
| L4 f(λ, A, ρ̄, η̄) | 8.7 bits | **4.0 bits** (I_meas ≈ 3.6) | 5.1 bits |

The optimistic column is not below the nominal one: the difference is within the binomial noise of 40 worlds per level (rates at 4 bits: 12/40 vs 11/40 for L2, 20/40 vs 14/40 for L4). At 12 bits future power reaches 0.93-1.0 (no plateau).

### 4. Descriptive (not pre-registered)

Over all 250 pipeline worlds, corr(pipeline rank, world λ) = −0.14. Only 2 worlds needed all 2000 fakes (the two detections).

## Interpretation

- **Exp 028's plain-N1 rank 0.38.** Against every (λ, A) or (λ, A, ρ̄, η̄) link injected here, the exp 028 pipeline performs at its false-positive rate: 1/200 random links up to 8 bits, 0/20 exact sin θ13 = λ/√2, 1/20 exact QLC + θ13. So a rank of 0.38 is what the pipeline returns whether or not a Cabibbo-scale link exists. It is not evidence against such links; it only says the real CKM is not special for the networks' learned (in-mixture) dependence.
- **Exp 025's plain-N1 rank 0.021** (conv, under-converged single flows) is also far from the 1% rule (needs < 0.0055) and was not reproduced by exp 028. Given that the better-trained pipeline has no measurable power against out-of-mixture λ links, exp 025's excess cannot be read as a link signal either; exp 025 itself attributed it to the hierarchy prior of the training families.
- **Minimal detectable Cabibbo-scale link today**, plain N1 null, 1% rule, perfect estimator: ≈ **7 bits** of link information about (λ, A) (6.9 bits L2, physical noise; ≈ 5 bits after measurement), with power saturating at ~60% because today's PMNS errors cap the usable information near 6 bits. A named exact relation such as sin θ13 = λ/√2 sits right at the threshold (≈ 50% power at our λ). With the exp 028 pipeline: not reached up to 8 bits (> 8 bits).
- **At JUNO / DUNE / Hyper-K precision** (oracle): ≈ **4-5 bits** for 50% power, and power for strong links approaches 1. The gain comes from s12², s23² and δ; links that act only through θ13 gain nothing.

## Verdict

No evidence for a Cabibbo-scale CKM → PMNS link, and the exp 028 instrument could not have found one: its power under the plain N1 null is at the false-positive level (2/240 linked worlds; CP95 upper ≤ 0.17 per level, 0.028 pooled over random links) for links up to 8 bits and for exact θ13 = λ/√2 and QLC relations. The information floor, independent of the estimator, is ≈ 7 bits (50% oracle power) today, with a ~60% power plateau, and ≈ 4-5 bits at JUNO / DUNE / Hyper-K precision. Exp 028's plain-N1 rank 0.38 and exp 025's 0.021 are therefore uninformative about λ-scale links. Nothing here is a candidate.

## Caveats

- One random-link family (smooth Fourier maps, amplitude 0.3 in target space, isotropic link noise); named links are only two relations through λ.
- Information is defined against the N1 box (λ ∈ (0.05, 0.5), A ∈ (0.3, 2)), so "bits" are relative to that broad prior; a narrower reference prior would give fewer bits for the same link.
- Future noise is Gaussian, linearised at today's best fit, CKM errors unchanged; today's oracle uses physical NuFIT offsets but a Gaussian density (non-Gaussian s23 / δ noise is the reason physical and Gaussian versions differ, e.g. KR-13 49% vs 34%).
- 40 oracle worlds per level: 50%-power points have roughly ±1-2 bits uncertainty from interpolation and binomial noise.
- Pipeline ranks of non-detected worlds are from sequential stopping (Deviation 1); detection is exact.

## Files

- `README.md` (pre-registration hash in `run.log`; `prereg_hash.py` recomputes it), `ceiling.py`, `run_all.py`, `restart_pipe.py`, `extra_pipe.py`, `analyze.py`, `diag_weak.py`.
- `results.json` (tables, per-world truth, measured CKM, I_link, I_meas, oracle ranks, pipeline T / k), `analyze.log`, `qcheck.log`, `diag_weak.log`, `run.log`.
- Gitignored `artifacts/034/`: `pipe/*.json`, `orc/*.json` (per world, with every evaluated fake T), `logs/*.log`.

To rerun from this folder with `../../.venv/Scripts/python` (needs exp 028 networks in `artifacts/028/models/` and exp 032's `artifacts/032/meas_cov.json`, which `ceiling.py` regenerates if missing): `ceiling.py qcheck`; `run_all.py <HH:MM UTC deadline>` with `EXP034_BLOCK=25` in the environment (~1.8 h on 5 cores; resumable); `analyze.py`; `diag_weak.py` (~1 min).

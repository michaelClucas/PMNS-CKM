# 20261119-028 Decisive NN information test: does the real CKM carry PMNS information beyond (λ, A)?

## Hypothesis

Exp 025 found a ~1.5-bit, ~2σ preference for the real CKM over broad N1 fakes, carried mostly by the size of λ. This experiment asks whether anything is left once the fakes share the real Cabibbo scale: within the exp 025 theory-space mixture, does the real CKM (λ, A, ρ̄, η̄) predict the real PMNS better than CKMs with the real λ and A but random ρ̄, η̄?

## Pre-registration

Fixed before any evaluation on real data. Nothing below is changed after the first real-data number is computed; any deviation is listed under "Deviations" with a reason.

**Data.** Exp 025 synthetic mixture (`artifacts/025/data.npz`, 9 families × 30,000, normal ordering only, exp 025 `gen.py` seed 25), exp 025 split (seed 2501, 80/10/10). No regeneration of NO data.

**Estimator.** Family-conditional flows q1(y | CKM ctx, mass ratios, f) and q0(y | mass ratios, f) (zuko NSF, exp 025 architecture and target map), plus family classifiers c1 = p(f | CKM ctx, mass ratios) and c0 = p(f | mass ratios) (MLP). For family weights w (a point on the 9-simplex), the mixture predictive is

q_w(y | c) = Σ_f π_f(c; w) q(y | c, f),  π_f(c; w) = w_f p(f | c) / Σ_g w_g p(g | c)

(training mixture is equal per family, so p_train(f) is constant and cancels). Equal weights w = 1/9 reproduce exp 025's target density. This decomposition lets one set of trained networks evaluate any family weighting without retraining.

**Training.** 5 seeds (0-4) for each of q1, q0, c1, c0. Adam 1e-3, batch 1024, ReduceLROnPlateau (factor 0.5, patience 2), early stopping when validation NLL has not improved by > 1e-3 for 6 epochs, epoch cap 80, wall cap 1 h per network. Best validation epoch kept. Epochs, stop reason and final validation NLL reported per network.

**Ensemble.** Per seed s, log q_w^(s) is computed with seed s's q and c. The ensemble log density is the mean over the 5 seeds of log q_w^(s) (pointwise, before any averaging over PMNS points).

**Statistic (conv T, as exp 025).** Real inputs: CKM global fit (`src.data.load_ckm`), M_Z same-scale mass ratios (`src.evaluate.mass_ratio_targets`), NuFIT 6.1 best fit (`src.evaluate.pmns_best`), Δm² ratio |Δm²21 / Δm²3ℓ| (`src.data.load_pmns`). 256 PMNS points from the NuFIT likelihood shape (`src.nulls.pmns_noise`, rng seed 2525, the exp 025 points).
T = log2 mean_j q1(y_j | CKM, m) − log2 mean_j q0(y_j | m), each q the ensemble density. The point statistic (NuFIT best fit only) is reported as secondary.

**Primary null: N1λ (λ-matched).** For fake i (rng `default_rng([28, 2, i])`): λ ~ split normal with centre and ± errors of the `lambda` record of `src.data.load_ckm()`, A likewise from the `A` record; then `src.nulls.n1_universe(Universe.real("NO"), rng, prior={"lambda": (λ, λ), "A": (A, A)})`, so ρ̄, η̄ ~ U[−1, 1] (N1_PRIOR) and everything else is as `n1_universe` (real PMNS, real masses). The fake's CKM context is read through `Universe.load_ckm()`.

**Secondary null: N1.** Plain `src.nulls.n1_universe` (rng `default_rng([28, 1, i])`).

**Fakes.** 2000 per null. Rank = #(T_fake ≥ T_real) / 2000.

**Pass rule.** A null passes if the one-sided 95% Clopper-Pearson upper limit of the rank is < 1% (`src.nulls.decide`, the criterion-4 rule; with 2000 fakes this is ≤ 11 fakes at or above the real T). Two-sided CP95 intervals are also reported.

**Prior sensitivity.** 500 draws w ~ Dirichlet(1, ..., 1) over the 9 families (rng seed 2828). For each draw, recompute T_real and every T_fake with that w (same networks, same fakes) and apply the pass rule. Report the fraction of draws in which the primary null passes, and the distribution of T and rank.

**Decision.**
- "Candidate signal" only if the primary null passes at equal weights AND the Dirichlet pass fraction of the primary null is ≥ 0.9.
- Otherwise "no evidence of CKM information beyond (λ, A)", stating which condition failed.
- The secondary N1 null is reported but does not enter the decision. Never "discovery".

**Sanity checks (reported, not part of the decision).** Held-out per-family information with family known (E_test[log2 q1(y|c,f) − log2 q0(y|m,f)]; the CTRL family should be ≈ 0) and with family marginalised at equal weights. HPD-rank calibration on 1000 test points. Per-seed T and ranks (single-seed mixtures).

**Inverted ordering (secondary, separate).** The exp 025 data are NO only (its filter rejects IO, and SU5, S4SO and the sampled MOD/FN produce essentially none). IO is added only where cheap: F1 and F2 (PMNS independent of the ordering label in those families), GEN and CTRL (IO is ~2-3% of filter-passing draws), and MN if ≥ 2000 IO samples are reached in the generation cap. Same estimator, seeds, statistic, nulls, fake count and Dirichlet rule, with NuFIT IO inputs. Reported separately; it does not enter the NO decision, and its mixture is narrower than NO's.

## Deviations

All made before any real-data evaluation with the trained networks. One pipeline smoke test did touch real data after the pre-registration was logged: 1-epoch throwaway networks (seed 77, in a scratch folder, not kept), 40 fakes, 10 Dirichlet draws, run only to check that `analyze.py` executes. No setting was changed in response to its numbers.
1. Wall cap per network lowered from 1 h to 20 min (`EXP028_MAX_SECONDS=1200`). The machine was shared with other agents at 100% CPU and epochs took ~35 s instead of 17 s; 10 flows at 1 h each did not fit the budget. Networks that hit the cap are flagged in the training table.
2. 3 seeds (0, 1, 2) instead of 5. With the 20-minute cap the first two flows (seed 0) stopped at 13 epochs with validation NLL still falling, which is no better converged than exp 025. The budget could not cover 5 seeds trained longer, so the plan changed to 3 seeds with 40 min per flow (seed 0 resumed from its best state for another 20 min, fresh Adam moments, last LR). Classifiers are unaffected (they early-stop in minutes). The ensemble is the mean over 3 seeds.
3. IO data keep every generated sample (unequal family counts) instead of truncating to the smallest family. The mixture weights use π_f ∝ w_f p(f | c) / p_train(f), which is the general form of the pre-registered formula (p_train is constant for NO).

## Method (as run)

- `train.py`: family-conditional zuko NSF flows (5 transforms, hidden (128, 128), 8 bins; exp 025 target map and context, family one-hot appended to the context) and 3×256 SiLU MLP family classifiers. 2 torch threads per job, at most 3 jobs.
- `run_train.py`: job queue. `gen_io.py`: IO data from the exp 025 generators with the ordering cut flipped (15 min cap per family).
- `analyze.py NO|IO`: everything in the pre-registration. Per-(seed, context, PMNS point, family) log densities are cached in `artifacts/028/evals_<ord>.npz`; each family weighting is a reweighting of that cache.
- Real inputs via `src` (CKM global fit, M_Z mass ratios, NuFIT 6.1 default analysis). `src.evaluate.pmns_best` returned NO s12² 0.3088, s13² 0.02249, s23² 0.47, δ 207°; IO s12² 0.3088, s13² 0.02261, s23² 0.555, δ 283°. |Δm²21/Δm²3ℓ| = 0.0299 (NO), 0.0301 (IO).
- `run.log` timestamps: driver lines are UTC (`Z`), `analyze.py` lines are machine-local time.

### Training (curves in `curves/`, logs in `logs/`)

| network | epochs | stop | best val NLL |
|---|---|---|---|
| NO q1 s0 / s1 / s2 | 47 / 57 / 63 | wall cap (40 min) | 0.513 / 0.303 / 0.491 |
| NO q0 s0 / s1 / s2 | 44 / 62 / 63 | wall cap (40 min) | 0.943 / 0.621 / 0.850 |
| NO c1 s0 / s1 / s2 | 30 / 30 / 36 | early stop | 0.486 / 0.485 / 0.489 |
| NO c0 s0 / s1 / s2 | 39 / 43 / 33 | early stop | 1.130 / 1.133 / 1.132 |
| IO q1 s0 / s1 / s2 | 32 / 26 / 34 | early stop | 3.850 / 3.880 / 3.867 |
| IO q0 s0 / s1 / s2 | 34 / 28 / 36 | early stop | 4.343 / 4.343 / 4.333 |
| IO c1 s0 / s1 / s2 | 34 / 25 / 26 | early stop | 0.592 / 0.599 / 0.594 |
| IO c0 s0 / s1 / s2 | 35 / 45 / 48 | early stop | 0.817 / 0.813 / 0.802 |

- Exp 025's flows reached val NLL ≈ 2.6 (q1) and 3.0 (q0) in 13 epochs. Here the NO flows reach 0.30-0.51 and 0.62-0.94. Family conditioning accounts for part of this; the classifiers add back 0.49 and 1.13 nats in the mixture (mixture test NLL q1 0.66-0.87, q0 1.42-1.71).
- **The NO flows are not fully converged.** All six hit the wall cap with the best epoch at or next to the last one, with LR already decayed to 1.6e-5 to 6e-5. Validation NLL is noisy epoch to epoch (±0.1-0.3 nats, mostly from the SU5 near-delta family), and best val NLL differs by 0.2-0.3 nats between seeds. The IO networks (19k training samples) and all classifiers early-stopped normally.

## Free-parameter count

Nothing is fitted to measured data. The networks learn only from synthetic samples. These choices can move the answer, and none was tuned on the result:
- Inherited from exp 025: the family set (9 inclusion bits), filter windows and NO cut, ~8 GEN/CTRL generator hyperparameters, and the MN broad box. That is about 12 bits plus ~20 generator hyperparameters.
- New here:
  - Family weights are no longer a single choice: the Dirichlet scan integrates over them (α = 1 is one choice).
  - Estimator form: family-conditional flows plus a classifier (1 bit).
  - Null definition (N1λ) and statistic (conv; point reported as secondary): both pre-registered.
- IO: the family set {F1, F2, GEN, CTRL, MN} is forced by which generators produce IO at affordable cost (5 inclusion bits, not selected on the data).

## Result

### 1. Held-out information on synthetic test data (bits, 3-seed ensemble; NO n ≈ 3000 per family, SE 0.01-0.05)

| | F1 | F2 | MN | FN | MOD | SU5 | S4SO | GEN | CTRL | all |
|---|---|---|---|---|---|---|---|---|---|---|
| I, family known | 0.46 | 0.68 | **3.75** | 0.93 | 0.04 | −1.25 | −0.03 | 0.33 | **−0.05** | 0.55 |
| I, family marginalised | 1.03 | 1.35 | 3.95 | 1.45 | 1.47 | −1.24 | 1.26 | 0.58 | 0.22 | 1.14 |

- **Control.** CTRL is ≈ 0 when the family is known: −0.05 ± 0.01 (per seed −0.04 / −0.06 / −0.05).
- **Family marginalised.** The CKM also tells the network which family it is in, so every family gains 0.2-1.4 bits (CTRL 0.22).
- **SU5** is negative in every seed (−0.6 / −2.2 / −0.9). Its near-delta CSD3 PMNS is resolved differently by q1 and q0, as in exp 025.
- **IO, family known** (F1 / F2 / GEN / CTRL / MN): 0.16 / 0.30 / 0.19 / −0.23 / 2.58 bits. The IO control is −0.23 ± 0.07, because q1 over-fits slightly on only 2,760 CTRL samples.

### 2. Calibration (HPD coverage on 500 test points, nominal 0.50 / 0.68 / 0.90 / 0.95)

- NO q1 mixture: 0.46-0.48 / 0.61-0.63 / 0.86-0.87 / 0.92. Slightly over-confident, as in exp 025.
- NO q0 mixture: 0.46-0.48 / 0.67 / 0.89-0.90 / 0.94.
- Family-conditional flows are within 0.01-0.03 of these. IO is within 0.05 of nominal everywhere.
- Classifiers, NO: c1 accuracy 0.77 (NLL 0.485), c0 0.55 (NLL 1.130).

### 3. Real data, equal family weights (NO; the pre-registered decision)

| | T (bits) | N1λ: fakes ≥ real / 2000 | N1λ one-sided CP95 upper | N1: fakes ≥ real / 2000 | N1 one-sided CP95 upper |
|---|---|---|---|---|---|
| **conv (primary)** | **−0.99** | **1043 → 0.522** [0.499, 0.544] | 0.540 | 754 → 0.377 [0.356, 0.399] | 0.395 |
| point | +0.61 | 876 → 0.438 | 0.457 | 366 → 0.183 | 0.198 |

- Fake T_conv distributions:
  - N1λ: median −0.96, 5-95% −1.35 to −0.24.
  - N1: median −1.20, 5-95% −2.68 to −0.40.
- Neither null passes. The real CKM sits in the middle of the λ-matched fakes and in the upper middle of the broad N1 fakes.
- Per seed (single-seed mixtures):
  - T_conv −0.42 / −1.78 / −1.07.
  - N1λ rank 0.39 / 0.89 / 0.41.
  - N1 rank 0.27 / 0.57 / 0.27.
- N1λ diagnostic:
  - Fake T rises with |ρ̄ + iη̄|: median −1.46 below 0.2, up to −0.70 at 1.0-1.5.
  - corr(T, ρ̄) = +0.66, corr(T, η̄) = −0.05.
  - The 31 fakes within 0.15 of the real (ρ̄, η̄) score at least as high as the real CKM 45% of the time.

### 4. Dirichlet(1) prior sensitivity (500 draws over the 9 family weights)

- Pass fractions: N1λ **0 / 500** (the rule needs ≥ 0.9), N1 0 / 500.
- T_conv 5 / 50 / 95%: −3.09 / −1.03 / +1.07 bits.
- N1λ rank 5 / 50 / 95%: 0.44 / 0.54 / 0.92. The lowest rank in any draw is 0.40.
- N1 rank 5 / 50 / 95%: 0.23 / 0.36 / 0.52.
- The N1λ rank moves mainly with the F2 weight (corr −0.62) and the FN weight (+0.57).

### 5. Inverted ordering (secondary; F1 / F2 / GEN / CTRL / MN mixture)

- IO data kept within the 15-min caps (`gen_io_stats.json`): F1 6000, F2 6000, MN 6000 (from 13.3 M raw draws), GEN 2897 (11.8 M), CTRL 2760 (9.3 M).
- T_conv +0.69 bits (point +0.92). N1λ rank 0.598 (CP95 upper 0.616). N1 rank 0.316 (CP95 upper 0.334).
- Per-seed T_conv: 1.38 / 0.15 / 0.32.
- Dirichlet: N1λ and N1 pass fractions both 0 / 500. N1λ rank 5 / 50 / 95%: 0.48 / 0.63 / 0.89.

### Comparison with exp 025

- Exp 025 (single under-converged flows, equal weights): T_conv +1.54 bits, N1 rank 0.021.
- This run: T_conv −0.99 and N1 rank 0.38. Same synthetic data and the same N1 prior, but family-conditional flows trained 3-5× more epochs and a 3-seed ensemble.
- Exp 025's ~2σ preference, which it already attributed to λ, does not reproduce with the better-trained estimator. Its excess was estimator noise or under-fitting, not a stable property of the mixture.
- Confound: the NuFIT NO best fit that `src` returned here (s23² 0.47, δ 207°) may differ from the one exp 025 read. Exp 025's `results.json` records its own inputs; this was not cross-checked.

## Verdict

**No evidence that the real CKM carries PMNS information beyond (λ, A).** Both pre-registered conditions fail:
- The primary λ-matched null does not pass at equal weights: rank 0.52, one-sided CP95 upper 0.54, against a threshold of 0.01.
- The Dirichlet pass fraction is 0 / 500, against a threshold of 0.9.

The secondary N1 null also fails (rank 0.38). In this run the real CKM is not even preferred over broad hierarchical CKMs. IO (secondary, narrower mixture) agrees: N1λ rank 0.60, N1 rank 0.32.

Scope: within the exp 025 family mixture, this excludes a CKM → PMNS link strong enough to make the real CKM a top-1% predictor of the real PMNS among λ-matched CKMs with random ρ̄, η̄. It is consistent with exp 021 (≲ 1-2 bits) and replaces exp 025's 2σ hint with a null result.

## Caveats

- **Convergence.** The NO flows hit the 40-min wall cap while validation NLL was still improving slowly. Seed-to-seed spread is ~1.4 bits in T_conv (−0.42 to −1.78) and 0.39-0.89 in N1λ rank. The verdict holds in every seed and in every Dirichlet draw (lowest N1λ rank 0.40), so better convergence would have to move the rank by more than 0.39 to change it.
- **Seeds.** 3 seeds, not the pre-registered 5 (Deviation 2).
- **Power.** The networks do learn CKM → PMNS structure where the mixture contains it (held-out MN 3.75 bits, family known). So the test has power against links of that size inside the mixture, but none against links outside it. No injection test was run (planting an MN-like universe as "real" would quantify the power; skipped for budget).
- **SU5.** Its held-out information is negative in every seed. Its Dirichlet weight barely moves the rank (corr −0.07).
- **Conv vs point.** For NO the two statistics disagree in sign (−0.99 vs +0.61). The conv statistic averages over the NuFIT likelihood, including the s23 octant ambiguity, and q1 at the real CKM puts more mass near the best fit than over the rest of the allowed region. The point ranks also fail (N1λ 0.44, N1 0.18).
- **Running.** No RG running. Real inputs mix M_Z values (CKM, masses) with low-energy values (PMNS), as in exp 025.
- **IO mixture.** It lacks FN, MOD, SU5 and S4SO (their priors give no IO at affordable cost), and the IO control is −0.23 bits.

## Files

- `README.md`. Pre-registration hash in `run.log`: SHA-256 prefix `ec31fbd8bd00cc76`, logged 2026-11-19T20:51:07Z before any real-data evaluation. The Deviations section and everything after it were added later.
- Scripts: `train.py`, `run_train.py`, `gen_io.py`, `analyze.py`.
- Outputs:
  - `results_NO.json`, `results_IO.json`: every number above, including the 2000 fake T values per null and the fakes' Wolfenstein parameters.
  - Logs: `analyze_NO.log`, `analyze_IO.log`, `logs/*.log`, `run.log`.
  - `gen_io_stats.json`, `curves/*.json`.
- Gitignored `artifacts/028/`:
  - `models/*.pt`
  - `data_io.npz`, `evals_NO.npz`, `evals_IO.npz`
  - `README_prereg.md` (the README as hashed)

To rerun from this folder with `../../.venv/Scripts/python`:
1. Exp 025 data must exist (`experiments/20261119-025-nn-ckm-info/gen.py`).
2. `python gen_io.py 6000 2` (~20 min).
3. Train:
   - `EXP028_MAX_SECONDS=2400 python run_train.py NO 3 0,1,2`
   - `python run_train.py IO 3 0,1,2`

   Expect ~1.5 h on 6 cores of a loaded machine. Seed 0 here was trained 20 min, then resumed for 20 min (`run_train.py NO 1 0 resume`).
4. `python analyze.py NO 2000 500` and `python analyze.py IO 2000 500` (~20 min each at 2-3 threads).

# 20261119-025 Model-agnostic NN test: how much does the CKM tell you about PMNS?

## Hypothesis

Within a broad theory-space prior (every family implemented in this repo plus generic shared-parameter textures plus a no-sharing control), the real CKM carries more information about the real PMNS than a random hierarchical CKM does. Measured by conditional normalizing flows: q1(PMNS | CKM, mass ratios) against q0(PMNS | mass ratios).

## Method

### Synthetic theory space (`gen.py` → `artifacts/025/data.npz`, `gen_stats.json`)

No measured number enters any generator. Nine families, 30,000 kept samples each (270,000 total). Model code is imported from the earlier experiments' files, not copied.

| family | source | parameters drawn | raw draws | acceptance |
|---|---|---|---|---|
| F1 shared rotation | exp 007 `f1lib` (x_catalog, 6×6 perms, 14 forms) | config uniform over 5040; α, β ~ U[0, 2π); V from the N1 prior | 66,651 | 0.495 |
| F2 Clebsch corrections | exp 009 `f2lib.configs/full_matrix` | config uniform over 17,316; φ ~ U[0, 2π); TM θν ~ U[0, 0.5]; CKM angles from an N1-prior V | 67,809 | 0.487 |
| MN minimal SO(10) 10+126 | exp 014 `so10fit.mn_mats` | exp 014 box. The data-centred quark windows are replaced by fixed broad ones (`mn_box_broad`): mass log-ratios ±1 decade around round values, s12q ∈ [0.05, 0.5], lg s13q ∈ [−4, −0.5], lg s23q ∈ [−3, −0.3] | 294,005 | 0.106 |
| FN real 10+126+120 | exp 014 `so10fit.fn_mats`, `fn_box("ours")` | uniform in the box | 46.7 M | 0.0006 |
| MOD common τ | exp 017 `models` (A4/S4/A5 catalogs, `tau_from_unit`) | group, quark and lepton assignments, and τ uniform; ln α ∈ [−14, 4], g ∈ [−6, 6] (exp 017 `search.py`). One τ per batch of 2000 coupling draws, so the forms are evaluated once per batch (exact prior sampling) | 24.1 M | 0.0012 |
| SU5 SU(5) + CSD3 | exp 022 `model.predict` | u_ij, d_ij ~ U[−10, 10], r ~ U[0, 1] (exp 022 `fit.BOUNDS`); θ, η, Clebsches frozen | 184,227 | 0.177 |
| S4SO S4×SO(10) | exp 023 `model.observables` | exp 023 `heldout.py` prior: every \|coupling\| log-uniform ±2 decades, centred on the paper's Table 4 point (not our fit), signs as published; 6 phases ~ U[0, 2π); no η̄_b | 757,415 | 0.041 |
| GEN shared textures (new) | `gen._gen_textures(shared=True)` | see below | 1.68 M | 0.018 |
| CTRL no sharing (new) | same generator; quark (Y_u, Y_d) and lepton (Y_e, Y_D, M_R) sectors from independent draws | 2.70 M | 0.011 |

**GEN.** K ∈ {2, 3} shared complex operator matrices Y_k. Each has:
- Froggatt-Nielsen suppression ε^{qL_i + qR_j}, with ε ~ U[0.05, 0.3]. qL = (a, b, 0), a ∈ 1..4, b ≤ a. qR = qL (SO(10)-like) with probability 0.5, else an independent (a', b', 0).
- O(1) complex coefficients: \|z\| log-uniform in [1/3, 3], uniform phase.
- A sparsity mask with zero probability 0 or 0.3.
- A type: symmetric (0.4), antisymmetric (0.2) or general (0.4).

Each sector is Y_f = Σ_k c_{f,k} [r_{f,k}] Y_k with GUT-like Clebsches c ∈ {±1, ±3, ±1/3, 2, ±1/2, 9/2, 4/9, 6, 0} (at least one non-zero). Up-type and Dirac sectors also get complex vev ratios r (log-uniform 0.1-10). M_R uses symmetrised Y_k. Light neutrinos: type-I seesaw m_ν = Y_D^T M_R^{-1} Y_D. One discrete structure per batch of 64.

**Observable engine** (`gen.obs_from_mats`, the exp 014 `raw_obs` convention): left rotations from M†M, CKM = U_u†U_d, PMNS = U_e†U_ν. The ordering comes from the spectrum: the closer pair is the solar pair. SU5 and S4SO use their own experiments' observable functions and conventions. A family-wide CP conjugation would only flip δ signs within that family.

**Filter** (`gen.FILTER`, log10 windows; fixed before any training and not tuned):
- m_u/m_c ∈ [−5, −1], m_c/m_t ∈ [−4, −1], m_d/m_s ∈ [−3, −0.5], m_s/m_b ∈ [−3, −0.5], m_e/m_μ ∈ [−4, −1], m_μ/m_τ ∈ [−3, −0.5]
- s12q ∈ [0.03, 0.6], s23q ∈ [3e-4, 0.6], s13q ∈ [1e-6, 0.4]. These cover the N1 prior.
- Δm²21/\|Δm²3ℓ\| ∈ [1e-3, 0.5]
- PMNS finite, and **normal ordering only**.

F1 and F2 have no mass model. Their mass ratios and Δm² ratio are drawn log-uniform inside the windows, independently of everything else.

### Flows (`train.py`, `run_train.py`; checkpoints in `artifacts/025/flows/`, curves in `flows/*.json`, logs in `logs/`)

- zuko 1.6 NSF: 5 transforms, hidden (128, 128), 8 bins. Adam 1e-3, batch 1024, plateau LR decay, 1 torch thread per job, 4 jobs in parallel.
- Wall cap of 420 s per flow (≈ 12-13 epochs). Best validation epoch kept.
- Target (5): logit s12², logit s13², logit s23², logit(δ/2π), log10 Δm² ratio, standardised. The Jacobian of this map cancels in every q1/q0 ratio.
- Context: q1 = (log10 s12q, s13q, s23q, cos δq, sin δq) + 6 log10 in-sector mass ratios; q0 = the 6 mass ratios. m_b/m_τ is excluded because it is normalisation-dependent in several families.
- Split 80/10/10, seed 2501, shared by every run.
- Runs: `main`, `drop_<F>` for each of the 9 families (q1 and q0 both retrained without F), and `seed1` (main with a different torch seed, to measure training noise).

### Information and real-data test (`analyze.py` → `results.json`, `analyze.log`; `diag_lambda.py` → `diag_lambda.json`)

- **I** = E_test[log2 q1 − log2 q0] per family.
- **Real data** (NO):
  - CKM global fit from `src.data.load_ckm`.
  - M_Z same-scale mass ratios from `src.evaluate.mass_ratio_targets`.
  - NuFIT 6.1 NO best fit from `src.evaluate.pmns_best`.
  - Δm² ratio from `src.data.load_pmns`.
- Two statistics:
  - **T_point** = log2 q1 − log2 q0 at the best fit.
  - **T_conv**: the same for the predictive densities, each averaged over 256 PMNS points drawn from the NuFIT likelihood shape (`src.nulls.pmns_noise`).
- **N1 calibration**: 2000 fakes from `src.nulls.n1_universe` (λ ~ U[0.05, 0.5], A ~ U[0.3, 2], ρ̄, η̄ ~ U[−1, 1]) with real PMNS and real masses, through the same trained flows.
  - Rank = fraction of fakes with T_fake ≥ T_real, two-sided 95% Clopper-Pearson interval.
  - B_ckm and S_rank from `src.infogain.null_info` with χ² ≡ −2T. B_ckm = log2[q1(real CKM) / mean q1(fake CKM)].
  - Support check: a fake counts as in support if its CKM context's 10-NN distance is ≤ the 99th percentile of training points' own 10-NN distance. 99.8% of fakes are in support, so restricting changes nothing.
- **Calibration**: HPD rank coverage on 1000 test points.

Running: the synthetic families have no scale. Real inputs are at M_Z (CKM, same-scale mass ratios) and at low energy (PMNS). Ignoring running shifts the heavy-quark ratios by tens of percent, which is small against the 1-4 decade training windows. It is not quantified further.

## Free-parameter count

Nothing is fitted to the measured data. The flows learn only from synthetic samples, and measured values enter only at evaluation. The "parameters" of this measurement are the prior-mixture choices. Each one changes the answer, and none was tuned on the result:
- Family set: 9 inclusion choices ≈ 9 bits.
- Weights: equal per family after the filter (1 choice).
- 10 filter windows and the NO-only restriction.
- About 8 discrete hyperparameters of the GEN/CTRL generator: K, the ε range, the charge range, the sparsity probabilities, the type probabilities, the Clebsch set, the vev-ratio range, and seesaw type I.
- The MN broad box, which replaced exp 014's data-centred box.
- The statistic choice (point vs conv: 1 bit, both reported) and the fake prior (N1 as in `src.nulls`, not chosen here).

Roughly 12 bits of discrete choices plus about 20 generator hyperparameters. This is why the result has to be read through the family-drop table, not the main number alone.

## Result

### 1. Information gain on held-out synthetic data (bits, main run, n ≈ 3000 per family, SE ≈ 0.03-0.06)

| | F1 | F2 | MN | FN | MOD | SU5 | S4SO | GEN | CTRL | all |
|---|---|---|---|---|---|---|---|---|---|---|
| I main | 0.36 | 0.50 | **2.94** | 0.67 | 0.72 | 0.47 | 0.28 | 0.31 | **−0.13** | 0.69 |
| I seed1 | 0.34 | 0.72 | 2.63 | 0.62 | 0.40 | −2.20 | 0.27 | 0.36 | −0.10 | 0.35 |

- The sanity check holds: the no-sharing control is ≈ 0, slightly negative. That is the estimator's bias: q1 has 5 more context inputs and is under-converged.
- The shared families are positive. MN, where the CKM is an explicit input of M_d, is the largest at about 3 bits.
- SU5 is unstable across seeds (+0.5 vs −2.2). Its PMNS is a near-delta CSD3 pattern (test NLL −14 nats), which the flows resolve differently each run.
- Training noise: about ±0.3 bits per family, and larger for SU5.
- In the drop runs, I for the dropped family is out-of-distribution, ranging from −1.9 (MOD) to +0.4 (GEN).

### 2. Real data (main flows; PMNS data hold 18.9 bits vs Haar, `src.infogain.pmns_data_bits`)

| | main | seed1 |
|---|---|---|
| T_point (bits) | +1.80 | +3.00 |
| T_conv (bits) | +1.54 | +2.73 |
| N1 rank, point (fakes ≥ real / 2000) | 63 → 0.032 [0.024, 0.040] | 86 → 0.043 |
| N1 rank, conv | 41 → 0.021 [0.015, 0.028] | 13 → 0.0065 |
| B_ckm (conv) | +1.68 bits | +2.27 bits |
| S_rank (conv) | 5.6 bits | |

- Real PMNS lies at HPD rank 0.17 in q1 at the real CKM (the 17% highest-density region) and at 0.07 in q0. Both are well inside the predictions.
- q1 median prediction at the real inputs: s12² 0.27, s13² 0.030, s23² 0.50, δ 258°. 68% bands are wide, e.g. s13² [5e-4, 0.11].
- Calibration (nominal 0.50 / 0.68 / 0.90 / 0.95):
  - q1: 0.49 / 0.66 / 0.88 / 0.93, slightly over-confident in the tails.
  - q0: 0.56 / 0.72 / 0.91 / 0.95.

### 3. Sensitivity: drop each family (both flows retrained)

| dropped | T_point | T_conv | rank point | rank conv | B_ckm conv |
|---|---|---|---|---|---|
| none (main) | 1.80 | 1.54 | 0.032 | 0.021 | 1.68 |
| F1 | 2.58 | 1.59 | 0.013 | 0.0095 | 1.88 |
| F2 | −0.10 | −0.60 | 0.050 | 0.028 | 1.62 |
| MN | 1.11 | 1.19 | 0.064 | 0.022 | 1.65 |
| FN | 0.15 | 1.40 | **0.578** | 0.037 | 1.40 |
| MOD | 1.31 | 1.35 | 0.056 | 0.039 | 1.44 |
| SU5 | 1.02 | 1.30 | 0.234 | 0.053 | 1.12 |
| S4SO | 2.06 | 2.04 | 0.066 | 0.016 | 1.83 |
| GEN | 1.83 | 1.78 | 0.162 | 0.040 | 1.35 |
| CTRL | 0.68 | 0.72 | 0.179 | 0.042 | 1.33 |

- The point rank is fragile: 0.013-0.58 across drops.
- The conv rank stays at 0.0065-0.053 in every variant.
- B_ckm stays at +1.1 to +2.3 bits.

### 4. What the preference is made of (`diag_lambda.json`, main flows, point statistic)

| fake λ bin | n | median T_fake | fraction ≥ real |
|---|---|---|---|
| 0.05-0.15 | 474 | +0.22 | 0.036 |
| 0.15-0.20 | 221 | +1.10 | 0.100 |
| 0.20-0.25 | 208 | +0.85 | 0.063 |
| 0.25-0.30 | 215 | +0.30 | 0.047 |
| 0.30-0.50 | 882 | −1.35 | 0.001 |
| λ within 0.03 and A within 0.2 of the real values | 50 | +1.20 | 0.10 |

- Most of the separation comes from the 44% of N1 fakes with λ > 0.3, which score clearly lower. Cabibbo-sized fakes score nearly as high as the real CKM.
- Among fakes with the real λ and A, 10% score at least as high (5/50). The real CKM's excess over them is about 0.6 bits.

## Verdict

**No evidence that the real CKM carries specific PMNS information beyond its hierarchy scale.**
- Under the broad N1 prior there is a weak, prior-conditional preference for the real CKM: conv rank 0.7-5% in all 11 flow variants, B_ckm +1.1 to +2.3 bits. That is about 2σ, and it is not a candidate signal under the criterion-4 protocol (needs < 1% with an upper CP limit < 1%; only drop_F1 and seed1 reach below 1%, with CP intervals crossing it).
- The diagnostic shows the preference is carried by λ ≈ 0.15-0.25 versus the N1 prior's large-λ tail, not by the detailed CKM. Within the families sampled here, "Cabibbo-sized quark mixing" co-occurs with real-like PMNS. That is the hierarchy information exp 021 flagged as unmeasured. It is a few bits at most, and conditional on the mixture.
- The number agrees with exp 021's profile bound (≲ 1-2 bits) from an independent, model-agnostic estimator.
- The result depends on the prior mixture:
  - Dropping FN moves the point rank from 3% to 58%.
  - Dropping F2 makes T negative.
  - Training noise between seeds is about 1 bit in T.

## Caveats

- Flows are under-converged: 420 s cap, 11-13 epochs, and validation loss still noisy at the last epoch (`flows/*.json`). Seed-to-seed T differs by 1.2 bits, comparable to the effect. A longer run with 3-5 seeds would tighten section 2.
- Normal ordering only. Most families produce NO almost exclusively in their priors, so an IO mixture would be dominated by F1/F2 and is not meaningful here.
- Family weights are equal after the filter. Raw acceptance varies from 0.06% (FN) to 49% (F1), so "equal post-filter weight" is a strong reweighting of the raw priors. Only the drop-one sensitivity was run, not continuous reweighting.
- MOD samples share τ within batches of 2000 (≈ 2.4 kept samples per τ on average). This is exact prior sampling, but the samples are clustered.
- F1/F2 masses and Δm² ratio are independent noise.
- MN uses a broad box chosen here, not exp 014's data-centred one.
- The real data are compared without RG running (see Method).
- `src/nulls.py` was being edited by another agent during this run. Only `n1_universe`, `pmns_noise`, `Universe.real` and `N1_PRIOR` were used.

## Files

- `gen.py`: generators, filter, features.
- `gen_stats.json`: per-family raw counts and acceptance.
- `train.py`, `run_train.py`: flows; per-run curves in `flows/*.json`, per-run logs in `logs/`.
- `analyze.py` → `results.json` (all numbers above, including the 2000 fake T values for main) and `analyze.log`.
- `diag_lambda.py` → `diag_lambda.json`.
- `run.log`: generation and training driver log.
- Gitignored: `artifacts/025/data.npz` (33 MB), `artifacts/025/flows/*.pt`.

To rerun from this folder:
1. `../../.venv/Scripts/python gen.py 30000 25` (about 21 min on 5 cores; FN dominates).
2. `python run_train.py 4` (22 flows, about 40 min on 4 cores).
3. `python analyze.py 2000`
4. `python diag_lambda.py`

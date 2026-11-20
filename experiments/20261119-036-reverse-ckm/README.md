# 20261119-036 Reverse direction: does the PMNS (plus masses) carry information about the CKM?

## Hypothesis

Exps 025/028 asked whether the CKM predicts the PMNS and found no evidence beyond the Cabibbo scale; exp 032 showed that test is limited by PMNS precision. The CKM is measured far more precisely, so the reverse question might be more sensitive: within a synthetic theory-space mixture, does the measured PMNS predict the measured CKM better than PMNS values drawn from a smooth prior (N2-type null)?

Expectation stated before running: under N2 the PMNS is the randomised side, so the sensitivity is set by how well the PMNS is measured relative to the N2 box, not by the CKM precision.

## Pre-registration

`prereg.md` (this folder, committed), SHA-256 `360edfc00cf5f254e7cc464b0f4c538eb20c7331e0147fc06ed0aa9cfec4aad3`, logged in `run.log` at 2026-11-20T02:04:55Z, before the oracle, the training and any real-data evaluation. Summary:
- Statistic: T = log2 mean_{i,j} Q1(c_i | p_j, m) − log2 mean_i Q0(c_i | m). c_i: CKM points from the PDG measurement (split normal per record). p_j: 128 PMNS posterior points (centre − `pmns_noise` offsets, rng 3602). m: measured mass ratios and Δm² ratio. Q: equal-weight mixture over 5 family-conditional flows; ensemble = mean over 3 seeds of log Q.
- Null: N2 (`src.nulls.n2_universe`: real CKM and masses, NO PMNS centre uniform on the NuFIT 3σ box widened ×3, δ uniform on [0, 360)), 2000 fakes `default_rng([36, 2, i])`. Under N2 the Q0 term is common to the real data and every fake, so the test ranks how well each PMNS centre predicts the measured CKM.
- Rule: pass iff `src.nulls.decide(k, 2000)` (one-sided CP95 upper < 1%; k ≤ 12 of 2000).
- Training mixture: exp 025 NO synthetic data (reproduced by `src.theoryspace`, exp 033) for GEN, CTRL, F1, F2, MN, 30,000 each. FN, MOD, SU5, S4SO were left out (budget; SU5 and S4SO are near-delta samples around published points).
- Oracle power before training; injection with 20 in-box GEN worlds.

## Deviations

All made before the real-data statistic was computed.
1. **Oracle settings (before any network result).** The first oracle run took about 15 min per level and was stopped twice during level 1 (`run.log`). Changes: 30 worlds per level instead of 50; the information bisection uses 100 points and 16 iterations, on a 5000-point prior subset below 8 bits; the marginal p(û) includes the true x (bounded Monte Carlo estimator, cap log2(20001) = 14.3 bits); levels 1, 2, 4, 6, 8, 12 bits instead of 1, 2, 4, 8, 12, 16 (the estimator cannot resolve 16 bits). The perfect-link part is identical across the three runs (same seeds, same numbers). Fakes are evaluated in blocks of 200 and stopped once k > 12, which fixes the outcome; ranks of stopped worlds are biased upward.
2. **CKM points 16 → 4** (c_0 = centre plus 3 split-normal draws, rng 3603). A timing check with untrained networks on fake centres (no real data) gave > 2 min per 100 fakes at 16 points, which did not fit the budget. CKM errors are 0.3-2%, the flows' CKM widths tens of percent, so the CKM average is close to a point evaluation either way (point and averaged T differ by 0.04 bits on real data).
3. **Rule threshold wording.** `prereg.md` says "k ≤ 11" and "stop once k ≥ 12"; `src.nulls.max_hits_to_pass(2000)` is 12. The rule used is `decide()` as pre-registered (k ≤ 12), and sequential stopping stops once k > 12.
4. **Training.** Every q1 flow and q0 s0 hit the 15-min wall cap (30-33 epochs, best epoch at or next to the last); q0 s1, s2 early-stopped. Jobs ran 2 at a time while the oracle used the third core, 1 torch thread each (2 threads per job would have allowed only one job within the 3-core budget).
5. **Injection worlds**: 20 in-box GEN worlds needed 26 batches (52,000 accepted, 3.6 M raw draws; acceptance into the box ≈ 4e-4 of accepted draws).
6. **F2 injection (secondary, pre-registered as optional)** ran after the GEN result: 20 in-box F2 worlds from 1 batch (2000 accepted, 4253 raw draws; F2's PMNS is often near the real values).

## Method (as run)

- `oracle.py`: oracle power under N2 with no networks (prereg "Oracle power"). Exact NuFIT NO Δχ² (1D s12², s13² + 2D (s23², δ), translated, no octant mirror; matches `src.data.pmns_delta_chi2` to 1e-15). Outputs `oracle.json`, `oracle.log`.
- `common.py`, `train.py`, `run_train.py`: data (exp 025 `data.npz`, 5 families, 150,000 rows, split seed 3601), family-conditional zuko NSF (4 transforms, (96, 96), 8 bins), target c = (log10 s12, log10 s13, log10 s23, logit δ_q/2π), contexts p = (logit s12², logit s13², logit s23², cos δ, sin δ), m = 6 log10 mass ratios + log10 Δm² ratio. Curves in `curves/`.
- `analyze.py heldout | real | inject <family>`; `worlds.py <family>` builds injection worlds from `src.theoryspace` with today's measurement noise.
- Real inputs via `src` only: PDG CKM records (s12 0.22517, s13 0.003763, s23 0.04189, δ 1.154 rad), `mass_ratio_targets`, `load_dm_ratio("NO")` (0.0299), `pmns_best("NO")` (0.3088, 0.02249, 0.47, 207°).

## Free-parameter count

Nothing is fitted to measured data; the networks learn only from synthetic samples and the measured values enter only as the evaluation point. Choices that shape the answer (fixed in `prereg.md` before any real-data number): family set (5 inclusion bits, budget-driven), network size and training caps, statistic form (conv, equal family weights, no classifier), N2 box (fixed by `src.nulls`), oracle link family and levels, injection family and the in-box condition.

## Result

### 1. Oracle ceiling (before training)

Information ceiling: I(x; ŷ) between the true and measured PMNS under the N2 box prior is **7.53 bits** (s12² 2.12, s13² 2.14, (s23², δ) 3.28). A PMNS → CKM link cannot deliver more than this about the measured PMNS centre, however precise the CKM. A 1% rank among 2000 fakes needs roughly −log2(13/2000) = 7.3 bits of evidence.

Perfect link (CKM pins the true PMNS exactly, known to the oracle): detected in **923/1000** worlds (CP95 [0.905, 0.939]) with x uniform in the box, 187/200 (0.935) with x at the NuFIT best fit. Rank q10/50/90: 0 / 0.001 / 0.005. So at today's precision a deterministic, invertible PMNS → CKM map would be detectable by an ideal estimator.

Smooth random links (û = a g(x) + ε, CKM in units of its errors, 30 worlds per level):

| I_link (bits) | I_meas median (bits) | oracle detected | CP95 | median rank |
|---|---|---|---|---|
| 1 | 0.50 | 0/30 | [0, 0.12] | 0.31 |
| 2 | 0.99 | 3/30 | [0.02, 0.27] | 0.14 |
| 4 | 1.96 | 3/30 | [0.02, 0.27] | 0.09 |
| 6 | 2.75 | 7/30 | [0.10, 0.42] | 0.031 |
| 8 | 3.38 | 15/30 | [0.31, 0.69] | 0.007 |
| 12 | 2.78* | 12/30 | [0.23, 0.59] | 0.011 |

- I_link is the information of the link about the true PMNS (with the CKM measurement noise included); I_meas the information about the measured PMNS centre. *At 12 bits the 512-point posterior estimate of p(û | ŷ) is under-resolved, so I_meas is underestimated there.
- **50% oracle power needs ≈ 8 bits of link information (≈ 3.4 bits about the measured PMNS).** Detection plateaus at 40-50% for smooth links: these maps fold the 4D box, so the measured CKM is consistent with several separated PMNS regions.
- For comparison, the training families carry I(CKM; PMNS | m, f) ≈ 0.1-0.3 bits (GEN, F1, F2) and ≈ 3.3 bits (MN) over their own priors (Result 2; exp 028 forward: 0.3-0.7 and 3.75). The ceiling is far above everything except MN, so the network part was kept small as pre-registered.

### 2. Held-out information (test split, family known, 3-seed ensemble, bits)

| GEN | CTRL | F1 | F2 | MN |
|---|---|---|---|---|
| 0.27 ± 0.02 | **−0.08 ± 0.01** | 0.08 ± 0.01 | 0.30 ± 0.02 | 3.29 ± 0.05 |

- CTRL ≈ 0 (slightly negative, per seed −0.07 / −0.09 / −0.08: q1 overfits a little with 7 extra context dimensions). Per-seed spread ≤ 0.15 bits elsewhere.
- Mutual information is symmetric, and the values sit close to exp 028's forward numbers for GEN and MN (0.33, 3.75); F1 and F2 are lower here (0.46 → 0.08, 0.68 → 0.30), consistent with these smaller, wall-capped flows being less converged (best val NLL q1 3.03-3.06, q0 3.58-3.59 nats).

### 3. Real data vs 2000 N2 fakes (pre-registered decision)

| | T (bits) | fakes ≥ real | rank | two-sided CP95 | one-sided CP95 upper | pass |
|---|---|---|---|---|---|---|
| **conv (primary)** | **0.284** | **486 / 2000** | **0.243** | [0.224, 0.262] | 0.259 | **no** |
| point (secondary) | 0.321 | 493 / 2000 | 0.247 | | | |

- Fake T: 5 / 50 / 95% = 0.05 / 0.22 / 0.35 bits; full range −0.05 to 0.39. Moving the PMNS centre anywhere in the N2 box changes the flows' CKM density by at most ~0.4 bits.
- Q0 term: log2 mean Q0(c | m) = −6.40 bits (common to real and fakes).

### 4. Pipeline power: 20 in-box GEN worlds

- Detected **0/20** (CP95 [0, 0.17]). Median world T minus median fake T: −0.06 bits; median rank 0.53 (sequential stopping, biased upward); no world reached rank ≤ 0.05; 17 of 20 stopped after 100 fakes.
- In-box GEN worlds are rare in the training data (≈ 17 of 30,000 GEN rows lie in the box), and GEN carries ≈ 0.3 bits over its whole prior, far below the ≈ 8 bits the oracle needs. So 0/20 is expected from both the information content and the missing training support.

### 5. Secondary: 20 in-box F2 worlds

- Detected **0/20** (CP95 [0, 0.17]). Median world T minus median fake T: −0.01 bits; median rank 0.52; 19 of 20 stopped after 100 fakes.
- F2 has ≈ 0.3 bits of CKM ↔ PMNS information over its prior (Result 2), and exp 032 found the CKM detail in F2 moves the PMNS mostly below the s13² error. 0/20 is again what the information content predicts.

## Verdict

**No evidence that the PMNS (plus masses) carries information about the CKM within this mixture** (GEN, CTRL, F1, F2, MN): the real PMNS predicts the real CKM no better than PMNS centres drawn from the N2 box (rank 0.243, one-sided CP95 upper 0.259; the rule needs < 0.01).

Power statement:
- The reverse direction is not more sensitive than the forward one under an N2-type null. The CKM precision does not help because the null randomises the PMNS; the ceiling is the 7.5 bits that today's PMNS measurement carries relative to the N2 box.
- An ideal estimator would detect a deterministic, invertible PMNS → CKM map 92% of the time, but a smooth random link needs ≈ 8 bits of link information for 50% oracle power.
- The trained pipeline has 0/20 power on in-box GEN worlds and 0/20 on in-box F2 worlds. The mixture's families carry ≲ 0.3 bits (GEN, F1, F2) or 3.3 bits (MN, with almost no support near the real PMNS), so within this theory space neither an ideal estimator nor these networks could have seen a link. The null result constrains only links of ≳ 8 bits, and only through the oracle, not through the networks.

Nothing here is a candidate.

## Caveats

- One oracle link family (smooth random Fourier maps of the 4 PMNS parameters). The perfect-link figure is an upper bound for invertible maps.
- The oracle uses pure translation of the NuFIT tables (no octant mirror), as `pmns_noise` injections do; the real N2 null mirrors s23² when the fake centre is in the other octant. The NN statistic uses the same translated posterior points for real and fake centres.
- Flows are small and under-converged (wall cap); held-out F1 and F2 information is below exp 028's forward values.
- Equal fixed family weights, no classifier: the mixture does not use the PMNS to identify the family (exp 028 did). This removes the "which family" information by design.
- NO only. Masses fixed at the measured values (not averaged over their errors).

## Files

- `prereg.md` (hashed pre-registration), `README.md`, `run.log`.
- `oracle.py`, `oracle.json`, `oracle.log`.
- `common.py`, `train.py`, `run_train.py`, `curves/*.json`.
- `analyze.py`, `worlds.py`, `heldout.json`, `real.json` (includes all 2000 fake T values), `worlds_GEN.json`, , `inject_GEN.json`, `worlds_F2.json`, `inject_F2.json`.
- Gitignored `artifacts/036/`: `models/*.pt`, `logs/*.log` (training), stdout captures.

To rerun from this folder with `../../.venv/Scripts/python`: `oracle.py` (~30 min, 1 core); `run_train.py 3` (~30 min); `analyze.py heldout`; `analyze.py real 3` (~6 min); `worlds.py GEN 20` (~9 min); `analyze.py inject GEN 3` (~7 min); same for F2 (`worlds.py F2 20` takes seconds).

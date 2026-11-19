# State

Iteration: 5 (2026-11-19)

## Status
Infrastructure built, verified, and tested (458 tests pass: `.venv/Scripts/python -m pytest -q`; `pytest.ini` limits collection to `tests/`). Exp 012 skeptic findings H2, H3, M1-M4, M6, M7, L1 fixed in exp 013 (see below).
- `data/` + `src/data.py`: PDG 2026 (CKM global fit, masses) and NuFIT 6.1 (both orderings, two analyses: default `IC23_woSKatm`, alt `IC24_wSKatm`), 1D and 2D Δχ² tables. Exp 20261118-001.
- `src/mixing.py`, `src/haar.py`, `src/stats.py`: angle extraction, Haar sampling, χ²/p-value/Šidák/Clopper-Pearson, `Objective`. Exp 20261118-004.
- Env: Python 3.14 `.venv`, torch 2.15 CPU, zuko, sbi, PySR (Julia bootstrapped into `.venv/julia_env`). Exp 20261118-002.
- Baseline catalog: 19 entries / 26 known relations, 6 model families, RG survey. Exp 20261118-003.
- `data/` independently verified: 100 records + all 483,654 χ² table rows, 0 mismatches. Exp 20261118-005.
- `src/evaluate.py` + `src/baselines.py` (37 registered baselines): the official scorer. PMNS scored jointly (2D s23-δ table), IO offset added once to `chi2_global` only. N counts scored observables; data inputs are not scored. Exp 20261118-006.
- Exp 013 hardening: `src/evaluate.py` alias groups (one measurement can't be scored twice; ≤4 independent CKM global keys; global and direct never mixed), explicit `use=` required, `ordering="best"` adds 1 bit and criterion 3 uses `p_global`, k_d = ceil(log2 #options), p/p_global use dof = N − k_cont, flags `compression_ok`/`fit_ok`. `src/leakprobe.py` (`leak_probe`, `evaluate_model(..., leak_check=True)`). `src/nulls.py` implements the criterion-4 protocol (`run_null` N1/N2 with translated NuFIT tables, `power_check`, `decide`); toy tests: planted relation passes 1000 fakes with 0 hits and power 0.60; CKM-blind family fails at fake 5. `src.rg` `masses=`/`sm_inputs=` overrides, `ckm_at` memoised; `src.data.pmns_profile_1d` vectorized (×100).
- `src/rg.py`: 2-loop gauge, 1-loop Yukawa/λ/κ running, SM and MSSM (single M_SUSY threshold). SM CKM running matches arXiv:2510.01312 to ≤0.14%; MSSM θ23^q, θ13^q ~2% high (1-loop Yukawas). `data/sm_inputs.json` added. Exp 20261118-008.

## Compute facts
- `torch.xpu.is_available()` = True (Intel iGPU) but unusable: first-call kernel JIT takes minutes per op, MLP training made no progress in 4 min. Use CPU build.
- MLP emulators: `torch.set_num_threads(2)` per job (2 threads ≈ 8 threads in throughput), up to ~5 parallel jobs.
- Batched 3x3 eigh/SVD: 1 thread per process + multiprocessing pool. torch is 2-10x faster than numpy for batched matmul.
- Iteration 3 oversubscribed the machine (agents saw 86-100% CPU; NPE got 1 epoch). Cap total agent cores at 10 and give heavy training jobs priority.
- PySR and sbi write `outputs/`, `sbi-logs/` to cwd by default: point them at `artifacts/` or the experiment folder.

## Data caveats
- CKM hand-transcription concern closed: exp 005 rechecked from rendered PDG pages, all match.
- No CKM correlation matrix (PDG does not publish one).
- `pmns_delta_chi2` uses the 2D (s23sq, δ) table when both are given, otherwise sums 1D profiles; ignores θ13-θ23, θ13-δ, Δm²3ℓ-θ23 correlations. Default returns 0 at each ordering's own best fit; `relative_to="global"` adds the IO penalty (1.49 default analysis, 5.91 with SK-atm). Fixed in iteration 2: `stats.Objective` now takes `pmns_ordering_offset` (added once) and refuses a `relative_to="global"` partial in sum1d mode; δ pull sign now wraps.

## Known infrastructure gaps (open)
- `src/nulls.py` (exp 018): criterion 4 is vacuous for any lead with k > N-2 (`is_hit`/`candidate_ok` require compression, so strict power is 0 by construction); `power_check` hardcodes `candidate_ok` (should take `ok_fn`) and needs a documented fit-only hit rule; `is_hit` compares p_global exactly (needs tolerance); no mass-side null (N3 redrawing mass ratios); `power_check` spawns a pool per injection (slow).
- `src.evaluate` cannot score same-scale mass ratios (flavour-atom alias rule blocks them), Δm² or Δm² ratio, has no theory floor on masses, and cannot score GUT-scale predictions. Exps 014-017 all scored these outside the official scorer.
- `src.rg.ckm_at` builds κ from PMNS data: 1e-12 crosstalk flagged by the leak probe. `rg.pmns_at` assumes m_lightest = 0. `derived_yukawas_mz` gives no uncertainties.
- Exp 012 L2-L4: `Objective` defaults to sum1d; Δm² not scorable via `evaluate` (also blocks the Δm²21/Δm²3ℓ ratio); undefined δ scored as 0.
- `src.evaluate` cannot score GUT-scale relations directly; `src.rg` lacks tanβ-enhanced SUSY threshold corrections and 2-loop MSSM Yukawa running.
- Exp 012's own test file now XPASSes (bugs fixed); the adapted tests live in `tests/test_evaluate_audit.py`.

## Criterion-4 protocol (adopted iteration 3, from exp 012)
1. Freeze pipeline, selection score, hit definition before running fakes. Each fake reruns everything, including discrete and ordering choices.
2. Mandatory nulls, per ordering: N1 CKM-side hierarchical (real PMNS + tables, fake CKM with λ 0.05-0.5, A 0.3-2, ρ̄, η̄ ∈ [-1,1]); N2 shifted data (real CKM, PMNS centre from a smooth prior not built from pattern values, scored with translated NuFIT tables). Haar-PMNS optional.
3. Hit: a fake's best relation has k ≤ N-2 under the same counting, max|pull| < 2, p_global ≥ the real candidate's.
4. 1000 fakes per null and ordering; pass if the one-sided 95% Clopper-Pearson upper limit < 1% (≤ 4 hits). Early stop only to declare failure.
5. Power check first: 20 injections of the candidate's own structure (best-fit prediction + measurement noise) through the full pipeline with 300 fakes each. If < 50% pass, the null cannot confirm or exclude this family.
6. Leak probe, alias check, ordering bit, k_d in bits run alongside.
7. Wording: a failed null = "no evidence". "Excluded" only for fit-quality failure or a null with demonstrated power.

## Current best candidates
None. See `CANDIDATES.md`.

## Ruled out (fit quality or compression) and no-evidence results
Official baseline scores (exp 006, no RG): no known relation has k ≤ N-2 with acceptable p. TM1 best (N=4, k = 2 cont + 1 bit; NO χ² 5.1, IO 1.8 (global 3.3); p_global 0.078 NO / 0.191 IO under exp 013 convention). Rough pulls (exp 003): QLC solar -4.3σ, self-complementarity +6.2σ, θ13=θ_C/√2 +4.9σ, exact TBM/GR1/GR2/HEX θ12 +3.7/-4.9/+5.5/-8.9σ, TM2 +4.8σ, leptonic GST -53σ.

- F1 shared rotation (exp 007): fails compression (k = 2 + 11.4 bits, N = 4). Null rates 36% (hierarchical) / 8% (Haar). Per exp 012 H1 that null had no power (a true F1 relation passes < 2% of the time), so this is "no evidence", not excluded. The CKM does supply θ13 in F1 (corrects iteration 2's "barely used"); any O(λ) hierarchical V does as well.
- F2 Clebsch charged-lepton corrections (exp 009): 17,316 configs (14.1 bits). Fails compression. LO-tier best: θ12e = θ_C, θ23e = θ_cb on TBM/GR2, s13 = sinθ_C sin(π/4 - θ_cb), χ² 7.2 (p 0.065, trials p 1). Hierarchical null 27-50%. Emulator inversion wants θ12e/θ_C = 0.961 ± 0.020, not a listed Clebsch value. Good overall fits come from free θν (TM-type), not the CKM. Sanity: TBC, solar sum rule, TM1/TM2 sum rules reproduced exactly.
- Single-observable symbolic formulas from CKM angles + mass ratios (exp 010): numerology at current precision. Up to 4.8M distinct functions (C ≤ 6); thousands per target within 1σ; no null cell below 1%. NN guidance found nothing brute force misses.
- Minimal SO(10) 10+126, type I (exps 011, 014): not compressive (k 17 for N 17). Published fit (arXiv:1306.4468) reproduced exactly (their point χ² 1.111 vs 1.103; our optimizer 1.097 with their inputs). With our inputs the large-mixing mode exists but χ² = 129.9 (N 17): the gap is the error budget (they use 30-43% light-quark errors; we use PDG + 2% floor; with their errors our data give 1.53). >5σ tension at no-running level, not a clean exclusion (running/thresholds > 2%). Held-out PMNS from quark + charged-lepton data fails under both error budgets (sin²θ23 95% ≤ 0.14 vs 0.47): lepton mixing is accommodated by tuning 5 phases + s, not predicted. Exp 011's NPE was compute-starved; its "search did not reach the fit" explanation was wrong.
- Real-parameter SO(10) 10+126+120 "FN" (Grimus-Kühböck hep-ph/0612132; exp 014): k 16 (+~3 bits) for N 17, not compressive. Fits our GUT data (χ² 0.30, 1 dof) but predicts neither δ_ℓ (exact fits for δ_ℓ 40°-317°) nor PMNS from the quark sector (posterior not converged; indicative). Posterior regression of PMNS on CKM + masses: R² ≤ 0.12, no better than prior; PMNS variation carried by free parameters, not CKM.
- BSV pure type-II b-τ (hep-ph/0210207; exp 015): fails fit quality in SM and MSSM tanβ 5-55, M_SUSY 1-10 TeV: χ² 2700-4600 for N 18, pulls ≥ 35σ. Real model forces m_e ≈ 3 m_d (data m_d/2.5); m_b/m_τ(GUT) = 0.78 in MSSM (needs 0.998 for large θ23); δ_ℓ predicted 0-20°. Agrees with literature (Bertolini et al. hep-ph/0605006, Babu-Macesanu hep-ph/0505200, Joshipura-Patel 1102.5148). General complex variant did not converge (untested here). Known baseline.

- Modular flavour, literature (exp 016, 39 papers in `survey.csv`): no published unified quark+lepton modular model has < 13 real parameters; with bits counted every one has k > N. Most compressive holomorphic model, 2O (arXiv:2307.14926, 2409.15823), reproduced exactly (their refit χ² 20.17 vs 20.3); with our data (2% light-quark floor) χ² 68.8, N 15, k_cont 10, p 1.8e-13; lepton sector alone χ² 14.4 (s12² 0.325). Published good fits rely on 12-53% light-quark errors. T' non-holomorphic (2506.19822) at its published point: χ² 170 with our data (not refit). Known baselines.
- Modular flavour, common-τ catalog (exp 017): A4/S4/A5 forms to weight 8 verified (121 tests). 4.12M assignments (22 bits + 1). k ≥ 31 vs N - 2 = 13: fails compression by count. Best joint common-τ fit A4 IO χ² 446.5 (N 15). Leptons fit easily with their own τ; quark sector never fits (best A4 quarks χ² 255; search-limited, 30 of 12,100 A4 quark assignments fit exactly). Emulator useful as a coarse filter only (Spearman 0.15-0.77).
- T1 and F2-LO under the protocol (exp 018): both "no evidence". T1: power 0/20, N1 rate 0.49 (T1 contains no CKM), N2 0.29-0.32. F2-LO: power 0-0.05, all nulls fail within 7-8 fakes, N2 rate 0.63. Dropped from CANDIDATES. Ran with a fit-only hit rule because the strict rule is vacuous for k > N-2.

## Leads (not candidates)
- S4 τ-only neutrino sector (exp 017, from exp 016 lead): type-I seesaw, RH neutrinos in a triplet, constant M_D, one weight-2 form in M_N; L ~ 3', NO, τ = -0.180 + 1.098i, χ² 5.84 for 7 lepton observables (k_cont 5, p 0.054), pulls s23² +1.52, δ +1.26 (238°). Lepton-only, not compressive; likely the known NPPT S4 gCP family. Sharp test available: fix τ from leptons, ask whether any quark assignment fits at the same τ.
- T1 (exp 010, GUT scale): θ12 = sin(3 (m_u/m_c)^{1/4}), θ13 = sin(3 (m_u/m_t)^{1/4}). Joint χ² 0.058 NO; null rate ~1% for that cell, ~8% after the 8-cell selection; fails at M_Z (s13² +2.2σ); the same f gives θ23 at -6.8σ. N=2, k ≈ 20 bits.
- F2 LO: θ12e ≈ 0.96 θ_C. Revives only if a theory fixes it in advance (Clebsch on sinθ, thresholds) or ties it to mass ratios.

## Open questions
- MSSM needs 2-loop Yukawa RGEs for < 2% accuracy on θ23^q, θ13^q.
- Can any family reach k ≤ N-2? Every family tried (F1, F2, symbolic, SO(10) MN/FN, BSV) either has free phases/θν/Yukawa entries that absorb the CKM, or fails fit quality. Need families where quark and lepton sectors share a small number of continuous parameters by construction. Best-known class in the literature: modular flavour symmetry with a common modulus τ for quarks and leptons (few couplings, many observables).

## Assessment after 5 iterations
Seven families searched, all negative: shared rotation (F1), Clebsch corrections (F2), symbolic formulas, minimal and real SO(10), BSV type-II, modular flavour (literature + common-τ catalog). The recurring pattern: lepton mixing is always fitted by sector-specific freedom (phases, θν, Yukawa entries, couplings); whenever the quark and lepton sectors are forced to share parameters (BSV, common τ), the quark sector fails at PDG precision. The CKM is never used to predict PMNS. Light-quark mass precision (PDG 2026, ~2%) is the binding constraint that published "good fits" evade with 12-53% errors.

## Next planned tasks (iteration 6)
1. Infra: fix the `src/nulls.py` and `src/evaluate.py` gaps above (ok_fn + fit-only hit rule, p tolerance, N3 mass-side null, shared pool; same-scale mass ratios, Δm² ratio, theory floor, GUT-scale scoring; κ crosstalk).
2. Sharp modular test: take τ from the S4 τ-only lepton lead (and the 2O lepton island) and run an exhaustive quark-assignment fit at that fixed τ (all 968 S4 quark assignments, exact fits, 2% floor). Couplings only, no τ freedom: this is the first test where leptons predict something on the quark side.
3. Theory scout: given the assessment, propose families where quarks and leptons share parameters by construction and the shared count makes k ≤ N-2 achievable (e.g. modular + GUT Clebsch with one τ and a fixed assignment; Pati-Salam with SU(4) relations; quark-lepton complementarity derived from a symmetry), with explicit counts, and say which look viable at 2% light-quark precision. Also quantify what the negative results so far imply: an upper bound on how much CKM information PMNS could carry within the families tried.

# State

Iteration: 3 (2026-11-18)

## Status
Infrastructure built, verified, and tested (401 tests pass: `.venv/Scripts/python -m pytest -q`; `pytest.ini` limits collection to `tests/`). Skeptic audit (exp 012) found bookkeeping holes in `src/evaluate.py`, listed under "Known infrastructure bugs"; they must be fixed before any candidate is scored.
- `data/` + `src/data.py`: PDG 2026 (CKM global fit, masses) and NuFIT 6.1 (both orderings, two analyses: default `IC23_woSKatm`, alt `IC24_wSKatm`), 1D and 2D Δχ² tables. Exp 20261118-001.
- `src/mixing.py`, `src/haar.py`, `src/stats.py`: angle extraction, Haar sampling, χ²/p-value/Šidák/Clopper-Pearson, `Objective`. Exp 20261118-004.
- Env: Python 3.14 `.venv`, torch 2.15 CPU, zuko, sbi, PySR (Julia bootstrapped into `.venv/julia_env`). Exp 20261118-002.
- Baseline catalog: 19 entries / 26 known relations, 6 model families, RG survey. Exp 20261118-003.
- `data/` independently verified: 100 records + all 483,654 χ² table rows, 0 mismatches. Exp 20261118-005.
- `src/evaluate.py` + `src/baselines.py` (37 registered baselines): the official scorer. PMNS scored jointly (2D s23-δ table), IO offset added once to `chi2_global` only. N counts scored observables; data inputs are not scored. Exp 20261118-006.
- `src/rg.py`: 2-loop gauge, 1-loop Yukawa/λ/κ running, SM and MSSM (single M_SUSY threshold). SM CKM running matches arXiv:2510.01312 to ≤0.14%; MSSM θ23^q, θ13^q ~2% high (1-loop Yukawas). `data/sm_inputs.json` added. Exp 20261118-008.

## Compute facts
- `torch.xpu.is_available()` = True (Intel iGPU) but unusable: first-call kernel JIT takes minutes per op, MLP training made no progress in 4 min. Use CPU build.
- MLP emulators: `torch.set_num_threads(2)` per job (2 threads ≈ 8 threads in throughput), up to ~5 parallel jobs.
- Batched 3x3 eigh/SVD: 1 thread per process + multiprocessing pool. torch is 2-10x faster than numpy for batched matmul.
- Iteration 3 oversubscribed the machine (agents saw 86-100% CPU; NPE got 1 epoch). Cap total agent cores at 10 and give heavy training jobs priority.
- `src.rg.ckm_at` takes ~10 s per call (re-solves y_t); cache it.
- PySR and sbi write `outputs/`, `sbi-logs/` to cwd by default: point them at `artifacts/` or the experiment folder.

## Data caveats
- CKM hand-transcription concern closed: exp 005 rechecked from rendered PDG pages, all match.
- No CKM correlation matrix (PDG does not publish one).
- `pmns_delta_chi2` uses the 2D (s23sq, δ) table when both are given, otherwise sums 1D profiles; ignores θ13-θ23, θ13-δ, Δm²3ℓ-θ23 correlations. Default returns 0 at each ordering's own best fit; `relative_to="global"` adds the IO penalty (1.49 default analysis, 5.91 with SK-atm). Fixed in iteration 2: `stats.Objective` now takes `pmns_ordering_offset` (added once) and refuses a `relative_to="global"` partial in sum1d mode; δ pull sign now wraps.

## Known infrastructure bugs (exp 012, open)
- H2: N can be inflated with aliased observables (ckm_lambda = s12 = Vus = Vcd = direct_Vus; m_c, m_s, m_c/m_s). Need alias groups; cap CKM global-fit keys at 4; never mix global and direct.
- H3: target leakage undetectable (`inputs` is only a declaration). `experiments/20261118-012-skeptic-infra/leak_probe.py` catches loader-mediated leaks; make it mandatory.
- M1: input/prediction overlap checked by name only. M6: `use=None` scores every known key in `pred`.
- M2: ordering choice is free; criterion 3 must use p_global or add 1 bit. M3: discrete choices counted 3 different ways; use k_d = ceil(log2 #options) for compression, p_fit (dof N - k_cont) for fit quality, and pay for discrete selection in the null.
- M4: Gaussian-scored Haar null (a) has the wrong shape vs NuFIT tables, worst in IO; score fakes with translated NuFIT tables. M5: Haar-PMNS null is informational only.
- L1: ckm_delta_rad not wrapped. 7 strict-xfail tests in exp 012 encode the correct behaviour.

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
Official baseline scores (exp 006, no RG): no known relation has k ≤ N-2 with acceptable p. TM1 best (k=3, N=4: NO χ² 5.1 p=0.024; IO 1.8 p=0.18, p_global 0.069). Rough pulls (exp 003): QLC solar -4.3σ, self-complementarity +6.2σ, θ13=θ_C/√2 +4.9σ, exact TBM/GR1/GR2/HEX θ12 +3.7/-4.9/+5.5/-8.9σ, TM2 +4.8σ, leptonic GST -53σ.

- F1 shared rotation (exp 007): fails compression (k = 2 + 11.4 bits, N = 4). Null rates 36% (hierarchical) / 8% (Haar). Per exp 012 H1 that null had no power (a true F1 relation passes < 2% of the time), so this is "no evidence", not excluded. The CKM does supply θ13 in F1 (corrects iteration 2's "barely used"); any O(λ) hierarchical V does as well.
- F2 Clebsch charged-lepton corrections (exp 009): 17,316 configs (14.1 bits). Fails compression. LO-tier best: θ12e = θ_C, θ23e = θ_cb on TBM/GR2, s13 = sinθ_C sin(π/4 - θ_cb), χ² 7.2 (p 0.065, trials p 1). Hierarchical null 27-50%. Emulator inversion wants θ12e/θ_C = 0.961 ± 0.020, not a listed Clebsch value. Good overall fits come from free θν (TM-type), not the CKM. Sanity: TBC, solar sum rule, TM1/TM2 sum rules reproduced exactly.
- Single-observable symbolic formulas from CKM angles + mass ratios (exp 010): numerology at current precision. Up to 4.8M distinct functions (C ≤ 6); thousands per target within 1σ; no null cell below 1%. NN guidance found nothing brute force misses.
- Minimal SO(10) 10+126 (exp 011): not compressive (k 17-19 for 17 observables). Held-out PMNS prediction from quark + charged-lepton data fails in every mode found (predicts small lepton mixing; sin²θ23 ≤ 0.1). Not an exclusion: the published type-I fit with χ² ≈ 1 (arXiv:1306.4468) was not reached by our search. NPE compute-starved (1 epoch); real data outside its effective support.

## Leads (not candidates)
- T1 (exp 010, GUT scale): θ12 = sin(3 (m_u/m_c)^{1/4}), θ13 = sin(3 (m_u/m_t)^{1/4}). Joint χ² 0.058 NO; null rate ~1% for that cell, ~8% after the 8-cell selection; fails at M_Z (s13² +2.2σ); the same f gives θ23 at -6.8σ. N=2, k ≈ 20 bits.
- F2 LO: θ12e ≈ 0.96 θ_C. Revives only if a theory fixes it in advance (Clebsch on sinθ, thresholds) or ties it to mass ratios.
- Type-II SO(10): large θ23 only near b-τ unification (BSV, hep-ph/0210207; known, baseline). SM running gives m_τ/m_b = 1.59 at GUT; score with MSSM running.
- Real-parameter SO(10) 10+126+120 ("FN", 18 params): no free Yukawa phases to absorb a link.

## Open questions
- MSSM needs 2-loop Yukawa RGEs for < 2% accuracy on θ23^q, θ13^q.
- Can any family reach k ≤ N-2? Every family tried so far has free phases or a free θν that absorb the CKM. Need families where those are fixed by structure, with masses as extra observables.

## Next planned tasks (iteration 4)
1. Infra hardening: fix exp 012 H2, H3, M1, M2, M3, M6, L1 in `src/evaluate.py`; move the 7 xfails into passing tests in `tests/`; add `src/nulls.py` implementing the criterion-4 protocol (N1, N2 with translated NuFIT tables, hit definition, power check); `src/rg` input overrides + caching; vectorized profile accessor in `src/data`.
2. Reproduce the published minimal SO(10) type-I fit (arXiv:1306.4468, χ² ≈ 1) with a global optimizer as pipeline sanity, then the real-parameter 10+126+120 model and its held-out PMNS prediction. Give it 5 cores.
3. BSV type-II with MSSM running (tanβ scan): score as a baseline with the official scorer.

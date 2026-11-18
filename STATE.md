# State

Iteration: 2 (2026-11-18)

## Status
Infrastructure built, verified, and tested (401 tests pass: `.venv/Scripts/python -m pytest -q`).
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
- PySR and sbi write `outputs/`, `sbi-logs/` to cwd by default: point them at `artifacts/` or the experiment folder.

## Data caveats
- CKM hand-transcription concern closed: exp 005 rechecked from rendered PDG pages, all match.
- No CKM correlation matrix (PDG does not publish one).
- `pmns_delta_chi2` uses the 2D (s23sq, δ) table when both are given, otherwise sums 1D profiles; ignores θ13-θ23, θ13-δ, Δm²3ℓ-θ23 correlations. Default returns 0 at each ordering's own best fit; `relative_to="global"` adds the IO penalty (1.49 default analysis, 5.91 with SK-atm). Fixed in iteration 2: `stats.Objective` now takes `pmns_ordering_offset` (added once) and refuses a `relative_to="global"` partial in sum1d mode; δ pull sign now wraps.
- Fake-universe statistics: 0 hits in 100 gives a 95% upper limit of ~3%. Use ≥300 fake universes to certify a rate < 1%.
- Haar fake CKMs are not hierarchical. Need a second null that keeps the quark hierarchy (conditioned sampler or shifted-data universes).

## Current best candidates
None. See `CANDIDATES.md` for baselines and leads.

## Ruled out
Official scores (exp 006, no RG running): no known relation has k ≤ N-2 with acceptable p. TM1 best symmetry baseline (k=3, N=4: NO χ² 5.1 p=0.024, IO 1.8 p=0.18). Earlier rough pulls (exp 003):
- QLC solar (-4.3σ), self-complementarity (+6.2σ), θ13 = θ_C/√2 (+4.9σ), exact TBM/GR1/GR2/HEX θ12 (+3.7/-4.9/+5.5/-8.9σ), TM2 (+4.8σ), leptonic GST (-53σ).
- U_PMNS = V_CKM† Φ X with X in {TBM, BM, GR1, GR2, HEX}: best Σpull²(angles) ≥ 27.
- No known relation both compresses (k ≤ N-2) and survives current low-scale data.

## F1 shared rotation: ruled out as searched (exp 007)
- U_PMNS = f(V_CKM, Φ, X), 14 compositions × 10 X × 36 permutations = 2784 distinct configs (11.4 bits) with fitted α, β; 114,059 (16.8 bits) with structured phases.
- Fits NO well (best χ² 0.79, 11 configs < 2) but the fit is driven by Φ: TBM/GR2/BM/HEX tie. Since V_CKM ≈ 1, sandwiches V Φ X V† reduce to ≈ Φ X; the CKM is barely used.
- Hierarchy-preserving null (random CKM-like V): 36% of fakes match the real best (fitted phases), 19% (structured). Haar-PMNS null: 8% (NO, fitted). 300 fakes each.
- Lesson: any family where free phases can absorb the CKM correction tests X, not a CKM-PMNS link. Future families need observables the phases cannot absorb (masses, δ fixed by structure), or no free phases.
- L1 (V Φ U_BM V†) dropped: k=4=N, NO χ² 5.89 with δ, not singled out by the null.

## Open questions
- Common scale: M_Z, plus a GUT-scale check of every candidate with `src/rg.py` (SM: θ23^q, θ13^q +13.4% to 2e16 GeV, θ_C +0.04%; PMNS running ≤0.12° in SM). MSSM needs 2-loop Yukawa RGEs for <2% accuracy.
- Exp 003 `rg_running.md` had sinθ12 mistaken for θ12; erratum appended.
- Unverified literature items listed in exp 003 README (KMS original reference, some recent arXiv IDs, SO(10) 10+126 Clebsch signs).

## Next planned tasks (iteration 3)
1. F2: GUT-Clebsch charged-lepton corrections on symmetry-fixed U_ν (θ12^e tied to θ_C via Clebsch factors). Small k, predicts δ through sum rules. Emulator + PySR distillation; must rediscover TBC/TM1/solar sum rule first. Include the hierarchy-preserving null from the start.
2. Neural-guided symbolic search on CKM angles + mass ratios (observables phases can't absorb): grammar of low-complexity expressions predicting PMNS angles, NN scorer trained on synthetic families, distilled by PySR, with fake-universe null.
3. F3 groundwork: SO(10)/SU(5) Yukawa-texture generator using `src/rg.py`, synthetic dataset, first NPE (sbi/zuko) to find posterior ridges linking PMNS and CKM quantities.
4. Skeptic audit of `src/evaluate.py` parameter-counting conventions and the exp 007 null design.

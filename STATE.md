# State

Iteration: 1 (2026-11-18)

## Status
Infrastructure built and tested (364 tests pass: `.venv/Scripts/python -m pytest -q`).
- `data/` + `src/data.py`: PDG 2026 (CKM global fit, masses) and NuFIT 6.1 (both orderings, two analyses: default `IC23_woSKatm`, alt `IC24_wSKatm`), 1D and 2D Δχ² tables. Exp 20261118-001.
- `src/mixing.py`, `src/haar.py`, `src/stats.py`: angle extraction, Haar sampling, χ²/p-value/Šidák/Clopper-Pearson, `Objective`. Exp 20261118-004.
- Env: Python 3.14 `.venv`, torch 2.15 CPU, zuko, sbi, PySR (Julia bootstrapped into `.venv/julia_env`). Exp 20261118-002.
- Baseline catalog: 19 entries / 26 known relations, 6 model families, RG survey. Exp 20261118-003.

## Compute facts
- `torch.xpu.is_available()` = True (Intel iGPU) but unusable: first-call kernel JIT takes minutes per op, MLP training made no progress in 4 min. Use CPU build.
- MLP emulators: `torch.set_num_threads(2)` per job (2 threads ≈ 8 threads in throughput), up to ~5 parallel jobs.
- Batched 3x3 eigh/SVD: 1 thread per process + multiprocessing pool. torch is 2-10x faster than numpy for batched matmul.
- PySR and sbi write `outputs/`, `sbi-logs/` to cwd by default: point them at `artifacts/` or the experiment folder.

## Data caveats (to resolve)
- CKM global-fit numbers were hand-transcribed from PDF text (PDG has no CKM entries in its database). Only cross-checked by unitarity and J tests. Needs independent recheck against PDG Eqs. (12.26)-(12.28).
- No CKM correlation matrix (PDG does not publish one).
- `pmns_delta_chi2` uses the 2D (s23sq, δ) table when both are given, otherwise sums 1D profiles; ignores θ13-θ23, θ13-δ, Δm²3ℓ-θ23 correlations. Default returns 0 at each ordering's own best fit; `relative_to="global"` adds the IO penalty (1.49 default analysis, 5.91 with SK-atm). When using `stats.Objective` with `pmns_mode="sum1d"`, make sure the IO offset is not counted once per parameter.
- Fake-universe statistics: 0 hits in 100 gives a 95% upper limit of ~3%. Use ≥300 fake universes to certify a rate < 1%.
- Haar fake CKMs are not hierarchical. Need a second null that keeps the quark hierarchy (conditioned sampler or shifted-data universes).

## Current best candidates
None. See `CANDIDATES.md` for baselines and leads.

## Ruled out (rough single-observable pulls, no RG running, exp 20261118-003 `quick_status.log`; to be redone by the evaluator)
- QLC solar (-4.3σ), self-complementarity (+6.2σ), θ13 = θ_C/√2 (+4.9σ), exact TBM/GR1/GR2/HEX θ12 (+3.7/-4.9/+5.5/-8.9σ), TM2 (+4.8σ), leptonic GST (-53σ).
- U_PMNS = V_CKM† Φ X with X in {TBM, BM, GR1, GR2, HEX}: best Σpull²(angles) ≥ 27.
- No known relation both compresses (k ≤ N-2) and survives current low-scale data.

## Leads (not candidates)
- U_PMNS = V_CKM Φ U_BM V_CKM†, Φ = diag(1,e^{iα},e^{iβ}): Σpull²(3 NO angles) = 0.20 with α, β on a 5° grid, but δ ≈ 82° vs 207° (NO). k = 2 continuous + ≥2 discrete, N = 4. Fails compression as is.
- TM1 (+1.4σ on s12²), GST (+0.2σ), Koide m_τ (+0.4σ) survive; within-sector, not CKM-PMNS links.

## Open questions
- Common scale: plan is M_Z with SM running, plus a GUT-scale check for every candidate. Need our own one-loop RGE code validated against arXiv:2510.01312 tables (CKM angles θ23, θ13 run ~13% in SM to GUT).
- Unverified literature items listed in exp 003 README (KMS original reference, some recent arXiv IDs, SO(10) 10+126 Clebsch signs).

## Next planned tasks (iteration 2)
1. Data verifier (fresh agent): recheck every number in `data/` against PDG 2026 / NuFIT 6.1 sources, especially the hand-transcribed CKM values.
2. Evaluator: `src/evaluate.py` + move `baselines.py` into `src/`; score every baseline with the loader + `stats.Objective` (joint PMNS Δχ²), both orderings, both NuFIT analyses; produce the official baseline table. Switch exp 003 scripts to the loader.
3. F1 shared-rotation search: enumerate finite X catalog × orderings × permutations × Φ phases for U_PMNS = f(V_CKM, Φ, X), with the same pipeline run on ≥300 Haar fake universes and a hierarchy-preserving null. Count trials.
4. RG module: one-loop SM (and MSSM tanβ option) running of CKM and PMNS (κ RGE, hep-ph/0108005), validated against 2510.01312.

# 20261118-003 Baseline catalog, model-family menu, RG recommendation

**Hypothesis:** N/A (literature survey and infrastructure).

**Method:** arXiv/INSPIRE-style search (web search plus the arXiv API for every cited ID; titles and authors checked against the API listing), transcription of each relation into one convention (PDG standard parametrization), implementation in `baselines.py` with structural self-tests, and a rough one-observable screen against `data/` (`quick_status.py`). RG magnitudes taken from Antusch-Hinze-Saad (arXiv:2510.01312) for CKM and estimated from the analytic RGEs of hep-ph/0305273 for PMNS (`rg_estimate.py`).

**Parameter count:** N/A. Each catalogued relation carries its own (N, k_c, k_d) in `baselines.md`.

## Files

| File | Content |
|---|---|
| `baselines.md` | 19 catalog entries (26 named relations/patterns) with formulas, N, k, scale, verified refs, status |
| `model_families.md` | 6 generator families + null families, ranked, with first-experiment recommendation |
| `rg_running.md` | comparison-scale recommendation, CKM and PMNS running magnitudes, implementation route |
| `baselines.py` | numpy implementation of every baseline; `python baselines.py` runs self-tests |
| `quick_status.py` / `quick_status.log` | rough pulls of each baseline vs data/ (no running, no correlations) |
| `rg_estimate.py` / `rg_estimate.log` | linear one-loop PMNS RG shift estimates, SM and MSSM |

Rerun (from repo root):
```
.venv/Scripts/python experiments/20261118-003-baseline-catalog/baselines.py
.venv/Scripts/python experiments/20261118-003-baseline-catalog/quick_status.py
.venv/Scripts/python experiments/20261118-003-baseline-catalog/rg_estimate.py
```
`quick_status.py` and `rg_estimate.py` read `data/*.json` directly because `src/` had no loader function when this was written; switch to the loader once it exists.

## Result

- No known relation in the catalog both meets k ≤ N − 2 and survives current data at low scale.
- Rough pulls (NuFIT 6.1 IC24, PDG 2026, no running): QLC solar −4.3σ; self-complementarity +6.2σ; TBC/KMS θ13 = θ_C/√2 +4.9σ (NO); TBM s12² +3.7σ, GR1 −4.9σ, GR2 +5.5σ, HEX −8.9σ, BM excluded; TM1 s12² +1.4σ (best surviving symmetry baseline), TM2 +4.8σ; GST +0.2σ; Koide m_τ +0.4σ; leptonic GST −53σ.
- QLC atmospheric relation needs θ23 = 45° − θ_cb for NO (−0.9σ) and 45° + θ_cb for IO (−0.5σ), i.e. one data-selected sign per ordering.
- δ_KM^q + δ_KM^l (2003.09921): −57° (NO), +5° (IO).
- Lead for the evaluator (not a candidate): U_PMNS = V_CKM · diag(1, e^{iα}, e^{iβ}) · U_BM · V_CKM† fits the three NO angles (Σpull² = 0.20 with α, β fitted on a 5° grid) but predicts δ ≈ 82° vs 212°. k = 2 continuous + 2 discrete for N = 4.
- RG: CKM θ23, θ13 run +13% (SM) / −8 to −12% (MSSM) to the GUT scale; θ_C +0.9%. PMNS running is negligible in the SM for m_lightest ≲ 0.05 eV; large in the MSSM for tan β ≳ 30 or quasi-degenerate masses. TBC-type θ_C → θ13 relations shift by 0.7σ from θ_C running alone, so they must be run.

## Verdict

Survey complete. Baselines and generator menu ready for the evaluator and for the first search experiments (F1 shared rotation with learned X, F2 Clebsch-corrected patterns with symbolic regression, F3 GUT textures with NPE). All arXiv IDs cited were checked against the arXiv API or arXiv abstract pages except where marked in `baselines.md`.

## Unverified items

- Original Mohapatra-Smirnov paper behind the "KMS" relation: not identified (attribution via arXiv:1206.4779).
- Leptonic GST: no canonical single reference identified.
- arXiv:2608.04266, 2603.06934, 2603.11341, 2407.18279, 1506.02324, 1412.0116, 1511.02143, 1711.08796, 1304.4526, 1402.5834: title/authors from an arXiv API search listing; content judged from abstract only.
- CKM running table values come from WebFetch extraction of the 2510.01312 HTML; spot-check against the paper before the RG integrator is validated against them.
- 10 + 126 SO(10) Clebsch signs in `model_families.md` F3 written from memory; fix against Babu-Mohapatra (hep-ph/9209215) before implementing.
- Claim that two-loop CKM-angle running effects are below CKM errors: not checked.

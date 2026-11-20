# 20261119-031 Skeptic audit of the whole program (exps 001-030, STATE, CANDIDATES)

## Hypothesis

The committed verdicts in `STATE.md` and `CANDIDATES.md` follow from what experiments 001-030 actually show: the wording follows criterion-4 protocol point 7, the k counts are honest, the exp 026 optimizer landscape bug does not invalidate any fit-quality verdict, and the quoted statistics match the logs. The audit tries to break each of these.

## Method

1. Read `OVERSEER_PROMPT.md`, `CLAUDE.md`, every experiment README (001-030), and the scripts and JSON behind each negative verdict. `STATE.md` and `CANDIDATES.md` were read only after that.
2. Wording: listed every negative statement in STATE and CANDIDATES and checked its basis. A fit-quality failure or a null with demonstrated power (≥ 50%) is the only basis allowed for "excluded" or "ruled out".
3. Parameter counts: compared every CANDIDATES row and the main STATE k values with the k each experiment's code and README use.
4. Optimizer bug (exp 026): traced every committed fit to the objective it minimizes (`grep -n "chi2_soft\|fit_and_evaluate\|least_squares\|Nelder"` over `experiments/*/*.py`, then read the residual functions). Reran the fits whose verdict depends most on fit quality with a global optimizer (`rerun_bsv_de.py`, `rerun_2o_de.py`, driver `run_all.sh`).
5. Statistics: `check_stats.py` recomputes every p, p_global, Clopper-Pearson interval, Šidák threshold and power number that STATE quotes, from the committed JSON. `bckm_injected.py` reads the B_ckm values of exp 030's injected-true universes. Checked the exp 028 pre-registration hash, and checked whether exps 025 and 028 read the same NuFIT point.
6. Coverage gaps: listed the relation classes no experiment tested.

No `src/` file, STATE, CANDIDATES or other experiment was edited. Measured values enter only through `src` loaders, imported by the original experiments' code.

## Parameter count

N/A (audit). The reruns refit published models with their original parameterisations (exp 015: 2-3 model parameters + 10 nuisances + 2 sign bits; exp 016: 10 continuous).

## Result

### Findings table (ranked; nothing here changes a candidate verdict, because there is no candidate)

| # | sev. | claim (where) | evidence | what changes |
|---|---|---|---|---|
| 1 | **high** | "The CKM carries ≲ 1-2 bits of PMNS information" is called a **bound** (STATE exp 021 entry and Assessment: "exp 021 bounds CKM-specific PMNS information at ≲ 1-2 bits"; exps 025 and 028 cite it as independent agreement). | The bound is the observed B_ckm through the F1, F2, T1 and symbolic search pipelines. None of those pipelines has power (exp 012 H1: a true F1 relation passes 0.7-15% of the time depending on null and variant; exp 018: T1 and F2-LO power 0-5%; exp 030: 0-5%). Exp 030's power runs inject universes where the bi-large relation is **true** and store B_ckm for each: median +1.3 bits (N1 NO), +0.7 (N1 IO), +1.1 (N2 NO), −0.4 (N2 IO); maximum 2.3 bits; 85-100% of injected-true universes give ≤ 2 bits (`bckm_injected.log`). A true CKM → PMNS relation therefore produces the same "≲ 1-2 bits" through these instruments. | The number is the ceiling of the instruments, not an upper limit on information. Reword as "observed B_ckm ≈ 0-1 bits; an injected true relation gives ≤ 2.3 bits through the same pipelines, so this is no evidence, not a bound". The statements "consistent with 0" and "≲ 10% of the budget" should go. Exp 028's agreement with it is not independent corroboration. |
| 2 | medium | STATE (Known infrastructure gaps, exp 026): "Earlier searches that started outside the tables (exps 011, 014, 017) may have been landscape-limited". Next-task item 1 asks for reruns of 011, 014, 017. | The bug only acts when **one** `Scorer` holds both NuFIT-table keys and finite keys (CKM, `mr_*`) and is minimized through `chi2_soft` without `include_finite`. Of the committed fits, only exp 022 (`fit.py`, `perturb.py`, `de.py`: N 14 with CKM + mass ratios + PMNS) does that, and exp 026 already reran it (DE and L-BFGS-B reach 22.40-22.42). Exps 011, 014, 015 and 023 use their own residual vectors, which always keep the CKM and mass terms. Exps 016, 017 and 020 use separate PMNS-only and CKM-only Scorers, with mass terms added outside `chi2_soft`. Exps 006, 013, 018, 029 and 030 fit PMNS-only use sets (no finite terms to drop). In exp 027 CSD2-GUT the only finite key (`mr_d_s`) does not depend on the fit parameters. | The bug invalidates no committed verdict. The STATE sentence names the wrong experiments. Exp 017's quark sector **is** search-limited (exp 020 side result: 2163 → 196 at a random τ), but for a different reason (local Nelder-Mead from emulator starts), and its verdict rests on compression anyway. |
| 3 | medium | BSV (exp 015; STATE, CANDIDATES): "χ² 2700-4600 for N 18". | Same objective as exp 015's fit stage, DE + least-squares polish (`rerun_bsv_de.py`). The real variants (R0, R0eta) at SM, tanβ 10 and tanβ 55 give DE χ² 1976-2194, against committed 3252-4607. Each still has an m_d pull of about −38σ. The general complex variant C gives 506 (SM) and 373 (tanβ 10), against committed 10564. | The quoted range is about 2× too high: exp 015's search was local and limited. The verdict "fails fit quality" stands (p = 0), and C stays non-compressive (k_cont = N). Quote ≈ 2000. |
| 4 | medium | STATE exp 010 entry: "Single-observable symbolic formulas ... numerology at current precision". | Basis: no null cell below 1%. Those nulls have no demonstrated power (exp 018 ran the protocol on the best symbolic lead, T1: power 0/20). | Protocol point 7 wording: "no evidence". "Numerology" asserts the formulas are coincidences, which the nulls cannot show. |
| 5 | medium | STATE Leads, T1: "null rate ~1% for that cell, ~8% after the 8-cell selection". | Superseded by exp 018 (N1 rate 0.49 over 1000 fakes, N2 0.29-0.32, power 0/20); T1 was dropped from CANDIDATES for that reason. | Replace with the exp 018 numbers, or move T1 out of Leads. As written it suggests a near-significant lead. |
| 6 | medium | A4-cat "fails fit and compression" (CANDIDATES; STATE exp 017 "Best joint common-τ fit A4 IO χ² 446.5"). | Exp 017's exact fits are local Nelder-Mead from emulator starts. Exp 020 found that the S4 quark sector at a random τ gives χ² 196 against exp 017's free-τ optimum 2163, so exp 017's joint χ² values are upper bounds on the minimum. | "Fails fit" is not demonstrated; "fails compression" (k ≥ 31 vs N − 2 = 13) is, and it carries the verdict. Reword as "fails compression; joint fit not converged (χ² ≤ 446.5)". |
| 7 | low | CANDIDATES BSV k "12-15 + 2 bits". | Exp 015: k_cont = 12 (R0), 13 (R0eta), 18 (C), each including 10 quark/CKM nuisances; discrete 2 sign bits + 1 ordering bit + 3 bits when tanβ is chosen from 7 by the data. Exp 015's own README: honest k = 18 (R0), 19 (R0eta) > N − 2 = 16. | k should read "12-13 (real variants), 18 (C) + 2-6 bits". With the honest count BSV fails compression as well as fit quality; CANDIDATES lists only fit quality. Verdict unchanged. |
| 8 | low | CANDIDATES L1 "covered by F1 null ... dropped (exp 006, 007)". | The F1 null has no power (exp 012 H1), so it cannot cover anything. The drop is justified by compression: k = 2 + 2 bits = 4 > N − 2 = 2. The fit alone is marginal: NO χ² 5.89 with dof 2 gives p = 0.053. | Cite compression, not the null. |
| 9 | low | Exp 028 README, Verdict: "this excludes a CKM → PMNS link strong enough to make the real CKM a top-1% predictor". | The same README states that no injection test was run. The sentence restates the observed rank (0.52), but "excludes" is protocol-reserved. STATE does not repeat it. | Wording only: "the real CKM is not a top-1% predictor (rank 0.52)". |
| 10 | low | 2O-mod fit (exp 016): χ² 68.8 from local least squares started at τ-profile points. | DE over the full τ domain and signed log-couplings (`rerun_2o_de.py`; see the rerun table) finds nothing lower: seed 1 gives 872.8, seed 2 gives 75.4 at a second τ island (0.171 + 0.901i), and seed 3, with the committed point in the initial population, returns 68.84. | Confirms the committed fit. |
| 11 | low | Exp 028 pre-registration. | `artifacts/028/README_prereg.md` hashes to `ec31fbd8bd00cc76…` as logged in `run.log`. Its 46 pre-registration lines are identical (CRLF aside) to the committed README; only the "Deviations" line differs, as disclosed. The hashed file lives only in gitignored `artifacts/`, so outside readers cannot check the hash. | Commit `README_prereg.md`, or log the full hash of a committed file, in future pre-registrations. |
| 12 | low | Exp 028 caveat: "pmns_best ... may differ from the one exp 025 read". | Both runs used the identical NO point: s12² 0.3088, s13² 0.02249, s23² 0.47, δ 207° (exp 025 `results.json` → `real`; exp 028 `results_NO.json` → `real.best`). The committed `src/data.py` and `data/` are unchanged since iteration 4. The working-tree edit by exp 033 during this audit only adds functions. | Confound closed. Exp 025's 2σ hint vanishing in exp 028 is not an input difference. |
| 13 | low | Leakage in the exp 025/028 theory-space mixture. | S4SO samples are log-uniform ±2 decades around the paper's Table 4 point, a fit to (2017) measured data, so that family's training set is built around real-like CKM and PMNS together. Exp 025's drop-S4SO run raises T (2.06 vs 1.80), so S4SO does not drive the hint; exp 028 is null. FN and MN boxes are data-independent (checked `so10fit.fn_box`, `gen.mn_box_broad`). The λ-matched null in exp 028 was chosen after exp 025's real-data diagnostic, but a stricter null can only make a pass harder. | No verdict changes. For future mixtures, centre literature-model priors on structure, not on fitted points. |
| 14 | info | p, p_global, CP, Šidák and power numbers in STATE. | `check_stats.py`: 45 rows, 0 mismatches. This includes SU5-CSD3 p_global 4.4e-4, 2O 1.8e-13, CDSV 0.042 / 0.10, TM1 0.078 / 0.191, F2-LO 0.065, exp 025 conv rank 0.021 with CP95 [0.015, 0.028], exp 028 ranks 0.52 / 0.38 / 0.60 with one-sided CP95 0.54, exp 020's [0.93, 0.998], the exp 029 Šidák 3.95σ (M = 35, the NO survivors), and the exp 030 one-sided CP uppers and powers. | None. One nuance: SU5-CSD3's analytic bound χ² ≥ 14.7 gives p ≤ 0.0117, which does not reach the pre-registered kill (p < 0.01) on its own. The kill rests on the fit (22.40), which exp 026's DE confirmed. |

### Verdict wording, item by item (protocol point 7)

| negative result | STATE / CANDIDATES wording | basis | ok? |
|---|---|---|---|
| Known baselines A1-C2 (exp 006/013) | "excluded" in exp 006 table | fit quality (p ≤ 1e-3) | yes |
| F1 (007) | "fails compression; no evidence" | compression; null without power | yes (STATE). The iteration-2 commit message "ruled out" remains in history. |
| L1 (006/007) | "dropped", "covered by F1 null" | compression | wording cites the wrong basis (#8) |
| F2 (009) | "Fails compression", null rates quoted | compression | yes |
| Symbolic (010) | "numerology at current precision" | null without power | **no** (#4) |
| MN SO(10) (011/014) | "not compressive", ">5σ tension, not a clean exclusion" | compression | yes |
| FN SO(10) (014) | "not compressive" | compression | yes |
| BSV (015) | "fails fit quality" | fit (p = 0) | yes; numbers not minimal (#3) |
| 2O (016) | "fails fit and compression" | fit p 1.8e-13 (confirmed by DE), compression with bits | yes |
| A4 catalog (017) | "fails fit and compression" | compression; fit not converged | partly (#6) |
| T1, F2-LO (018) | "no evidence", dropped | nulls, power 0-5% | yes; T1 Lead text stale (#5) |
| S4 fixed τ (020) | "fails fit and compression"; "τ not special" | fit p = 0 with exact per-assignment fits; random-τ rank | yes |
| CKM information "bound" (021) | "carries ≲ 1-2 bits", "bounds" | pipelines without power | **no** (#1) |
| SU5-CSD3 (022) | "fails fit quality" | fit p 4.4e-4 (DE-confirmed) + analytic bound | yes |
| S4×SO(10) (023/026) | "fails compression" | k_cont 17 ≥ N 16 | yes |
| NN tests (025, 028) | "no evidence" | nulls without injection power | yes in STATE; exp 028 README uses "excludes" (#9) |
| Bi-large (027/030) | "no evidence (not excluded; no power)", "class closed as a candidate source" | power 0-5%; compression with honest count | yes: "closed" follows from compression, not from the null |

### Parameter counts

Spot-checked against code and READMEs: F1 (2 + log2 2784 = 11.4 bits), F2 (17,316 configs = 14.1 bits), 2O (k_cont 10: τ + 8 coupling ratios; ~15 bits estimated), A4-cat, S4-fixτ (k_cont 4-6 + 10/14 bits; τ fixed by leptons and not counted, correctly, since leptons are not in N), SU5-CSD3 (9 + 1, 22 strict), S4×SO(10) (`fit_ours.k_of`: 10 + 6 phases + η̄_b = 17, η̄_b counted as data-selected per exp 026), BL-CDSV (1 continuous + 11 grammar bits = 12; exp 030 counts 14 with ordering and grammar), FN (16 + ~3), TM1 (2 + 1). Apart from #7 (BSV, a misquote that understates the failure), no under- or over-count changes `compression_ok`. The one count that is a judgement call is BL-CDSV's 11 grammar bits. With the published count (k = 1 + 1 bit) NO-only is compressive (2 ≤ 2) but fails fit (p_global 0.042), and IO needs the ordering bit (3 > 2). The verdict is the same under either count.

### Optimizer bug (exp 026): which committed fits are exposed

| exp | objective minimized | exposed to the dropped-terms landscape? | verdict hinges on fit? | action |
|---|---|---|---|---|
| 006/013 baselines | `fit_and_evaluate` NM, PMNS-only `use` | no (no finite terms) | yes (low-dim, 32 starts) | none |
| 007, 009 | own NM on `pmns_delta_chi2` | no | no (compression) | none |
| 011 | own least squares / DE-MC on GUT Gaussian pulls | no | no (k = N); NPE compute-starved (already disclosed) | none |
| 014 | own `Problem.residuals` (Gaussian, all terms) | no | no (k = N) | none |
| 015 | own `common.residuals` (all terms, fallback penalty only on PMNS) | no | **yes** at fixed tanβ (k 14-15 ≤ 16) | DE rerun |
| 016 | separate PMNS / CKM Scorers + external mass terms, least squares | no | partly (compression fails with bits; fit decides if bits are waived) | DE rerun |
| 017, 020 | separate sector Scorers + external mass terms, NM | no | no (compression); fit search-limited (#6) | none |
| 022 | `fit_and_evaluate` NM, joint CKM + mr + PMNS | **yes** | yes | already rerun in exp 026: 22.40-22.42, verdict stands |
| 023 | own residuals (CKM and mass terms always kept) | no | no (k_cont ≥ N) | none (exp 026 refit) |
| 027 | `fit_and_evaluate` NM; CSD2 `use6` has one constant finite key | no in effect | CDSV/RMSV yes (1-2 params, 16 starts) | none |
| 029, 030 | PMNS-only fits | no | n/a | none |

### Reruns (`run_all.sh` → `run_bsv_*.log`, `run_2o.log`; JSON in gitignored `artifacts/031/`)

| exp | fit | seed | committed χ² | DE χ² | note |
|---|---|---|---|---|---|
| 015 BSV | C MSSM_tb10_ms3TeV NO | 1 | - | 372.6 | fit-stage chi2_gut, N 18; max nuisance pull 3.3; DE popsize 15 maxiter 600 |
| 015 BSV | C SM NO | 1 | 10564.1 | 506.2 | fit-stage chi2_gut, N 18; max nuisance pull 5.1; DE popsize 15 maxiter 600 |
| 015 BSV | R0 MSSM_tb10_ms3TeV NO | 1 | 3752.6 | 1997.5 | fit-stage chi2_gut, N 18; max nuisance pull 37.9; DE popsize 15 maxiter 400 |
| 015 BSV | R0 MSSM_tb55_ms3TeV NO | 1 | 3511.6 | 1993.7 | fit-stage chi2_gut, N 18; max nuisance pull 37.8; DE popsize 15 maxiter 400 |
| 015 BSV | R0 SM NO | 1 | 4607.4 | 2194.0 | fit-stage chi2_gut, N 18; max nuisance pull 38.5; DE popsize 15 maxiter 400 |
| 015 BSV | R0eta MSSM_tb10_ms3TeV NO | 1 | 3542.9 | 1989.8 | fit-stage chi2_gut, N 18; max nuisance pull 37.8; DE popsize 15 maxiter 400 |
| 015 BSV | R0eta MSSM_tb55_ms3TeV NO | 1 | 3252.3 | 1975.8 | fit-stage chi2_gut, N 18; max nuisance pull 37.7; DE popsize 15 maxiter 400 |
| 015 BSV | R0eta SM NO | 1 | 3689.0 | 2007.3 | fit-stage chi2_gut, N 18; max nuisance pull 38.1; DE popsize 15 maxiter 400 |
| 016 2O | joint, N 15 | 1 | 68.8 | 872.8 | tau +0.064+1.174i; worst pull s13sq -18.5 |
| 016 2O | joint, N 15 | 2 | 68.8 | 75.4 | tau +0.171+0.901i; worst pull s12sq +5.0 |
| 016 2O | joint, N 15 (committed point in DE init) | 3 | 68.8 | 68.8 | tau -0.205+1.069i; worst pull s12sq +5.3 |

The exp 015 numbers are the fit-stage GUT χ² (same `common.residuals`, same box, nuisances in ±60σ). The committed value is the best over sign patterns in `fits_R0*.jsonl` / `fits_C.jsonl`. The DE value is the best over the same sign patterns, one seed.
- **BSV R0 / R0eta:** DE roughly halves the committed χ² in every scenario, from 3252-4607 down to 1976-2194. Exp 015's 5-start local search was far from the minimum. Every DE optimum still carries an m_d pull of about −38σ (the m_e ≈ 3 m_d problem from exp 015), so p = 0 and the fit-quality verdict stands. The STATE/CANDIDATES range "2700-4600" should read "≈ 2000 (DE) for the real variants".
- **BSV C (general complex, k_cont 18 = N):** DE gives 506 (SM) and 373 (MSSM tanβ 10), against exp 015's unconverged 10564 and exp 011's DE-MC 462. It is not compressive at any χ², so the verdict is unchanged.
- **2O (exp 016):** none of the three DE runs beats the committed 68.84. Seed 2 finds a second τ island (0.171 + 0.901i, χ² 75.4). The seed seeded with the committed point returns 68.84 (Δ −0.001). The committed fit is the global minimum as far as DE can tell.

No χ² dropped far enough to change a verdict.

### Coverage gaps (relation classes never tested)

1. **Group-fixed quark angles.** Residual-symmetry models where one finite group fixes both mixings, e.g. D_n / Δ(6n²) giving θ_C from group data (|V_us| = sin(π/14)-type, Lam; Holthausen-Lim; Talbert). Every family tested so far takes the CKM from data.
2. **Rephasing invariants and moduli.** J_ℓ vs J_q, element-wise |U_ij|² vs |V_ij|² relations, unitarity-triangle angles. The exp 010 grammar uses only angle and phase leaves and targets only s12², s13², s23², δ.
3. **Δm² ratio as a target.** Δm²21/Δm²3ℓ against quark mass ratios or λ^n. It enters model fits only as a scored observable, never as a symbolic or relation target.
4. **Other neutrino-mass mechanisms with quark-lepton sharing.** Dirac neutrinos with Y_ν = Y_u (Pati-Salam), inverse or linear seesaw, type III, and radiative models (Zee, Zee-Babu) where PMNS structure follows charged-lepton masses. Only type I and type II were tested.
5. **Intermediate scales and seesaw thresholds.** Relations imposed at M_R or a Pati-Salam scale, and κ running between the M_R thresholds. All comparisons are at M_Z or 2e16 GeV with no seesaw thresholds.
6. **Shared zero textures.** Fritzsch or four-zero textures imposed on all four sectors at once, and GST in both sectors jointly. Only the leptonic GST baseline was scored.
7. **The CKM ← PMNS direction outside modular τ.** Only exp 020 predicts quark data from lepton data.

## Verdict

**Hypothesis partly rejected.** No committed verdict about a candidate changes: no family reaches k ≤ N − 2 with acceptable fit, and the exp 026 landscape bug touches only exp 022, which exp 026 already reran. Two program-level statements are not supported as written:
- The ≲ 1-2 bit "bound" on CKM-specific PMNS information (#1). Injected-true relations give the same value through the same pipelines.
- Exp 010's "numerology" (#4), plus stale or misattributed wording in #5, #6 and #8.

The BSV χ² range (#3) is not a minimum. The fit-quality verdict holds.

## Files

- `README.md`
- `check_stats.py` → `check_stats.json`, `check_stats.log`: recomputes the quoted statistics.
- `bckm_injected.py` → `bckm_injected.json`, `bckm_injected.log`: B_ckm of exp 030's injected-true universes.
- `rerun_bsv_de.py`: DE + least-squares rerun of exp 015's fit-stage objective. Usage: `<variant> <scenario> <seed> [maxiter] [popsize]`.
- `rerun_2o_de.py`: DE + trf rerun of exp 016's joint 2O fit. Usage: `<seed> [maxiter] [use_x0]`.
- `run_all.sh`: the reruns as run (4 processes, about 30 min) → `run_bsv_R0eta.log`, `run_bsv_R0.log`, `run_bsv_C.log`, `run_2o.log`. Per-run JSON in `artifacts/031/` (gitignored).
- `summarize_reruns.py` → `reruns_summary.json`, `reruns_summary.md` (the rerun table; needs the artifacts, so run it with `PYTHONIOENCODING=utf-8` on Windows).

Run everything from the repo root with `.venv/Scripts/python`.

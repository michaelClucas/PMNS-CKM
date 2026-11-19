# 20261119-030 Yukawa engine, native flow log-ratio statistic, bi-large grammar under the criterion-4 protocol

Part B pre-registration below was written and frozen before any fake universe was run; Part A and the results were added afterwards.

## Part B pre-registration (frozen before running fakes)

**Hypothesis.** The bi-large Cabibbo-seed grammar (CDSV 1902.08962 forms I/II, RMSV 1410.3658 types 1-4, as implemented in exp 027) singles out the real CKM: fake CKMs (N1) or fake PMNS data (N2) fed through the same search rarely give an equally good fit.

**Pipeline** (`bilarge.py`, `pipeline(universe)`):
- Grammar: U_PMNS = U_l(λ, A, φ)† U_ν.
  - CDSV part: U_l ∈ {form I = R23(θ23^q) F R12(θ12^q) F*, form II = F R12(θ12^q)^T F* R23(θ23^q)^T}, F = diag(e^{−iφ/2}, e^{iφ/2}, 1), sin θ12^q = λ, sin θ23^q = Aλ². U_ν = R23 U13(δ_ν) R12 with each sin θ_ij^ν ∈ {nλ, 1 − nλ : n = 1..4}, δ_ν ∈ {0, π}. 2048 configurations, φ continuous.
  - RMSV part: sin θ13^ν = λ, sin θ12^ν = sin θ23^ν = ψλ, δ_ν = 0, U_l types 1-4 (exp 027 `verify_bilarge.rmsv`); ψ ∈ [1, 4.4] and φ continuous. 4 configurations.
  - Options with a sine outside (0, 1) are skipped on that universe.
- λ and A are read from `universe.load_ckm()` (declared inputs `ckm_lambda`, `ckm_A`). PMNS is scored only through the universe's scorer (translated NuFIT tables on N2 and injected universes).
- Search: φ on a 1° grid, ψ on a 0.025 grid. Table χ² per ordering (vectorised copy of `Universe.pmns_delta_chi2`, agreement 3e-14 in `check_pipeline.log`). Per ordering, the 6 configurations with the highest coarse p_global are polished (CDSV: bounded Brent on φ within ±1°; RMSV: Nelder-Mead on (ψ, φ)) with `Universe.scorer(...).chi2`, then scored with `src.evaluate.Scorer`.
- Selection score: largest p_global over both orderings (ordering chosen inside the pipeline; ties by lower χ²_global).
- Count: N = 4 (s12², s13², s23², δ). k = k_cont (1 CDSV, 2 RMSV) + 12 bits (ceil log2 2052) + 1 ordering bit = 14 or 15. k > N − 2 on every universe, so the strict hit rule is vacuous.
- Hit rule: `src.nulls.is_hit_fit_only` (max|pull| < 2 and p_global ≥ the reference's p_global × (1 − 1e-5)). **Informational**: it cannot satisfy criterion 4, which needs k ≤ N − 2. Power check uses `candidate_ok_fit_only` (max|pull| < 2, p_global > 0.05). Strict hits are also counted from the stored summaries.
- Real-data reference (from `check_pipeline.log`, run before any fake): CDSV-I (s12 = 2λ, s13 = λ, s23 = 1 − λ, δ_ν = π) in IO, χ² 4.710, χ²_global 6.195, p_global 0.1025, max pull 1.68. Best NO entry: RMSV type 1/3, χ² 4.790, p_global 0.0912. Exp 027's fits are reproduced to 3 decimals.
- Protocol runs (`run_protocol.py`, 2 worker processes):
  1. Power first: `power_check`, 20 injections × 300 fakes, for N1 and N2 in each ordering. Injection = `make_injector` of the real best result of that ordering (PMNS prediction) plus ckm_lambda, ckm_A at their real values (PDG noise on both). Seed [30, 1].
  2. N1 and N2, 1000 fakes per null per ordering, early stop only to declare failure (> 4 hits). Seed [30, 0].
  3. Rate runs: 200 fakes per null and ordering without early stop (seed [30, 2]), only to estimate the hit rate and B_ckm. They cannot change a verdict.
- Decision: pass if the one-sided 95% CP upper limit < 1% (≤ 4 hits in 1000). Power < 50% means the null cannot confirm or exclude the family. Wording per protocol point 7: a failed null is "no evidence"; "excluded" only for fit-quality failure or a null with demonstrated power.

## Part A: infrastructure (additive)

### A1. `src/yukawa.py`: one Yukawa-matrices → observables engine

`observables(Y_u, Y_d, Y_e, m_nu=None, *, Y_nu=None, M_R=None, ordering="auto", nu_mass_scale_eV=None, details=False)` returns a dict keyed as `src.evaluate` expects: `ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, s12sq, s13sq, s23sq, dcp_deg, mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau, R_clebsch, dm_ratio` (+ `dm21sq_eV2, dm3l_sq_eV2` when `nu_mass_scale_eV` is given) plus `alpha21_deg, alpha31_deg` (not scored; `evaluate` ignores keys outside `use`). Helpers: `left_rotation`, `takagi`, `seesaw_type1`, `majorana_phases`.

Convention (module docstring has the full statement):
- Dirac Yukawas LR, −L = conj(f_L) Y_f f_R H; left rotation diagonalises Y Y†.
- V_CKM = U_u† U_d, U_PMNS = U_e† U_ν.
- Majorana: −L = ½ νᵀ m_ν ν, m_ν = U_ν* diag(m_i) U_ν† (PDG). U_ν is the Takagi factor (eigenvectors of m_ν† m_ν, column phases fixed so U_νᵀ m_ν U_ν is real non-negative).
- Type-I seesaw: m_ν = −Y_ν* M_R⁻¹ Y_ν† (Y_ν LR, M_R the two-component matrix of N = ν_R^c).
- δ, angles: `src.mixing` (PDG, δ ∈ [0, 2π)). Majorana phases PDG, U = U_std diag(1, e^{iα21/2}, e^{iα31/2}): α21 = 2 arg(U_e2 U_e1*), α31 = 2[arg(U_e3 U_e1*) + δ].
- Ordering: "auto" labels NO if m_b² − m_a² < m_c² − m_b² (ascending a < b < c), else IO (columns reordered (1, 2, 0)); dm_ratio = Δm²21/|Δm²3ℓ| as in `src.evaluate`.

Convention mapping of the earlier experiments (each reproduced in `tests/test_yukawa.py` from its stored inputs):

| exp | its convention | engine input | reproduced |
|---|---|---|---|
| 014 SO(10) FN (`so10fit.py`) | Weyl RL f^c M f, left rotation from M†M, m_ν = M_Dᵀ M_R⁻¹ M_D, U_ν from M_ν†M_ν | Y_f = M_f†; m_ν = M_ν, or seesaw Y_ν = M_D†, M_R = F | `fit_FN_ours.json` (z vector, 14 dimensionless obs + δ_q, δ_ℓ) to 1e-8; `fit_FN_DR.json` (1306.4468 observables, ratios) to 1e-7 rel; both m_ν paths |
| 022 SU(5)+CSD3 (`model.py`) | LR Yukawas, U_ν = eigenvectors of m m† | Y_f as given, m_ν = conj(m) | `fit_primary.json` pred_gut (all 17 keys) to 1e-9; paper benchmark (`reproduce.json`, tb5) θ13, δ_ℓ, Δm² to 1e-7; unconjugated m flips the δ_ℓ sign (exp 022's rejected choice) |
| 023 S4×SO(10) (`model.py`) | LR Yukawas, PMNS = Takagi of U_eᵀ m* U_e | Y_f as given, m_ν = conj(m) | published point (`reproduce.json`): angles, δ_q, δ_ℓ, α21, α31, Δm², y_b to ≤ 1e-6; `fit_eb08.json` pred_gut to 1e-9 (with exp 023's linear η̄_b factors on s13, s23, m_s/m_b, m_b/m_τ) |

So exps 022 and 023 both used the complex conjugate of the engine's m_ν (they fixed it by reproducing published CP signs), and exp 014 used the Hermitian conjugate (RL) Yukawas. The physics of all three is the same as the engine's once mapped.

### A2. `src.infogain.log_ratio_rank`: native flow log-ratio statistic

`log_ratio_rank(real_T, fake_Ts, unit="nats"|"bits")`: rank of a real log-likelihood-ratio statistic T (larger = real CKM explains PMNS better) among fake values, with hits = #(fake T ≥ real T), two-sided CP interval, S_rank, and B_ckm = log2[e^{T_real} / mean e^{T_fake}]. Identical to `null_info(−2T, −2T_fake)` (tested), so it replaces exp 025's χ² = −2T workaround without changing any number.
- Against exp 025 `results.json` (main tag, point statistic, 2000 N1 fakes): hits 63/2000, CP95 [0.0243, 0.0401], S_rank 4.989 bits exactly; B_ckm 1.5022 bits to 2e-4 (the stored fakes are rounded to 1e-4 bits); fake median and 5/95% quantiles to 1e-4.

### A3. Tests and score stability

- `.venv/Scripts/python -m pytest -q`: **539 passed** (`pytest.log`): 529 before + 8 in `tests/test_yukawa.py` + 2 in `tests/test_infogain_030.py`. No existing test or src function changed; `src/infogain.py` only gains `log_ratio_rank`.
- `check_ref_numbers.py` (copy of exp 026's, output kept here): exp 019 `ref_numbers.json` recomputed, max abs/rel difference 0.0, `pass_1e-9` true (`ref_check.json`).
- `check_committed.py` (copy of exp 026's): exp 022 and 023 committed fit points rescored, max |Δχ²| 0.0 (`committed_check.json`).

## Part B: results (bi-large grammar, criterion-4 protocol, fit-only rule, informational)

Logs: `run.log`; per-task JSON `power_*.json`, `null_*.json`, `rate_*_200.json`. Wall ~1 h on 2 workers.

Real data (frozen reference): CDSV-I IO, χ²_global 6.195, p_global 0.1025, max pull 1.68. N 4, k 14 (1 cont + 12 bits + 1 ordering bit), so k > N − 2 and the strict criterion-4 rule is vacuous (strict hits 0 in every run, as it must be).

**Power check first** (20 injections × 300 fakes each; an injection passes if the pipeline's own result is fit-ok and the null then has 0 hits in 300):

| null | ordering | injections fit-ok | pass | power |
|---|---|---|---|---|
| N1 | NO | 20/20 | 1 | 0.05 |
| N1 | IO | 20/20 | 0 | 0.00 |
| N2 | NO | 20/20 | 0 | 0.00 |
| N2 | IO | 20/20 | 0 | 0.00 |

Injected fits are good (median p_global 0.60 NO, 0.25 IO), but in 79/80 injections a fake reaches an equally good fit within 1-136 fakes. Even when the bi-large structure is true, this grammar cannot be told apart from fake CKMs or fake PMNS data. Power is far below 50%.

**Nulls** (1000 planned per null and ordering, early stop at > 4 hits, which only declares failure):

| null | ordering | hits / run | one-sided CP95 upper | passed |
|---|---|---|---|---|
| N1 | NO | 5/26 | 0.36 | no |
| N1 | IO | 5/11 | 0.73 | no |
| N2 | NO | 5/6 | 0.99 | no |
| N2 | IO | 5/6 | 0.99 | no |

**Rate runs** (200 fakes, no early stop; cannot change a verdict):

| null | ordering | hits / 200 | rate, two-sided CP95 | B_ckm (bits) |
|---|---|---|---|---|
| N1 | NO | 74 | 0.37 [0.30, 0.44] | −0.82 |
| N1 | IO | 73 | 0.37 [0.30, 0.44] | −1.14 |
| N2 | NO | 163 | 0.82 [0.75, 0.87] | −2.51 |
| N2 | IO | 156 | 0.78 [0.72, 0.84] | −2.20 |

- N1 hit rate ~37% agrees with the exp 027 pilot (35/100, χ² rule, NO only, coarser grid). About a third of random hierarchical CKMs (λ up to 0.5) give a grammar fit at least as good as the real one.
- N2: ~80% of random PMNS centres are fitted at least as well. The grammar fits most of PMNS space.
- B_ckm < 0 everywhere. The real CKM fits worse than a typical fake (N1) and the real PMNS fits worse than a typical fake PMNS (N2).

## Parameter count

N = 4. k = k_cont (1 for CDSV, 2 for RMSV) + 12 bits (2052 configurations) + 1 ordering bit = 14 (selected CDSV-I IO). Fails criterion 2 (k ≤ 2) by 12. As published, without the grammar (CDSV I vs II and the ordering): k = 3.

## Verdict

- Infrastructure: the shared engine `src.yukawa` and `src.infogain.log_ratio_rank` are in. Exps 014, 022 and 023 are reproduced from stored inputs with each convention mapping documented. No committed score moved.
- Bi-large Cabibbo-seed grammar: **no evidence** of a CKM-PMNS link. All four nulls fail (N1 rate ~37%, N2 ~80%). The power check is 0-5% in all four null/ordering cells, so under protocol point 7 the family is **not "excluded"** by the nulls: the protocol cannot confirm or exclude it. It fails criterion 2 by count (k 14 vs N − 2 = 2) and only narrowly passes criterion 3 (p_global 0.10, IO with the ordering bit; NO best 0.09). This closes the class as a candidate source. It stays a known baseline (published 2014/2019).

## Remaining gaps

- `src.yukawa` does not run the matrices. High-scale models still go through `src.evaluate(scale=...)` translation. There is no matrix-level RG. Degenerate neutrino masses: Takagi phases are not rotated within a degenerate block (angles there are unphysical anyway).
- Exps 014, 022 and 023 were not switched to the engine. Their scripts are unchanged, and the tests pin the mapping.
- `log_ratio_rank` is checked only against exp 025's point statistic: the conv fakes are not stored. Exp 025's `analyze.py` is not rewritten.
- Still open from STATE: 2-loop MSSM Yukawa running, undefined δ (L4), `derived_yukawas_mz` errors, low-scale PMNS + Δm² ratio helper.

## Files

- `bilarge.py`: frozen pipeline. `check_pipeline.py` → `check_pipeline.log/.json`: table check (3e-14), exp 027 fits rescored, leak probe, real-data selection.
- `run_protocol.py` → `run.log`, `power_*.json`, `null_*.json`, `rate_*_200.json`. Rerun: `../../.venv/Scripts/python run_protocol.py power:N1:IO power:N1:NO power:N2:IO power:N2:NO null:N1:NO null:N1:IO null:N2:NO null:N2:IO rate:N1:NO:200 rate:N1:IO:200 rate:N2:NO:200 rate:N2:IO:200` from this folder (~1 h, 2 processes).
- `check_ref_numbers.py` → `ref_check.json`, `check_ref_numbers.log`; `check_committed.py` → `committed_check.json` (run from the repo root); `pytest.log`.
- Source: `src/yukawa.py` (new), `src/infogain.py` (`log_ratio_rank` added). Tests: `tests/test_yukawa.py`, `tests/test_infogain_030.py`.

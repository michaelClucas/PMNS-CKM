# 20261118-012 Skeptic audit: scoring infrastructure and the exp 007 null

## Hypothesis

The shared scoring infrastructure (`src/evaluate.py`, `src/baselines.py`, `src/stats.py`, `src/data.py`) and the one completed search (exp 007, F1 shared rotation) are sound: they cannot let a false candidate through, and the past verdicts follow from the numbers.

## Method

- Read the four modules and exp 006/007 code and outputs. No other experiment folder was touched; exp 007 code is imported read-only by `null_checks.py`.
- `test_infra_findings.py`: one test per finding. `xfail(strict=True)` tests state the correct behaviour and fail today (they demonstrate the bug); `_demo` tests pass today and show the exploit; the rest check edge cases that are already right. Result: **11 passed, 7 xfailed**.
- `spot_check.py`: independent recomputation of three exp 006 scores. Own matrix construction, own angle extraction (self-tested by round trip), own NuFIT CSV reader and bilinear interpolation, own χ² and p. Only the JSON readers `src.data.load_ckm/load_masses` are shared. Output `spot_check.log`.
- `interp_sensitivity.py`: linear vs cubic interpolation of the NuFIT tables.
- `power_estimate.py`: how often an exactly true F1 relation would pass criterion 4 under exp 007's nulls, from exp 007's own fake-best-χ² distributions. Output `power_estimate.log`.
- `null_checks.py`: exp 007 pipeline with V_CKM replaced by the identity (real data, and null (a) on the same 100 fake seeds, paired); Gaussian vs table χ² at the real best points. Output `null_checks.json`, `null_a_identity.csv`, `run.log`.
- `leak_probe.py`: prototype detector for undeclared data reads (proposed fix for F3).

Rerun (repo root, 1 core):
```
.venv/Scripts/python -m pytest -q experiments/20261118-012-skeptic-infra/test_infra_findings.py
.venv/Scripts/python experiments/20261118-012-skeptic-infra/spot_check.py
cd experiments/20261118-012-skeptic-infra && ../../.venv/Scripts/python interp_sensitivity.py && cd ../..
.venv/Scripts/python experiments/20261118-012-skeptic-infra/power_estimate.py
.venv/Scripts/python experiments/20261118-012-skeptic-infra/null_checks.py --n 100   # ~13 min
```

**Parameter count:** N/A (audit).

## Result

### Spot checks (independent code): all three reproduce exp 006

| id | ordering | exp 006 χ² / p | this audit χ² / p |
|---|---|---|---|
| B2-TM1 | NO | 5.1146 / 0.0237 | 5.1146 / 0.0237 (θ = 0.2628, φ = 4.520) |
| B2-TM1 | IO | 1.8208 / 0.177 (χ²_g 3.307) | 1.8208 / 0.177 (χ²_g 3.307) |
| A1s QLC solar | NO / IO | 19.207 / 19.186 | 19.207 / 19.186 (s12² pred 0.28061) |
| C1 GST | - | 0.01485 / 0.903 | 0.01485 / 0.903 (linearized Gaussian also 0.0148) |

Interpolation: cubic instead of linear/bilinear tables lowers the TM1 minima by 0.03 (NO) / 0.02 (IO); inside 3σ the two schemes differ by at most 0.12 in χ². Fitted minima landing on grid nodes is a harmless artifact.

### Findings, ranked by severity

**High**

**H1. Exp 007's "ruled out" is not supported: the null has almost no power against F1.** (`experiments/20261118-007-f1-shared-rotation/README.md` verdict; `STATE.md` "F1 shared rotation: ruled out"; `CANDIDATES.md` status.) If F1 with fitted phases were exactly true, the real best χ² would be roughly χ²(2), and the fakes reach that easily (null (b) NO: fake median best χ² 1.78, 1st percentile 0.016). From exp 007's own fake distributions (`power_estimate.log`):

| null | ord | variant | E[rate] if F1 were true | P(rate < 1%) if F1 were true |
|---|---|---|---|---|
| (b) | NO | cont | 0.44 | 0.007 |
| (b) | NO | struct | 0.23 | 0.08 |
| (b) | IO | cont | 0.12 | 0.03 |
| (a) | NO | cont | 0.16 | 0.012 |
| (a) | NO | struct | 0.035 | 0.15 |

The observed 36% (b, NO, cont) is what a true F1 relation would give. The search fails criterion 4, so "no candidate" stands, but the data cannot tell F1-true from F1-false. Fix: change the wording to "untestable with this pipeline (null has no power; observed rate equals the rate expected if true)", and make a power check (injection test, protocol step 5 below) mandatory before a search is run, so that a family which cannot pass criterion 4 even when true is not spent compute on and not reported as excluded.

**H2. N inflation through aliased observables.** `src/evaluate.py:173-180` (`Scorer.__init__`) and `:225` (`N = len(self.use)`) accept any set of keys. `ckm_lambda`, `ckm_s12`, `ckm_Vus`, `ckm_Vcd`, `ckm_direct_Vus`, `ckm_direct_Vcd` are six records of essentially one measurement; `m_c`, `m_s`, `m_c_over_m_s` count one ratio twice. Failing scenario (`test_F1_alias_observables_fake_compression_demo`): a one-number "relation" x predicting all six |V_us| aliases scores N = 6, k = 1, p > 0.05, max|pull| < 2, i.e. it passes criteria 2 and 3. Fix: an alias table in `src/evaluate.py` mapping each key to a measurement group (CKM global fit: at most 4 independent keys in total since the fit has 4 parameters, and never global and direct versions of the same element together; masses: an individual mass and a ratio containing it are one group). `Scorer` raises on two keys from one group unless a covariance is supplied.

**H3. Target leakage is undetectable.** `src/evaluate.py:247-266`: `inputs` is a declaration only, nothing checks it. Failing scenario (`test_F3_leak_accepted_by_evaluate_demo`): a model that returns `pmns_best("NO")` scores N = 4, k = 0, p ≈ 1. This is the exact failure mode `CLAUDE.md` forbids, and a neural or symbolic search can do it implicitly (e.g. a grammar terminal or a constant initialised from data). Fix: run `leak_probe.py` (or an equivalent in `src/`) on every candidate: shift each record in `data/*.json` by +1σ, one at a time, through the official loader, and fail the candidate if any undeclared record moves a prediction. Tests show it flags the leaky model, flags an undeclared ckm_s12 read in QLC, and passes QLC with its declared input. It does not catch hardcoded numbers or direct CSV reads, so candidate code still needs a read for literals.

**Medium**

**M1. Input/prediction overlap is checked by name only.** `src/evaluate.py:262-265`. `inputs=("ckm_s12",)` with predictions `ckm_Vus`, `ckm_lambda` passes and scores χ² = 0, N = 2, k = 0 (`test_F2_input_alias_demo`). Fix: check overlap on the alias groups of H2. Also: for official numbers always treat inputs as nuisances (fit and score them, N+1 and k+1). That costs no dof and removes the "input error is small" judgment call (exp 006 found input shifts move χ² by up to 2.6 for A3).

**M2. The ordering choice is free, and the criterion-3 p uses the within-ordering χ².** `src/evaluate.py:25-35, 236-241`. TM1 has p = 0.024 in NO and p = 0.18 in IO, but p_global (IO) = 0.069 (`test_F6_ordering_choice_not_counted_demo`). STATE quotes "IO 1.8 p = 0.18". A candidate that only works in IO gets a p that ignores the Δχ² = 1.49 (5.91 with SK-atm) penalty and the selection. Fix: criterion 3 uses `p_global`, unless the ordering was fixed before looking at the data, in which case it is stated. If the better ordering is reported after seeing both, add 1 to k_discrete automatically (an `ordering_selected=True` argument).

**M3. Discrete choices are counted three different ways.** `src/baselines.py:503` gives the A4 family k_d = 2 for picking 1 of 15 options (5 X × 3 orders, log2 15 = 3.9 bits); exp 007 counts log2(2784) = 11.4 bits; `src/stats.py:220` takes `int(k)`, which truncates fractional bits; and `p` subtracts discrete choices from the χ² dof, which has no statistical meaning. Fix: one convention. Criterion 2 counts k_d = ceil(log2(number of options considered)); criterion 3 uses `p_fit` (dof = N − k_cont); the discrete selection is paid for in criterion 4 (the null reruns the same choice). `test_F5_k_discrete_counts_bits` encodes that convention.

**M4. Null (a) scores fakes with a different χ² shape than the real claim, and it differs most in IO.** `f1lib.GaussChi2` vs `TableChi2`. At the real IO table-best point (s23² = 0.475, δ = 278°) the table gives χ² = 2.70 and the Gaussian 25.4, because the Gaussian has no first-octant minimum in IO. The Gaussian real best is a different config (χ² 6.12, s23² = 0.559). So null (a) IO tests a different search than the claimed IO result, and its high reference χ² inflates the IO rate (36%). In NO the two agree (0.785 table vs 0.685 Gaussian at the same point). This does not change the exp 007 verdict. Fix: score fakes with the NuFIT table translated to the fake centre (Δχ²_fake(x) = Δχ²_NuFIT(x − t_fake + t_real), with δ periodic and s23² mirrored about 0.5 when the fake sits in the other octant), so real and fake χ² have the same non-Gaussian shape. Or drop null (a) for PMNS-side claims (M5).

**M5. A Haar-PMNS null is anti-conservative for families built from PMNS-motivated patterns, and it is not a CKM test.** TBM, BM, GR, HEX and TM were constructed after, and to match, the measured PMNS. A Haar target is far from all of them, so the real data looks special for reasons unrelated to the CKM. `null_checks.json` (`real_identity`): with V_CKM replaced by the identity, the real best χ² is 300 (NO) / 290 (IO), because without V nothing supplies θ13. So the CKM does matter in F1, contrary to STATE's "the CKM is barely used": it supplies an O(λ) rotation, and the free phases tune its size. Any O(λ) hierarchical matrix does as well (null (b)). The paired V = I run of null (a) is in the table below. Fix: null (a) is never sufficient on its own. A CKM-side null (b) is mandatory for any claimed CKM-PMNS link.

**M6. `use=None` silently scores every known key in `pred`.** `src/evaluate.py:258-259`. A model that echoes its input (`{"s12sq": ..., "ckm_s12": <data>}`) gets N = 2 with a zero-χ² term, and no error is raised because nothing was declared as an input (`test_F4_use_none_does_not_count_echoed_inputs`). Fix: require an explicit `use`.

**M7. Criterion 4 as written cannot be certified.** `OVERSEER_PROMPT.md` says "≥ 100 fakes, rate < 1%". With 0 hits in 100 the 95% one-sided Clopper-Pearson upper limit is 2.95%. 300 fakes certify < 1% only with 0 hits (upper 0.99%; 1 hit gives 1.57%). So a good candidate with a true rate of 0.5% passes 300 fakes only 22% of the time (0.995^300). With 1000 fakes up to 4 hits pass (upper < 1%), and a true rate of 0.2% passes 95% of the time (`test_C4_clopper_pearson_thresholds`). Protocol below.

**Low**

- **L1.** `ckm_delta_rad` is a plain Gaussian (`src/stats.py:161-167` via `Objective._gauss`), so δ_q + 2π is penalized (`test_F9_ckm_delta_wraps`). Fix: wrap like the PMNS δ.
- **L2.** `src/stats.py:118` defaults `Objective(pmns_mode="sum1d")`, which drops the s23-δ correlation that `evaluate` keeps. Code that uses `Objective` directly gets a different χ² from the official one. Fix: default to `"joint"` or make the argument required.
- **L3.** `evaluate` cannot score Δm²21 or Δm²3ℓ (`PMNS_KEYS` has 4 entries; `pmns_delta_chi2` supports them). A mass-splitting candidate would need a side path. Fix: add `dm21sq_eV2` and `dm3l_sq_eV2` keys passed through to the joint call.
- **L4.** A prediction with undefined δ (s13 = 0) gets δ = 0 from `extract_angles` and is scored at δ = 0 (TM1 at θ = 0 gives χ² 1868, no crash). Harmless for current fits.
- **L5.** Exp 007 dedup (2784 classes) cannot undercount the trials that matter, because the null reruns the identical pipeline on every fake. Dedup only enters the Šidák p, which is 1 anyway. The "best χ² ≤ real best χ²" statistic is correct for a single family with fixed k_cont; for a search over relations of different k it must be the score the search used to pick the candidate (protocol step 3).

Checked and right: δ wrap (−150° ≡ 210° ≡ 570°), both θ23 octants allowed by the 2D table, out-of-table points give χ² = inf and p = 0, IO offset added exactly once and only with a PMNS term, NO offset = 0, `ordering_offset` = 1.486 matches the raw IO table floor, 0/300 Clopper-Pearson arithmetic, the exp 007 vectorized scorer (spot-checked by exp 007 and consistent with my table reader).

### Null (a) with and without the CKM (`null_checks.json`, 100 paired fakes)

Same 100 Haar targets as exp 007 (seeds `[1001, i]`, i < 100), Gaussian scorer, identical pipeline. The reference is the real-data best from the same pipeline with the same V.

| ordering | variant | real best, V = V_CKM | hits V = V_CKM | real best, V = I | hits V = I |
|---|---|---|---|---|---|
| NO | cont | 0.67 | 6/100 | 166.9 | 10/100 |
| NO | struct | 5.63 | 3/100 | 166.9 | 10/100 |
| IO | cont | 6.12 | 28/100 | 152.4 | 9/100 |
| IO | struct | 8.36 | 7/100 | 152.4 | 9/100 |

The CKM-free family misses the real data by Δχ² ≈ 150-170 (θ13 = 0 for most X) and still gets a null-(a) rate of 9-10%, about the same as the real F1 family. The null (a) rate does not separate a family that fits the data from one that fails it by 12σ, so it carries almost no information about a CKM-PMNS link. That supports M5.

## Verdict

Hypothesis rejected in part. The scoring arithmetic is right: three baselines reproduce exactly with independent code, and the δ wrap, octant, table-range and IO-offset handling are correct. The bookkeeping that criteria 2-4 depend on is open to abuse: N can be inflated with aliased observables (H2), data can leak into a "prediction" undetected (H3, M1, M6), the ordering choice is free (M2), and discrete choices are counted inconsistently (M3). No past verdict that was negative becomes positive, but exp 007's "ruled out" should read "untestable by this pipeline" (H1). No candidate is affected, since none exists.

## Recommended criterion-4 protocol (for every future candidate)

1. **Freeze before looking at fakes:** the search pipeline (catalog, grammar, optimizer settings, seeds), the selection score, and the hit definition. The pipeline that produced the real candidate is rerun unchanged on every fake, including all discrete choices and the ordering choice.
2. **Mandatory nulls** (all must pass; report each separately for NO and IO):
   - (N1) **CKM-side hierarchical null**: real PMNS data and tables, fake CKM from a broad hierarchy-preserving prior (as exp 007 (b), or a wider one: λ in [0.05, 0.5], A in [0.3, 2], ρ̄, η̄ in [−1, 1]), plus fake quark masses with the same treatment when masses are used. This is the test of a CKM-PMNS link.
   - (N2) **Shifted-data null**: real CKM, PMNS data replaced by a pseudo-measurement whose centre is drawn from a smooth prior that is not built from pattern-motivated values (e.g. sin² uniform over the 3σ-wide NuFIT ranges widened ×3, δ uniform), scored with the NuFIT tables translated to that centre (M4 fix). This replaces the Haar-PMNS Gaussian null.
   - Haar-PMNS (exp 007 (a)) is optional and informational only.
3. **Statistic:** the search's own selection score, applied identically to real data and fakes. A hit is a fake whose best relation (a) satisfies k ≤ N − 2 with k counted by the same rules, (b) has max|pull| < 2, and (c) has p_global ≥ the real candidate's p_global. For a single family with fixed k_cont this reduces to best χ² ≤ real best χ².
4. **Size and decision:** 1000 fakes per null and ordering. Pass if the one-sided 95% Clopper-Pearson upper limit is < 1% (≤ 4 hits in 1000). Stop early only to declare a fail (≥ 5 hits). Report hits, n, rate and upper limit.
5. **Power / injection check (before step 4, mandatory):** generate 20 synthetic datasets in which the candidate's own structure is true (its best-fit prediction plus noise drawn from the NuFIT/PDG errors), run the full pipeline and nulls on each with 300 fakes, and report the fraction that pass. If fewer than half pass, criterion 4 cannot confirm this family and it must not be reported as excluded on the basis of the null. Exp 007 F1 would have failed this check (P(pass) ≈ 1%).
6. **Bookkeeping checks run with the null:** `leak_probe` on the candidate with its declared inputs (H3), alias-group check of `use` and `inputs` (H2, M1), ordering stated or counted (M2), k_d in bits (M3).
7. **Wording:** a failed null means "no evidence"; "excluded" is reserved for a fit-quality failure (p_global < 0.05 for every member of the family) or a null with demonstrated power (step 5).

## Files

`test_infra_findings.py`, `leak_probe.py`, `spot_check.py` (+ `.log`), `interp_sensitivity.py`, `power_estimate.py` (+ `.log`), `null_checks.py` → `null_checks.json`, `null_a_identity.csv`, `run.log`.

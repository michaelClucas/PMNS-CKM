# Exp 033: infra, λ-matched null, IO-aware Δm² ratio, shared theory-space generator

Infrastructure only. Closes three gaps listed in STATE.md ("Still open"): no `n1lam_universe`, no shared synthetic theory-space generator in `src`, no IO-aware Δm² ratio helper. All changes are additive; no default changed; no committed score moved.

**Hypothesis.** None (infrastructure). Requirement: the new code reproduces what exps 025 and 028 built by hand, exactly where the seed path allows and statistically otherwise.

**Free parameters.** Not applicable (no candidate is scored).

## What changed

### `src.nulls`: N1lam (λ-matched N1)
- `n1lam_universe(base, rng, prior=None)`: λ and A drawn from the base universe's CKM posterior, split normal with the centre and ± errors of its `lambda` and `A` records (real PDG values on real data; injected values on an injected base), one `rng.standard_normal()` each in that order; then exactly `n1_universe` with λ and A pinned, so ρ̄, η̄ ~ U[−1, 1] and PMNS, tables and masses are the base's. `kind="N1lam"`, `truth` = N1's plus `lambda_A_centre`.
- `split_normal(rng, rec, width=1)` (the exp 028 helper). `N1LAM_PRIOR = {"rhobar", "etabar", "width"}` (width scales the λ, A errors; 0 pins them at the centre). Unknown prior keys raise.
- Wired into `make_fake` / `run_null` / `power_check` as null `"N1lam"`. `NULLS` is now `("N1", "N2", "N3", "N1lam")` (existing entries first and unchanged); seed code 5, so fake i of a run uses `default_rng([*seed, 5, ordering code, i])`. N1, N2, N3 streams are untouched.
- With `rng = default_rng([28, 2, i])` it reproduces exp 028's fake i exactly (test against all 2000 stored fakes per ordering in `results_{NO,IO}.json`, max difference ≤ 5e-6 = the stored rounding).

### `src.data`: Δm² ratio
- `dm_ratio(dm21sq, dm3l_sq)`: Δm²21 / |Δm²3ℓ|, scalar or array, positive in both orderings.
- `load_dm_ratio(ordering, analysis)`: record with `value` (= `src.evaluate`'s `dm_ratio` best value), linear-propagation `err_plus` / `err_minus` taken from the side of each splitting that moves the ratio that way (for IO, Δm²32 < 0, so err_plus uses Δm²32's err_plus), plus `dm21sq_eV2`, signed `dm3l_sq_eV2`, `dm3l_abs_eV2`. Scoring should still go through `src.evaluate` (`dm_ratio` key, profiled from the two NuFIT tables); the test checks that the scorer's Δχ² at value ± err is between 0.5 and 1.6.
- NO 0.02990 +0.00043 −0.00050; IO 0.03015 +0.00047 −0.00049 (default analysis).

### `src.yukawa.observables_batch`
Batched version of `observables` (same LR convention, Takagi phase fixing, ordering rule and keys, plus `io`), leading batch axis, nan rows for non-finite or non-symmetric input. Agrees with the scalar path row by row to ≤ 1e-12 (relative; phases ≤ 1e-9 deg), tested for ordering auto / NO / IO.

### `src/theoryspace.py`: shared generator API
- Registry of `Family(name, sample, description, source, shared, native)`; `families()`, `get`, `sample(name, n, rng)` → dict of (n,) arrays with `src.evaluate` keys (`src.yukawa.OUTPUT_KEYS` + `io`), `register`.
- `accept(obs, window=FILTER_025, ordering="NO"|"IO"|None)`: exp 025's physical filter. `sample_accepted(name, n, rng, ...)`: exp 025's chunking. `features_025(obs)`: the exp 025 / 028 flow features (ctx_ckm, ctx_m, y).
- **Native (ported):** GEN (shared textures) and CTRL (no sharing), rng-call for rng-call from exp 025 `gen.py`, observables through `observables_batch` with Y_f = M_f† and m_nu = M_ν (exp 025's RL matrices; the exp 014 mapping documented in `src/yukawa.py`). Also returns `mr_b_tau`, `R_clebsch`, Majorana phases, which exp 025 did not.
- **Adapters (not ported):** F1, F2, MN, FN, MOD, SU5, S4SO via `exp025_adapter(name)`, which loads exp 025 `gen.py` lazily and renames keys (`ckm_delta` → `ckm_delta_rad`, `dcp` rad → `dcp_deg`, adds `R_clebsch`; `mr_b_tau` and Majorana phases nan). Same rng stream as exp 025. A new heavy family is added with `register(Family(name, sample_fn, ...))` where `sample_fn(n, rng)` returns the keys above.
- Speed: GEN 8000 raw draws ≈ 0.5 s (exp 025's own code ≈ 0.4 s).

## Results (tests)
`tests/test_theoryspace.py` (19) and `tests/test_infra_033.py` (15):
- Same seed: GEN and CTRL ported draws equal exp 025's code (8000 draws, identical accept mask, observables to 1e-9 relative, δ to 1e-7 deg). `features_025` equals exp 025 `gen.features` to 1e-12.
- Training rows: with exp 025's own seeds (`[25, 7]` GEN, `[25, 8]` CTRL, first chunk 7500) the accepted rows equal the first rows of each family in `artifacts/025/data.npz` to 1e-9 (skips if the local artifact is absent).
- Acceptance on 64,000 fresh-seed draws vs `gen_stats.json` (GEN 0.01781, CTRL 0.01113): within 4 cluster-robust SE (clusters of 20 batches, since draws in a batch of 64 share one discrete structure). Development runs over 3 seeds: z = −0.5 to +2.4.
- Marginals: all 16 flow features' means within 4.5 cluster-robust SE of exp 025's 30k accepted rows, KS p > 1e-3 on the 5 PMNS targets (skips without the artifact). Plain KS on the CKM features is not used: it ignores the batch clustering and gave p ≈ 0 on one seed for log s23 while the cluster-robust z was −2.0.
- N1lam: exact reproduction of exp 028 (above); fresh-seed λ, A split-normal shape (half-means within 12% of err·√(2/π), 2000 fakes), ρ̄, η̄ uniform (KS), and per-coordinate KS against exp 028's stored fakes; a CKM-blind relation fails N1lam in 5 fakes like N1; a λ-only relation (s13² = λ²/2.25) is untouched by N1lam (median fake χ² < real + 1) and separated by N1.
- Δm² ratio: both orderings and analyses, IO error sides, vectorised `dm_ratio`.

## Checks
- Full suite: 573 passed (`pytest.log`, `.venv/Scripts/python -m pytest -q`; 539 before + 34 new).
- `check_ref_numbers.py` (copy of exp 026's): exp 019 `ref_numbers.json` recomputed, max abs/rel difference 0.0, `pass_1e-9` true (`ref_check.json`, `check_ref_numbers.log`).
- `check_committed.py` (copy of exp 026's): exp 022 / 023 committed fits rescored, max |Δχ²| 0.0 (`committed_check.json`, `check_committed.log`).

## Verdict
Gaps closed: `n1lam_universe` / null "N1lam", IO-aware Δm² ratio, shared generator API with GEN and CTRL native.

## Remaining gaps
- F1, F2, MN, FN, MOD, SU5, S4SO are adapters to exp 025 `gen.py` (and through it to exps 007, 009, 014, 017, 022, 023), not ports onto `src/yukawa.py`. F1 and F2 have no mass model (masses log-uniform inside the filter windows), so they cannot go through the Yukawa engine as they stand. Their `mr_b_tau` and Majorana phases are nan.
- The adapter families' acceptance and marginals were not re-tested here (only key mapping and same-stream equality for F1, F2); FN and MOD take 10-20 min per 30k accepted.
- `observables_batch` has no seesaw input (pass m_ν); no RG running in the generator (all families are at one scale, as in exp 025).
- `load_dm_ratio` errors ignore the Δm²21-Δm²3ℓ correlation (NuFIT does not publish it in the tables used).

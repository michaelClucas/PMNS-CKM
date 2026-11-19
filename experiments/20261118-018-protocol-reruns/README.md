# 20261118-018 Criterion-4 protocol re-runs: leads T1 and F2-LO

## Hypothesis

The two leads whose null tests had no demonstrated power (exp 012 H1) get a definitive criterion-4 status when their original searches are wrapped as frozen `pipeline(universe)` functions and run through `src.nulls` (power check, N1, N2 per ordering, leak probe, alias check):

- **T1** (exp 010): GUT scale, θ12 = sin(3 (m_u/m_c)^{1/4}), θ13 = sin(3 (m_u/m_t)^{1/4}).
- **F2-LO** (exp 009): U_PMNS = U_e(θ12e = θ_C, θ23e = θ_cb)† D(φ) U_GR2, φ fitted, M_Z.

## Method

### Pipelines (`pipelines.py`)

Each pipeline reads measured numbers only from the `Universe` (`context()`, `load_ckm()`, `load_masses()`, `load_pmns()`, `map_to_real()`) and scores the final result with `universe.evaluate`. Every discrete choice is rerun on every universe.

**T1** = exp 010 `templates.py`, unchanged except the score:
- Same grammar: one-variable f(x), C ≤ 5, 29,704 distinct f, built once with exp 010's `symlib.Enum` and cached in `artifacts/protocol/`.
- Same 4 sector mappings (ckm, down, lepton, up), 4 angle subsets, 2 scales (M_Z and 2e16 GeV). Inputs come from `src.rg.ckm_at` with the universe's CKM and masses.
- Same validity rules. The input-error model is exp 010's relative 1σ directions, scaled to the universe's inputs as exp 010 did for its fakes.
- Cell score = sum of the 1D NuFIT profiles of the subset's angles. This is what `src.evaluate` uses for s_ij² without δ. A vectorized screen applies `map_to_real` to the shifted tables, and the winner is rescored with `universe.evaluate`. The screen and `evaluate` agree to all printed digits on real data.
- Selection, frozen before any fake: best template per (scale, subset, ordering) by χ², then the cell with the largest p_global (IO cells pay the NuFIT ordering offset).
- Count: k = ceil(log2(118,816 templates × 4 subsets × 2 scales)) = 20 bits + 1 ordering bit = 21; k_cont = 0; N = 2 or 3.

**Change vs exp 010:** the score does not profile the propagated input errors, because `src.evaluate` cannot. Profiling lowers T1's χ² from 0.097 to 0.058.

**F2-LO** = exp 009 `f2lib.search`, fit variant:
- Tiers A+B+C: 17,220 Clebsch configs. M_Z CKM angles from `universe.context()`. 180-point φ grid, then golden refinement of the 100 best rows (`f2lib` code unchanged).
- Scored per ordering with the universe's translated tables.
- Ordering pick: per-ordering best config, then the lower chi2_global.
- Count: k = 1 (φ) + 15 bits (ceil log2 17,220) + 1 ordering bit = 17; N = 4.
- Scale fixed to M_Z, as in CANDIDATES. Exp 009's GUT variant is not part of the selection.

### Hit rules (`run_protocol.py`)

The protocol hit (`src.nulls.is_hit`) requires k ≤ N − 2. Both leads have the same k on every universe and fail that clause by construction (21 > 0, 17 > 2). So:
- No fake can ever be a strict hit.
- No injection can pass `candidate_ok`, so the strict power check is 0 by construction.

The strict protocol therefore says nothing about either lead beyond "fails criterion 2". To get an informative criterion-4 answer I used a **fit-only** rule: the protocol rule minus the compression clause (max|pull| < 2 and p_global ≥ the reference's). Since k is constant this changes nothing else.
- The fit-only rule drives early stopping and the verdicts.
- Strict hit counts are recomputed from the stored per-fake summaries and are 0 everywhere, as expected.
- For the power check, `src.nulls.candidate_ok` was replaced in-process by the fit-only version (max|pull| < 2, p_global > 0.05). `src/` was not edited.

### Runs

- Seeds: [18, 0] for the nulls, [18, 1] for power, [18, 2] for the fixed-length rate runs.
- 3 worker processes.
- Nulls: 1000 fakes planned per null and ordering, with early stop at > 4 hits (the protocol's failure bound).
- Power: `power_check` with 20 injections × 300 fakes (early stop at the first hit), built from the real-data best prediction with `make_injector`.
- Rate runs: because every null stopped early, fixed-length runs without early stop estimate the actual hit rate. They cannot change a verdict.

Both pipelines take the ordering choice internally (`ordering_selected`). So the NO and IO null runs differ only in seed and in which ordering's centre the N2 draw is anchored to.

## Parameter count

- **T1:** N = 2 (s12², s13²), k = 21 (20 discrete bits for template/subset/scale, 1 ordering bit), k_cont = 0. Fails k ≤ N − 2.
- **F2-LO:** N = 4, k = 17 (1 continuous φ, 15 bits config, 1 bit ordering). Fails k ≤ N − 2.

## Result

### 1. Reproduction on real data (`repro.json`, `repro_t1_profiled.json`)

| lead | pipeline result | original | difference |
|---|---|---|---|
| F2-LO | GR2, c12 = c23 = 1, std, φ = 287.080°, NO; χ² 7.210758003788 (IO best 7.164892852043, rejected by chi2_global +1.49) | exp 009 `tier_best.csv` MZ ABC fit: NO 7.210758003745, IO 7.164892852011 | 4e-11 (golden-section endpoint rounding) |
| T1 | GUT, 12+13, up, f = sin(3√x), NO; official χ² 0.097246, p_global 0.9525, pulls +0.27, +0.15 | exp 010 `templates_summary.json` GUT NO 12+13: 0.058221118313 (IO 0.046355) | Same template and cell. The official score does not profile input errors. With exp 010's `ProfileScorer` the pipeline's own predictions give 0.05822111831316642 (NO) and 0.04635452900377803 (IO), bit-identical to exp 010. RG inputs agree with exp 010's cache to 0 relative difference. |

Real-data reference p_global: T1 0.9525, F2-LO 0.0655 (max|pull| 1.84, s23²).

### 2. Leak probe and alias check (`leak_alias.json`)

**F2-LO** (inputs ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad declared):
- 0 flags. The alias check (`check_observables`) passes.

**T1** (m_u, m_c, m_t_direct declared): the alias check passes. 16 flags, all explained:
- m_b moves s_ij² by 7-9e-6 (relative). This is real RG dependence through y_b in the running.
- CKM s23 moves them by ~1e-9. Other masses and CKM records move them by 1e-12 to 1e-10. Both come from the Yukawa running.
- 4 PMNS records (θ12, θ13, θ23, δ NO) and both neutrino splittings move s12² by 2.4e-12 relative. This is integrator crosstalk: `src.rg.build_mz_state` puts the NuFIT angles and splittings into κ, and the adaptive step control sees κ. It is not a leak at any meaningful level (NuFIT s12² 1σ ≈ 4% relative).

Neither model reads PMNS data in substance.

### 3. Power check (fit-only rule; `power_*.json`)

| lead | null | ord | injected fits OK | nulls passed | power | fakes run per injection |
|---|---|---|---|---|---|---|
| T1 | N1 | NO | 20/20 | 0/20 | 0.00 | 1-4 |
| T1 | N1 | IO | 20/20 | 0/20 | 0.00 | 1-9 |
| T1 | N2 | NO | 20/20 | 0/20 | 0.00 | 1-99 |
| T1 | N2 | IO | 20/20 | 0/20 | 0.00 | 1-14 |
| F2-LO | N1 | NO | 19/20 | 0/20 | 0.00 | 1-26 |
| F2-LO | N2 | NO | 19/20 | 1/20 | 0.05 | 1-300 |

F2-LO power was not run for IO. The pause killed the first attempt, and after the NO rerun (N2 took 54 min) the IO runs did not fit the budget. Both pipelines choose the ordering internally, so the IO runs would differ only in which ordering's centre is injected.

All powers are far below 50%. Per protocol step 5, **neither null can confirm or exclude either family**. The reasons:

- **T1 N1:** T1 does not contain the CKM. A fake CKM leaves the same template available with χ² changed only at the 1e-7 level through Yukawa running. Each fake is a "hit" or not depending on the sign of that numerical noise (49% hit rate in the fixed-length run).
- **T1 N2:** with 118,816 templates × 16 cells, a random PMNS centre is matched with p_global ≥ 0.95 often enough to stop every injected null early. All but a few hits come from the 12+23 cells, where the broad θ23 profile makes a two-angle fit easy.
- **F2-LO:** the Clebsch lattice × 5 patterns × φ fits a random CKM (N1) or a random PMNS centre (N2) about as well as the truth. Only the one injection with p_global 0.996 survived 300 N2 fakes.

### 4. N1 / N2 nulls on real data (`null_*.json`, `rate_*.json`)

1000 fakes planned. Early stop at > 4 hits, the protocol's fail bound. Clopper-Pearson upper limits are one-sided 95%.

| lead | null | ord | hits / fakes run | early stop | one-sided 95% upper | passed |
|---|---|---|---|---|---|---|
| T1 | N1 | NO | 5/7 | yes | 0.947 | no |
| T1 | N1 | IO | 5/10 | yes | 0.778 | no |
| T1 | N2 | NO | 5/16 | yes | 0.548 | no |
| T1 | N2 | IO | 5/21 | yes | 0.437 | no |
| F2-LO | N1 | NO | 5/7 | yes | 0.947 | no |
| F2-LO | N1 | IO | 5/8 | yes | 0.889 | no |
| F2-LO | N2 | NO | 5/7 | yes | 0.947 | no |
| F2-LO | N2 | IO | 5/7 | yes | 0.947 | no |

Strict (`is_hit`) hits: 0 in every run, by construction (see Method).

Fixed-length hit rates (no early stop, seed [18, 2]; 95% Clopper-Pearson two-sided):

| lead | null | ord | hits / n | rate [95% CI] |
|---|---|---|---|---|
| T1 | N1 | NO | 490/1000 | 0.490 [0.459, 0.522] |
| T1 | N2 | NO | 288/1000 | 0.288 [0.260, 0.317] |
| T1 | N2 | IO | 315/1000 | 0.315 [0.286, 0.345] |
| F2-LO | N2 | NO | 63/100 | 0.63 [0.53, 0.72] |
| F2-LO | N1 | NO | not finished (killed at the budget limit) | exp 009 null (b), narrower prior: 0.46 |

For comparison, exp 010 quoted ~1% for T1's own cell and ~8% after an estimated 8-cell correction. With the selection actually rerun on each fake (16 cells, both orderings, p_global as the cross-cell score), the rate is 29-32%. The ~8% estimate was too low because the 12+23 cells are much easier to hit than the 12+13 cell.

### Wall time

- Real-data pipeline: T1 1.2 s (0.1 s with cached inputs); F2-LO 9 s (N1 fakes need new CKM observables; N2 fakes reuse them).
- Power checks: T1 3.5-4.5 min each; F2-LO N1 12 min, N2 54 min (dominated by one injection that ran all 300 fakes plus pool start-up per injection).

## Verdict

**T1: no evidence.**
- Fails criterion 2 (k = 21 vs N = 2), so it cannot be a candidate.
- Criterion 4: N1 and N2 fail in both orderings. The rerun look-elsewhere rate is 29-32% (N2), and N1 is a coin flip because the relation does not involve the CKM.
- Power is 0/20 for both nulls, so neither null can exclude the family. Under the protocol wording this is "no evidence", not "excluded".
- It is not a CKM-PMNS relation in any case: it ties PMNS angles to up-quark mass ratios.

**F2-LO: no evidence.**
- Fails criterion 2 (k = 17 vs N = 4).
- Criterion 4: N1 and N2 fail in both orderings at the first handful of fakes (5 hits in 7-8 fakes).
- Power: 0/20 (N1) and 1/20 (N2), so it is not excluded either: "no evidence".
- Fit quality is marginal: p_global 0.065, s23² pull −1.84.

**CANDIDATES.md recommendation:** drop both from the leaderboard as leads. Record them as "fails compression; criterion-4 nulls fail (exp 018), nulls have no power for these families: no evidence".
- T1 is cheap to re-test with future m_u and θ12 data, but only as a single pre-registered formula (k = 0) under a mass-side null.
- F2-LO stays dead unless a theory fixes θ12e = 0.96 θ_C in advance (STATE already says so).

Neither result is "excluded": no null with demonstrated power exists for these families.

## src/nulls.py issues (proposed fixes, not applied)

1. **The protocol is vacuous for any lead with k > N − 2.** `is_hit` and `candidate_ok` both require compression, so the nulls "pass" trivially and power is 0 by construction. Fix: add `ok_fn` to `power_check` (it currently hardcodes the module-level `candidate_ok`, so I had to monkeypatch it). Also add a documented `hit_fit` (no compression clause), and state in STATE that criterion 4 is only run on results that pass criterion 2, or uses the fit-only rule.
2. **Ties at numerical-noise level.** `is_hit` compares p_global exactly. For a family that ignores the null's faked sector (T1 under N1), the hit flips on 1e-7 RG noise. Fix: `pf >= pr * (1 - 1e-6)` or a χ² tolerance, so an unchanged relation always counts as a hit.
3. **No mass-side null.** N1 fakes only the CKM and N2 only the PMNS centre. T1's actual claim (PMNS from up-quark mass ratios) is tested by neither. Fix: an N3 that redraws mass ratios, log-uniform over a decade as in exp 010 null (b), keeping record errors.
4. **Speed.** `power_check` creates a new spawn pool per injection (about 20 s start-up each here, plus per-worker cache rebuilds). Fix: one pool for the whole power check, or accept a `pool=` argument.
5. **src.rg (not nulls):** `ckm_at` builds κ from the NuFIT angles and splittings, so quark-sector outputs depend on PMNS records at 1e-12 through adaptive step control, and the leak probe flags it. Fix: let `ckm_at` take `pmns=`/`m_lightest=`, or exclude κ from the error norm when only quark outputs are needed.

## Files

- `pipelines.py`: the two frozen pipelines and the vectorized universe scorer.
- `repro.py` → `repro.json`: real-data reproduction.
- `repro_t1_profiled.py` → `repro_t1_profiled.json`: exp 010 profiled-score check.
- `leak_alias.py` → `leak_alias.json`.
- `run_protocol.py`: nulls, power, fixed-length rates.
- Outputs: `null_<lead>_<null>_<ord>.json`, `power_<lead>_<null>_<ord>.json`, `rate_<lead>_<null>_<ord>.json`.
- `run.log`: all results, including a pause in which the background jobs were killed. The F2-LO power checks were rerun from scratch with the same seeds. The first F2-LO power N1 NO attempt left no output. Driver stdout duplicated `run.log` and was deleted because its tracebacks held local paths.

Rerun from this folder with `../../.venv/Scripts/python`:

```
repro.py
repro_t1_profiled.py
leak_alias.py
run_protocol.py T1 null:N1:NO null:N1:IO null:N2:NO null:N2:IO power:N1:NO power:N2:NO power:N1:IO power:N2:IO rate:N1:NO:1000 rate:N2:NO:1000 rate:N2:IO:1000
run_protocol.py F2LO null:N1:NO null:N1:IO null:N2:NO null:N2:IO power:N1:NO power:N2:NO rate:N2:NO:100   # rate:N1:NO:100 did not finish in budget
```

`artifacts/exp010/real_inputs_*.json` (exp 010 cache, rebuilt by its `symlib` if missing) and `artifacts/protocol/t1_templates_C5.npz` are regenerated on first use.

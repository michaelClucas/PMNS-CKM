# Exp 037 pre-registration: targeted exact-likelihood tests of the frozen CKM-linked relations

Written before any real-data rank was computed. Design checks run before this file (no real-data rank, no fake compared with real data): `check.py` (`check.log`), `check_margin.py` (`check_margin.log`), `run.py timing` on injected worlds only. The SHA-256 of this file is recorded in `README.md` and `run.log` before the run starts.

## Relation list

`experiments/20261119-029-forecast-freeze/frozen_relations.csv`, SHA-256 `04d6574e93619ec122bc2c2b7cc48b7200283e51958e4bdac2f3af7113734217` (verified by `lib37.load_frozen`, file not modified). All 37 relations, both orderings: **74 rows**, including the 6 rows with max|pull| > 2 today (kept, flagged as fit failures). Every relation has a CKM input, so none is skipped. S-dcp-3 and S-dcp-4 (same function mod 2π) are both tested and both counted.

## Statistic

For a data set D (PMNS data of one ordering) and a CKM context c:
T(D, c) = −½ min_θ χ²_D(pred(θ, c)), θ the relation's own continuous parameters (k_cont; none for symbolic rows).
- χ²_D: official within-ordering NuFIT 6.1 Δχ² (`src.data` tables via `src.nulls.Universe`; 4-key rows use the 1D s12², s13² profiles + 2D (s23², δ) table as `src.evaluate.Scorer`; symbolic rows the 1D profile of their one observable). Translated tables for shifted data (N2, injected worlds). Projected-precision worlds: Gaussian with exp 029 `sensitivities.json` σ (δ residual wrapped).
- Profile: fixed coarse grid then local nested grids (`lib37.Fitted.profile`: A4 48×48 phases, E1 19³ angles, F2LO/CDSV 360 phases, RMSV 69 ψ × 72 φ; n_start best separated coarse points refined by 3 stages of 11^d points, E1 6 stages of 7³). Reproduces the frozen χ²_today within +0.004 for every fitted row (`check.log`). The same procedure is applied to real data, worlds and fakes.
- Symbolic rows: exp 010 reading (angle value in [0, π/2] → sin², δ value in [−2π, 2π] → mod 2π); outside → no prediction, χ² = ∞. Mass-ratio leaves fixed at the real M_Z values (no null moves masses).
- CKM context from the universe's records: standard angles (s12, s13, s23, δ records) for A4, E1, F2LO, symbolic; λ, A records for CDSV, RMSV.

Hit: fake at least as good, χ²_fake ≤ χ²_data (relative tolerance 1e-9). Invalid fakes (relation undefined, e.g. CDSV-II with 3λ ≥ 1) are not hits; their count is reported.

## Nulls (fake j shared by every row and data set, rng `default_rng([37, code, j])`, banks in `artifacts/037/`)

- **N1** (code 1): `src.nulls.n1_universe`, (λ, A, ρ̄, η̄) ~ U(0.05, 0.5) × U(0.3, 2) × U(−1, 1)². Real-PMNS data, fake CKM.
- **N1lam** (code 5): `src.nulls.n1lam_universe`, λ, A from the real records' split normal, ρ̄, η̄ ~ U(−1, 1). Tests information beyond (λ, A); by construction it has no power against relations that use only λ (and A).
- **N2** (code 2): `src.nulls.n2_universe` from the real universe (octant rule), real CKM, fake PMNS centre per ordering. The N2 fake statistic does not depend on the tested data set, so the same N2 sequence is used for real data and every world of that ordering (future versions: Gaussian with future σ around the same fake centres).

## Decision rule (primary)

- M = 74 rows × 3 nulls = **222** tests. Global α = 0.01, Šidák local α_loc = 1 − 0.99^(1/222) = **4.53e-5**.
- Sequential Monte Carlo p-value per (row, null): at least 1000 fakes; continue while k ≤ 1, up to N_MAX = 50,000; stop as soon as k ≥ 2 (outcome fixed). p̂ = (k+1)/(n+1) at stop. **Pass iff k ≤ 1 at n = 50,000** (p̂ ≤ 4.0e-5 < α_loc). Šidák p = 1 − (1 − p̂)^222.
- Reported rank: k/1000 at n = 1000.
- Secondary (history) correction, not a pass condition: p_hist = 1 − (1 − p̂)^(2^b), b = k_honest − k_cont (data-selected bits from exp 029: 15 for symbolic, 4 A4, 1 E1, 15 F2LO, 11 CDSV, 5 RMSV). The relations were selected because the real CKM fits, so the real rank is biased toward small values; the list-level Šidák does not remove this selection, the history factor bounds it.

## Power (first, same rule, per row and null)

- 20 injected worlds per row: truth = the row's frozen best-fit prediction (`pred_*`) at the real CKM and frozen θ; data centre = truth + `src.nulls.pmns_noise` (via `make_injector`, translated tables without mirror), rng `[37, 7, relation index, ordering, i]`; CKM = real records (CKM measurement noise ignored: PDG errors are ≪ the N1 spread; for N1lam the fakes are centred on the same records).
- **P_300**: fraction of worlds with 0 hits in the first 300 fakes (the program's standard power check, one-sided CP95 < 1%). Necessary for passing the global rule, so an upper bound on global power.
- **P_glob**: fraction of worlds passing the primary rule (k ≤ 1 at 50,000). Symbolic rows: computed exactly. Fitted rows: worlds are extended at most to 2,000 fakes (budget), so reported as the upper bound P_2000 = fraction with k ≤ 1 at 2,000.
- **Projected**: the same with worlds centred at truth + N(0, σ_fut), Gaussian likelihood, nominal scenario (all rows) and optimistic (symbolic rows only), rng `[37, 8, relation index, ordering, version, i]`.

## Wording

- Pass with P_glob (or its bound) ≥ 0.5: reported plainly as a candidate-level signal with a skeptic checklist; no BREAKTHROUGH.md.
- Pass with lower power: reported, flagged as passing a null without demonstrated power.
- No pass: "no evidence".
- "Excluded (as an exact CKM link, this null)" only if P_300 ≥ 0.5 and none of the 20 injected worlds has a 300-fake rank k/300 ≥ the real rank k/1000 (p_excl = 1/21 < 0.05). Fit failures (max|pull| > 2 today) are labelled as such from exp 029.

## Budget

4 worker processes, 1 thread each; global deadline set on the command line (≤ 3 h from the task start). A (row, null) still alive at the deadline is reported as truncated with its n and k (p̂ then a lower bound on what the full run could give), and the world counts reached.

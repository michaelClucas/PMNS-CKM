# Candidates

Leaderboard of candidate relations. k = fitted free parameters (incl. data-selected discrete choices), N = observables reproduced. Fake-universe rate = fraction of synthetic datasets where the same search finds an equally good relation. Per the criterion-4 protocol in `STATE.md`, null rates from exps 007-010 come from nulls without a power check, so a high rate means "no evidence", not "excluded".

| id | relation | k | N | χ² / p | fake-universe rate | status |
|----|----------|---|---|--------|--------------------|--------|
| T1 | GUT scale: θ12 = sin(3 (m_u/m_c)^{1/4}), θ13 = sin(3 (m_u/m_t)^{1/4}) | ≈20 bits | 2 | NO 0.058, IO 0.046 | ~1% for its cell, ~8% after cell selection | lead; fails compression; fails at M_Z (exp 010) |
| F2-LO | U_e(θ12e=θ_C, θ23e=θ_cb)† Φ U_TBM/GR2 | 1 + 14.1 bits | 4 | NO 7.21, p 0.065, trials p 1 | 46% hierarchical null | fails compression; no evidence (exp 009) |
| F1-best | U_PMNS = V X Φ V†, X = DEM, α, β fitted | 2 + 11.4 bits | 4 | NO 0.79, local p 0.68, trials p 1 | 36% hierarchical, 8% Haar | fails compression; no evidence (exp 007) |
| L1 | V_CKM Φ U_BM V_CKM† | 2 + 2 discrete | 4 | NO 5.89, IO 4.17 | covered by F1 null | dropped (exp 006, 007) |

| BSV | SO(10) pure type-II: m_ν ∝ M_d − M_e (hep-ph/0210207) | 12-15 + 2 bits | 18 | 2700-4600, p 0 | not run | known baseline; fails fit quality (exp 015) |
| SO10-FN | real 10+126+120 SO(10) (hep-ph/0612132) | 16 + ~3 bits | 17 | 0.30 (1 dof) | not run | known model; not compressive; predicts neither δ_ℓ nor PMNS from quarks (exp 014) |

Known baselines: `experiments/20261118-013-infra-hardening/results.csv` (current convention; supersedes exp 006 p-values). None passes k ≤ N-2 with acceptable p.

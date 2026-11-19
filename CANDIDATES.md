# Candidates

Leaderboard of candidate relations. k = fitted free parameters (incl. data-selected discrete choices), N = observables reproduced. Fake-universe rate = fraction of synthetic datasets where the same search finds an equally good relation. Per the criterion-4 protocol in `STATE.md`, null rates from exps 007-010 come from nulls without a power check, so a high rate means "no evidence", not "excluded".

| id | relation | k | N | χ² / p | fake-universe rate | status |
|----|----------|---|---|--------|--------------------|--------|
| F1-best | U_PMNS = V X Φ V†, X = DEM, α, β fitted | 2 + 11.4 bits | 4 | NO 0.79, local p 0.68, trials p 1 | 36% hierarchical, 8% Haar | fails compression; no evidence (exp 007) |
| L1 | V_CKM Φ U_BM V_CKM† | 2 + 2 discrete | 4 | NO 5.89, IO 4.17 | covered by F1 null | dropped (exp 006, 007) |
| BSV | SO(10) pure type-II: m_ν ∝ M_d − M_e (hep-ph/0210207) | 12-15 + 2 bits | 18 | 2700-4600, p 0 | not run | known baseline; fails fit quality (exp 015) |
| 2O-mod | modular 2O, common τ, type-I seesaw (arXiv:2307.14926) | 10 + ~15 bits | 15 | 68.8, p 1.8e-13 | not run | known model; fails fit and compression (exp 016) |
| A4-cat | best common-τ A4 catalog model (exp 017) | 13 + 23 bits | 15 | 446.5 | not run | fails fit and compression |
| SO10-FN | real 10+126+120 SO(10) (hep-ph/0612132) | 16 + ~3 bits | 17 | 0.30 (1 dof) | not run | known model; not compressive; predicts neither δ_ℓ nor PMNS from quarks (exp 014) |

Dropped in iteration 5: T1 and F2-LO ("no evidence" under the protocol, exp 018).

Known baselines: `experiments/20261118-013-infra-hardening/results.csv` (current convention; supersedes exp 006 p-values). None passes k ≤ N-2 with acceptable p.

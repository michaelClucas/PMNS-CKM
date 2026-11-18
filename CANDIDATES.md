# Candidates

Leaderboard of candidate relations. k = fitted free parameters (incl. data-selected discrete choices), N = observables reproduced. Fake-universe rate = fraction of synthetic datasets where the same search finds an equally good relation.

| id | relation | k | N | χ² / p | fake-universe rate | status |
|----|----------|---|---|--------|--------------------|--------|
| L1 | U_PMNS = V_CKM Φ U_BM V_CKM†, Φ = diag(1,e^{iα},e^{iβ}) | 2 cont + 2 discrete | 4 | NO 5.89, IO 4.17 (global 5.65); p undefined (dof 0) | covered by F1 null: 36% | dropped (exp 006, 007) |
| F1-best | U_PMNS = V X Φ V†, X = DEM (perm r4c5), α, β fitted | 2 cont + 11.4 bits | 4 | NO 0.79, local p 0.68, trials p 1 | 36% hierarchical null, 8% Haar null | ruled out (exp 007) |
| F1-struct | V X Φ V†, X = HEX r1c0, α=180°, β=240° | 16.8 bits | 4 | NO 2.94, local p 0.57, trials p 1 | 19% hierarchical, 5% Haar | ruled out (exp 007) |

Known baselines: official scores in `experiments/20261118-006-baseline-scores/results.csv`. None passes k ≤ N-2 with acceptable p.

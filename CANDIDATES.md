# Candidates

Leaderboard of candidate relations. k = fitted free parameters (incl. data-selected discrete choices), N = observables reproduced. Fake-universe rate = fraction of synthetic datasets where the same search finds an equally good relation. Per the criterion-4 protocol in `STATE.md`, null rates from exps 007-010 come from nulls without a power check, so a high rate means "no evidence", not "excluded".

| id | relation | k | N | χ² / p | fake-universe rate | status |
|----|----------|---|---|--------|--------------------|--------|
| F1-best | U_PMNS = V X Φ V†, X = DEM, α, β fitted | 2 + 11.4 bits | 4 | NO 0.79, local p 0.68, trials p 1 | 36% hierarchical, 8% Haar | fails compression; no evidence (exp 007) |
| L1 | V_CKM Φ U_BM V_CKM† | 2 + 2 discrete | 4 | NO 5.89, IO 4.17 | covered by F1 null (no power) | dropped: fails compression (k 4 > N-2 = 2) (exps 006, 007) |
| BSV | SO(10) pure type-II: m_ν ∝ M_d − M_e (hep-ph/0210207) | 12-18 + 2-6 bits | 18 | ≈ 2000 (DE rerun, exp 031), p 0 | not run | known baseline; fails fit quality and compression (exps 015, 031) |
| 2O-mod | modular 2O, common τ, type-I seesaw (arXiv:2307.14926) | 10 + ~15 bits | 15 | 68.8, p 1.8e-13 | not run | known model; fails fit and compression (exp 016) |
| A4-cat | best common-τ A4 catalog model (exp 017) | 13 + 23 bits | 15 | 446.5 (local fit, not converged) | not run | fails compression (k ≥ 31 vs 13); fit not converged (exp 031) |
| S4-fixτ | S4 lepton τ fixed, best quark assignment (exp 020) | 6 + 10 bits | 8 | 3167.5, p 0 | lepton τ worse than 98/100 random τ | fails fit and compression |
| SU5-CSD3 | SU(5) Clebsch (4/9, 9/2) + CSD3 seesaw (arXiv:1503.03306), pre-registered (exp 022) | 9 + 1 bit (22 strict) | 14 | 22.40, p_global 4.4e-4 | not run (kill) | known model; fails fit quality; no CKM → PMNS link |
| S4xSO10 | S4×SO(10) (arXiv:1705.01555; exps 023, 026) | 17 + ~12 bits | 16 | 11.59 full thresholds (9.53 linear; dof < 0); Z3 phases 36.96, p 6e-7 | not run | known model; fails compression; PMNS not predicted from quarks |
| BL-CDSV | bi-large mixing, CKM charged-lepton correction (1902.08962; exps 027, 030) | 1 cont, 12 honest (14 with grammar + ordering) | 4 | NO 8.21, p_global 0.042; IO global 6.20, p 0.10 | N1 37%, N2 78-82%, power 0-5% (fit-only); likelihood (exp 037): NO N2 excluded as exact (power 0.55-0.60) | known model; fails compression with honest count; marginal fit |
| YUK | Koide-Nishiura yukawaon (1510.05370; exp 038) | 6 + 4 bits | 13 | 1027, p 1.7e-217 (30% floor: 41.1, p 7.7e-7) | not run | known model; passes compression; fails fit quality (m_u/m_c, m_s/m_b) |
| FZ4U | universal four-zero texture (hep-ph/0212195; exp 039) | 11 + 2 bits | 15 | ∞ (sin²θ23 ≈ 0.01), Gaussian ≈ 1159 | not run | passes compression; fails fit quality |
| DEMU | universal democratic breaking (hep-ph/9912358; exp 039) | 10-11 | 15 | ∞ (V_ub -20σ, δ_q = 0) | not run | fails fit quality |
| SO10-FN | real 10+126+120 SO(10) (hep-ph/0612132) | 16 + ~3 bits | 17 | 0.30 (1 dof) | not run | known model; not compressive; predicts neither δ_ℓ nor PMNS from quarks (exp 014) |

Dropped in iteration 5: T1 and F2-LO ("no evidence" under the protocol, exp 018).

Known baselines: `experiments/20261118-013-infra-hardening/results.csv` (current convention; supersedes exp 006 p-values). None passes k ≤ N-2 with acceptable p.

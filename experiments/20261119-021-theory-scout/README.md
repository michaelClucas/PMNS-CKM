# 20261119-021 Theory scout: compressive quark-lepton families and an information bound from the negative results

## Hypothesis

1. After seven negative families, a family where quark and lepton observables share parameters by construction can still reach k ≤ N − 2 and survive PDG 2026 light-quark precision.
2. The negative results so far can be turned into a quantitative upper bound on how much PMNS information the CKM (+ masses) carries within the families tried.

## Method

- Read STATE, CANDIDATES, OVERSEER_PROMPT and exps 003-018.
- Literature: arXiv abstract pages for 1406.7005, 1305.6612, 1705.01555, 1503.03306, 2309.15901, 0902.4644, hep-ph/0405088, 1304.6264, 1512.00850, 1512.07531. PDF text (pdftotext, kept outside the repo) for 1305.6612, 1503.03306 and 1705.01555, for parameter counts, Clebsches and fit errors. 1004.3781, hep-ph/0212195 and 1404.5726 were checked from a search listing only (titles and abstract snippets).
- Quick numerical checks, all single-process and ≤ 10 min each:
  - `checks.py`:
    - C1: Clebsch double-ratio filter on an 11-value Clebsch lattice in the SU(5) 1-2 texture.
    - C2: universal four-zero texture fits, quark and lepton.
    - C3: QLC with symmetry-fixed phases.
  - `checks2.py`:
    - C4: CSD(n) Littlest-Seesaw lepton fits.
    - C5: published GUT predictions vs NuFIT 6.1.
    - C6: diagonal-Clebsch double ratios, and AGMS m_s/m_d.
  - `checks3.py`:
    - C7: the charged-lepton 1-2 correction CSD3 would need.
- Bound (`bound.py`): no new fits. It reads the committed null CSVs and JSONs of exps 007, 009, 010 and 018, the exp 014 PT samples (`artifacts/so10/`, present), and the exp 016/017 χ² values. It computes:
  - B_ckm: profile log Bayes factor of real vs random hierarchical CKM.
  - S_rank: rank surprisal.
  - G_pred: held-out PMNS predictive gain from quark data vs prior.
  - KL_shift: how far quark data move the PMNS prediction.
  - The net bits of sharing τ.
  - The PMNS information budget vs Haar.

All measured numbers come from `src.data` / `src.rg`. Published model predictions and parameters quoted from papers are labelled as such.

## Parameter count

N/A (scouting). Every proposal in `proposals.md` carries its own N, k_cont and bits.

## Result

### A. Proposals (`proposals.md`)

Ranked by viability × (N − k):

| rank | family | N | k | predictions | check result | literature |
|---|---|---|---|---|---|---|
| 1 | P1 SU(5) sparse Clebsch textures + CSD3 seesaw | 13 | ≈ 10 if Clebsch/n/η fixed by a published model (≈ 20 if scanned) | R = (m_e/m_μ)/(m_d/m_s), 5 lepton obs from 1 parameter, θ12e and θ23e from Y_d | R +1.9σ (2% floor) / +3.2σ (PDG) for (4/9, 9/2). Pure CSD3 χ² 12.0 for 5 obs (p 0.017). θ12e correction only reaches 10.2 and must be ≤ 0.03 rad; θ23e is needed. | 1503.03306, 1512.07531, 2309.15901 |
| 2 | P3 S4×SO(10), all Yukawas from the same CSD3 flavons | ≈ 15 | 19 as published (k > N); ≈ 10 + 6-10 bits with Z_n-fixed phases | δ_ℓ windows, R, PMNS from shared alignments | δ lower window Δχ² 0.7-1.8, upper 3.9-4.3. Not yet fitted with 2% light-quark floor. | 1705.01555 |
| 3 | P5 modular SU(5)/PS, one τ, minimal-weight rule | 15 | ≈ 11 | R, CKM + PMNS from τ | Y_e = Y_d^T dead by R (+342σ). Common-τ quarks fail (exp 017). | 2101.12724, 2309.15901 |
| 4 | P1b AGMS SU(5)×A4, θ13 = θ_C/√2 | 15 | ≈ 10 + ~7 bits | θ13, δ_ℓ, m_s/m_d, δ_q | R passes. Published PMNS point χ² 120 vs NuFIT 6.1. m_s/m_d 18.95 is −2.0σ. | 1305.6612 |
| 5 | P4 universal four-zero texture (shared r) | 8 | 7 | CKM, PMNS (1 dof each) | Quark χ² 77.8 (1 dof); leptons 67.9. Dead. | 1004.3781, hep-ph/0212195 |
| 6 | P2 Pati-Salam A4×Z5 CSD4 | ? | ? | θ23, θ13, δ, θ12, θ_C ≈ 1/4 | CSD4 χ² 1080. Benchmarks χ² 29-55. Dead at LO. | 1406.7005 |
| 7 | P6 QLC with fixed phases | 4 | 5 bits | 4 PMNS | best χ² 25.3. Dead. | hep-ph/0405088 |

New general tool: the **Clebsch double ratio** R = (m_e/m_μ)/(m_d/m_s) = 0.0938 ± 0.0027 (2% floor; ± 0.0016 PDG). It is RG-invariant in the SM, and SUSY-threshold-invariant for the first two generations. Results:
- Georgi-Jarlskog: +6.5σ diagonal, ≥ +7.8σ in the [[0,a],[a,c]] texture.
- Minimal SU(5): +342σ.
- 29 of 1331 lattice triples (2.2%) survive.

Any GUT family with fixed 1-2 Clebsches should be filtered on R before any fit.

### B. Bound (`bound.md`)

- PMNS data hold about 19 bits relative to Haar (s13² 7.8, s12² 5.2, s23² 4.0, δ 2.0).
- Real CKM vs random hierarchical CKM through the same searches: B_ckm ≤ +1.2 bits (F1, NO, structured). F2 is −15 to +0.2, T1 ≤ +0.25, symbolic ≤ +0.09. Best single-cell rank surprisal is ≤ 4 bits in the multi-observable searches. Exp 010 has one 8.2-bit cell (s23², 1/300) among 160 cells, ≲ 2 bits after the cell penalty.
- So within F1, F2, T1 and the exp 010 grammar, the CKM carries ≲ 1-2 bits of PMNS information beyond a random hierarchical CKM: ≲ 10% of the budget, consistent with 0.
- SO(10) held-out predictive gain:
  - Quark data move the PMNS prediction by 1-3 bits per observable (KL_shift), but jointly away from the data.
  - G_pred = −76 bits (MN, our errors), −857 (MN, DR errors), −1132 (FN). With 3σ smoothing: −7, −93, −127.
  - Within the posterior, I_gauss ≤ 0.09 bits per observable.
- Sharing τ (2O, A4 catalog): the net gain is ≤ −28 bits against a maximum Occam gain of about 11 bits.
- Not covered:
  - Information carried by the hierarchy itself.
  - Families outside the list (e.g. P1-P3).
  - IO for SO(10).
  - Effects beyond one-loop running.

## Verdict

No candidate (scouting experiment).
- P1 is the only family where the CKM enters PMNS through a fixed map and k ≤ N − 2 is reachable on paper. Its lepton core (CSD3) is already at p ≈ 0.02 with NuFIT 6.1, and its published Clebsch pair is +2-3σ on R. Expected viability is low (~15%), but it is the best available.
- P3 is the best-specified "same alignments" model and is untested at 2% light-quark precision.
- P4, P2 and P6 are dead on quick checks. P1b and P5 are dead as published.
- The bound gives the negative results a single number: ≲ 1-2 bits of CKM-specific PMNS information in every family tried.

## Recommended next experiments (dispatchable briefs)

1. **P1 pre-registered test: CSD3 + SU(5) Clebsch texture with θ23e (2-3 cores, 2 h).**
   - Freeze the 1503.03306 Yukawa structure (texture zeros, Clebsch 4/9 and 9/2, b-τ = 1, η = 2π/3, n = 3) before fitting. Copy the structure from the PDF into the brief.
   - Fit at M_GUT with MSSM `src.rg` running (tan β ∈ {5, 10} as the paper fixes it, not fitted) against our GUT-scale data with the 2% floor.
   - Score N = 13 (exclude m_μ/m_τ or add the threshold parameter to k).
   - Report χ², p, R pull, and the θ12e/θ23e it produces vs the C7 window (θ12e ≤ 0.03).
   - Then run N1 (fake CKM) with the protocol: P1 has k ≤ N − 2 on paper, so the strict hit rule applies, and the power check is meaningful for once.
   - Kill criterion: χ² p < 0.01 at the frozen structure.
2. **P3 reproduction + refit: S4×SO(10) CSD3 (arXiv:1705.01555) (2 cores, 2-3 h).**
   - Reproduce their best fit (χ² 3.4) with their inputs, as exp 016 did for 2O.
   - Refit with PDG 2026 + 2% floor.
   - Extract the 45_H Clebsches and compute R.
   - If χ² stays acceptable, run the fixed-phase variant (each of the 6 phases restricted to Z3 or Z4 roots; count the bits) and see whether k ≤ N − 2 with p > 0.05 is reachable.
3. **Infra (small): add R and in-sector mass ratios to `src.evaluate` (1 core, 1 h).**
   - R as a scored, scale-free observable, with the 2% floor documented.
   - Add `bound.py`'s B_ckm and S_rank as a standard output of `src.nulls`, so every future family reports its CKM-specific bits.
   - This also covers the open STATE gap on same-scale mass ratios.

## Files

- `proposals.md`: families, constructions, counts, checks, literature.
- `bound.md`: information-bound write-up (intended as a section of the final write-up).
- `bound.py` → `bound.json`, `bound.log`.
- `checks.py` → `checks.json`, `checks.log` (C1-C3; ~10 min, dominated by C2's lepton Nelder-Mead).
  - The named-Clebsch entry "1305.6612 AGMS" was added to the script after the run. In checks.json the AGMS triple appears in `surviving` as (c12, c21, c22) = (0.5, 6, 6) in this script's convention, min pull 0.002.
- `checks2.py` → `checks2.json`, `checks2.log` (C4-C6, < 1 min).
- `checks3.py` → `checks3.json`, `checks3.log` (C7, ~2 min).

Rerun from this folder with `../../.venv/Scripts/python <script>`. No background processes remain.

## Caveats

- C1 uses a 2×2 (first two generations) texture and ignores 1-3/2-3 mixing feedback on the light eigenvalues (O(θ23²) ~ 0.2%).
- C2 used 60 starts per fit. The shared-r quark χ² 77.8 is a best-found value, not a proven minimum.
- C4/C7 use U_e = 1 or a pure 1-2 rotation, scored with the NuFIT tables at low scale (no ν running: SM, m1 = 0, negligible).
- Published-point checks (C5) use printed central values only.
- Viability percentages in `proposals.md` are judgement calls, not computed.

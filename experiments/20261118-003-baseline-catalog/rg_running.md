# RG running: comparison scale and method

## Recommendation

1. **Default comparison at low scale (M_Z, data as published), SM, no running of PMNS.** Every candidate also reports a **GUT-scale check** (SM at 10^16 GeV; MSSM at 2×10^16 GeV for tan β ∈ {5, 10, 30, 50}) using the CKM values of Antusch-Hinze-Saad and the PMNS running estimate below.
2. **Running of CKM θ23 and θ13 is never negligible** for relations evaluated at a high scale (13% in SM, 8-12% in MSSM). θ_C runs < 1%.
3. **PMNS running is negligible in the SM** for any ordering when the lightest mass ≲ 0.05 eV, and in the MSSM for tan β ≲ 10 with hierarchical masses. It is large (≫ 1σ, nonlinear) in the MSSM for quasi-degenerate masses or tan β ≳ 30, so any MSSM/GUT-scale candidate adds tan β and m_lightest (and Majorana phases) to its parameter list.
4. **Implementation:** write a one-loop SM (+ optional MSSM) RGE integrator in numpy/scipy for (g_i, Y_u, Y_d, Y_e, λ, κ) and validate it against the Antusch-Hinze-Saad tables. Do not depend on REAP (Mathematica).

## CKM running, M_Z → GUT

From Antusch, Hinze, Saad, "Updated running quark and lepton parameters at various scales", arXiv:2510.01312 (two-loop RGEs via REAP, SMDR/RunDec matching; UTfit 2023 inputs). Values as extracted from the HTML version (2024 dataset):

| | θ12 | θ23 (×10⁻²) | θ13 (×10⁻³) | δ (rad) |
|---|---|---|---|---|
| M_Z | 0.2251 ± 0.0008 | 4.193 ± 0.041 | 3.70 ± 0.08 | 1.139 ± 0.023 |
| SM, 10^16 GeV | 0.22708 ± 0.00082 | 4.743 ± 0.047 | 4.19 ± 0.09 | 1.139 |
| MSSM, 2×10^16, M_SUSY = 3 TeV, tan β = 5 | 0.22702 | 3.862 | 3.40 | 1.139 |
| same, tan β = 10 | 0.22702 | 3.874 | 3.41 | 1.139 |
| same, tan β = 30 | 0.22702 | 3.823 | 3.37 | 1.139 |
| same, tan β = 50 | 0.22702 | 3.679 | 3.24 | 1.139 |
| MSSM, M_SUSY = 10 TeV, tan β = 10 | 0.22702 | 3.905 | 3.44 | 1.139 |

Relative shifts (my arithmetic from the table): SM θ12 +0.9%, θ23 +13.1%, θ13 +13.2%; MSSM (3 TeV) θ12 +0.85%, θ23 −7.6% (tan β 10) to −12.3% (tan β 50), θ13 −7.8% to −12.4%; δ unchanged at the quoted precision. Earlier tables: Antusch, Maurer, arXiv:1306.6879.

In units of current errors (data/ckm.json, PDG 2026: σ(θ_C) ≈ 0.0007, σ(s23)/s23 ≈ 1.9%, σ(s13)/s13 ≈ 2.3%): the θ23, θ13 shifts are 4-7σ of the CKM errors, θ_C ~3σ. So CKM must be run whenever a relation is claimed at the GUT scale.

## PMNS running, M_Z → GUT

Analytic one-loop RGEs for the Weinberg operator (Antusch, Kersten, Lindner, Ratz, arXiv:hep-ph/0305273, eqs. 8-10):

`dθ12/dt = −C y_τ²/(32π²) · sin 2θ12 · s23² · |m1 e^{iφ1} + m2 e^{iφ2}|² / Δm²_sol + O(θ13)`,

C = −3/2 (SM), C = 1 (MSSM), y_τ² ∝ (1 + tan²β) in the MSSM, t = ln(μ/μ0). The 1/Δm²_sol enhancement makes θ12 the most RG-sensitive angle; φ1 − φ2 = π suppresses it. θ13 and θ23 have 1/Δm²_atm and are far smaller.

`rg_estimate.py` (log: `rg_estimate.log`) evaluates eqs. 8-10 with constant right-hand side over t = ln(M_GUT/M_Z) = 33.0, using NuFIT 6.1 IC24 central values from `data/`. Linear approximation; entries above ~2° only mean "large, integrate numerically". Selected rows (shift M_Z → M_GUT, degrees):

| Model | Ordering | m_lightest | Δθ12 (φ=0 / φ1−φ2=π) | Δθ13 (max of the two) | Δθ23 (max) |
|---|---|---|---|---|---|
| SM | NO | 0 | 0.0004 | 0.0002 | 0.001 |
| SM | NO | 0.05 eV | 0.055 / 0.00 | 0.002 | 0.006 |
| SM | IO | 0 | 0.062 / 0.00 | 0 | −0.001 |
| SM | IO | 0.1 eV | 0.32 / 0.00 | 0.0005 | −0.017 |
| MSSM tan β = 10 | NO | 0 | −0.027 | 0.010 | −0.08 |
| MSSM tan β = 10 | NO | 0.05 eV | −3.7 / 0.00 | 0.15 | −0.38 |
| MSSM tan β = 10 | IO | 0 | −4.2 / 0.00 | 0 | 0.06 |
| MSSM tan β = 30 | NO | 0 | −0.25 | 0.09 | −0.73 |
| MSSM tan β = 50 | NO | 0 | −0.68 | 0.26 | −2.0 |
| MSSM tan β = 50 | IO | 0 | −103 (nonlinear) / −0.006 | 0 | 1.6 |

Compare with NuFIT 6.1 1σ: θ12 0.42°, θ13 0.11°, θ23 ~0.9°.

Literature agreement: hep-ph/0305273 and the review Ohlsson, Zhou, "Renormalization group running of neutrino parameters", arXiv:1311.3846 (Nature Commun. 5, 5153) state that SM running is small and MSSM running is enhanced by tan β and by quasi-degenerate masses, with θ12 the most sensitive. Mei, arXiv:hep-ph/0502015, does the M_Z → Λ_GUT analysis for NH/IH/degenerate in SM and MSSM. Caveats of my estimate: no seesaw thresholds (above the lightest M_R the running changes, Antusch et al. arXiv:hep-ph/0501272), y_τ not run, MSSM running applied from M_Z instead of from M_SUSY.

**Dirac neutrinos:** analytic RGEs in Lindner, Ratz, Schmidt, arXiv:hep-ph/0506280: mixing-angle running is enhanced relative to quarks (milder hierarchy, larger angles) and in the MSSM at large tan β comparable to future experimental precision. Same qualitative pattern; no Majorana phases to suppress θ12 running.

## When ignoring running is acceptable

Rule: the RG shift of every quantity entering a relation must be < 0.3σ of the error on the predicted observable.

| Relation type | SM, NO hierarchical | MSSM tan β ≲ 10, hierarchical | MSSM large tan β or degenerate |
|---|---|---|---|
| uses PMNS angles only | ignore | ignore except θ12 at IO (4°) | must run |
| uses θ_C → PMNS θ12 (QLC) | θ_C shift 0.11° = 0.27σ of θ12: borderline, run it | same | must run |
| uses θ_C → PMNS θ13 (TBC/KMS) | shift 0.08° = 0.7σ of θ13: **must run** | must run | must run |
| uses θ_cb, θ_ub (QLC atmospheric, V† X) | θ_cb +0.31° = 0.3-0.4σ of θ23: run it | must run | must run |
| uses δ_q | ignore (δ_q invariant at quoted precision) | ignore | ignore for δ_q |
| mass ratios (GUT Yukawa relations) | always run (QCD running is O(1)) | always | always |

The relation is assumed to hold at a stated scale, so a candidate states whether it is a low-scale or high-scale relation; scale choice is a discrete parameter (k_d += 1) if chosen after looking at the data.

## Implementation route

1. **Own numpy/scipy integrator (recommended).** One-loop SM RGEs for g1, g2, g3, Y_u, Y_d, Y_e, λ (Machacek-Vaughn, standard) and κ (Antusch, Drees, Kersten, Lindner, Ratz, arXiv:hep-ph/0108005, which corrected the earlier κ RGE). MSSM one-loop is equally short. Integrate in t = ln μ with `scipy.integrate.solve_ivp`; extract angles with `src/mixing.py` after each run. ~200 lines, ms per run, vectorizable for synthetic data. Boundary values at M_Z from `data/` (masses need MS-bar conversion; take the M_Z Yukawas from 2510.01312 tables to start, then reproduce them).
2. **Validation targets:** the CKM table above (SM 10^16 GeV, MSSM tan β = 10, 50), and the analytic PMNS shifts in `rg_estimate.log` for the SM.
3. **Two-loop:** not needed for the angles at current precision (one-loop vs two-loop differences are below the CKM-error level for angles per the structure of the RGEs; **not checked numerically here**, verify by comparing our one-loop output against the two-loop 2510.01312 table, which is the actual test).
4. **Existing packages:**
   - REAP/MPT (Mathematica; Antusch et al., arXiv:hep-ph/0501272; project page at MPI-K Heidelberg) handles seesaw thresholds and two loops but needs Mathematica or the free Wolfram Engine + `wolframclient` from Python. Not recommended as a dependency; useful as a cross-check if someone has Mathematica.
   - SusyTC (REAP extension for SUSY thresholds, arXiv:1512.06727): Mathematica.
   - PyR@TE 3 (Python, arXiv:2007.12700): generates RGEs for non-supersymmetric gauge theories up to three loops for gauge couplings; can be used to generate/check the SM beta functions (and non-SUSY extensions such as type-II seesaw triplets). Not for the MSSM.
   - SMDR (C library, Martin-Robertson arXiv:1907.02500): precise SM MS-bar inputs at M_Z; overkill for angles.
5. **Neural emulator:** once the integrator exists, the RG map (low-scale angles, tan β, m_lightest, Majorana phases) → GUT-scale angles is smooth and cheap to emulate; that makes F3/F4 generators fast enough for SBI.

## Erratum (overseer, iteration 2)
"M_Z θ12 = 0.2251" above is sin θ12; the angle is 0.2270 rad. With that fixed, SM θ_C runs +0.04% from M_Z to 1e16 GeV, not +0.9%, and the MSSM +0.85% figure is wrong the same way. Validated running numbers: `experiments/20261118-008-rg-module/README.md`.

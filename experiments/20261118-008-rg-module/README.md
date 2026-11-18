# 008: RG module (`src/rg.py`)

## Hypothesis

Our numpy/scipy RG code reproduces published running of CKM parameters and Yukawas (arXiv:2510.01312) and the analytic one-loop running of PMNS parameters (hep-ph/0305273), so the program can compare CKM and PMNS at a common scale.

## Method

- RGEs (all in the `src/rg.py` docstring with conventions): SM gauge couplings at two loops, Yukawas `Y_u, Y_d, Y_e`, Higgs quartic and Weinberg operator `kappa` at one loop (kappa RGE from hep-ph/0108005). MSSM above a single threshold `M_SUSY` (default 3 TeV) with tree-level matching `Y_u/sin(beta)`, `Y_{d,e}/cos(beta)`, `kappa/sin^2(beta)`, no tan-beta-enhanced threshold corrections, no MSbar/DRbar conversion. Integration with `solve_ivp` (DOP853, rtol 1e-10) in `t = ln mu`, about 0.1 s per M_Z to GUT run.
- M_Z boundary values:
  - Gauge couplings, Higgs quartic, MSbar vev v(M_Z), v(M_t): full-SM MSbar set of arXiv:2510.01312 Eq. (4) (derived from PDG 2024 inputs), stored in `data/sm_inputs.json`. PDG 2026 raw values (alpha_s, sin^2 theta_W MSbar, M_Z, M_W, M_H, G_F) are stored next to them for reference.
  - Quark masses from `src.data.load_masses()` (PDG 2026), run with four-loop QCD and two-loop decoupling at m_c(m_c), m_b(m_b). Top: pole mass to MSbar with three-loop QCD, `y_t(M_t) = sqrt2 m_t(m_t)/v(M_t)`, then `y_t(M_Z)` is solved through our own SM RGEs.
  - Charged leptons: one-loop QED pole to MSbar.
  - CKM from PDG 2026 (`data/ckm.json`, taken to be the M_Z value). PMNS from NuFIT 6.1 at M_Z, with m_lightest, ordering and Majorana phases as inputs.
- Validation: (1) our M_Z Yukawas vs 2510.01312 Eq. (4). (2), (3) Running from the paper's own M_Z inputs (Eq. 4 Yukawas, Eq. 6 angles) compared with their Tables 2 and 6, which isolates the integrator. (4) d(theta)/d ln(mu) at M_Z vs the analytic formulas of hep-ph/0305273 eqs. (8)-(10), with theta13 set to 1e-4 rad because those formulas are leading order in theta13. I checked every published number used here against the arXiv HTML of 2510.01312v2 (Tables 2 and 6, Eqs. 1-8).

Parameter count: N/A (infrastructure).

Rerun: `.venv/Scripts/python experiments/20261118-008-rg-module/run_tables.py > experiments/20261118-008-rg-module/run.log` (about 10 s). Tests: `tests/test_rg.py`.

## Validation table (selected; full table in `run.log`)

The "shift ratio" column is (ours − M_Z value)/(published − M_Z value): the fraction of the published M_Z→μ shift that we reproduce.

| quantity | ours | published | rel. diff | shift ratio |
|---|---|---|---|---|
| y_t(M_Z) from data/ | 0.96683 | 0.967 | −0.02% | |
| y_b(M_Z) from data/ | 0.016329 | 0.0163 | +0.18% | |
| y_c, y_u (M_Z) from data/ | | | +1.1%, +1.0% | |
| y_tau(M_Z) from data/ | 0.0099542 | 0.0099378 | +0.17% | |
| SM 1e16: θ12 | 0.22704 | 0.22708 | −0.02% | 0.46 |
| SM 1e16: θ23 | 0.047486 | 0.04743 | +0.12% | 1.010 |
| SM 1e16: θ13 | 0.0041890 | 0.00419 | −0.02% | 0.998 |
| SM 1e16: δ | 1.1389 | 1.139 | 0.00% | |
| SM 1e12: θ23, θ13 | 0.046526, 0.0041044 | 0.04650, 0.00411 | +0.06%, −0.14% | 1.006, 0.986 |
| SM 1e16: y_t, y_b, y_tau | 0.4506, 0.006122, 0.009652 | 0.4454, 0.00606, 0.009639 | +1.2%, +1.0%, +0.13% | |
| SM 1e16: g1, g2, g3 | 0.5768, 0.5245, 0.5283 | 0.5752, 0.5230, 0.5310 | +0.27%, +0.30%, −0.50% | |
| MSSM tanβ=10, 2e16: θ23 | 0.039528 | 0.03874 | +2.0% | 0.75 |
| MSSM tanβ=10, 2e16: θ13 | 0.0034873 | 0.00341 | +2.3% | 0.73 |
| MSSM tanβ=50, 2e16: θ23 | 0.037431 | 0.03679 | +1.7% | 0.88 |
| MSSM tanβ=50, 2e16: θ13 | 0.0033021 | 0.00324 | +1.9% | 0.87 |
| MSSM tanβ=10/50: θ12, δ | 0.22699, 1.1390 | 0.22702, 1.139 | −0.02%, 0.00% | |
| MSSM tanβ=10/50, 2e16: y_t | 0.5165 / 0.5599 | 0.4947 / 0.5312 | +4.4% / +5.4% | |
| PMNS dθ12/dt, SM NO m1=0.05 eV | 2.762e-5 | 2.751e-5 (analytic) | +0.4% | |
| PMNS dθ23/dt, SM NO m1=0.05 eV | 2.794e-6 | 2.804e-6 (analytic) | −0.3% | |
| PMNS dθ13/dt, SM NO m1=0.05 eV | 2.963e-8 | 2.964e-8 (analytic) | 0.0% | |
| PMNS dθ12, dθ13, dθ23/dt, MSSM tanβ=50 NO m1=0 | −3.419e-4, −8.154e-5, −1.006e-3 | −3.406e-4, −8.175e-5, −1.010e-3 | +0.4%, −0.3%, −0.3% | |
| PMNS dθ12/dt, MSSM tanβ=30 NO m1=0.1, φ=(0.5,1.0) | −6.161e-2 | −6.137e-2 | +0.4% | |

The ~0.4% residual on the PMNS rates is the same in SM and MSSM, so it comes from the analytic approximation, not from the y_tau normalisation. In the suppressed case (IO, φ1−φ2=π) θ12 differs by 2% and θ13 has a nonzero rate where the LO formula gives 0. That is expected, because only sub-leading terms survive there.

Also checked (tests): up-then-down running returns the M_Z state to 1e-8 relative (SM, MSSM tanβ 10 and 50); mixing matrices stay unitary to 1e-12; kappa stays symmetric; the M_Z state reproduces the input angles, δ and masses to 1e-9.

## Headline running, M_Z → 2×10^16 GeV (inputs from data/, Majorana phases 0)

| | θ_C | θ13^q | θ23^q |
|---|---|---|---|
| M_Z | 13.0129° | 0.2156° | 2.4008° |
| SM | 13.0150° (+0.002°) | 0.2444° (+13.4%) | 2.7217° (+13.4%) |
| MSSM tanβ=10 | 13.0120° | 0.2032° (−5.7%) | 2.2629° (−5.7%) |
| MSSM tanβ=50 | 13.0119° | 0.1924° (−10.8%) | 2.1422° (−10.8%) |

The published two-loop MSSM shifts are −7.8% (tanβ=10) and −12.3% (tanβ=50). See limitations.

PMNS (degrees, θ12/θ13/θ23 at 2e16 GeV; M_Z values NO: 33.76/8.62/43.27, IO: 33.76/8.65/48.15):

| model | NO m1=0 | NO m1=0.05 eV | IO m3=0 | IO m3=0.05 eV |
|---|---|---|---|---|
| SM | 33.761/8.620/43.271 | 33.819/8.621/43.275 | 33.819/8.650/48.149 | 33.879/8.650/48.145 |
| MSSM tanβ=10 | 33.733/8.620/43.218 | 31.129/8.594/43.026 | 31.225/8.655/48.191 | 28.881/8.652/48.390 |
| MSSM tanβ=50 | 32.847/8.606/41.451 | 0.646/7.631/35.006 | 7.821/8.840/49.588 | 4.554/8.630/56.118 |

SM running of PMNS is at most 0.12° (θ12, IO, m3 = 0.05 eV), which is below 0.3σ of NuFIT. In the MSSM at tanβ=50, θ12 collapses for every non-hierarchical case. With Majorana phases φ1−φ2 = π that running is suppressed; `pmns_at(..., majorana=(phi1, phi2))` covers it. The full grid (m_lightest ∈ {0, 0.01, 0.05} eV, 1e12 GeV, mass ratios) is in `run.log`.

## Verdict

**SM: validated.** CKM angle running agrees with 2510.01312 to ≤0.14% at 1e12 and 1e16 GeV, which is 1.0 ± 0.015 of the published shift and far below the CKM errors (~1-2%). PMNS rates match the analytic one-loop formulas to 0.4%. Yukawas at the GUT scale are within ~1.4% (one-loop Yukawa running), and gauge couplings within 0.5%.

**MSSM: usable with a stated error.** θ12 and δ match. θ23 and θ13 at 2e16 GeV come out 1.7-2.3% too high, because we reproduce only 73-88% of the published two-loop MSSM shift. That is ~2× the published 1σ (1%). The cause is the one-loop MSSM Yukawa running, which leaves y_t at the GUT scale 4-5% above the published value; I did not track it further. Any candidate that uses MSSM GUT-scale θ23^q or θ13^q must add this ±2% systematic, or get two-loop MSSM Yukawa RGEs first.

## Known limitations

- Yukawas, λ and κ run at one loop. In the SM this costs ~1% on GUT-scale Yukawas and nothing visible on angles. In the MSSM it costs 2% on θ23^q and θ13^q (above).
- MSSM: single threshold, tree-level matching, no tanβ-enhanced η_b or η_τ threshold corrections (2510.01312 Eqs. 13-16 give the a-posteriori recipe), no MSbar→DRbar conversion. λ is frozen above M_SUSY.
- No seesaw thresholds: κ is the effective operator all the way to the output scale. This is wrong above the lightest right-handed neutrino mass (hep-ph/0501272).
- M_Z boundary: gauge couplings, λ and v come from the 2510.01312 Eq. (4) derived set, not from our own matching. Charged-lepton pole→MSbar uses one-loop QED (0.1-0.2% vs the paper). Light-quark Yukawas differ from the paper by ≤1.1%, which is mostly the PDG 2024→2026 input change. The CKM input is treated as the M_Z value.
- `m_lightest = 0` comes back as ~1e-9 eV numerical noise at high scales (eigenvalues of κ†κ). Treat values below 1e-8 eV as 0.
- Neutrino masses are reported as κ v(M_Z)²/4 with fixed v, so only ratios and angles are meaningful at high scales.
- Correction to `experiments/20261118-003-baseline-catalog/rg_running.md`: its "M_Z θ12 = 0.2251" is sin θ12. The paper's θ12(M_Z) = 0.2270, so SM θ_C runs +0.04% to 1e16 GeV, not +0.9%.

## Data sources (new file `data/sm_inputs.json`, accessed 2026-11-18)

- `pdg2026`: PDG 2026 "Physical constants" table (https://pdg.lbl.gov/2026/reviews/rpp2026-rev-phys-constants.pdf): α_s(M_Z) = 0.1180(9), sin²θ̂_W(M_Z) MSbar = 0.23122(6) (also EW review Table 10.2), M_Z = 91.1879(20), M_W = 80.3625(77), G_F, α(0). M_H = 125.13(11) from the PDG 2026 API (S126M). I did not locate α̂(M_Z)⁻¹ MSbar in the text extraction of the 2026 EW review, so it is not stored.
- `ahs2025_msbar_MZ`: S. Antusch, K. Hinze, S. Saad, arXiv:2510.01312v2, Eq. (4): g1 = 0.461228, g2 = 0.65096, g3 = 1.2123, λ = 0.55853 (REAP convention), v(M_Z) = 248.401 GeV; v(M_t) = 246.604 GeV from the text. These are derived MSbar parameters from PDG 2024 inputs, not direct measurements.
- The literature reference values used for validation (Tables 2 and 6, Eq. 4 Yukawas, Eq. 6 angles) live in `run_tables.py`, `tests/test_rg.py` and `src/rg.py` (`AHS_EQ4_YUKAWAS`, `AHS_EQ6_CKM`, marked as validation-only literature values).

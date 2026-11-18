# 20261118-001: measured-data layer

## Hypothesis
N/A, infrastructure.

## Method
- PDG 2026 Review of Particle Physics (Int. J. Mod. Phys. A 41, 2630011 (2026)):
  - CKM review PDF (rpp2026-rev-ckm-matrix.pdf, revised March 2026), Sec. 12.4 global fit and Sec. 12.2 direct determinations, transcribed into `data/ckm.json` from `pdftotext` output.
  - PDG API SQLite `pdg-2026.0.sqlite` for quark and charged-lepton masses, read by `build_masses.py` (summary-table values, scheme/scale from the PDG listing notes).
- NuFIT 6.1 (2025, data to November 2025):
  - Parameter table `v61.tbl-parameters.pdf` transcribed in `make_pmns_json.py` into `data/pmns.json`, both analyses, both orderings.
  - The four Delta chi^2 release files downloaded and split per projection by `build_nufit.py` into `data/nufit/` (format in `data/nufit/FORMAT.md`).
- Default PMNS analysis: IC23 w/o SK-atm. NuFIT does not name a recommended one; this is the upper section of their table and the filled regions in their plots. IC24 with SK-atm is stored too and selectable with `analysis="IC24_wSKatm"`.
- Loader `src/data.py`, tests `tests/test_data.py`.

Rebuild from repo root:

    .venv/Scripts/python experiments/20261118-001-data-build/make_pmns_json.py
    .venv/Scripts/python experiments/20261118-001-data-build/build_masses.py
    .venv/Scripts/python experiments/20261118-001-data-build/build_nufit.py [--full]

Downloads cache in gitignored `artifacts/`. No extra packages (stdlib + numpy). `pdftotext` (poppler) was used once to read the PDFs; it is not needed to rebuild.

## Parameter count
N/A.

## Result

CKM (PDG 2026, SM global fit):

| quantity | value |
|---|---|
| lambda | 0.22517 +- 0.00068 |
| A | 0.826 +0.017 -0.015 |
| rhobar | 0.1576 +0.0092 -0.0091 |
| etabar | 0.3556 +0.0071 -0.0069 |
| s12 | 0.22517 +- 0.00068 |
| s13 | 0.003763 +0.000088 -0.000083 |
| s23 | 0.04189 +0.00081 -0.00069 |
| delta | 1.154 +- 0.025 rad |
| J | (3.16 +0.13 -0.11) e-5 |

PMNS (NuFIT 6.1, IC23 w/o SK-atm; IC24 with SK-atm in brackets where different):

| quantity | NO | IO |
|---|---|---|
| sin^2 theta12 | 0.3088 +0.0067 -0.0066 | same |
| sin^2 theta13 | 0.02249 +- 0.00057 [0.02248 +0.00055 -0.00059] | 0.02261 +- 0.00056 [0.02262 +0.00057 -0.00056] |
| sin^2 theta23 | 0.470 +0.017 -0.014 [same] | 0.555 +0.013 -0.016 [0.550] |
| delta_CP (deg) | 207 +23 -20 [212 +26 -36] | 283 +24 -28 [274 +22 -25] |
| Delta m^2_21 (1e-5 eV^2) | 7.537 +0.094 -0.10 | same |
| Delta m^2_3l (1e-3 eV^2) | +2.521 +0.026 -0.018 [+2.511] | -2.500 +0.024 -0.023 [-2.483] |
| Delta chi^2 (IO - NO) | | 1.5 [5.9] |

Cross-check: best fits and 1 sigma / 3 sigma ranges read off the 1D chi^2 tables by `build_nufit.py` agree with the transcribed table to grid resolution; the IO floors in the tables are 1.486 and 5.908. `pytest tests/test_data.py`: 13 passed.

`pmns_delta_chi2` is an approximation: NuFIT publishes 1D, 2D and one 3D projection, not the full 6D chi^2. Default method uses the 2D (s23sq, dcp) table when both are given and adds 1D profiles for the rest, ignoring the remaining correlations (s13-s23, s13-dcp, dm3l-s23). `method="1d"` sums 1D profiles only.

## Verdict
Data layer built. Ready for independent verification by a second subagent (planned in STATE.md).

## Gaps and substitutions
- CKM correlation matrix: PDG does not publish it. `load_ckm()["correlations"]` is None. Options for later: CKMfitter or UTfit published results (not a PDG primary source), or treat the PDG errors as independent.
- CKM values are transcribed from the PDF (PDG API SQLite has no CKM global-fit entries). The verifier should recheck these by eye against Eqs. (12.26) to (12.28).
- NuFIT parameter table values are transcribed from the PDF; the chi^2 tables independently confirm them.
- Top MSbar mass (Q007TP2): PDG text says "MSbar mass" without stating the scale. Recorded as such.
- Masses are at their PDG schemes/scales; no running to a common scale has been done. That belongs to the RG-running task.
- NuFIT 3 sigma |U_PMNS| ranges (`v61.tbl-mixing.pdf`) not stored; the PDF layout did not extract cleanly and the loader does not need it.
- Full 6D NuFIT chi^2 is not published, so correlations beyond the 2D/3D projections cannot be recovered.
- The 3D (s23sq, dm3l, dcp) table is not committed (40 MB per file). `build_nufit.py --full` regenerates it into `artifacts/nufit_full/`.

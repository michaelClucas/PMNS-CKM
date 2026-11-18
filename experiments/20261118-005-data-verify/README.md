# 20261118-005: independent verification of data/

## Hypothesis
`data/` (ckm.json, pmns.json, masses.json, nufit/*.csv) matches the primary sources, and `src/data.py` reads it correctly.

## Method
Sources re-downloaded on 2026-11-18 into gitignored `artifacts/verify/dl/`, without using the build agent's cache:
- PDG 2026 CKM review, https://pdg.lbl.gov/2026/reviews/rpp2026-rev-ckm-matrix.pdf ("Revised March 2026", Ceccucci, Ligeti, Sakai). Pages 12-13 rendered to PNG (`pdftoppm`) and read from the images; Sec. 12.2 direct values read from `pdftotext -layout`.
- PDG 2026 API, https://pdg.lbl.gov/2026/api/pdg-2026.0.sqlite (edition 2026, release 2026-05-30). Mass rows found by description search (`%QUARK MASS%`, `%MASS RATIO%`, e/mu/tau), not by the identifiers listed in SOURCES.md; scheme/scale checked against `pdgtext`.
- NuFIT 6.1, http://www.nu-fit.org/sites/default/files/v61.tbl-parameters.pdf rendered to PNG and read from the image; the four `v61.release-{TBoff,TByes}-{NO,IO}.txt.xz` files decompressed and parsed by an independent parser.

Scripts (run from repo root):
- `verify_tables.py`: source values typed from the page images, compared to ckm.json, pmns.json and the NuFIT part of masses.json (value, +err, -err, 3 sigma range).
- `verify_nufit_csv.py artifacts/verify/dl`: every committed CSV compared row by row to the raw release section (rel. tol. 1e-6), plus a random sample printed.
- `verify_loader.py`: 71 logic checks of `src/data.py` and the `stats.Objective` convention.

## Parameter count
N/A.

## Result

Summary: 85 transcribed records (CKM 39, NuFIT table 46) and 15 PDG mass records, every value, error and 3 sigma range: 0 mismatches. 56 CSV files, 483,654 rows, compared in full against the raw NuFIT files: 0 differences. 71/71 loader checks pass. One convention hazard in `stats.Objective` (below).

### CKM and NuFIT parameter table (`verify_tables.py` output)
Tuples are (value, +err, -err[, (3 sigma low, high)]). CKM locations are pages of rpp2026-rev-ckm-matrix.pdf.

| field | data/ (value, +err, -err[, 3sigma]) | source | location | match |
|---|---|---|---|---|
| ckm.wolfenstein.lambda | (0.22517, 0.00068, 0.00068) | (0.22517, 0.00068, 0.00068) | Eq.(12.26) p.12 | yes |
| ckm.wolfenstein.A | (0.826, 0.017, 0.015) | (0.826, 0.017, 0.015) | Eq.(12.26) p.12 | yes |
| ckm.wolfenstein.rhobar | (0.1576, 0.0092, 0.0091) | (0.1576, 0.0092, 0.0091) | Eq.(12.26) p.12 | yes |
| ckm.wolfenstein.etabar | (0.3556, 0.0071, 0.0069) | (0.3556, 0.0071, 0.0069) | Eq.(12.26) p.12 | yes |
| ckm.alt.UTfit.lambda | (0.22519, 0.00068, 0.00068) | (0.22519, 0.00068, 0.00068) | Sec.12.4 text p.12 | yes |
| ckm.alt.UTfit.A | (0.82, 0.011, 0.011) | (0.82, 0.011, 0.011) | Sec.12.4 text p.12 | yes |
| ckm.alt.UTfit.rhobar | (0.1561, 0.0088, 0.0088) | (0.1561, 0.0088, 0.0088) | Sec.12.4 text p.12 | yes |
| ckm.alt.UTfit.etabar | (0.3587, 0.0069, 0.0069) | (0.3587, 0.0069, 0.0069) | Sec.12.4 text p.12 | yes |
| ckm.alt.tree.lambda | (0.22528, 0.00069, 0.00069) | (0.22528, 0.00069, 0.00069) | Sec.12.4 text p.13 | yes |
| ckm.alt.tree.A | (0.802, 0.026, 0.026) | (0.802, 0.026, 0.026) | Sec.12.4 text p.13 | yes |
| ckm.alt.tree.rhobar | (0.166, 0.021, 0.02) | (0.166, 0.021, 0.02) | Sec.12.4 text p.13 | yes |
| ckm.alt.tree.etabar | (0.379, 0.022, 0.021) | (0.379, 0.022, 0.021) | Sec.12.4 text p.13 | yes |
| ckm.global.Vud | (0.97431, 0.00016, 0.00016) | (0.97431, 0.00016, 0.00016) | Eq.(12.27) p.12 | yes |
| ckm.global.Vus | (0.22517, 0.00068, 0.00068) | (0.22517, 0.00068, 0.00068) | Eq.(12.27) p.12 | yes |
| ckm.global.Vub | (0.003763, 8.8e-05, 8.3e-05) | (0.003763, 8.8e-05, 8.3e-05) | Eq.(12.27) p.12 | yes |
| ckm.global.Vcd | (0.22503, 0.00068, 0.00068) | (0.22503, 0.00068, 0.00068) | Eq.(12.27) p.12 | yes |
| ckm.global.Vcs | (0.97345, 0.00016, 0.00016) | (0.97345, 0.00016, 0.00016) | Eq.(12.27) p.12 | yes |
| ckm.global.Vcb | (0.04189, 0.00081, 0.00069) | (0.04189, 0.00081, 0.00069) | Eq.(12.27) p.12 | yes |
| ckm.global.Vtd | (0.00863, 0.00019, 0.00017) | (0.00863, 0.00019, 0.00017) | Eq.(12.27) p.12 | yes |
| ckm.global.Vts | (0.04117, 0.00079, 0.00068) | (0.04117, 0.00079, 0.00068) | Eq.(12.27) p.12 | yes |
| ckm.global.Vtb | (0.999115, 2.9e-05, 3.4e-05) | (0.999115, 2.9e-05, 3.4e-05) | Eq.(12.27) p.12 | yes |
| ckm.J | (3.16e-05, 1.3e-06, 1.1e-06) | (3.16e-05, 1.3e-06, 1.1e-06) | text before Eq.(12.28) p.13 | yes |
| ckm.s12 | (0.22517, 0.00068, 0.00068) | (0.22517, 0.00068, 0.00068) | Eq.(12.28) p.13 | yes |
| ckm.s13 | (0.003763, 8.8e-05, 8.3e-05) | (0.003763, 8.8e-05, 8.3e-05) | Eq.(12.28) p.13 | yes |
| ckm.s23 | (0.04189, 0.00081, 0.00069) | (0.04189, 0.00081, 0.00069) | Eq.(12.28) p.13 | yes |
| ckm.delta_rad | (1.154, 0.025, 0.025) | (1.154, 0.025, 0.025) | Eq.(12.28) p.13 | yes |
| ckm.direct.Vud | (0.97367, 0.00032, 0.00032) | (0.97367, 0.00032, 0.00032) | Eq.(12.7) | yes |
| ckm.direct.Vus | (0.22431, 0.00085, 0.00085) | (0.22431, 0.00085, 0.00085) | Eq.(12.8) | yes |
| ckm.direct.Vcd | (0.2247, 0.0022, 0.0022) | (0.2247, 0.0022, 0.0022) | Eq.(12.9) | yes |
| ckm.direct.Vcs | (0.969, 0.005, 0.005) | (0.969, 0.005, 0.005) | Eq.(12.10) | yes |
| ckm.direct.Vcb | (0.0407, 0.0013, 0.0013) | (0.0407, 0.0013, 0.0013) | Eq.(12.11) | yes |
| ckm.direct.Vub | (0.00389, 0.00016, 0.00016) | (0.00389, 0.00016, 0.00016) | Eq.(12.12) | yes |
| ckm.direct.Vtd | (0.0086, 0.0002, 0.0002) | (0.0086, 0.0002, 0.0002) | Eq.(12.13) | yes |
| ckm.direct.Vts | (0.0415, 0.0009, 0.0009) | (0.0415, 0.0009, 0.0009) | Eq.(12.13) | yes |
| ckm.direct.Vtb | (1.002, 0.024, 0.024) | (1.002, 0.024, 0.024) | Eq.(12.15) | yes |
| ckm.unitarity.row1 | (0.9984, 0.0007, 0.0007) | (0.9984, 0.0007, 0.0007) | Sec.12.4 text p.12 | yes |
| ckm.unitarity.row2 | (0.991, 0.01, 0.01) | (0.991, 0.01, 0.01) | Sec.12.4 text p.12 | yes |
| ckm.unitarity.col1 | (0.9986, 0.0011, 0.0011) | (0.9986, 0.0011, 0.0011) | Sec.12.4 text p.12 | yes |
| ckm.unitarity.col2 | (0.991, 0.01, 0.01) | (0.991, 0.01, 0.01) | Sec.12.4 text p.12 | yes |
| pmns.IC23_woSKatm.NO.s12sq | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.theta12_deg | (33.76, 0.42, 0.41, (32.54, 35.03)) | (33.76, 0.42, 0.41, (32.54, 35.03)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.s23sq | (0.47, 0.017, 0.014, (0.432, 0.587)) | (0.47, 0.017, 0.014, (0.432, 0.587)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.theta23_deg | (43.27, 1.0, 0.82, (41.11, 50.02)) | (43.27, 1.0, 0.82, (41.11, 50.02)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.s13sq | (0.02249, 0.00057, 0.00057, (0.0207, 0.0242)) | (0.02249, 0.00057, 0.00057, (0.0207, 0.0242)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.theta13_deg | (8.62, 0.11, 0.11, (8.27, 8.95)) | (8.62, 0.11, 0.11, (8.27, 8.95)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.dcp_deg | (207, 23, 20, (114, 405)) | (207, 23, 20, (114, 405)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.dm21sq_1e-5eV2 | (7.537, 0.094, 0.1, (7.236, 7.823)) | (7.537, 0.094, 0.1, (7.236, 7.823)) | tbl-parameters IC23 NO | yes |
| masses.IC23_woSKatm.NO.dm21sq | (7.537e-05, 9.4e-07, 1e-06, (7.236e-05, 7.823e-05)) | (7.537e-05, 9.400000000000001e-07, 1.0000000000000002e-06, (7.236e-05, 7.823000000000002e-05)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.NO.dm3l_sq_1e-3eV2 | (2.521, 0.026, 0.018, (2.454, 2.592)) | (2.521, 0.026, 0.018, (2.454, 2.592)) | tbl-parameters IC23 NO | yes |
| masses.IC23_woSKatm.NO.dm31sq | (0.002521, 2.6e-05, 1.8e-05, (0.002454, 0.002592)) | (0.002521, 2.6e-05, 1.8e-05, (0.002454, 0.002592)) | tbl-parameters IC23 NO | yes |
| pmns.IC23_woSKatm.IO.s12sq | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.theta12_deg | (33.76, 0.42, 0.41, (32.54, 35.03)) | (33.76, 0.42, 0.41, (32.54, 35.03)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.s23sq | (0.555, 0.013, 0.016, (0.437, 0.59)) | (0.555, 0.013, 0.016, (0.437, 0.59)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.theta23_deg | (48.15, 0.75, 0.92, (41.4, 50.21)) | (48.15, 0.75, 0.92, (41.4, 50.21)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.s13sq | (0.02261, 0.00056, 0.00056, (0.02091, 0.02433)) | (0.02261, 0.00056, 0.00056, (0.02091, 0.02433)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.theta13_deg | (8.65, 0.11, 0.11, (8.31, 8.97)) | (8.65, 0.11, 0.11, (8.31, 8.97)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.dcp_deg | (283, 24, 28, (202, 347)) | (283, 24, 28, (202, 347)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.dm21sq_1e-5eV2 | (7.537, 0.094, 0.1, (7.236, 7.822)) | (7.537, 0.094, 0.1, (7.236, 7.822)) | tbl-parameters IC23 IO | yes |
| masses.IC23_woSKatm.IO.dm21sq | (7.537e-05, 9.4e-07, 1e-06, (7.236e-05, 7.822e-05)) | (7.537e-05, 9.400000000000001e-07, 1.0000000000000002e-06, (7.236e-05, 7.822e-05)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.IO.dm3l_sq_1e-3eV2 | (-2.5, 0.024, 0.023, (-2.569, -2.43)) | (-2.5, 0.024, 0.023, (-2.569, -2.43)) | tbl-parameters IC23 IO | yes |
| masses.IC23_woSKatm.IO.dm32sq | (-0.0025, 2.4e-05, 2.3e-05, (-0.002569, -0.00243)) | (-0.0025, 2.4e-05, 2.3e-05, (-0.002569, -0.0024300000000000003)) | tbl-parameters IC23 IO | yes |
| pmns.IC23_woSKatm.delta_chi2_IO_minus_NO | 1.5 | 1.5 | tbl-parameters IC23_woSKatm header | yes |
| pmns.IC24_wSKatm.NO.s12sq | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.theta12_deg | (33.76, 0.42, 0.41, (32.54, 35.03)) | (33.76, 0.42, 0.41, (32.54, 35.03)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.s23sq | (0.47, 0.017, 0.014, (0.435, 0.584)) | (0.47, 0.017, 0.014, (0.435, 0.584)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.theta23_deg | (43.29, 0.96, 0.79, (41.27, 49.86)) | (43.29, 0.96, 0.79, (41.27, 49.86)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.s13sq | (0.02248, 0.00055, 0.00059, (0.02064, 0.02418)) | (0.02248, 0.00055, 0.00059, (0.02064, 0.02418)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.theta13_deg | (8.62, 0.11, 0.11, (8.26, 8.95)) | (8.62, 0.11, 0.11, (8.26, 8.95)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.dcp_deg | (212, 26, 36, (125, 365)) | (212, 26, 36, (125, 365)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.dm21sq_1e-5eV2 | (7.537, 0.094, 0.1, (7.236, 7.823)) | (7.537, 0.094, 0.1, (7.236, 7.823)) | tbl-parameters IC24 NO | yes |
| masses.IC24_wSKatm.NO.dm21sq | (7.537e-05, 9.4e-07, 1e-06, (7.236e-05, 7.823e-05)) | (7.537e-05, 9.400000000000001e-07, 1.0000000000000002e-06, (7.236e-05, 7.823000000000002e-05)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.NO.dm3l_sq_1e-3eV2 | (2.511, 0.021, 0.02, (2.45, 2.576)) | (2.511, 0.021, 0.02, (2.45, 2.576)) | tbl-parameters IC24 NO | yes |
| masses.IC24_wSKatm.NO.dm31sq | (0.002511, 2.1e-05, 2e-05, (0.00245, 0.002576)) | (0.002511, 2.1000000000000002e-05, 2e-05, (0.0024500000000000004, 0.002576)) | tbl-parameters IC24 NO | yes |
| pmns.IC24_wSKatm.IO.s12sq | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | (0.3088, 0.0067, 0.0066, (0.2893, 0.3295)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.theta12_deg | (33.76, 0.42, 0.41, (32.54, 35.03)) | (33.76, 0.42, 0.41, (32.54, 35.03)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.s23sq | (0.55, 0.013, 0.016, (0.439, 0.584)) | (0.55, 0.013, 0.016, (0.439, 0.584)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.theta23_deg | (47.9, 0.73, 0.92, (41.51, 49.83)) | (47.9, 0.73, 0.92, (41.51, 49.83)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.s13sq | (0.02262, 0.00057, 0.00056, (0.02093, 0.02441)) | (0.02262, 0.00057, 0.00056, (0.02093, 0.02441)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.theta13_deg | (8.65, 0.11, 0.11, (8.32, 8.99)) | (8.65, 0.11, 0.11, (8.32, 8.99)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.dcp_deg | (274, 22, 25, (203, 335)) | (274, 22, 25, (203, 335)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.dm21sq_1e-5eV2 | (7.537, 0.094, 0.1, (7.236, 7.822)) | (7.537, 0.094, 0.1, (7.236, 7.822)) | tbl-parameters IC24 IO | yes |
| masses.IC24_wSKatm.IO.dm21sq | (7.537e-05, 9.4e-07, 1e-06, (7.236e-05, 7.822e-05)) | (7.537e-05, 9.400000000000001e-07, 1.0000000000000002e-06, (7.236e-05, 7.822e-05)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.IO.dm3l_sq_1e-3eV2 | (-2.483, 0.02, 0.02, (-2.547, -2.421)) | (-2.483, 0.02, 0.02, (-2.547, -2.421)) | tbl-parameters IC24 IO | yes |
| masses.IC24_wSKatm.IO.dm32sq | (-0.002483, 2e-05, 2e-05, (-0.002547, -0.002421)) | (-0.002483, 2e-05, 2e-05, (-0.0025470000000000002, -0.002421)) | tbl-parameters IC24 IO | yes |
| pmns.IC24_wSKatm.delta_chi2_IO_minus_NO | 5.9 | 5.9 | tbl-parameters IC24_wSKatm header | yes |

### Masses (PDG 2026 API, `pdgdata` summary-table rows, edition 2026)

| field | data/ (value, +err, -err) GeV | source row | location | match |
|---|---|---|---|---|
| m_u | 0.00216, 7e-5, 7e-5 | 2.16 +- 0.07 MeV | Q123UM | yes |
| m_d | 0.0047, 7e-5, 7e-5 | 4.70 +- 0.07 MeV | Q123DM | yes |
| m_s | 0.0929, 7e-4, 7e-4 | 92.9 +- 0.7 MeV | Q123SM | yes |
| m_c | 1.2729, 0.0045, 0.0045 | 1.2729 +- 0.0045 GeV | Q004M | yes |
| m_b | 4.186, 0.006, 0.006 | 4.186 +- 0.006 GeV | Q005M | yes |
| m_t_direct | 172.6035596, 0.2713750, 0.2730478 | 172.6035595874743 +0.2713750 -0.2730478 GeV | Q007TP | yes |
| m_t_pole_xsec | 172.0542188, 0.6140864, 0.5896776 | same | Q007TP4 | yes |
| m_t_msbar_xsec | 162.5002847, 2.1316137, 1.5324334 | same | Q007TP2 | yes |
| m_e | 0.00051099895069, 1.6e-13 | 0.51099895069 +- 1.6e-10 MeV | S003M | yes |
| m_mu | 0.1056583755, 2.3e-9 | 105.6583755 +- 2.3e-6 MeV | S004M | yes |
| m_tau | 1.77693246513, 8.4915e-5, 8.7580e-5 | 1776.93246513 +0.0849150 -0.0875802 MeV | S035M | yes |
| m_u/m_d | 0.462 +- 0.020 | same (V row) | Q123MR0 | yes |
| m_c/m_s | 11.761 +- 0.047 | same (V row) | Q004MSR | yes |
| m_b/m_c | 4.58 +- 0.01 | same (V row) | Q004MBR | yes |
| m_b/m_s | 53.876838, 0.117242 | same (AC row, only row) | Q005A00 | yes |

Schemes and scales match the PDG notes: light quarks MSbar at mu = 2 GeV (Q123UM/DM/SM notes), m_c = mbar_c(mbar_c), m_b = mbar_b(mbar_b), Q007TP "usually interpreted as the pole mass", Q007TP2 "MSbar mass" with no scale stated, Q004MSR/Q004MBR "MSbar masses at a common scale". Q001M/Q002M/Q003M carry the same values as Q123*M.

### NuFIT chi^2 tables (`verify_nufit_csv.py`)
- 56 committed CSVs, 483,654 rows, all identical to the raw sections within 1e-6 relative. Column order matches the raw section header (e.g. `DMS-T23` is log10(dm21sq), s23sq).
- Random sample, one row per file, 16 shown (csv line: data/ row = raw row): TBoff IO T13 l.29 (0.0203, 18.00857); TBoff IO DMS-DCP l.5827 (-4.144, 110, 59.32356); TBoff IO T23 l.25 (0.365, 111.1306); TByes NO DCP l.73 (175, 1.02045); TBoff NO T12-DMS l.16211 (0.662, -4.35, 860.7709); TBoff IO DMS-DMA l.24363 (-3.98, -2.35, 481.2827); TByes NO T23-DCP l.7066 (0.73, 100, 533.3986); TBoff NO T13 l.31 (0.0205, 11.01316); TByes NO DMA l.67 (2.385, 34.90535); TByes NO DMS l.161 (-3.86, 577.1025); TBoff IO DCP l.2 (-180, 17.52683); TBoff IO T23-DMA l.15234 (0.71, -2.62, 294.1989); TBoff NO T12-DCP l.6249 (0.635, 30, 259.8056); TBoff NO T13-DCP l.5750 (0.027, 90, 71.0419); TBoff IO T13-T12 l.8097 (0.0233, 0.38, 87.0642); TByes NO T23 l.50 (0.49, 1.279303). All match.
- TBoff/TByes to analysis mapping confirmed independently: IO floors 1.4859 (TBoff) and 5.9082 (TByes) match the table headers 1.5 (IC23 w/o SK-atm) and 5.9 (IC24 with SK-atm); the argmin of each 1D table matches the tbl-parameters best fit to within one grid step.
- Not committed, as documented in FORMAT.md: the 3D T23/DMA/DCP projection (both analyses) and 14 TByes 2D projections per ordering.

### src/data.py logic (`verify_loader.py`, 71/71 pass)
- 1D interpolation reproduces every sampled grid point of all 6 profiles x 4 (analysis, ordering) exactly after subtracting the table minimum.
- DCP grid is -180..180 step 5 with equal endpoints, so `_wrap_dcp` (maps to [-180,180)) never falls off the grid; a 2D point at dcp+360 reproduces the grid value.
- T23-DCP 2D table: its minimum equals the 1D floor; profiling it over dcp reproduces the T23 1D table, and over s23sq the DCP table (max diff < 2e-3), so axes and orientation (x = s23sq, y = dcp) are read correctly.
- DMA: NO grid 0.2..7.0, IO grid -7.0..-0.2 (Delta m^2_32 < 0), consistent with the eV^2 x 1e3 conversion and the sign convention in the docstring. A wrong-sign dm3l for the ordering returns inf.
- At the published best fit of each (analysis, ordering), the full 6-parameter `pmns_delta_chi2` is < 0.1. At best fit +- 1 sigma (tbl-parameters) the 1D Delta chi^2 is 0.96 to 1.10.
- `ordering_offset`: IO 1.4859 / 5.9082, NO 0. `relative_to="global"` adds it exactly once per call.
- `stats.Objective` convention: with the default `pmns_delta_chi2` (relative_to="local"), neither mode includes the ordering penalty. If a caller passes `functools.partial(pmns_delta_chi2, relative_to="global")`, `pmns_mode="sum1d"` adds the offset once per PMNS parameter: at the IO best fit, sum1d gives 5.958 (= 4 x 1.486 + 0.014) while joint gives 1.500. This is the hazard noted in STATE.md, confirmed numerically. It is a usage issue, not a data error.

## Verdict
data/ matches the primary sources. No transcription errors in the CKM global fit (Eqs. 12.26-12.28, J), the direct |V_ij|, the unitarity sums, the NuFIT 6.1 parameter table (both analyses, both orderings, 1 sigma, 3 sigma, ordering Delta chi^2), the PDG masses, or the chi^2 tables. The loader reads the tables correctly. The "Data caveats" item about the hand-transcribed CKM numbers can be closed.

## Proposed corrections
None to `data/` or `src/data.py`.

Proposed guard in `src/stats.py` (overseer's call, not a data fix): in `Objective.__init__`, when `pmns_mode == "sum1d"`, reject a callable carrying `relative_to="global"` (e.g. `getattr(f, "keywords", {}).get("relative_to") == "global"`), and add the ordering penalty once through an explicit argument, e.g. `pmns_ordering_offset: float = 0.0` stored as `res.terms["ordering"]`. Without it, NO-vs-IO comparisons through `Objective` either omit the 1.49 (5.91) penalty (default) or count it 4 times (sum1d with global).

Documentation note: the CKM review PDF header says "Revised March 2026", matching SOURCES.md; the page footers read "1st June, 2026" (posting date). No change needed.

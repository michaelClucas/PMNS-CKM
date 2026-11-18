# Data sources

All files accessed 2026-11-18. Build scripts: `experiments/20261118-001-data-build/`.

## PDG2026_CKM: PDG Review of Particle Physics 2026, review 12 "CKM Quark-Mixing Matrix"

- Edition: F. Takahashi et al. (Particle Data Group), Int. J. Mod. Phys. A 41, 2630011 (2026). Review revised March 2026 by A. Ceccucci, Z. Ligeti, Y. Sakai.
- URL: https://pdg.lbl.gov/2026/reviews/rpp2026-rev-ckm-matrix.pdf
- Used for `data/ckm.json`, all fields:
  - `wolfenstein` (lambda, A, rhobar, etabar): Eq. (12.26), SM global fit, CKMfitter method.
  - `wolfenstein_alternatives`: UTfit-method and tree-level-only fits, quoted in the Sec. 12.4 text.
  - `magnitudes_global_fit` (|V_ij|): Eq. (12.27).
  - `jarlskog` (J) and `standard_parametrization` (s12, s13, s23, delta): Sec. 12.4 text and Eq. (12.28).
  - `magnitudes_direct`: Eqs. (12.7) to (12.13) and (12.15).
  - `unitarity_checks_direct`: Sec. 12.4 text.
- Transcribed by hand from `pdftotext` output of the PDF; signs of asymmetric errors checked against both layout and raw text extraction.
- Not available: correlation matrix of the global fit (PDG does not publish it).

## PDG2026 API: PDG 2026 machine-readable database

- File: `pdg-2026.0.sqlite`, schema 0.3, data release 2026-05-30, edition 2026, licence CC BY 4.0.
- URL: https://pdg.lbl.gov/2026/api/pdg-2026.0.sqlite (index: https://pdg.lbl.gov/2026/api/index.html)
- Used for `data/masses.json` fields `quarks`, `charged_leptons`, `quark_mass_ratios`, read by `build_masses.py` from summary-table rows (`pdgdata.in_summary_table = 1`):
  - m_u Q123UM, m_d Q123DM, m_s Q123SM (MSbar at mu = 2 GeV per the PDG listing note)
  - m_c Q004M (MSbar mbar_c(mbar_c)), m_b Q005M (MSbar mbar_b(mbar_b))
  - m_t_direct Q007TP, m_t_pole_xsec Q007TP4, m_t_msbar_xsec Q007TP2
  - m_e S003M, m_mu S004M, m_tau S035M
  - ratios Q123MR0 (m_u/m_d), Q004MSR (m_c/m_s), Q004MBR (m_b/m_c), Q005A00 (m_b/m_s)
- Scheme and scale strings come from the PDG listing notes (`pdgtext` table) for those identifiers.

## NuFIT6.1: NuFIT 6.1 (2025), three-neutrino fit to data available November 2025

- Citation requested by NuFIT: I. Esteban et al., JHEP 12 (2024) 216 [arXiv:2410.05380], and NuFIT 6.1 (2025), www.nu-fit.org.
- Release page: http://www.nu-fit.org/?q=node/309
- Files:
  - http://www.nu-fit.org/sites/default/files/v61.tbl-parameters.pdf: best fit, 1 sigma, 3 sigma ranges for both analyses and both orderings, and the IO-vs-NO Delta chi^2. Used for `data/pmns.json` (transcribed in `make_pmns_json.py`) and the `neutrino_splittings` field of `data/masses.json`.
  - http://www.nu-fit.org/sites/default/files/v61.release-TBoff-NO.txt.xz, `-TBoff-IO`, `-TByes-NO`, `-TByes-IO`: Delta chi^2 tables (1D, 2D, 3D projections). Split into `data/nufit/*.csv` by `build_nufit.py`. TBoff = "IC23 w/o SK-atm", TByes = "IC24 with SK-atm".
  - http://www.nu-fit.org/sites/default/files/v61.release-notes.pdf: data sets included and the table format description.
- Not used: v61.tbl-mixing.pdf (3 sigma |U| ranges), not needed by the loader and its layout did not extract cleanly.

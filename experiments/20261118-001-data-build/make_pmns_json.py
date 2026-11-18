"""Write data/pmns.json from values transcribed from NuFIT 6.1 v61.tbl-parameters.pdf.

Transcribed from pdftotext output of http://www.nu-fit.org/sites/default/files/v61.tbl-parameters.pdf
(accessed 2026-11-18). Cross-checked against the 1D chi^2 tables by tests/test_data.py and
by build_nufit.py output. Run from repo root.
"""
import json
# (value, err_minus, err_plus, lo3, hi3)
T = {
 "IC23_woSKatm": {
  "NO": {"s12sq": (0.3088,0.0066,0.0067,0.2893,0.3295), "theta12_deg": (33.76,0.41,0.42,32.54,35.03),
         "s23sq": (0.470,0.014,0.017,0.432,0.587), "theta23_deg": (43.27,0.82,1.0,41.11,50.02),
         "s13sq": (0.02249,0.00057,0.00057,0.02070,0.02420), "theta13_deg": (8.62,0.11,0.11,8.27,8.95),
         "dcp_deg": (207,20,23,114,405), "dm21sq_1e-5eV2": (7.537,0.10,0.094,7.236,7.823),
         "dm3l_sq_1e-3eV2": (2.521,0.018,0.026,2.454,2.592)},
  "IO": {"s12sq": (0.3088,0.0066,0.0067,0.2893,0.3295), "theta12_deg": (33.76,0.41,0.42,32.54,35.03),
         "s23sq": (0.555,0.016,0.013,0.437,0.590), "theta23_deg": (48.15,0.92,0.75,41.40,50.21),
         "s13sq": (0.02261,0.00056,0.00056,0.02091,0.02433), "theta13_deg": (8.65,0.11,0.11,8.31,8.97),
         "dcp_deg": (283,28,24,202,347), "dm21sq_1e-5eV2": (7.537,0.10,0.094,7.236,7.822),
         "dm3l_sq_1e-3eV2": (-2.500,0.023,0.024,-2.569,-2.430)},
  "delta_chi2_IO_minus_NO": 1.5},
 "IC24_wSKatm": {
  "NO": {"s12sq": (0.3088,0.0066,0.0067,0.2893,0.3295), "theta12_deg": (33.76,0.41,0.42,32.54,35.03),
         "s23sq": (0.470,0.014,0.017,0.435,0.584), "theta23_deg": (43.29,0.79,0.96,41.27,49.86),
         "s13sq": (0.02248,0.00059,0.00055,0.02064,0.02418), "theta13_deg": (8.62,0.11,0.11,8.26,8.95),
         "dcp_deg": (212,36,26,125,365), "dm21sq_1e-5eV2": (7.537,0.10,0.094,7.236,7.823),
         "dm3l_sq_1e-3eV2": (2.511,0.020,0.021,2.450,2.576)},
  "IO": {"s12sq": (0.3088,0.0066,0.0067,0.2893,0.3295), "theta12_deg": (33.76,0.41,0.42,32.54,35.03),
         "s23sq": (0.550,0.016,0.013,0.439,0.584), "theta23_deg": (47.90,0.92,0.73,41.51,49.83),
         "s13sq": (0.02262,0.00056,0.00057,0.02093,0.02441), "theta13_deg": (8.65,0.11,0.11,8.32,8.99),
         "dcp_deg": (274,25,22,203,335), "dm21sq_1e-5eV2": (7.537,0.10,0.094,7.236,7.822),
         "dm3l_sq_1e-3eV2": (-2.483,0.020,0.020,-2.547,-2.421)},
  "delta_chi2_IO_minus_NO": 5.9},
}
SCHEME = {
 "s12sq": "sin^2(theta12), PDG/NuFIT standard parametrization", "s23sq": "sin^2(theta23)", "s13sq": "sin^2(theta13)",
 "theta12_deg": "theta12 in degrees", "theta23_deg": "theta23 in degrees", "theta13_deg": "theta13 in degrees",
 "dcp_deg": "Dirac CP phase delta_CP in degrees, NuFIT convention [0,360); 3sigma range may exceed 360 when it wraps",
 "dm21sq_1e-5eV2": "Delta m^2_21 in units of 1e-5 eV^2",
 "dm3l_sq_1e-3eV2": "Delta m^2_3l in units of 1e-3 eV^2: Delta m^2_31 > 0 for NO, Delta m^2_32 < 0 for IO",
}
out = {
 "meta": {
  "description": "Three-flavour neutrino oscillation parameters from NuFIT 6.1 (2025), fit to data available November 2025, table v61.tbl-parameters.pdf.",
  "source_key": "NuFIT6.1",
  "default_analysis": "IC23_woSKatm",
  "default_analysis_reason": "NuFIT does not name a recommended analysis. IC23 w/o SK-atm is the upper section of the NuFIT parameter table and the analysis used for the filled regions and the mass-scale plots on the release page; IC24 with SK-atm additionally includes the tabulated atmospheric chi^2 data provided by the Super-Kamiokande and IceCube collaborations. Both are stored.",
  "ranges": "value = best fit; err_plus/err_minus = 1 sigma (Delta chi^2 = 1) relative to the local minimum of that ordering; range_3sigma = Delta chi^2 = 9 range (1 dof).",
  "delta_chi2_IO_minus_NO_note": "Delta chi^2 of the IO local minimum relative to the global (NO) minimum, as printed in the table header. The chi^2 data tables give 1.486 (IC23 w/o SK-atm) and 5.908 (IC24 with SK-atm).",
  "scale": "low energy (oscillation experiments, neutrino energies MeV-GeV)",
 },
 "analyses": {},
}
for ana, d in T.items():
    out["analyses"][ana] = {"delta_chi2_IO_minus_NO": d["delta_chi2_IO_minus_NO"]}
    for o in ("NO", "IO"):
        out["analyses"][ana][o] = {k: {"value": v, "err_plus": ep, "err_minus": em, "range_3sigma": [lo, hi],
                                       "scale": "low energy", "scheme": SCHEME[k],
                                       "source": f"NuFIT6.1 v61.tbl-parameters.pdf, {ana}, {o}"}
                                   for k, (v, em, ep, lo, hi) in d[o].items()}
json.dump(out, open("data/pmns.json", "w"), indent=1)

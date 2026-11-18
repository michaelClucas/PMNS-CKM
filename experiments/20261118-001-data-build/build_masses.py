"""Build data/masses.json from the PDG 2026 API SQLite file plus NuFIT 6.1 splittings.

Usage (from repo root):
    .venv/Scripts/python experiments/20261118-001-data-build/build_masses.py

Downloads https://pdg.lbl.gov/2026/api/pdg-2026.0.sqlite into artifacts/pdg/ (gitignored)
and reads the summary-table values (pdgdata.in_summary_table = 1). Neutrino splittings are
copied from data/pmns.json (NuFIT 6.1), so run after pmns.json exists.
"""
import json
import pathlib
import sqlite3
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_URL = "https://pdg.lbl.gov/2026/api/pdg-2026.0.sqlite"
DB = ROOT / "artifacts" / "pdg" / "pdg-2026.0.sqlite"

# key: (PDG identifier, unit conversion to GeV, scale, scheme)
ENTRIES = {
    "m_u": ("Q123UM", 1e-3, "mu = 2 GeV", "MSbar (PDG normalizes light-quark MSbar masses at mu = 2 GeV)"),
    "m_d": ("Q123DM", 1e-3, "mu = 2 GeV", "MSbar (PDG normalizes light-quark MSbar masses at mu = 2 GeV)"),
    "m_s": ("Q123SM", 1e-3, "mu = 2 GeV", "MSbar (PDG normalizes light-quark MSbar masses at mu = 2 GeV)"),
    "m_c": ("Q004M", 1.0, "mu = mbar_c", "MSbar running mass mbar_c(mbar_c)"),
    "m_b": ("Q005M", 1.0, "mu = mbar_b", "MSbar running mass mbar_b(mbar_b)"),
    "m_t_direct": ("Q007TP", 1.0, "pole-like", "top mass from kinematic reconstruction (MC mass, usually interpreted as pole mass; PDG says the interpretation uncertainty is hard to quantify)"),
    "m_t_pole_xsec": ("Q007TP4", 1.0, "pole", "top pole mass from t-tbar cross-section measurements"),
    "m_t_msbar_xsec": ("Q007TP2", 1.0, "MSbar, scale not stated in PDG text (conventionally mu = mbar_t)", "top MSbar mass from t-tbar cross-section measurements"),
    "m_e": ("S003M", 1e-3, "low energy", "pole (physical) mass"),
    "m_mu": ("S004M", 1e-3, "low energy", "pole (physical) mass"),
    "m_tau": ("S035M", 1e-3, "low energy", "pole (physical) mass"),
}

RATIOS = {
    "m_u_over_m_d": "Q123MR0",
    "m_c_over_m_s": "Q004MSR",
    "m_b_over_m_c": "Q004MBR",
    "m_b_over_m_s": "Q005A00",
}


def r(x):
    """Strip float noise from unit conversion."""
    return float(f"{x:.12g}")


def main():
    if not DB.exists():
        DB.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", DB_URL)
        urllib.request.urlretrieve(DB_URL, DB)
    con = sqlite3.connect(DB)
    info = dict(con.execute("select name, value from pdginfo").fetchall())

    def get(pdgid):
        rows = con.execute(
            "select d.value, d.error_positive, d.error_negative, d.unit_text, d.display_value_text, i.description "
            "from pdgdata d join pdgid i on d.pdgid_id = i.id where d.pdgid = ? and d.in_summary_table = 1",
            (pdgid,)).fetchall()
        assert len(rows) == 1, (pdgid, rows)
        return rows[0]

    out = {"meta": {
        "description": "Fermion masses. Quark and charged-lepton masses from the PDG 2026 summary tables (PDG API SQLite pdg-2026.0); neutrino mass-squared splittings from NuFIT 6.1.",
        "pdg_edition": info["edition"], "pdg_data_release": info["data_release_timestamp"],
        "pdg_citation": info["citation"].replace("<i>", "").replace("</i>", ""),
        "units": "GeV for masses; eV^2 for splittings",
        "note": "Quark masses are in different schemes and at different scales; run them to a common scale before comparing (not done here).",
    }, "quarks": {}, "charged_leptons": {}, "quark_mass_ratios": {}, "neutrino_splittings": {}}

    for key, (pid, conv, scale, scheme) in ENTRIES.items():
        v, ep, em, unit, disp, desc = get(pid)
        group = "charged_leptons" if pid.startswith("S") else "quarks"
        out[group][key] = {"value": r(v * conv), "err_plus": r(ep * conv), "err_minus": r(em * conv), "unit": "GeV",
                           "scale": scale, "scheme": scheme,
                           "pdg_display": f"{disp} {unit}", "source": f"PDG2026 API {pid} ({desc})"}
    for key, pid in RATIOS.items():
        v, ep, em, unit, disp, desc = get(pid)
        out["quark_mass_ratios"][key] = {"value": v, "err_plus": ep, "err_minus": em, "unit": "",
                                         "scale": "scale independent (ratio of masses in the same scheme at the same scale)",
                                         "scheme": "MSbar ratio", "pdg_display": disp,
                                         "source": f"PDG2026 API {pid} ({desc})"}

    pmns = json.load(open(ROOT / "data" / "pmns.json"))
    for ana, d in pmns["analyses"].items():
        for o in ("NO", "IO"):
            r21 = d[o]["dm21sq_1e-5eV2"]
            r3l = d[o]["dm3l_sq_1e-3eV2"]
            name3 = "dm31sq" if o == "NO" else "dm32sq"
            out["neutrino_splittings"].setdefault(ana, {})[o] = {
                "dm21sq": {"value": r(r21["value"] * 1e-5), "err_plus": r(r21["err_plus"] * 1e-5), "err_minus": r(r21["err_minus"] * 1e-5),
                           "range_3sigma": [r(x * 1e-5) for x in r21["range_3sigma"]], "unit": "eV^2", "scale": "low energy",
                           "scheme": "Delta m^2_21 = m2^2 - m1^2", "source": r21["source"]},
                name3: {"value": r(r3l["value"] * 1e-3), "err_plus": r(r3l["err_plus"] * 1e-3), "err_minus": r(r3l["err_minus"] * 1e-3),
                        "range_3sigma": [r(x * 1e-3) for x in r3l["range_3sigma"]], "unit": "eV^2", "scale": "low energy",
                        "scheme": "Delta m^2_31 = m3^2 - m1^2 (NO)" if o == "NO" else "Delta m^2_32 = m3^2 - m2^2 (IO)",
                        "source": r3l["source"]},
            }
    out["neutrino_splittings"]["default_analysis"] = pmns["meta"]["default_analysis"]
    json.dump(out, open(ROOT / "data" / "masses.json", "w"), indent=1)
    print(json.dumps(out["quarks"], indent=1)[:1500])


if __name__ == "__main__":
    main()

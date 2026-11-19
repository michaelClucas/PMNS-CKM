"""Build survey.csv from survey_raw.csv (literature-agent output) plus corrections checked against
the LaTeX sources and our-counting columns. Counting (CLAUDE.md / STATE.md rules):
  N_dimless   observables scored in dimensionless form: PMNS s12, s13, s23, delta_l, Dm21/Dm31,
              me/mmu, mmu/mtau, CKM 4, mu/mc, mc/mt, md/ms, ms/mb (15 for a full SM-flavour fit);
              GUT models add mb/mtau-type cross-sector ratios only when a coupling is shared.
  k_cont      claimed real parameters minus the overall dimensionful scales (each scale is fixed by
              one absolute mass that we do not score).
  bits_est    ceil(log2 #alternatives the paper or its series scanned) + 3 (choice of finite modular
              group among ~8 tried in the literature) + 1 (gCP or not) [+1 holomorphic vs not].
              Estimates; marked est.
"""
import csv
import math
import pathlib

HERE = pathlib.Path(__file__).parent
FIX = {
    "2506.19822": {"group": "Gamma3'=T' non-holomorphic (polyharmonic Maass forms, weights 0-4 incl. weight 1)",
                   "seesaw": "type I, two RH neutrinos (m1 = 0)",
                   "N_fitted": "19 measured (9 lepton incl. delta_l, 10 quark incl. mt, mb); me fixes lepton scale",
                   "quark_error_budget": "GUT values of Ross-Serna type: md/ms +-53%, mu/mc +-42%, ms/mb +-23%, th23q +-3%, th13q +-9%",
                   "fit_result": "chi2_tot = 22.65 (gCP, 13 params; chi2_l 8.86 + chi2_q 13.79; p ~ 9e-4 for 6 dof) / 13.33 (no gCP, 14 params, p ~ 0.02); NO only"},
    "2409.15823": {"gCP": "yes", "seesaw": "type I, three RH neutrinos (M_nu fixed by tau and one scale)",
                   "fit_result": "chi2_comb = 20.3 for 18 obs / 14 params (p ~ 4e-4, 4 dof); s12 +3.2 sigma, ms/mb +2.7 sigma; lepton-only 4.74, quark-only 0.0025 (reproduced here: 20.17 / 4.67 / 0)"},
    "2307.14926": {"gCP": "yes"},
}
# (N_dimless, k_cont, alternatives_for_bits, extra_bits) for unified rows with usable counts
OURS = {
    "2506.19822": (15, 9, 100, 5, "alternatives est. ~100 lepton x quark combos; +1 bit non-holomorphic"),
    "2409.15823": (15, 10, 2000, 4, "same model as 2307.14926: 'thousands' of 2O assignments classified"),
    "2307.14926": (15, 10, 2000, 4, "'hundreds' lepton, 'thousands' overall"),
    "2006.10722": (15, 11, 72 * 30, 4, "72 lepton models x est. 30 quark models"),
    "2012.13390": (15, 12, 380 * 4000, 4, "380 viable lepton x 4000 quark models"),
    "2311.10136": (15, 12, 6300, 4, "~6300 neutrino models + quark scans"),
    "1912.07573": (15, 12, 50, 4, "5 quark textures x est. 10 lepton models; fit only within 5 sigma"),
    "2101.02266": (16, 14, 20, 4, "scales est. 3; alternatives unknown, est. 20"),
    "2011.03501": (15, 15, 720 * 892, 4, "720 lepton x 892 quark"),
    "2404.06520": (16, 16, 100, 4, "PS shares d/e scale -> mb/mtau scored; alternatives est. 100"),
    "2207.10677": (16, 16, 10, 4, "~20 params minus est. 4 scales; Kahler params trimmed by hand"),
    "2101.12724": (16, 15, 53 * 7, 4, "18-param model; good fits need 24"),
    "2108.09655": (17, 23, 36, 4, "27 params (est. 4 scales)"),
}


def main():
    rows = list(csv.DictReader(open(HERE / "survey_raw.csv", encoding="utf-8")))
    cols = list(rows[0].keys()) + ["N_dimless_ours", "k_cont_ours", "bits_est_ours", "N_minus_k_cont",
                                    "N_minus_k_total_est", "counting_note"]
    for r in rows:
        r.update(FIX.get(r["arxiv_id"], {}))
        o = OURS.get(r["arxiv_id"])
        if o:
            N, k, alt, extra, note = o
            bits = math.ceil(math.log2(alt)) + extra
            r.update({"N_dimless_ours": N, "k_cont_ours": k, "bits_est_ours": f"{bits} est.",
                      "N_minus_k_cont": N - k, "N_minus_k_total_est": N - k - bits, "counting_note": note})
        else:
            r.update({c: "" for c in cols[-6:]})
            r["counting_note"] = "not ranked (counts unknown, k >= N, or not a unified quark+lepton fit)"
    rows.sort(key=lambda r: -(r["N_minus_k_cont"] if r["N_minus_k_cont"] != "" else -99))
    with open(HERE / "survey.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows[:12]:
        print(r["arxiv_id"], r["group"][:30], r["N_dimless_ours"], r["k_cont_ours"], r["bits_est_ours"],
              r["N_minus_k_cont"], r["N_minus_k_total_est"])


if __name__ == "__main__":
    main()

"""Filters (b)/(c) from published structural predictions: score each published number with the
official scorer (single key, NuFIT 6.1 1D profiles for PMNS angles; R_clebsch with the sqrt(2)*2%
floor). These are the papers' own predicted values or parameter-independent bounds (models.md).

    python pub_filters.py -> pub_filters.json, stdout (log)
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import evaluate as ev  # noqa: E402

HERE = Path(__file__).resolve().parent
V, _, _ = ev.mass_ratio_targets()
FR = {"R_clebsch": 0.02 * math.sqrt(2)}

ITEMS = [
    # (model, what, key, value, source)
    ("DR", "sin th13 ~ 0.05", "s13sq", 0.05 ** 2, "hep-ph/0507045 abstract"),
    ("DMM", "U_e3 ~ th_C/(3 sqrt2) ~ 0.05", "s13sq", 0.05 ** 2, "0911.2242 abstract"),
    ("CM-T' (2007)", "|U_e3| = 0.055", "s13sq", 0.055 ** 2, "0705.0714 numerical result"),
    ("YANG", "sin th13 = 0.0576", "s13sq", 0.0576 ** 2, "1011.4573"),
    ("KN23 case i", "|U13|^2 <= m_e/m_mu (upper bound)", "s13sq", V["mr_e_mu"], "hep-ph/0403008 Eq. (45)"),
    ("KN23 case ii", "|U13|^2 = 0.00014", "s13sq", 0.00014, "hep-ph/0403008 Eq. (36)"),
    ("KN23 nonsym", "|U13|^2 = 1.3e-10", "s13sq", 1.3e-10, "hep-ph/0506192 abstract"),
    ("MP", "TBM, th13 = 0 (use 1e-6)", "s13sq", 1e-6, "hep-ph/0505113"),
    ("AB", "sin^2 th13 <= 1e-2 (paper's general scan, upper end)", "s13sq", 1e-2, "hep-ph/0502161 abstract"),
    ("AB (restricted)", "0.002 <= sin^2 th13 <= 0.003 (upper end)", "s13sq", 0.003, "hep-ph/0502161 abstract"),
    ("BPW (1998)", "theta_e-mu ~ 0.06 (solar, SMA)", "s12sq", math.sin(0.06) ** 2, "hep-ph/9812538 Eq. (35)"),
    ("JLM", "sin^2 2th13 = 0.074", "s13sq", (1 - math.sqrt(1 - 0.074)) / 2, "hep-ph/0510353"),
    ("AtoZ", "th23 = 45 +- 0.5 deg (lower end 44.4)", "s23sq", math.sin(math.radians(44.4)) ** 2, "1406.7005 Eq. (120)"),
    ("R = 1/9 (GJ 1-2 block)", "CM-T', AKS, MP, JLM-LO, UTZ-LO", "R_clebsch", 1 / 9, "structural (models.md)"),
    ("CM-T' fit", "m_e/m_mu 0.0061, m_d/m_s 0.050", "R_clebsch", 0.0061 / 0.050, "0705.0714 Eq. (41)-(42)"),
    ("AKS fit", "y_e/y_mu 4.74e-3, y_d/y_s 11.7/295", "R_clebsch", (2.79 / 589) / (11.7 / 295), "1301.6764 Table 5"),
    ("AtoZ", "y_mu = 4.1 y_s, y_e = y_d/2.6", "R_clebsch", 1 / (2.6 * 4.1), "1406.7005 Eq. (116)"),
]

if __name__ == "__main__":
    rows = []
    for model, what, key, val, src in ITEMS:
        out = {}
        for o in ("NO", "IO"):
            kw = {"mass_floor": FR} if key == "R_clebsch" else {}
            r = ev.evaluate({key: val}, o, use=[key], **kw)
            out[o] = {"chi2": r.chi2, "pull": r.pulls[key]}
            if key == "R_clebsch":
                break
        rows.append({"model": model, "what": what, "key": key, "value": val, "source": src, **out})
        print(f"{model:24s} {key:9s} {val:.3g}  " + "  ".join(f"{o}: chi2 {d['chi2']:.1f} pull {d['pull']:+.1f}"
                                                         for o, d in out.items()))
    (HERE / "pub_filters.json").write_text(json.dumps(rows, indent=1, default=float))

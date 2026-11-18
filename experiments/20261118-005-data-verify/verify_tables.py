"""Compare data/ckm.json, data/pmns.json, data/masses.json (NuFIT part) against values
read by the verifier from the rendered source pages (PDG 2026 CKM review pp. 11-13,
NuFIT 6.1 v61.tbl-parameters.pdf). Source values below were typed independently from
page images, not copied from data/.

Run from repo root: .venv/Scripts/python experiments/20261118-005-data-verify/verify_tables.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "data"
ckm = json.loads((D / "ckm.json").read_text())
pmns = json.loads((D / "pmns.json").read_text())
masses = json.loads((D / "masses.json").read_text())

rows = []


def chk(field, rec, val, ep, em, loc, rng=None):
    got = (rec["value"], rec.get("err_plus"), rec.get("err_minus"))
    ok = all(abs(a - b) <= 1e-12 * max(1, abs(b)) for a, b in zip(got, (val, ep, em)))
    if rng is not None:
        r = rec.get("range_3sigma")
        ok = ok and r is not None and all(abs(a - b) < 1e-12 for a, b in zip(r, rng))
        got = got + (tuple(r) if r else None,)
    rows.append((field, got, (val, ep, em) + ((tuple(rng),) if rng else ()), loc, ok))


# ---- PDG 2026 CKM review (rpp2026-rev-ckm-matrix.pdf)
W = ckm["wolfenstein"]
chk("ckm.wolfenstein.lambda", W["lambda"], 0.22517, 0.00068, 0.00068, "Eq.(12.26) p.12")
chk("ckm.wolfenstein.A", W["A"], 0.826, 0.017, 0.015, "Eq.(12.26) p.12")
chk("ckm.wolfenstein.rhobar", W["rhobar"], 0.1576, 0.0092, 0.0091, "Eq.(12.26) p.12")
chk("ckm.wolfenstein.etabar", W["etabar"], 0.3556, 0.0071, 0.0069, "Eq.(12.26) p.12")
U = ckm["wolfenstein_alternatives"]["UTfit_method"]
for k, v in {"lambda": (0.22519, .00068), "A": (0.820, .011), "rhobar": (0.1561, .0088), "etabar": (0.3587, .0069)}.items():
    chk(f"ckm.alt.UTfit.{k}", U[k], v[0], v[1], v[1], "Sec.12.4 text p.12")
T = ckm["wolfenstein_alternatives"]["tree_level_only"]
for k, v in {"lambda": (0.22528, .00069, .00069), "A": (0.802, .026, .026),
             "rhobar": (0.166, .021, .020), "etabar": (0.379, .022, .021)}.items():
    chk(f"ckm.alt.tree.{k}", T[k], *v, "Sec.12.4 text p.13")
M = ckm["magnitudes_global_fit"]
src27 = {"Vud": (0.97431, .00016, .00016), "Vus": (0.22517, .00068, .00068),
         "Vub": (0.003763, .000088, .000083), "Vcd": (0.22503, .00068, .00068),
         "Vcs": (0.97345, .00016, .00016), "Vcb": (0.04189, .00081, .00069),
         "Vtd": (0.00863, .00019, .00017), "Vts": (0.04117, .00079, .00068),
         "Vtb": (0.999115, .000029, .000034)}
for k, v in src27.items():
    chk(f"ckm.global.{k}", M[k], *v, "Eq.(12.27) p.12")
chk("ckm.J", ckm["jarlskog"]["J"], 3.16e-5, 0.13e-5, 0.11e-5, "text before Eq.(12.28) p.13")
S = ckm["standard_parametrization"]
chk("ckm.s12", S["s12"], 0.22517, .00068, .00068, "Eq.(12.28) p.13")
chk("ckm.s13", S["s13"], 0.003763, .000088, .000083, "Eq.(12.28) p.13")
chk("ckm.s23", S["s23"], 0.04189, .00081, .00069, "Eq.(12.28) p.13")
chk("ckm.delta_rad", S["delta_rad"], 1.154, .025, .025, "Eq.(12.28) p.13")
Dd = ckm["magnitudes_direct"]
for k, v, eq in [("Vud", (0.97367, .00032), "12.7"), ("Vus", (0.22431, .00085), "12.8"),
                 ("Vcd", (0.2247, .0022), "12.9"), ("Vcs", (0.969, .005), "12.10"),
                 ("Vcb", (40.7e-3, 1.3e-3), "12.11"), ("Vub", (3.89e-3, .16e-3), "12.12"),
                 ("Vtd", (8.6e-3, .2e-3), "12.13"), ("Vts", (41.5e-3, .9e-3), "12.13"),
                 ("Vtb", (1.002, .024), "12.15")]:
    chk(f"ckm.direct.{k}", Dd[k], v[0], v[1], v[1], f"Eq.({eq})")
X = ckm["unitarity_checks_direct"]
for k, v in {"row1": (0.9984, .0007), "row2": (0.991, .010), "col1": (0.9986, .0011), "col2": (0.991, .010)}.items():
    chk(f"ckm.unitarity.{k}", X[k], v[0], v[1], v[1], "Sec.12.4 text p.12")

# ---- NuFIT 6.1 v61.tbl-parameters.pdf
# (value, err_plus, err_minus, lo3, hi3)
NF = {
    ("IC23_woSKatm", "NO"): {
        "s12sq": (0.3088, .0067, .0066, .2893, .3295), "theta12_deg": (33.76, .42, .41, 32.54, 35.03),
        "s23sq": (0.470, .017, .014, .432, .587), "theta23_deg": (43.27, 1.0, .82, 41.11, 50.02),
        "s13sq": (0.02249, .00057, .00057, .02070, .02420), "theta13_deg": (8.62, .11, .11, 8.27, 8.95),
        "dcp_deg": (207, 23, 20, 114, 405), "dm21sq_1e-5eV2": (7.537, .094, .10, 7.236, 7.823),
        "dm3l_sq_1e-3eV2": (2.521, .026, .018, 2.454, 2.592)},
    ("IC23_woSKatm", "IO"): {
        "s12sq": (0.3088, .0067, .0066, .2893, .3295), "theta12_deg": (33.76, .42, .41, 32.54, 35.03),
        "s23sq": (0.555, .013, .016, .437, .590), "theta23_deg": (48.15, .75, .92, 41.40, 50.21),
        "s13sq": (0.02261, .00056, .00056, .02091, .02433), "theta13_deg": (8.65, .11, .11, 8.31, 8.97),
        "dcp_deg": (283, 24, 28, 202, 347), "dm21sq_1e-5eV2": (7.537, .094, .10, 7.236, 7.822),
        "dm3l_sq_1e-3eV2": (-2.500, .024, .023, -2.569, -2.430)},
    ("IC24_wSKatm", "NO"): {
        "s12sq": (0.3088, .0067, .0066, .2893, .3295), "theta12_deg": (33.76, .42, .41, 32.54, 35.03),
        "s23sq": (0.470, .017, .014, .435, .584), "theta23_deg": (43.29, .96, .79, 41.27, 49.86),
        "s13sq": (0.02248, .00055, .00059, .02064, .02418), "theta13_deg": (8.62, .11, .11, 8.26, 8.95),
        "dcp_deg": (212, 26, 36, 125, 365), "dm21sq_1e-5eV2": (7.537, .094, .10, 7.236, 7.823),
        "dm3l_sq_1e-3eV2": (2.511, .021, .020, 2.450, 2.576)},
    ("IC24_wSKatm", "IO"): {
        "s12sq": (0.3088, .0067, .0066, .2893, .3295), "theta12_deg": (33.76, .42, .41, 32.54, 35.03),
        "s23sq": (0.550, .013, .016, .439, .584), "theta23_deg": (47.90, .73, .92, 41.51, 49.83),
        "s13sq": (0.02262, .00057, .00056, .02093, .02441), "theta13_deg": (8.65, .11, .11, 8.32, 8.99),
        "dcp_deg": (274, 22, 25, 203, 335), "dm21sq_1e-5eV2": (7.537, .094, .10, 7.236, 7.822),
        "dm3l_sq_1e-3eV2": (-2.483, .020, .020, -2.547, -2.421)},
}
DCHI = {"IC23_woSKatm": 1.5, "IC24_wSKatm": 5.9}
for (an, o), recs in NF.items():
    loc = f"tbl-parameters {an.split('_')[0]} {o}"
    for k, (v, ep, em, lo, hi) in recs.items():
        chk(f"pmns.{an}.{o}.{k}", pmns["analyses"][an][o][k], v, ep, em, loc, rng=(lo, hi))
        if k == "dm21sq_1e-5eV2":
            chk(f"masses.{an}.{o}.dm21sq", masses["neutrino_splittings"][an][o]["dm21sq"],
                v * 1e-5, ep * 1e-5, em * 1e-5, loc, rng=(lo * 1e-5, hi * 1e-5))
        if k == "dm3l_sq_1e-3eV2":
            key = "dm31sq" if o == "NO" else "dm32sq"
            chk(f"masses.{an}.{o}.{key}", masses["neutrino_splittings"][an][o][key],
                v * 1e-3, ep * 1e-3, em * 1e-3, loc, rng=(lo * 1e-3, hi * 1e-3))
    if o == "IO":
        got = pmns["analyses"][an]["delta_chi2_IO_minus_NO"]
        rows.append((f"pmns.{an}.delta_chi2_IO_minus_NO", got, DCHI[an], f"tbl-parameters {an} header",
                     abs(got - DCHI[an]) < 1e-12))


def fmt(x):
    return str(x).replace("|", "/")


bad = [r for r in rows if not r[4]]
print(f"checked {len(rows)} records, {len(bad)} mismatches")
print("| field | data/ (value, +err, -err[, 3sigma]) | source | location | match |")
print("|---|---|---|---|---|")
for f, g, s, loc, ok in rows:
    print(f"| {f} | {fmt(g)} | {fmt(s)} | {loc} | {'yes' if ok else 'NO'} |")

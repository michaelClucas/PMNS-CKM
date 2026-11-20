"""Exp 031: recompute the p, p_global, Clopper-Pearson, Sidak and power numbers that STATE.md /
CANDIDATES.md quote, from the committed logs/JSON of the source experiments (no measured data used).

usage (repo root): .venv/Scripts/python experiments/20261119-031-skeptic-audit/check_stats.py
Output: experiments/20261119-031-skeptic-audit/check_stats.json and a table on stdout.
"""
import json
import pathlib

from scipy.stats import beta, chi2, norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
E = ROOT / "experiments"
OUT = pathlib.Path(__file__).with_name("check_stats.json")


def j(path):
    return json.loads((E / path).read_text())


def cp2(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def cp_up1(k, n):
    return 1.0 if k == n else beta.ppf(0.95, k + 1, n - k)


rows = []


def row(claim, quoted, recomputed, source, tol):
    ok = abs(quoted - recomputed) <= tol
    rows.append(dict(claim=claim, quoted=quoted, recomputed=recomputed, source=source, match=bool(ok)))


# --- p-values quoted in STATE / CANDIDATES (dof = N - k_cont)
for claim, c2, dof, quoted, src in [
    ("SU5-CSD3 p_global (chi2 22.40, N14, k_cont 9)", 22.4019, 5, 4.4e-4, "exp 022 README"),
    ("SU5-CSD3 analytic bound chi2>=14.7 -> best possible p", 14.7, 5, None, "exp 022 README"),
    ("2O p (chi2 68.84, N15, k_cont 10)", 68.8406, 5, 1.8e-13, "exp 016 fit_ours_ours.json"),
    ("BL-CDSV NO p_global (8.21, N4, k_cont 1)", 8.21, 3, 0.042, "exp 027 README"),
    ("BL-CDSV IO p_global (6.20, N4, k_cont 1)", 6.20, 3, 0.10, "exp 030 README"),
    ("TM1 NO p_global (5.1, N4, k_cont 2)", 5.1, 2, 0.078, "exp 013 results.csv"),
    ("TM1 IO p_global (3.3, N4, k_cont 2)", 3.3, 2, 0.191, "exp 013 results.csv"),
    ("F2-LO p (7.21, N4, k_cont 1)", 7.2108, 3, 0.065, "exp 018 README"),
    ("S4 tau-only lepton lead p (5.84, N7, k_cont 5)", 5.84, 2, 0.054, "exp 017 README"),
    ("RMSV p (4.79, N4, k_cont 2)", 4.79, 2, 0.091, "exp 027 README"),
]:
    p = float(chi2.sf(c2, dof))
    if quoted is None:
        rows.append(dict(claim=claim, quoted=None, recomputed=p, source=src, match=None))
    else:
        row(claim, quoted, p, src, max(0.06 * quoted, 1e-3 * quoted + 1e-16) if quoted > 0.01 else 0.1 * quoted)

# --- exp 016 committed JSON self-consistency
d = j("20261118-016-modular-literature/fit_ours_ours.json")
row("2O p from JSON chi2", d["p"], float(chi2.sf(d["chi2"], d["dof"])), "exp 016 fit_ours_ours.json", 1e-15)

# --- exp 025 ranks and CP intervals
d = j("20261119-025-nn-ckm-info/results.json")["main"]
rc = d["rank_conv_all"]
lo, hi = cp2(rc["n_fakes_ge_real"], rc["n"])
row("025 conv rank", 0.021, rc["n_fakes_ge_real"] / rc["n"], "exp 025 results.json", 6e-4)
row("025 conv CP95 lo", 0.015, lo, "exp 025 results.json", 5e-4)
row("025 conv CP95 hi", 0.028, hi, "exp 025 results.json", 5e-4)
rp = d["rank_point_all"]
row("025 point rank", 0.032, rp["n_fakes_ge_real"] / rp["n"], "exp 025 results.json", 6e-4)
row("025 T_point bits", 1.80, d["T_point_bits"], "exp 025 results.json", 0.005)
row("025 T_conv bits", 1.54, d["T_conv_bits"], "exp 025 results.json", 0.005)

# --- exp 028
for o, quoted_T, quoted_rank in (("NO", -0.99, 0.52), ("IO", 0.69, 0.60)):
    d = j(f"20261119-028-nn-info-decisive/results_{o}.json")
    e = d["equal"]
    row(f"028 {o} T_conv", quoted_T, e["T_conv"], f"exp 028 results_{o}.json", 0.005)
    r = e["N1lam_conv"]
    row(f"028 {o} N1lam rank", quoted_rank, r["n_ge_real"] / r["n"], f"exp 028 results_{o}.json", 0.005)
    if o == "NO":
        row("028 NO N1lam CP95 one-sided upper", 0.54, cp_up1(r["n_ge_real"], r["n"]), "exp 028 results_NO.json", 0.005)
        row("028 NO plain N1 rank", 0.38, e["N1_conv"]["n_ge_real"] / e["N1_conv"]["n"], "exp 028 results_NO.json", 0.005)
        row("028 NO Dirichlet N1lam pass fraction", 0.0, d["dirichlet"]["N1lam_pass_fraction"], "exp 028", 0)
        pb = d["real"]["best"]
        rows.append(dict(claim="028 pmns_best used (NO)", quoted=None, recomputed=pb, source="results_NO.json", match=None))
pb25 = j("20261119-025-nn-ckm-info/results.json")["real"]
same = all(abs(pb25[k] - j("20261119-028-nn-info-decisive/results_NO.json")["real"]["best"][k]) < 1e-12
           for k in ("s12sq", "s13sq", "s23sq", "dcp_deg"))
rows.append(dict(claim="pmns_best identical in exps 025 and 028 (NO)", quoted=None, recomputed=same,
                 source="025 results.json real vs 028 results_NO.json real.best", match=same))

# --- exp 030 power and null rates
for nul in ("N1", "N2"):
    for o in ("NO", "IO"):
        pw = j(f"20261119-030-engine-bilarge/power_{nul}_{o}.json")
        rt = j(f"20261119-030-engine-bilarge/rate_{nul}_{o}_200.json")
        rows.append(dict(claim=f"030 {nul} {o} power / 200-fake rate", quoted=None,
                         recomputed=[pw["fraction"], rt["rate"]], source="exp 030 JSON", match=None))
        row(f"030 {nul} {o} one-sided CP95 upper of rate", rt["upper"], cp_up1(rt["n_hits"], rt["n_run"]), "exp 030 JSON", 1e-6)

# --- exp 018 power
for f in ("T1_N1_NO", "T1_N1_IO", "T1_N2_NO", "T1_N2_IO", "F2LO_N1_NO", "F2LO_N2_NO"):
    pw = j(f"20261118-018-protocol-reruns/power_{f}.json")
    rows.append(dict(claim=f"018 power {f}", quoted=None, recomputed=pw["fraction"], source="exp 018", match=None))

# --- exp 020 random-tau CP interval (98/100)
lo, hi = cp2(98, 100)
row("020 random-tau CP95 lo (98/100)", 0.93, lo, "exp 020 README", 5e-3)
row("020 random-tau CP95 hi (98/100)", 0.998, hi, "exp 020 README", 5e-3)

# --- exp 029 Sidak
a3 = 2 * norm.sf(3)
for M, quoted in ((35, 3.95), (34, 3.95)):
    pl = 1 - (1 - a3) ** (1 / M)
    row(f"029 Sidak local Z for global 3 sigma, M={M}", quoted, float(norm.isf(pl / 2)), "exp 029 README", 0.01)
rows.append(dict(claim="029 Sidak if M counted all 37 relations x 2 orderings (74)", quoted=None,
                 recomputed=float(norm.isf((1 - (1 - a3) ** (1 / 74)) / 2)), source="sensitivity", match=None))

OUT.write_text(json.dumps(rows, indent=1, default=float))
bad = [r for r in rows if r["match"] is False]
for r in rows:
    flag = {True: "ok ", False: "BAD", None: "-- "}[r["match"]]
    print(f"{flag} {r['claim']:60s} quoted {r['quoted']!s:>10}  recomputed {r['recomputed']}")
print(f"\n{len(rows)} rows, {len(bad)} mismatches")

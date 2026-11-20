"""Exp 037 analysis: artifacts/037/rel_*.json -> results.csv, results.json, analyze.log (rules in prereg.md).
Run from this folder: PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python analyze.py > analyze.log
"""
import csv
import json
import math

import numpy as np

import lib37 as L

M = 74 * 3
ALPHA_LOC = L.sidak_local(M)
NULLS = ("N1", "N1lam", "N2")


def main():
    rows = L.load_frozen()
    out, missing = [], []
    for rid in dict.fromkeys(r["id"] for r in rows):
        f = L.BANK_DIR / f"rel_{rid}.json"
        if not f.exists():
            missing.append(rid)
            continue
        d = json.loads(f.read_text())
        for r in (x for x in rows if x["id"] == rid):
            o = r["ordering"]
            bits = int(r["k_honest"]) - int(r["k_cont"])
            rec = dict(id=rid, ordering=o, k_cont=int(r["k_cont"]), bits=bits, N=int(r["N"]),
                       survives_today=r["survives_today"] == "True")
            for null in NULLS:
                res = d["nulls"][null]["results"]
                real = next(x for x in res if x["name"] == f"{o}/real")
                k, n = real["k"], real["n"]
                p = (k + 1) / (n + 1)
                rec["chi2"] = real["chi2_data"]
                rec[f"{null}_k1000"] = real["k1000"]
                rec[f"{null}_rank"] = real["k1000"] / 1000 if real["k1000"] >= 0 else float("nan")
                rec[f"{null}_k"], rec[f"{null}_n"] = k, n
                rec[f"{null}_p"] = p
                rec[f"{null}_p_sidak"] = 1 - (1 - p) ** M
                rec[f"{null}_p_hist"] = 1 - (1 - p) ** (2 ** bits)
                rec[f"{null}_pass"] = bool(k <= 1 and n >= L.N_MAX)
                rec[f"{null}_truncated"] = bool(real["truncated"])
                for v in ("today", "nom", "opt"):
                    w = [x for x in res if x["o"] == o and x["version"] == v]
                    if not w:
                        continue
                    p300 = np.mean([x["k300"] == 0 for x in w])
                    if d["kind"] == "symbolic":
                        pg = np.mean([x["k"] <= 1 and x["n"] >= L.N_MAX for x in w])
                        tag = "glob"
                    else:
                        pg = np.mean([0 <= x["k2000"] <= 1 or (x["k"] <= 1 and x["n"] >= 2000) for x in w])
                        tag = "le2000"
                    rec[f"{null}_{v}_P300"] = float(p300)
                    rec[f"{null}_{v}_P{tag}"] = float(pg)
                    if v == "today":
                        r_real = real["k1000"] / 1000
                        worse = sum(x["k300"] / 300 >= r_real for x in w if x["k300"] >= 0)
                        rec[f"{null}_worlds_as_bad"] = int(worse)
                        rec[f"{null}_excluded"] = bool(p300 >= 0.5 and worse == 0)
                        rec[f"{null}_world_rank_median"] = float(np.median([x["k300"] / 300 for x in w]))
            rec["pass_any"] = any(rec[f"{n}_pass"] for n in NULLS)
            out.append(rec)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(L.HERE / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = dict(M=M, alpha_local=ALPHA_LOC, n_rows=len(out), missing=missing,
                   passes=[(r["id"], r["ordering"], n) for r in out for n in NULLS if r[f"{n}_pass"]],
                   excluded=[(r["id"], r["ordering"], n) for r in out for n in NULLS if r.get(f"{n}_excluded")],
                   truncated=[(r["id"], r["ordering"], n) for r in out for n in NULLS if r[f"{n}_truncated"]],
                   min_p=min((r[f"{n}_p"], r["id"], r["ordering"], n) for r in out for n in NULLS) if out else None)
    (L.HERE / "results.json").write_text(json.dumps(dict(summary=summary, rows=out), indent=1))
    print(json.dumps(summary, indent=1))
    hdr = ("id", "ord", "chi2", "N1 rank", "N1lam rank", "N2 rank", "min p_sid", "P300 N1/N1lam/N2 today",
           "Pglob/le2000 today", "P300 nom")
    print(" | ".join(hdr))
    for r in out:
        def g(n, v, t):
            return r.get(f"{n}_{v}_{t}", float("nan"))
        pgk = "Pglob" if r["id"].startswith("S-") else "Ple2000"
        print(f"{r['id']:11s} {r['ordering']} {r['chi2']:7.3f} | {r['N1_rank']:.3f} {r['N1lam_rank']:.3f} {r['N2_rank']:.3f}"
              f" | {min(r[f'{n}_p_sidak'] for n in NULLS):.3g}"
              f" | {g('N1','today','P300'):.2f} {g('N1lam','today','P300'):.2f} {g('N2','today','P300'):.2f}"
              f" | {g('N1','today',pgk):.2f} {g('N1lam','today',pgk):.2f} {g('N2','today',pgk):.2f}"
              f" | {g('N1','nom','P300'):.2f} {g('N1lam','nom','P300'):.2f} {g('N2','nom','P300'):.2f}"
              f" | excl {[n for n in NULLS if r.get(f'{n}_excluded')]}")


if __name__ == "__main__":
    main()


def power_summary():
    d = json.loads((L.HERE / "results.json").read_text())["rows"]
    print("\n== power summary (cells = row x null) ==")
    for v in ("today", "nom", "opt"):
        for n in NULLS:
            cells = [r for r in d if f"{n}_{v}_P300" in r]
            if not cells:
                continue
            p3 = np.array([r[f"{n}_{v}_P300"] for r in cells])
            pg = np.array([r.get(f"{n}_{v}_Pglob", r.get(f"{n}_{v}_Ple2000")) for r in cells])
            best = max(cells, key=lambda r: r[f"{n}_{v}_P300"])
            print(f"{v:5s} {n:5s}: cells {len(cells):2d}, median P300 {np.median(p3):.2f}, cells P300>=0.5: {int(np.sum(p3 >= 0.5))},"
                  f" max P300 {p3.max():.2f} ({best['id']} {best['ordering']}), max global-rule power (or <=2000 bound) {pg.max():.2f},"
                  f" cells >= 0.5: {int(np.sum(pg >= 0.5))}")
    print("\n== smallest real p-hat per null ==")
    for n in NULLS:
        r = min(d, key=lambda r: r[f"{n}_p"])
        print(f"{n}: {r['id']} {r['ordering']} k {r[f'{n}_k']}/{r[f'{n}_n']} p {r[f'{n}_p']:.4g} rank {r[f'{n}_rank']:.3f}"
              f" Sidak {r[f'{n}_p_sidak']:.3g} hist ({r['bits']} bits) {r[f'{n}_p_hist']:.3g}")
    allp = [r[f"{n}_p"] for r in d for n in NULLS]
    print(f"real p-hat over 222 cells: min {min(allp):.4g}, fraction < 0.01: {np.mean(np.array(allp) < 0.01):.3f},"
          f" < 0.05: {np.mean(np.array(allp) < 0.05):.3f}")


if __name__ == "__main__":
    power_summary()


def markdown_table():
    """Result table for README (results.csv numbers)."""
    d = json.loads((L.HERE / "results.json").read_text())["rows"]
    fr = {(r["id"], r["ordering"]): r for r in L.load_frozen()}
    print("\n== README table ==")
    print("| relation | ord | k_cont + bits | N | χ² | P300 today N1 / N1lam / N2 | P300 nominal N1 / N1lam / N2 "
          "| rank N1 / N1lam / N2 | min Šidák p | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in d:
        ex = [n for n in NULLS if r.get(f"{n}_excluded")]
        fit_fail = fr[(r["id"], r["ordering"])]["survives_today"] != "True"
        v = "no evidence"
        if ex:
            v += f"; excluded as exact link by {', '.join(ex)} (fit-quality driven)"
        if fit_fail:
            v += "; fit failure today (max pull > 2)"
        f = lambda n, t: r.get(f"{n}_{t}_P300", float("nan"))
        print(f"| {r['id']} | {r['ordering']} | {r['k_cont']} + {r['bits']} | {r['N']} | {r['chi2']:.2f} "
              f"| {f('N1','today'):.2f} / {f('N1lam','today'):.2f} / {f('N2','today'):.2f} "
              f"| {f('N1','nom'):.2f} / {f('N1lam','nom'):.2f} / {f('N2','nom'):.2f} "
              f"| {r['N1_rank']:.3f} / {r['N1lam_rank']:.3f} / {r['N2_rank']:.3f} "
              f"| {min(r[f'{n}_p_sidak'] for n in NULLS):.2f} | {v} |")


if __name__ == "__main__":
    markdown_table()

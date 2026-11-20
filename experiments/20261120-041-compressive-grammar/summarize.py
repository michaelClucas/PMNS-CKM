"""Exp 041: collect filter.json, fits.jsonl (DE), fits_emu.jsonl (emulator-start polish), search.json
into summary.json: best exact fit per entry and per family, G8 results, emulator ranking quality."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grammar as g  # noqa: E402


def rows(fn):
    p = HERE / fn
    return [json.loads(x) for x in open(p)] if p.exists() else []


def score(r):
    """Sort key: official chi2 if finite, else 1e4 + Gaussian fallback (the soft objective)."""
    c = r.get("chi2", np.inf)
    return c if np.isfinite(c) else r.get("soft_chi2", np.inf)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    fits = [r for r in rows("fits.jsonl") + rows("fits_emu.jsonl") if "error" not in r]
    filt = json.load(open(HERE / "filter.json")) if (HERE / "filter.json").exists() else None
    S = json.load(open(HERE / "search.json")) if (HERE / "search.json").exists() else []
    emu = {r["eid"]: r for r in S}
    best = {}
    for r in fits:
        if r["eid"] not in best or score(r) < score(best[r["eid"]]):
            best[r["eid"]] = r
    table = []
    for e in g.ENTRIES:
        r = best.get(e["id"])
        if r is None:
            continue
        k, kd = g.k_of(e)
        worst = sorted(r["pulls"].items(), key=lambda kv: -abs(kv[1]))[:3]
        table.append(dict(eid=e["id"], entry=g.name(e), in_G8=g.in_g8(e), k=k, N=r["N"], chi2=r["chi2"],
                          soft_chi2=r["soft_chi2"], p_global=r["p_global"], max_pull=r["max_pull"],
                          compression_ok=bool(k <= g.N_OBS - 2), fit_ok=r["fit_ok"],
                          worst_pulls={a: round(b, 2) for a, b in worst},
                          pred={a: round(b, 5) for a, b in r["pred"].items()},
                          n_runs=sum(1 for x in fits if x["eid"] == e["id"]),
                          best_from=("emulator start" if r.get("from_start") else f"DE seed {r['seed']}"),
                          emu_chi2=emu.get(e["id"], {}).get("emu_chi2"), emu_rank=emu.get(e["id"], {}).get("rank"),
                          killed_by_filter=(next(x["killed"] for x in filt["entries"] if x["eid"] == e["id"])
                                            if filt else None)))
    table.sort(key=lambda t: score(t) if np.isfinite(t["chi2"]) else t["soft_chi2"])
    fam = {}
    for t in table:
        e = g.ENTRIES[t["eid"]]
        for key in ("nu=" + e["nu"], "reps=" + "".join("H" if r else "h" for r in e["reps"])):
            if key not in fam or score(t) < score(fam[key]):
                fam[key] = t
    out = dict(k_table=g.k_table(), n_fit_runs=len(fits), entries=table,
               best_per_family={k: dict(entry=v["entry"], chi2=v["chi2"], soft_chi2=v["soft_chi2"],
                                        p_global=v["p_global"], max_pull=v["max_pull"]) for k, v in sorted(fam.items())},
               n_fit_ok=sum(t["fit_ok"] for t in table),
               n_fit_and_compression=sum(t["fit_ok"] and t["compression_ok"] for t in table),
               n_finite_chi2=sum(np.isfinite(t["chi2"]) for t in table))
    if S and table:
        ex = np.array([t["soft_chi2"] for t in sorted(table, key=lambda t: t["eid"])])
        em = np.array([emu[t["eid"]]["emu_chi2"] for t in sorted(table, key=lambda t: t["eid"])])
        order_exact = np.argsort(ex)
        top_exact = set(order_exact[:5].tolist())
        top_emu = set(np.argsort(em)[:5].tolist())
        de_only = {r["eid"]: min(score(x) for x in fits if x["eid"] == r["eid"] and not x.get("from_start"))
                   for r in S if any(x["eid"] == r["eid"] and not x.get("from_start") for x in fits)}
        emu_only = {r["eid"]: min(score(x) for x in fits if x["eid"] == r["eid"] and x.get("from_start"))
                    for r in S if any(x["eid"] == r["eid"] and x.get("from_start") for x in fits)}
        out["emulator"] = dict(spearman_emu_vs_exact=spearman(em, ex),
                               top5_overlap=len(top_exact & top_emu),
                               emu_start_better_than_DE=sum(emu_only[k] < de_only[k] - 1e-6 for k in emu_only
                                                            if k in de_only),
                               n_compared=len(set(emu_only) & set(de_only)))
    if filt:
        out["filter"] = dict(n_killed=filt["n_killed"], s23_2sigma=filt["s23_2sigma"],
                             killed_entries_fit_ok=[t["entry"] for t in table if t["killed_by_filter"] and t["fit_ok"]])
    json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=float)
    for t in table:
        print(t["eid"], t["entry"], "G8" if t["in_G8"] else "  ", "k", t["k"], "chi2", round(t["chi2"], 2),
              "soft", round(t["soft_chi2"], 1), "p_glob", t["p_global"], "maxpull", round(t["max_pull"], 2),
              t["worst_pulls"], "emu", t["emu_chi2"] and round(t["emu_chi2"], 1), t["best_from"])
    print({k: v for k, v in out.items() if k not in ("entries", "k_table")})


if __name__ == "__main__":
    main()

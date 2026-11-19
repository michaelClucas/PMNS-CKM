"""Markdown tables for the README from frozen_relations.csv, forecast.csv, windows.csv, plus the
SHA-256 of frozen_relations.csv. Run: ../../.venv/Scripts/python tables.py > tables.md"""
import hashlib
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PMNS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
NUFIT_SIG = {"s12sq": 0.00665, "s13sq": 0.00057, "s23sq": 0.0155, "dcp_deg": 21.5}  # NO symmetrized, for labels only


def fmt(k, v, s):
    if not np.isfinite(v):
        return ""
    if k == "dcp_deg":
        return f"{v:.0f} ± {s:.0f}"
    if k == "s13sq":
        return f"{v:.5f} ± {s:.5f}"
    return f"{v:.4f} ± {s:.4f}"


def main():
    raw = (HERE / "frozen_relations.csv").read_bytes()
    print("SHA-256 frozen_relations.csv:", hashlib.sha256(raw).hexdigest(), "\n")
    d = pd.read_csv(HERE / "frozen_relations.csv")
    f = pd.read_csv(HERE / "forecast.csv")
    print("| id | ord | formula | k_cont / k_honest | today max|pull| (p_glob) | s12² | s13² | s23² | δ (deg) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        cells = []
        for k in PMNS:
            v = r[f"pred_{k}"]
            s = np.hypot(r[f"sig_ckm_{k}"], r[f"sig_fit_{k}"]) if np.isfinite(v) else np.nan
            c = fmt(k, v, s)
            if c and r["family"] != "S" and s > 0.5 * NUFIT_SIG[k]:
                c += " (fit)"
            cells.append(c)
        surv = "" if r["survives_today"] else " ✗"
        print(f"| {r['id']} | {r['ordering']} | {r['formula'][:90]} | {r['k_cont']} / {r['k_honest']} | "
              f"{r['max_abs_pull_today']:.2f}{surv} ({r['p_global_today']:.2g}) | " + " | ".join(cells) + " |")
    print()
    print("| id | ord | nominal Z | optimistic Z | nominal alt-octant Z | optimistic alt-octant Z | worst obs (nominal) |")
    print("|---|---|---|---|---|---|---|")
    for (rid, o), g in f.groupby(["id", "ordering"], sort=False):
        g = g.set_index("scenario")
        if not bool(g.loc["nominal", "survives_today"]):
            continue
        pulls = {k: g.loc["nominal", f"pull_{k}"] for k in PMNS}
        pulls = {k: v for k, v in pulls.items() if np.isfinite(v)}
        worst = max(pulls, key=lambda k: abs(pulls[k]))
        print(f"| {rid} | {o} | {g.loc['nominal', 'Z']:.1f} | {g.loc['optimistic', 'Z']:.1f} | "
              f"{g.loc['nominal_altoct', 'Z']:.1f} | {g.loc['optimistic_altoct', 'Z']:.1f} | "
              f"{worst} {pulls[worst]:+.1f} |")
    print()
    w = pd.read_csv(HERE / "windows.csv")
    print("Inverse scan (nominal sigmas): values where 1-2 frozen relations stay within Z < 2")
    for (o, t), g in w.groupby(["ordering", "observable"]):
        g = g.sort_values("value")
        sel = g[(g.n_compatible >= 1) & (g.n_compatible <= 2)]
        # merge contiguous runs with the same compatible set
        runs, prev = [], None
        step = np.diff(g.value.values).min()
        for _, r in sel.iterrows():
            if prev is not None and runs and abs(r.value - runs[-1][1] - step) < 1e-9 and r.compatible == runs[-1][2]:
                runs[-1][1] = r.value
            else:
                runs.append([r.value, r.value, r.compatible])
            prev = r
        print(f"- {o} {t}: count range {g.n_compatible.min()}-{g.n_compatible.max()}; "
              + "; ".join(f"[{a:g}, {b:g}] -> {c}" for a, b, c in runs))


if __name__ == "__main__":
    main()

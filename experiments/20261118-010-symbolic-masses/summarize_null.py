"""null_{a,b}_{MZ,GUT}.csv + levels_real.csv -> null_summary.json and null_summary.csv.

Per (kind, scale, target, ordering, C): hit = fake best |pull| (complexity <= C) <= real best |pull|
from the same pipeline (kind a: Gaussian real row, universe -1; kind b: NuFIT-table real values from
levels_real.csv). Also: c1 = lowest complexity reaching |pull| < 1; hit_c1 = fake c1 <= real c1.
Rates with src.stats.fake_universe_rate (95% Clopper-Pearson).
"""
import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src.stats import fake_universe_rate  # noqa: E402

CMAX = 5


def c1(best):
    k = np.nonzero(np.asarray(best) < 1)[0]
    return int(k[0]) + 1 if len(k) else 99


def main():
    real_b = {}
    with open(HERE / "levels_real.csv") as fh:
        for r in csv.DictReader(fh):
            real_b.setdefault((r["scale"], r["target"], r["ordering"]), {})[int(r["complexity_max"])] = \
                float(r["best_abs_pull"])
    out, rows = [], []
    for kind in ("a", "b"):
        for scale in ("MZ", "GUT"):
            f = HERE / f"null_{kind}_{scale}.csv"
            if not f.exists():
                continue
            data = {}
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    best = [float(r[f"best_C{c}"]) for c in range(1, CMAX + 1)]
                    data.setdefault((r["target"], r["ordering"]), {})[int(r["universe"])] = best
            for (tgt, o), d in data.items():
                if kind == "a":
                    real = d.pop(-1)
                else:
                    real = [real_b[(scale, tgt, o)][c] for c in range(1, CMAX + 1)]
                fakes = np.array(list(d.values()))
                n = len(fakes)
                for c in range(1, CMAX + 1):
                    h = int((fakes[:, c - 1] <= real[c - 1]).sum())
                    r = fake_universe_rate(h, n)
                    rec = {"kind": kind, "scale": scale, "target": tgt, "ordering": o, "C": c,
                           "real_best": real[c - 1], "fake_median": float(np.median(fakes[:, c - 1])),
                           "hits": h, "n": n, **r}
                    out.append(rec)
                rc = c1(real)
                fc = np.array([c1(x) for x in fakes])
                h = int((fc <= rc).sum())
                r = fake_universe_rate(h, n)
                out.append({"kind": kind, "scale": scale, "target": tgt, "ordering": o, "C": "c1",
                            "real_c1": rc, "fake_c1_median": float(np.median(fc)), "hits": h, "n": n, **r})
    (HERE / "null_summary.json").write_text(json.dumps(out, indent=1))
    keys = ["kind", "scale", "target", "ordering", "C", "real_best", "fake_median", "real_c1", "fake_c1_median",
            "hits", "n", "rate", "lo", "hi", "upper"]
    with open(HERE / "null_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for rec in out:
            w.writerow({k: rec.get(k, "") for k in keys})
    for rec in out:
        if rec["C"] in (1, 2, 3, "c1"):
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})


if __name__ == "__main__":
    main()

"""Compare this experiment's results.csv with exp 006's, cell by cell.

    .venv/Scripts/python experiments/20261118-013-infra-hardening/diff_vs_006.py

Writes diffs_vs_006.csv (one row per changed cell) and diff_summary.log.
A numeric cell counts as changed if |new - old| > 1e-9 * max(1, |old|), or nan-ness differs.
"""
import csv
import math
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
OLD = HERE.parent / "20261118-006-baseline-scores" / "results.csv"
NEW = HERE / "results.csv"
COLS = ("N", "k_c", "k_d", "k", "dof", "chi2", "p", "p_fit", "chi2_global", "p_global", "max_abs_pull",
        "input_dchi2", "params")


def load(path):
    with open(path, newline="") as fh:
        return {(r["id"], r["ordering"], r["analysis"]): r for r in csv.DictReader(fh)}


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def changed(a, b):
    fa, fb = num(a), num(b)
    if fa is None or fb is None:
        return a != b
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) != math.isnan(fb)
    return abs(fa - fb) > 1e-9 * max(1.0, abs(fa))


def main():
    old, new = load(OLD), load(NEW)
    rows, by_col = [], defaultdict(set)
    for key, o in old.items():
        n = new.get(key)
        if n is None:
            rows.append(dict(id=key[0], ordering=key[1], analysis=key[2], column="<row>", old="present", new="missing"))
            continue
        for c in COLS:
            if changed(o.get(c, ""), n.get(c, "")):
                rows.append(dict(id=key[0], ordering=key[1], analysis=key[2], column=c, old=o.get(c), new=n.get(c)))
                by_col[c].add(key[0])
    with open(HERE / "diffs_vs_006.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "ordering", "analysis", "column", "old", "new"])
        w.writeheader()
        w.writerows(rows)
    lines = [f"{len(old)} exp-006 rows compared, {len(rows)} changed cells"]
    for c in COLS:
        ids = sorted(by_col.get(c, ()))
        lines.append(f"{c:13s} changed for {len(ids):2d} baselines: {' '.join(ids)}")
    text = "\n".join(lines)
    (HERE / "diff_summary.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

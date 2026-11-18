"""Compare every committed data/nufit/*.csv against the NuFIT 6.1 release files, plus a
printed random sample of rows.

Usage (repo root):
    .venv/Scripts/python experiments/20261118-005-data-verify/verify_nufit_csv.py RAW_DIR
RAW_DIR holds v61.release-{TBoff,TByes}-{NO,IO}.txt decompressed from
http://www.nu-fit.org/sites/default/files/v61.release-*.txt.xz
"""
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
NUFIT = ROOT / "data" / "nufit"
RAW = pathlib.Path(sys.argv[1])


def parse_raw(path, skip=("T23/DMA/DCP",)):
    sec, cur, rows = {}, None, None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("# ") and "projection:" in s:
                tag = s[2:].split(" projection")[0]
                cur = None if tag in skip else tag
                if cur:
                    rows = sec.setdefault(cur, [])
                continue
            if not s or s.startswith("#") or cur is None:
                continue
            rows.append([float(x) for x in s.split()])
    return {k: np.array(v) for k, v in sec.items()}


random.seed(20261118)
n_files = n_rows = 0
problems, sample = [], []
for an in ("TBoff", "TByes"):
    for o in ("NO", "IO"):
        raw = parse_raw(RAW / f"v61.release-{an}-{o}.txt")
        mins = {k: v[:, -1].min() for k, v in raw.items()}
        print(f"{an} {o}: global min of Delta chi^2 over 1D tables = "
              f"{min(mins[t] for t in ('T12', 'T13', 'T23', 'DCP', 'DMS', 'DMA')):.4f}")
        for csvp in sorted(NUFIT.glob(f"v61_{an}_{o}_*.csv")):
            tag = csvp.stem.split("_", 3)[3].replace("-", "/")
            a = np.loadtxt(csvp, delimiter=",", skiprows=1, ndmin=2)
            b = raw.get(tag)
            n_files += 1
            if b is None:
                problems.append(f"{csvp.name}: section {tag} not in raw file")
                continue
            if a.shape != b.shape:
                problems.append(f"{csvp.name}: shape {a.shape} vs raw {b.shape}")
                continue
            err = np.abs(a - b) / np.maximum(1.0, np.abs(b))
            n_rows += len(a)
            if err.max() > 1e-6:
                i = int(err.max(axis=1).argmax())
                problems.append(f"{csvp.name}: max rel diff {err.max():.2e} at row {i}: {a[i]} vs {b[i]}")
            for i in random.sample(range(len(a)), 1):
                sample.append((csvp.name, i + 2, a[i].tolist(), b[i].tolist()))
        # sections in raw but not committed
        have = {p.stem.split("_", 3)[3].replace("-", "/") for p in NUFIT.glob(f"v61_{an}_{o}_*.csv")}
        print(f"  not committed: {sorted(set(raw) - have)}")

print(f"\ncompared {n_files} files, {n_rows} rows in full; problems: {len(problems)}")
for p in problems:
    print("  ", p)
print("\nRandom sample (one row per file), csv line number, data/ row, raw row:")
for s in random.sample(sample, 16):
    print("  ", s)

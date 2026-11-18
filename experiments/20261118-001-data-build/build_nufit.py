"""Download the NuFIT 6.1 Delta-chi^2 tables and split them into per-projection CSVs.

Usage (from repo root):
    .venv/Scripts/python experiments/20261118-001-data-build/build_nufit.py [--full]

Raw .xz files are cached in artifacts/nufit_raw/ (gitignored).
Committed output (data/nufit/):
    - every 1D projection, both analyses (TBoff = IC23 w/o SK-atm, TByes = IC24 with SK-atm), NO and IO
    - every 2D projection for the default analysis (TBoff), NO and IO
    - the T23/DCP 2D projection for the TByes analysis too (used by src.data.pmns_delta_chi2)
--full additionally writes the 3D T23/DMA/DCP projection and the remaining TByes 2D
projections to artifacts/nufit_full/ (about 40 MB per file, not committed).

Prints the best fit and 1sigma/3sigma ranges read off each 1D profile so they can be
compared with the NuFIT parameter table (data/pmns.json).
"""
import argparse
import csv
import lzma
import pathlib
import re
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts" / "nufit_raw"
FULL = ROOT / "artifacts" / "nufit_full"
OUT = ROOT / "data" / "nufit"
BASE = "http://www.nu-fit.org/sites/default/files/"
FILES = {(a, o): f"v61.release-{a}-{o}.txt.xz" for a in ("TBoff", "TByes") for o in ("NO", "IO")}

COLNAMES = {
    "sin^2(theta23)": "s23sq",
    "sin^2(theta13)": "s13sq",
    "sin^2(theta12)": "s12sq",
    "Delta_CP/deg": "dcp_deg",
    "Log10(Delta_m21^2/[eV^2])": "log10_dm21sq_eV2",
    "Delta_m31^2/[1e-3_eV^2]": "dm3l_sq_1e-3eV2",
    "Delta_m32^2/[1e-3_eV^2]": "dm3l_sq_1e-3eV2",
    "Delta_chi^2": "delta_chi2",
}


def fetch(name):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        print("downloading", BASE + name)
        urllib.request.urlretrieve(BASE + name, path)
    return path


def parse(path):
    """Return {tag: (column_names, ndarray)} for one NuFIT release file."""
    sections, tag, cols, rows = {}, None, None, []
    with lzma.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("# "):
                if tag:
                    sections[tag] = (cols, np.array(rows, dtype=float))
                m = re.match(r"# (\S+) projection: (.*)$", line.strip())
                tag = m.group(1)
                cols = [COLNAMES[c] for c in m.group(2).split()]
                rows = []
            elif line.startswith("#") or not line.strip():
                continue
            else:
                rows.append([float(x) for x in line.split()])
    sections[tag] = (cols, np.array(rows, dtype=float))
    return sections


def write_csv(path, cols, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in arr:
            w.writerow([f"{x:.10g}" for x in r])


def interval(x, y, level):
    """Range of x where y - min(y) < level, by linear interpolation (outermost crossings)."""
    y = y - y.min()
    inside = np.where(y < level)[0]
    i0, i1 = inside[0], inside[-1]
    lo = x[i0] if i0 == 0 else np.interp(level, [y[i0], y[i0 - 1]], [x[i0], x[i0 - 1]])
    hi = x[i1] if i1 == len(x) - 1 else np.interp(level, [y[i1], y[i1 + 1]], [x[i1], x[i1 + 1]])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    for (ana, order), name in FILES.items():
        secs = parse(fetch(name))
        offset = min(arr[:, -1].min() for _, arr in secs.values())
        print(f"\n== {ana} {order}: min Delta_chi2 over all sections = {offset:.3f}")
        for tag, (cols, arr) in secs.items():
            fname = f"v61_{ana}_{order}_{tag.replace('/', '-')}.csv"
            ndim = len(cols) - 1
            commit = ndim == 1 or (ndim == 2 and (ana == "TBoff" or tag == "T23/DCP"))
            if commit:
                write_csv(OUT / fname, cols, arr)
            elif args.full:
                write_csv(FULL / fname, cols, arr)
            if ndim == 1:
                x, y = arr[:, 0], arr[:, 1]
                bf = x[np.argmin(y)]
                l1, h1 = interval(x, y, 1.0)
                l3, h3 = interval(x, y, 9.0)
                print(f"  {cols[0]:>18}: bf {bf:.5g}  1s [{l1:.5g}, {h1:.5g}]  3s [{l3:.5g}, {h3:.5g}]  min {y.min():.3f}")


if __name__ == "__main__":
    main()

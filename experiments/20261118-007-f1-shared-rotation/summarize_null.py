"""Fake-universe hit rates: fraction of fakes whose best chi^2 <= the real best chi^2.

Kind a (Haar PMNS target, Gaussian chi^2) is compared with the real-data pipeline run with
the same Gaussian scorer (gauss_NO / gauss_IO in real_pipeline.json), like with like.
Kind b (hierarchical fake CKM, NuFIT tables) is compared with table_NO / table_IO.
Writes null_summary.json and prints a table.
"""
import csv
import json

import common  # noqa: F401
import numpy as np

from common import HERE
from src.stats import fake_universe_rate


def main():
    real = json.load(open(HERE / "real_pipeline.json"))["pipeline"]
    out = {}
    for kind, pre in (("a", "gauss"), ("b", "table")):
        path = HERE / f"null_{kind}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path)))
        n = len(rows)
        for o in ("NO", "IO"):
            for var in ("cont", "struct"):
                ref = real[f"{pre}_{o}"][f"best_{var}"]
                vals = np.array([float(r[f"best_{var}_{o}"]) for r in rows])
                hits = int(np.sum(vals <= ref))
                r = fake_universe_rate(hits, n)
                key = f"{kind}_{o}_{var}"
                out[key] = {"n_fakes": n, "real_best_chi2": ref, "hits": hits, **r,
                            "fake_best_chi2_median": float(np.median(vals)),
                            "fake_best_chi2_p10": float(np.percentile(vals, 10))}
                print(f"null {kind} {o} {var:6s}: real {ref:7.3f}  hits {hits:3d}/{n}  "
                      f"rate {r['rate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] (95% CP), "
                      f"one-sided 95% upper {r['upper']:.3f}; fake median {np.median(vals):.2f}")
    json.dump(out, open(HERE / "null_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

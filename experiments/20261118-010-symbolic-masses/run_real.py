"""Brute-force search on the real inputs at M_Z and 2e16 GeV (SM running).

Outputs: grammar_counts.json, levels_real.csv, top_<scale>.csv, known_relations.csv.
Requires artifacts/exp010/enum_C6.npz (run_enum.py).
"""
import csv
import json
import time

import numpy as np

import pipeline as P
import symlib as S

CMAX = 6
TOP_PER_LEVEL = 5

# known relations, written in the grammar (value = angle in rad); complexity counted by hand
KNOWN = [
    ("s12sq", "QLC solar: theta12 = pi/4 - t12q", 3, lambda x: np.pi / 4 - x[0]),
    ("s23sq", "QLC atm: theta23 = pi/4 - t23q", 3, lambda x: np.pi / 4 - x[2]),
    ("s23sq", "QLC atm: theta23 = pi/4 + t23q", 3, lambda x: np.pi / 4 + x[2]),
    ("s13sq", "KMS: theta13 = t12q / sqrt2", 3, lambda x: x[0] / np.sqrt(2)),
    ("s13sq", "KMS exact: theta13 = asin(sin(t12q)/sqrt2)", 5, lambda x: np.arcsin(np.sin(x[0]) / np.sqrt(2))),
    ("s13sq", "lepton GST: theta13 = asin(sqrt(me/mmu)/sqrt2)", 4, lambda x: np.arcsin(np.sqrt(x[8]) / np.sqrt(2))),
    ("s13sq", "theta13 = sqrt(me/mmu)", 1, lambda x: np.sqrt(x[8])),
    ("s13sq", "theta13 = sqrt(md/ms)/sqrt2 (down-GST analogue)", 3, lambda x: np.sqrt(x[6]) / np.sqrt(2)),
    ("s12sq", "TBM: theta12 = asin(sqrt(1/3))", 3, lambda x: np.arcsin(np.sqrt(1 / 3))),
    ("s23sq", "maximal: theta23 = pi/4", 1, lambda x: np.pi / 4),
    ("dcp_deg", "maximal: delta = 3pi/2 = pi + pi/2", 3, lambda x: 1.5 * np.pi),
    ("dcp_deg", "delta = -dq", 2, lambda x: -x[3]),
    ("dcp_deg", "delta = pi + dq", 3, lambda x: np.pi + x[3]),
]


def fd_tangent(f, base, dirs):
    t = []
    for d in dirs:
        t.append((f(base + 1e-3 * d) - f(base - 1e-3 * d)) / 2e-3)
    return np.array(t)[None, :]


def main():
    E = S.Loaded(S.ART / f"enum_C{CMAX}.npz")
    counts = {"per_level_distinct": E.counts, "per_level_raw_candidates": E.counts_raw,
              "cumulative_distinct": {c: int(sum(E.counts[k] for k in E.counts if k <= c)) for c in E.counts},
              "leaves": S.LEAF_NAMES, "unary": list(S.UNARY.values()), "binary": list(S.BINARY.values())}
    (S.HERE / "grammar_counts.json").write_text(json.dumps(counts, indent=1))
    slices = P.level_slices(E.comp)
    lv_rows, known_rows = [], []
    for scale in ("MZ", "GUT"):
        t0 = time.time()
        r = S.real_inputs(scale)
        v, T = E.evaluate(r["base"], r["dirs"])
        res = P.run_profile(v, T, E.comp, keep_pulls=True)
        top_rows = []
        for (tgt, o), (best, n1, n2, nv, pull) in res.items():
            p, s = S.predictions(v, T, tgt)
            for c in range(1, CMAX + 1):
                lv_rows.append([scale, tgt, o, c, float(best[c - 1]), int(n1[c - 1]), int(n2[c - 1]),
                                int(nv[c - 1]), counts["cumulative_distinct"][c]])
                sl = slices[c - 1]
                ap = np.abs(pull[sl])
                ap = np.where(np.isfinite(ap), ap, np.inf)
                order = np.argsort(ap)[:TOP_PER_LEVEL]
                for rank, j in enumerate(order):
                    i = sl.start + j
                    top_rows.append([tgt, o, c, rank + 1, E.expr(i), float(v[i]), float(p[i]), float(s[i]),
                                     float(pull[i])])
            # known relations
            for (kt, name, kc, f) in KNOWN:
                if kt != tgt:
                    continue
                val = np.array([f(r["base"])])
                tan = fd_tangent(f, r["base"], r["dirs"])
                pk, sk = S.predictions(val, tan, tgt)
                pl = float(P.profile_scorer(tgt, o).pull(pk, sk)[0])
                in_enum = bool(np.any(np.isclose(v[E.comp <= kc], val[0], rtol=1e-12, atol=1e-14)))
                rank = int((np.abs(pull[E.comp <= kc]) < abs(pl)).sum())  # nan (invalid) never counts
                known_rows.append([scale, tgt, o, name, kc, float(pk[0]), float(sk[0]), pl, in_enum, rank,
                                   counts["cumulative_distinct"][kc]])
        with open(S.HERE / f"top_{scale}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target", "ordering", "complexity", "rank_in_level", "expression", "value_rad",
                        "prediction", "sigma_pred", "pull"])
            w.writerows(top_rows)
        print(f"[real] {scale}: {len(E)} expressions scored in {time.time() - t0:.1f}s", flush=True)
    with open(S.HERE / "levels_real.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "target", "ordering", "complexity_max", "best_abs_pull", "n_abs_pull_lt1",
                    "n_abs_pull_lt2", "n_valid_cum", "n_expressions_cum"])
        w.writerows(lv_rows)
    with open(S.HERE / "known_relations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "target", "ordering", "relation", "complexity", "prediction", "sigma_pred", "pull",
                    "found_in_enumeration", "n_better_at_le_complexity", "n_expressions_le_complexity"])
        w.writerows(known_rows)
    for row in lv_rows:
        if row[3] in (1, 2, 3, 4, 6):
            print("  ", row)


if __name__ == "__main__":
    main()

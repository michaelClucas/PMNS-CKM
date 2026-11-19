"""Exp 010 symbolic survivors at M_Z, selection rule fixed here (see README):
  1. complexity C <= 4 in the exp 010 grammar (27,051 distinct functions);
  2. the expression contains at least one CKM leaf (t12q, t13q, t23q, dq);
  3. exp 010 validity rules (symlib.predictions);
  4. official scorer src.evaluate (1D NuFIT profile, CKM/mass inputs declared, central values)
     gives |pull| <= 2 in BOTH orderings (no ordering choice), and the exp 010 input-error-convolved
     pull is also <= 2 in both orderings.
Output: symbolic_survivors_all.csv (every survivor), printed counts per target.
Run from this folder: ../../.venv/Scripts/python symbolic_survivors.py
"""
import csv
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E10 = ROOT / "experiments" / "20261118-010-symbolic-masses"
sys.path.insert(0, str(E10))
sys.path.insert(0, str(ROOT))
import symlib as S  # noqa: E402
import pipeline as P  # noqa: E402
from src.evaluate import Scorer  # noqa: E402

CMAX = 4
CKM_LEAVES = {0, 1, 2, 3}
INPUTS_ALL = {0: "ckm_s12", 1: "ckm_s13", 2: "ckm_s23", 3: "ckm_delta_rad"}


def leaves(E, i, acc=None):
    acc = set() if acc is None else acc
    o = int(E.op[i])
    if o == 0:
        acc.add(int(E.a[i]))
    else:
        leaves(E, int(E.a[i]), acc)
        if o > 6:
            leaves(E, int(E.b[i]), acc)
    return acc


def main():
    path = S.ART / "enum_C6.npz"
    if path.exists():
        E = S.Loaded(path, cmax=CMAX)
    else:
        E = S.Enum(CMAX)
    r = S.real_inputs("MZ")
    v, T = E.evaluate(r["base"], r["dirs"])
    n_total = len(E)
    rows = []
    counts = {}
    for tgt in S.TARGETS:
        p, s = S.predictions(v, T, tgt)
        pulls = {o: P.profile_scorer(tgt, o).pull(p, s) for o in ("NO", "IO")}
        cand = np.nonzero(np.isfinite(p) & (np.abs(pulls["NO"]) <= 2) & (np.abs(pulls["IO"]) <= 2))[0]
        sc = {}
        n_ckm = 0
        for i in cand:
            lv = leaves(E, i)
            if not (lv & CKM_LEAVES):
                continue
            n_ckm += 1
            inputs = tuple(sorted({INPUTS_ALL[x] for x in lv if x in INPUTS_ALL}))
            off = {}
            for o in ("NO", "IO"):
                key = (o, inputs)
                if key not in sc:
                    sc[key] = Scorer((tgt,), o, inputs=inputs)
                off[o] = sc[key]({tgt: float(p[i])})
            if max(abs(off[o].pulls[tgt]) for o in off) > 2:
                continue
            uses_mass = bool(lv - CKM_LEAVES - set(range(16, 26)))
            rows.append(dict(target=tgt, complexity=int(E.comp[i]), expression=E.expr(i),
                             leaves=" ".join(S.LEAF_NAMES[x] for x in sorted(lv)), uses_mass_ratio=uses_mass,
                             pred=float(p[i]), sigma_pred=float(s[i]),
                             pull_official_NO=off["NO"].pulls[tgt], pull_official_IO=off["IO"].pulls[tgt],
                             pull_conv_NO=float(pulls["NO"][i]), pull_conv_IO=float(pulls["IO"][i])))
        counts[tgt] = dict(valid_pull_le2_both=int(len(cand)), with_ckm_leaf=n_ckm,
                           survivors=sum(1 for q in rows if q["target"] == tgt))
    with open(HERE / "symbolic_survivors_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"distinct functions C<={CMAX}: {n_total}")
    for t, c in counts.items():
        print(t, c)
    print("total survivors", len(rows))


if __name__ == "__main__":
    main()

"""Exp 047 (a): where A6842 sits in the arXiv:2110.05065 classification, and why that paper excludes it.

1. Permutation classes: the 6 common weak-basis permutations P (Y_q -> P Y_q P^T, same P for u and d).
   Checks whether any maps A6842's zero pair onto a same-structure (Fritzsch-like I_x I_x) pair.
2. Paper labels: u pattern = S6 (class I_f), d pattern = S3 (class I_c); 2110.05065 Table 2 puts I_f I_c in
   the p_k = p2 row, excluded analytically on p. 12 by the leading-order U^a ~ 1 of Eqs. (24)-(25), which
   assume the (3,3) element C of the class-I_a form is ~ m3.
3. At the refitted A6842 point (textures.json): map each sector to the I_a form and report C / m3 and the
   largest |U^a_ij| off the diagonal (U^a diagonalises the I_a-form real matrix, ascending masses).
"""
import itertools
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

U_NZ = {(0, 0), (0, 2), (1, 2), (2, 2)}          # A6842 u nonzeros (upper triangle, 0-based)
D_NZ = {(0, 0), (0, 1), (0, 2), (2, 2)}          # A6842 d nonzeros
FRITZSCH_NZ = {(0, 1), (1, 1), (1, 2), (2, 2)}   # class I_a = S1


def sym(nz):
    return frozenset((min(i, j), max(i, j)) for i, j in nz)


def perm(nz, p):
    return sym({(p[i], p[j]) for i, j in nz})


def class_I():
    return {perm(FRITZSCH_NZ, p) for p in itertools.permutations(range(3))}


def main():
    out = {}
    CI = class_I()
    out["u_in_class_I"] = sym(U_NZ) in CI
    out["d_in_class_I"] = sym(D_NZ) in CI
    same = [p for p in itertools.permutations(range(3)) if perm(U_NZ, p) == perm(D_NZ, p)]
    out["common_perm_to_same_structure"] = same
    # permutation q with q(I_a) = pattern, for each sector
    qu = [p for p in itertools.permutations(range(3)) if perm(FRITZSCH_NZ, p) == sym(U_NZ)]
    qd = [p for p in itertools.permutations(range(3)) if perm(FRITZSCH_NZ, p) == sym(D_NZ)]
    out["perm_Ia_to_u"], out["perm_Ia_to_d"] = qu, qd
    t = json.loads((HERE / "textures.json").read_text())["A6842"]
    for s, q in (("u", qu[0]), ("d", qd[0])):
        M = np.array(t["Yu_abs" if s == "u" else "Yd_abs"])
        # I_a form: Ma[i, j] = M[q(i), q(j)]
        Ma = M[np.ix_(q, q)]
        w, U = np.linalg.eigh(Ma)
        m = np.abs(w)
        o = np.argsort(m)
        U = U[:, o]
        off = np.abs(U - np.diag(np.diag(U)))
        out[s] = {"Ia_form_abs": Ma.round(6).tolist(), "C_over_m3": float(Ma[2, 2] / m[o][2]),
                  "D_over_m3": float(Ma[1, 1] / m[o][2]), "masses_over_m3": (m[o] / m[o][2]).tolist(),
                  "max_offdiag_absU": float(off.max()), "absU": np.abs(U).round(4).tolist()}
    print(json.dumps(out, indent=1, default=str))
    (HERE / "equivalence.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

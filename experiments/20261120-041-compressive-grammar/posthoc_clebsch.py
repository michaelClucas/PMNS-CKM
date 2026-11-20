"""Exp 041 post-hoc diagnostic (NOT part of the grammar; looked at after the grammar failed):
which real Clebsch (l12, l22, l23) on Y_e = Clebsch * Y_d would reproduce the charged-lepton mass
ratios at the quark-only best fit? Scans l in a fine grid and in the SU(5) dim-5 Clebsch list
{+-1/2, +-1, +-3/2, +-2, +-3, +-9/2, +-6} (Antusch-Spinrath type values). -> posthoc_clebsch.json"""
import itertools
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import grammar as g  # noqa: E402
from src import evaluate as E  # noqa: E402
from src.yukawa import left_rotation  # noqa: E402

th = json.load(open(HERE / "filter.json"))["quark_fits"][0]["params"]
Yu, Yd = g.quark_mats(th)
sc = E.Scorer(["mr_e_mu", "mr_mu_tau"], "NO", mass_floor=g.FLOOR)


def chi(l12, l22, l23):
    S = np.ones((3, 3))
    S[0, 1] = S[1, 0] = l12
    S[1, 1] = l22
    S[1, 2] = S[2, 1] = l23
    _, y = left_rotation(Yd * S)
    p = {"mr_e_mu": y[0] / y[1], "mr_mu_tau": y[1] / y[2]}
    return sc.chi2(p), p


L = [s * v for v in (0.5, 1, 1.5, 2, 3, 4.5, 6) for s in (1, -1)]
res = []
for l12, l22, l23 in itertools.product(L, L, L):
    c, p = chi(l12, l22, l23)
    res.append(dict(l12=l12, l22=l22, l23=l23, chi2=c, **p))
res.sort(key=lambda r: r["chi2"])
out = dict(note=__doc__, quark_params=th, best_from_list=res[:10],
           n_list_combos=len(res), n_chi2_below_4=sum(r["chi2"] < 4 for r in res),
           ps_grammar_values={str(k): chi(*k)[0] for k in itertools.product((1, -3), repeat=3)})
json.dump(out, open(HERE / "posthoc_clebsch.json", "w"), indent=1)
for r in res[:10]:
    print(r)
print(out["n_chi2_below_4"], "of", len(res), "list combos give chi2 < 4 on the two charged-lepton ratios")
print({k: round(v, 1) for k, v in out["ps_grammar_values"].items()})

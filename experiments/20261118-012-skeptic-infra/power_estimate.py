"""Power of the exp 007 nulls: how often would an exactly TRUE F1 relation pass criterion 4?

If the relation were true, the real-data best chi^2 would be about chi^2 with dof = N - k_cont
(2 for fitted phases, 4 for structured phases), and smaller still after picking the best of
K configurations. The fake best-chi^2 distribution is taken from exp 007's null_a.csv /
null_b.csv (it describes how well the search fits data unrelated to the relation). Expected
hit rate under truth = E[F_fake(X)], X ~ chi^2(dof). P(pass) = P(F_fake(X) < 1%).
Approximation: the fake distribution under truth-data equals the one observed on real data.

    .venv/Scripts/python experiments/20261118-012-skeptic-infra/power_estimate.py
"""
import csv
import pathlib

import numpy as np
from scipy.stats import chi2

EXP7 = pathlib.Path(__file__).resolve().parents[1] / "20261118-007-f1-shared-rotation"

print("null ord var     fake_p1  fake_med  E[rate|true]  P(rate<1%|true)")
for kind in ("a", "b"):
    with open(EXP7 / f"null_{kind}.csv") as fh:
        rows = list(csv.DictReader(fh))
    for o in ("NO", "IO"):
        for var, dof in (("cont", 2), ("struct", 4)):
            v = np.sort([float(r[f"best_{var}_{o}"]) for r in rows])
            x = chi2.rvs(dof, size=200000, random_state=1)
            F = np.searchsorted(v, x, side="right") / len(v)
            print(f"({kind})  {o}  {var:6s}  {np.percentile(v, 1):7.3f}  {np.median(v):8.2f}  "
                  f"{F.mean():12.3f}  {np.mean(F < 0.01):15.4f}")

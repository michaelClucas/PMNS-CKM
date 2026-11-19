"""BSV Eq. (24), 2-3 generation limit with eps_u = eps_d = eps_e = 0: tan 2 theta_l = sin 2 theta_q / (2 sin^2 theta_q - eps),
eps = (y_b - y_tau)/y_b at M_GUT, theta_q = theta23^q(M_GUT). Uses rg_scan.csv. Writes bsv_analytic.csv."""
import csv
import numpy as np
from src import data as D
rows = list(csv.DictReader(open("rg_scan.csv")))
s23_meas = D.load_pmns("NO")["s23sq"]["value"]
out = []
for r in rows:
    tq, eps = float(r["theta23_q"]), float(r["eps_btau"])
    tl = 0.5 * np.arctan2(np.sin(2 * tq), 2 * np.sin(tq) ** 2 - eps)
    t_meas = np.arcsin(np.sqrt(s23_meas))
    eps_need = 2 * np.sin(tq) ** 2 - np.sin(2 * tq) / np.tan(2 * t_meas)
    out.append({"scenario": r["scenario"], "mb_over_mtau": float(r["mb_over_mtau"]), "eps": eps, "theta23_q": tq,
                "s23sq_lepton_BSV24": np.sin(tl) ** 2, "eps_needed_for_measured_s23sq": eps_need,
                "mb_over_mtau_needed": 1 / (1 - eps_need)})
    print(f"{r['scenario']:24s} mb/mtau={float(r['mb_over_mtau']):.4f} eps={eps:+.4f} -> s23sq(BSV24)={np.sin(tl)**2:.4f}; "
          f"needs eps={eps_need:+.4f} (mb/mtau={1/(1-eps_need):.4f})")
with open("bsv_analytic.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

"""Type-II b-tau / theta23 ridge check (Bajc-Senjanovic-Vissani mechanism) in the broad so10_II and so10_I datasets:
median and 90th percentile of sin^2 theta23^l, and fraction with sin^2 theta23 > 0.3, in bins of log10(m_tau/m_b) at M_GUT,
restricted to samples whose other charged-lepton ratios are within 0.3 dex of the data. Writes btau_ridge.json/.png."""
import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import gen
ART = HERE.parents[1] / "artifacts" / "f3"
z0 = np.array(gen.gut_reference()["z"])
bins = np.array([-1.0, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 1.0])
out = {}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=80)
for k, v in enumerate(("so10_II", "so10_I")):
    z = np.load(ART / f"ds_broad_{v}.npz")["z"].astype(float)
    m = (np.abs(z[:, 5] - z0[5]) < 0.3) & (np.abs(z[:, 6] - z0[6]) < 0.3)
    x, y = z[m, 7], z[m, 14]
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (x >= a) & (x < b)
        rows.append({"lg_mtau_mb": [a, b], "n": int(s.sum()),
                     "s23sq_median": round(float(np.median(y[s])), 4) if s.sum() else None,
                     "s23sq_q90": round(float(np.quantile(y[s], 0.9)), 4) if s.sum() else None,
                     "frac_s23sq_gt_0.3": round(float(np.mean(y[s] > 0.3)), 4) if s.sum() else None})
    out[v] = {"n_selected": int(m.sum()), "bins": rows, "observed_lg_mtau_mb_GUT": round(float(z0[7]), 3)}
    c = 0.5 * (bins[:-1] + bins[1:])
    ax[k].plot(c, [r["frac_s23sq_gt_0.3"] or 0 for r in rows], "o-")
    ax[k].axvline(z0[7], color="k", lw=1)
    ax[k].set(title=f"{v}: P(sin²θ23 > 0.3)", xlabel="log10(m_τ/m_b) at M_GUT")
fig.tight_layout(); fig.savefig(HERE / "btau_ridge.png")
(HERE / "btau_ridge.json").write_text(json.dumps(out, indent=1))
for v in out:
    print(v, out[v]["n_selected"], [(r["lg_mtau_mb"], r["n"], r["s23sq_median"], r["frac_s23sq_gt_0.3"]) for r in out[v]["bins"]])

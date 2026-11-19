"""Pull plot of the global best fits (fit_*.json) -> fit_pulls.png and fit_summary.csv."""
import json

import matplotlib
import numpy as np

import so10fit as S

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAGS = ["MN_ours", "MN_ours_nodl", "MN_ours_drrel", "FN_ours"]
rows = []
fig, ax = plt.subplots(figsize=(10, 4))
w = 0.8 / len(TAGS)
for i, t in enumerate(TAGS):
    p = S.HERE / f"fit_{t}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    pu = np.array([d["best_pulls"][n] for n in S.Z_NAMES])
    used = np.ones(17, bool)
    if d["subset"] == "nodl":
        used[15] = False
    ax.bar(np.arange(17) + (i - len(TAGS) / 2 + 0.5) * w, np.where(used, pu, 0), w,
           label=f"{t}: χ²={d['best_chi2']:.1f} (N={d['n_obs']}, k={d['n_params']})")
for t in ["MN_DR", "FN_DR"] + TAGS:
    p = S.HERE / f"fit_{t}.json"
    if p.exists():
        d = json.loads(p.read_text())
        rows.append(f"{t},{d['model']},{d['target']},{d['subset']},{d['errs']},{d['n_obs']},{d['n_params']},"
                    f"{d['best_chi2']:.4f},{d['best_IO']}")
ax.axhline(2, color="0.5", ls=":"); ax.axhline(-2, color="0.5", ls=":")
ax.set_xticks(range(17)); ax.set_xticklabels(S.Z_NAMES, rotation=60, fontsize=7)
ax.set_ylabel("pull at best fit"); ax.legend(fontsize=7)
ax.set_title("SO(10) global best fits to GUT-scale data (SM running), our errors unless 'drrel'")
fig.tight_layout(); fig.savefig(S.HERE / "fit_pulls.png", dpi=80)
(S.HERE / "fit_summary.csv").write_text("tag,model,target,subset,errs,N,k,chi2_min,IO\n" + "\n".join(rows) + "\n")
print("\n".join(rows))

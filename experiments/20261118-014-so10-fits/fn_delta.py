"""FN (real 10+126+120): δ_PMNS held out. 16 parameters are fitted to the other 16 dimensionless observables
(fit_FN_ours_nodl); list the distinct good minima and the δ_PMNS each one predicts. Also the same table for the
all-17 fit. Output: fn_delta.json.

Caveat: the minima catalogue comes from random LM starts + basin hopping around the 32 best, so the counts are
not posterior weights and minima far from the best ones are under-sampled.
"""
import json

import numpy as np

import so10fit as S

out = {}
for tag in ("FN_ours_nodl", "FN_ours"):
    p = S.ART / f"fit_{tag}.npz"
    if not p.exists():
        continue
    d = np.load(p)
    X, C = d["theta"], d["chi2"]
    prob = S.Problem("FN", "ours")
    sel = C < 4.0
    y, o = prob.predict(X[sel])
    v = S.fn_values(X[sel])
    key = np.round(np.column_stack([y[:, 15], v[:, 11], v[:, 12]]), 2)
    u, ix, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    rows = []
    for k, i, c in zip(u, ix, cnt):
        rows.append(dict(dl_deg=float(np.degrees(y[i, 15])), r=float(k[1]), s=float(k[2]), chi2_fit=float(C[sel][i]),
                         dl_pull=float(S.wrap(y[i, 15] - prob.val[15]) / prob.err[15]), n_hits=int(c),
                         s23sq=float(y[i, 14]), IO=bool(o["io"][i])))
    rows.sort(key=lambda r: r["chi2_fit"])
    dls = np.array([r["dl_deg"] for r in rows])
    out[tag] = dict(n_minima_chi2_lt4=int(sel.sum()), n_distinct=len(rows), minima=rows[:40],
                    dl_range_deg=[float(dls.min()), float(dls.max())] if len(dls) else None,
                    measured_dl_deg=float(np.degrees(prob.val[15])), sigma_deg=float(np.degrees(prob.err[15])))
    print(f"{tag}: {sel.sum()} minima with chi2<4, {len(rows)} distinct; δ_PMNS range "
          f"{out[tag]['dl_range_deg']} deg (measured {out[tag]['measured_dl_deg']:.0f} ± {out[tag]['sigma_deg']:.0f})")
    for r in rows[:12]:
        print(f"   chi2 {r['chi2_fit']:.2f} δ {r['dl_deg']:.0f} deg pull {r['dl_pull']:+.2f} r {r['r']:.1f} s {r['s']:.2f} hits {r['n_hits']}")
(S.HERE / "fn_delta.json").write_text(json.dumps(out, indent=1))

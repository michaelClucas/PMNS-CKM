"""Exp 037 design checks (no real-data rank is computed here):
  1. frozen-list hash;
  2. vectorised chi2 (TableData, translated_chi2_many) == official src scorer on random points,
     real and N2-translated universes, 4-key and 1-key;
  3. the profile procedure reproduces the exp 029 frozen chi2_today on real data for all 74 rows
     (and the symbolic predictions);
  4. timing of one fake profile per relation.
Run from this folder: ../../.venv/Scripts/python check.py > check.log
"""
import time

import numpy as np

import lib37 as L
from src import nulls as nl

rows = L.load_frozen()
print("frozen hash OK", L.FROZEN_SHA, "rows", len(rows), "relations", len({r["id"] for r in rows}))

# ---- 2. chi2 agreement
rng = np.random.default_rng(1)
maxdiff = 0.0
for o in L.ORDS:
    for kind in ("real", "N2a", "N2b"):
        u = nl.Universe.real(o) if kind == "real" else nl.n2_universe(nl.Universe.real(o), np.random.default_rng([99, len(kind), o == "NO"]))
        D = L.TableData(u, o)
        c = u.pmns_data_centre(o)
        for keys in (L.PMNS, ("s12sq",), ("s13sq",), ("s23sq",), ("dcp_deg",)):
            sc = u.scorer(keys, o)
            pts = {"s12sq": c["s12sq"] + rng.normal(0, 0.02, 40), "s13sq": c["s13sq"] + rng.normal(0, 0.001, 40),
                   "s23sq": c["s23sq"] + rng.normal(0, 0.05, 40), "dcp_deg": rng.uniform(0, 360, 40)}
            v = D.chi2(pts, keys)
            ref = np.array([sc.chi2({k: float(pts[k][i]) for k in keys}) for i in range(40)])
            fin = np.isfinite(ref)
            assert np.array_equal(fin, np.isfinite(v)), (o, kind, keys)
            maxdiff = max(maxdiff, float(np.max(np.abs(v[fin] - ref[fin]))) if fin.any() else 0.0)
            if len(keys) == 1 and kind != "real":
                w = L.translated_chi2_many(o, keys[0], float(pts[keys[0]][0]), [c[keys[0]]] * 3)
                assert abs(w[0] - v[0]) < 1e-9 or (not np.isfinite(w[0]) and not np.isfinite(v[0])), (o, keys)
print(f"vector chi2 vs official scorer: max |diff| {maxdiff:.2e}")

# ---- 3. reproduce frozen chi2_today
ctx = L.real_ctx()
worst = 0.0
for r in rows:
    rel = L.make_relation(r)
    o = r["ordering"]
    D = L.TableData(nl.Universe.real(o), o)
    if isinstance(rel, L.Symbolic):
        p = float(rel.predict_ctx(ctx))
        c2 = float(D.chi2({rel.key: np.array([p])}, rel.keys)[0])
        dp = abs(p - r["pred"][rel.key])
        line = f"pred {p:.6g} (frozen {r['pred'][rel.key]:.6g}, |d| {dp:.1e})"
    else:
        best, arg = rel.profile(ctx, [D])
        c2 = float(best[0])
        line = f"theta {np.round(arg[0], 4).tolist()} (frozen {r['params']})"
    d = c2 - float(r["chi2_today"])
    worst = max(worst, abs(d)) if not isinstance(rel, L.Fitted) else worst
    print(f"{r['id']:12s} {o} chi2 {c2:9.4f} frozen {float(r['chi2_today']):9.4f} diff {d:+.4f}  {line}")
print("max |diff| symbolic", worst)

# ---- 4. timing
D2 = [L.TableData(nl.Universe.real(o), o) for o in L.ORDS]
bank = L.fake_bank("N1", 5)
for rid in ("K-A4a-TBM", "K-A4b-BM", "K-E1", "F2LO-GR2", "BL-CDSV-I", "BL-RMSV-1"):
    r = next(x for x in rows if x["id"] == rid)
    rel = L.make_relation(r)
    t0 = time.perf_counter()
    for c in bank:
        rel.profile(c, D2)
    dt = (time.perf_counter() - t0) / len(bank)
    t0 = time.perf_counter()
    obs0 = rel.obs(rel.P0, bank[0])
    tc = time.perf_counter() - t0
    print(f"timing {rid:10s}: {dt * 1e3:7.1f} ms per fake for 2 data sets (coarse obs {tc * 1e3:.1f} ms, G={len(rel.P0)})")

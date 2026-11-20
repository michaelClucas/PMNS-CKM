"""Exp 037 design check: how much can the local refinement lower chi2 below the coarse-grid minimum?
Sets the lazy-refinement margin (a fake whose coarse minimum exceeds chi2_data + MARGIN cannot be a
hit). Uses N1 fakes j = 0..199 on the real NO and IO tables and an N2-translated table; no rank
of real data is computed. Run: ../../.venv/Scripts/python check_margin.py > check_margin.log
"""
import numpy as np

import lib37 as L
from src import nulls as nl

rows = {r["id"]: r for r in L.load_frozen()}
fitted = [rid for rid in rows if not rid.startswith("S-")]
bank = L.fake_bank("N1", 200)
Ds = [L.TableData(nl.Universe.real(o), o) for o in L.ORDS]
Ds += [L.TableData(nl.n2_universe(nl.Universe.real(), np.random.default_rng([37, 99, i])), o)
       for i, o in enumerate(L.ORDS)]
for rid in fitted:
    rel = L.make_relation(rows[rid])
    gains, near = [], []
    for c in bank:
        if not rel.valid(c):
            continue
        obs0 = rel.obs(rel.P0, c)
        best, _ = rel.profile(c, Ds, coarse=obs0)
        for D, b in zip(Ds, best):
            c0 = D.chi2(obs0, rel.keys)
            m = np.min(np.where(np.isfinite(c0), c0, np.inf))
            if np.isfinite(m) and np.isfinite(b):
                gains.append(m - b)
                if b < 60:
                    near.append(m - b)
    g = np.array(gains)
    nr = np.array(near) if near else np.array([0.0])
    print(f"{rid:12s} n {len(g):4d} gain median {np.median(g):.3f} 99% {np.quantile(g, 0.99):.3f} max {g.max():.3f}"
          f" | refined chi2 < 60: n {len(near)} max gain {nr.max():.3f}", flush=True)

"""Fast observable path vs src.yukawa.observables_batch on random G43 structures (all nu options)."""
import time
import numpy as np
import lib43 as L

rng = np.random.default_rng(7)
worst = {}
for nu in ("D", "I1", "IR", "II", "F", "P"):
    for t in range(20):
        Z = list(rng.choice(L.P_ALL, rng.integers(3, 5), replace=False))
        nz = len(Z)
        spec = {"Z": Z, "e": "P", "ce": [None if rng.uniform() < 0.3 else float(rng.choice(L.CSET)) for _ in range(nz + 1)],
                "nu": "IR" if nu == "P" else nu, "cnu": list(rng.choice(L.CSET, nz + 1)), "cR": list(rng.choice(L.CSET, nz + 1))}
        if nu == "P":
            spec["extra"] = list(rng.choice(L.NU_POS, 2, replace=False))
        lo, hi = L.layout(spec)
        X = rng.uniform(lo, hi, (50, len(lo)))
        X[:, :nz] = rng.choice([-1, 1], (50, nz)) * rng.uniform(1, 7, (50, nz))
        a = L.observe(spec, X)
        b = L.observe(spec, X, exact=True)
        for k in a:
            ok = np.isfinite(b[k]) & np.isfinite(a[k])
            if k in ("ckm_delta_rad",):
                d = np.abs(L._wrap(a[k] - b[k], 2 * np.pi))
            elif k == "dcp_deg":
                d = np.abs(L._wrap(a[k] - b[k], 360))
            else:
                d = np.abs(a[k] - b[k]) / np.maximum(np.abs(b[k]), 1e-12)
            # angles undefined when s13 ~ 0 or degenerate spectra: ignore those rows for delta
            if k in ("ckm_delta_rad", "dcp_deg"):
                s = "ckm_s13" if k == "ckm_delta_rad" else "s13sq"
                ok &= b[s] > 1e-8
            worst[k] = max(worst.get(k, 0), float(np.max(d[ok])) if ok.any() else 0)
for k, v in worst.items():
    print(f"{k:14s} max rel/abs diff {v:.2e}")
spec = {"Z": ["12", "22", "23"], "e": "S", "ce": [1.0] * 4, "nu": None}
X = np.tile([3.0, 5, 6, -4, -2, -1, 0.1, 0.2, 0.3], (10, 1))
t = time.time()
for _ in range(200):
    L.observe(spec, X)
print("fast observe B=10 ms", (time.time() - t) * 5)

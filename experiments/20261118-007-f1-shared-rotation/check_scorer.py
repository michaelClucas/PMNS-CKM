"""Check the vectorized TableChi2 against src.data.pmns_delta_chi2 on random points."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from src import data
from f1lib import TableChi2

rng = np.random.default_rng(7)
for o in ("NO", "IO"):
    f = TableChi2(o)
    pts = np.column_stack([rng.uniform(0.15, 0.85, 4000), rng.uniform(-0.01, 0.08, 4000),
                           rng.uniform(0.2, 0.8, 4000), rng.uniform(-400, 400, 4000)])
    v = f(*pts.T)
    ref = np.array([data.pmns_delta_chi2(o, s12sq=a, s13sq=b, s23sq=c, dcp_deg=d) for a, b, c, d in pts])
    fin = np.isfinite(ref)
    assert np.array_equal(fin, np.isfinite(v)), o
    err = np.max(np.abs(v[fin] - ref[fin]))
    print(f"{o}: {fin.sum()} finite points, max |vec - loader| = {err:.2e}")
    assert err < 1e-9

# fast_observables vs extract_angles-based observables on random A Phi B products
from f1lib import fast_observables, observables
from src.haar import haar_unitary
U = haar_unitary(rng, 20000)
U[:50] = np.eye(3)[[1, 0, 2]]  # degenerate cases
U[50:100] = np.eye(3)[[0, 2, 1]]
ref = np.array(observables(U))
fast = np.array(fast_observables(*U[:, :2].reshape(-1, 6).T))
dd = np.abs((fast[3] - ref[3] + 180) % 360 - 180)
print("fast_observables max diff: angles", np.max(np.abs(fast[:3] - ref[:3])), "delta(deg)", dd.max())
assert np.max(np.abs(fast[:3] - ref[:3])) < 1e-10 and dd.max() < 1e-6

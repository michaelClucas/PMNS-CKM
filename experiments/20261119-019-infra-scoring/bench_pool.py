"""Shared pool vs one pool per injection in src.nulls.power_check (2 workers).

Run from the repo root: .venv/Scripts/python experiments/20261119-019-infra-scoring/bench_pool.py
Uses the toy pipeline of tests/test_nulls_019.py (fit-only rule, limit 0.3 so 24 fakes can pass).
"""
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from src import nulls  # noqa: E402
from src.nulls import FIT_ONLY, Universe, make_injector, power_check, run_null  # noqa: E402
from test_nulls_019 import planted_k3_pipeline  # noqa: E402
from test_nulls import planted_pred  # noqa: E402


def per_injection_pools(pipeline, inj, n_inject, n_fakes, seed, **kw):
    """The pre-019 behaviour: run_null creates and tears down its own pool for each injection."""
    import numpy as np
    base = Universe.real("NO")
    out = []
    for j in range(n_inject):
        u = inj(np.random.default_rng([seed, 3, 0, j]), base)
        ref = nulls.summarize(pipeline(u))
        if not FIT_ONLY["ok_fn"](ref):
            out.append(None)
            continue
        out.append(run_null(pipeline, "N1", "NO", n_fakes, ref=ref, base=u, seed=[seed, 1000 + j], workers=2,
                            hit_fn=FIT_ONLY["hit_fn"], limit=0.3))
    return out


if __name__ == "__main__":
    inj = make_injector(planted_pred(Universe.real().context()))
    t = time.time()
    power_check(planted_k3_pipeline, inj, n_inject=6, n_fakes=24, seed=19, workers=2, limit=0.3, **FIT_ONLY)
    shared = time.time() - t
    t = time.time()
    per_injection_pools(planted_k3_pipeline, inj, 6, 24, 19)
    per = time.time() - t
    print(f"6 injections x 24 fakes, 2 workers: shared pool {shared:.1f} s, pool per injection {per:.1f} s")

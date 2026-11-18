"""Timing benchmark for the core library. Run from repo root:
    python experiments/20261118-004-core-lib/bench.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

from src.haar import haar_unitary, sample_haar_angles  # noqa: E402
from src.mixing import extract_angles, jarlskog, rephase_to_standard, standard_matrix  # noqa: E402


def t(label, f, reps=3):
    best = min(_time(f) for _ in range(reps))
    print(f"{label:45s} {best:8.3f} s")
    return best


def _time(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


rng = np.random.default_rng(0)
N = 1_000_000
print(f"numpy {np.__version__}, N = {N}")
U = haar_unitary(rng, N)
ang = sample_haar_angles(rng, N)
t("haar_unitary(1e6)", lambda: haar_unitary(rng, N))
t("sample_haar_angles(1e6)", lambda: sample_haar_angles(rng, N))
t("standard_matrix(1e6)", lambda: standard_matrix(*ang))
t("extract_angles(1e6)", lambda: extract_angles(U))
t("jarlskog(1e6)", lambda: jarlskog(U))
t("rephase_to_standard(1e5)", lambda: rephase_to_standard(U[:100_000]), reps=1)

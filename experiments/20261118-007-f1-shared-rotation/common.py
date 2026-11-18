"""Shared setup: repo path, configuration list and deduplication."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from f1lib import build_T, fingerprints, raw_configs  # noqa: E402
from src.haar import haar_unitary  # noqa: E402

DEDUP_SEED = 0


def configs():
    """(raw, distinct, class_of_raw). Distinct = first raw config of each fingerprint class,
    fingerprints computed with a fixed Haar-random V (seed DEDUP_SEED) so the classes are
    structural, not tied to the measured CKM."""
    raw = raw_configs()
    fp = fingerprints(build_T(haar_unitary(np.random.default_rng(DEDUP_SEED)), raw))
    cls, rep = {}, []
    class_of = []
    for i, h in enumerate(fp):
        if h not in cls:
            cls[h] = len(rep)
            rep.append(i)
        class_of.append(cls[h])
    return raw, [raw[i] for i in rep], class_of

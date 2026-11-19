"""Timing of the exp 013 performance changes (1 core). Writes bench.log next to this file.

    .venv/Scripts/python experiments/20261118-013-infra-hardening/bench.py

Old implementations are rebuilt inline (no edits to src): the pre-013 pmns_delta_chi2
path is tests/test_data_profile.py::_ref_pmns; the uncached ckm_at is ckm_at(cache=False).
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src import data, evaluate as ev, rg  # noqa: E402
from src.nulls import Universe, run_null  # noqa: E402
from test_data_profile import _ref_pmns  # noqa: E402
from test_nulls import planted_pipeline, _planted_universe  # noqa: E402


def timeit(f, n):
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


def main():
    out = []
    p = dict(s12sq=0.3, s13sq=0.022, s23sq=0.5, dcp_deg=200.0)
    t_new = timeit(lambda: data.pmns_delta_chi2("NO", **p), 20000)
    t_old = timeit(lambda: _ref_pmns("NO", **p), 20000)
    out.append(f"pmns_delta_chi2 joint (2D + two 1D): old {t_old * 1e6:.1f} us, new {t_new * 1e6:.1f} us, x{t_old / t_new:.2f}")
    v = np.random.default_rng(0).uniform(0.2, 0.4, 100000)
    t_vec = timeit(lambda: data.pmns_profile_1d("NO", "s12sq", v), 20)
    t_sca = timeit(lambda: [data.pmns_delta_chi2("NO", s12sq=float(a)) for a in v[:5000]], 3) / 5000 * len(v)
    out.append(f"1D profile on 1e5 points: scalar loop {t_sca:.3f} s, pmns_profile_1d {t_vec * 1e3:.2f} ms, "
               f"x{t_sca / t_vec:.0f}")
    t0 = time.perf_counter()
    rg.ckm_at(2e16)
    t_first = time.perf_counter() - t0
    t_unc = timeit(lambda: rg.ckm_at(2e16, cache=False), 10)
    t_c = timeit(lambda: rg.ckm_at(2e16), 1000)
    out.append(f"ckm_at(2e16): first call in process {t_first:.3f} s (includes the y_t solve), "
               f"uncached repeat {t_unc * 1e3:.1f} ms, memoised {t_c * 1e6:.1f} us")
    t_ev = timeit(lambda: ev.evaluate(p, "NO", use=ev.PMNS_KEYS), 2000)
    out.append(f"evaluate (4 PMNS keys, Scorer built per call): {t_ev * 1e6:.0f} us")
    u = _planted_universe()
    t0 = time.perf_counter()
    r = run_null(planted_pipeline, "N1", "NO", n_fakes=1000, base=u, seed=11)
    out.append(f"run_null toy pipeline (2 relations), 1000 N1 fakes, 1 process: {time.perf_counter() - t0:.1f} s, "
               f"{r.n_hits} hits, passed={r.passed}")
    text = "\n".join(out)
    (HERE / "bench.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

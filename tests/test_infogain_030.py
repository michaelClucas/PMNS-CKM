"""src.infogain.log_ratio_rank (exp 030) against exp 025 results.json (flow log-ratio statistic, N1 fakes)."""
import json
import math
import pathlib

import numpy as np
import pytest

from src import infogain as ig

RES = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "20261119-025-nn-ckm-info" / "results.json"


def test_reproduces_exp025_point_rank():
    d = json.loads(RES.read_text())["main"]
    ref = d["rank_point_all"]
    fakes = np.array(d["fake_T_point_bits"])  # stored rounded to 1e-4 bits
    r = ig.log_ratio_rank(d["T_point_bits"], fakes, unit="bits")
    assert r["n"] == ref["n"] == 2000
    assert r["hits"] == ref["n_fakes_ge_real"] == 63
    assert r["rate"] == pytest.approx(ref["frac"])
    assert r["rate_ci95"] == pytest.approx(ref["cp95"], rel=1e-12)
    assert r["S_rank"] == pytest.approx(ref["S_rank_bits"], rel=1e-12)
    assert r["B_ckm"] == pytest.approx(ref["B_ckm_bits"], abs=2e-4)  # fakes rounded to 1e-4 bits
    assert r["fake_T_bits_median"] == pytest.approx(ref["fake_T_bits_median"], abs=1e-4)
    assert r["fake_T_bits_q05_q95"] == pytest.approx(ref["fake_T_bits_q05_q95"], abs=1e-4)
    for tag in ("drop", "seed1"):  # other tags only store the summary; the main fakes do not apply
        assert tag in json.loads(RES.read_text())


def test_equals_null_info_with_minus_2T_and_units():
    rng = np.random.default_rng(5)
    fakes = rng.normal(0, 1.5, 500)
    real = 1.9
    a = ig.log_ratio_rank(real, fakes, unit="nats")
    b = ig.null_info(-2 * real, -2 * fakes)
    for k in ("n", "hits", "rate", "S_rank", "S_rank_max_ci"):
        assert a[k] == b[k]
    assert a["rate_ci95"] == b["rate_ci95"]
    assert a["B_ckm"] == pytest.approx(b["B_ckm"], abs=1e-12)
    c = ig.log_ratio_rank(real / math.log(2), fakes / math.log(2), unit="bits")
    assert c["B_ckm"] == pytest.approx(a["B_ckm"], abs=1e-12) and c["hits"] == a["hits"]
    assert ig.log_ratio_rank(1.0, [np.nan, 0.5, 2.0])["n"] == 2
    with pytest.raises(ValueError):
        ig.log_ratio_rank(1.0, [np.inf, np.nan])
    with pytest.raises(ValueError):
        ig.log_ratio_rank(1.0, [0.0], unit="dB")

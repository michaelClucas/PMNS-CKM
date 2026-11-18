"""Tests for the exp 012 skeptic audit. Run from the repo root:

    .venv/Scripts/python -m pytest -q experiments/20261118-012-skeptic-infra/test_infra_findings.py

Tests marked xfail(strict=True) state the CORRECT behaviour and currently fail, so they
demonstrate a bug. Once the bug is fixed they pass, pytest reports XPASS(strict) as a
failure, and the marker should be removed. `_demo` tests pass today and show the exploit.
Unmarked non-demo tests check behaviour that is already right.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from src import data, evaluate as ev  # noqa: E402
from src.baselines import BASELINES, data_context  # noqa: E402
from src.stats import fake_universe_rate  # noqa: E402

from leak_probe import leak_probe  # noqa: E402

CKM = data.load_ckm()
VUS_ALIASES = ("ckm_lambda", "ckm_s12", "ckm_Vus", "ckm_Vcd", "ckm_direct_Vus", "ckm_direct_Vcd")


# ------------------------------------------------------------ F1: N inflation by aliases

def _one_param_vus_model(th):
    """A 'relation' with one fitted number x that predicts every alias of |V_us|."""
    return {k: th["x"] for k in VUS_ALIASES}


def test_F1_alias_observables_fake_compression_demo():
    """1 fitted number, N = 6: passes k <= N - 2, p > 0.05 and max|pull| < 2 today."""
    r = ev.fit_and_evaluate(_one_param_vus_model, {"x": (0.2, 0.25)}, "NO", VUS_ALIASES, n_starts=4)
    res = r.result
    assert res.N == 6 and res.k == 1
    assert res.k <= res.N - 2 and res.p > 0.05 and res.max_abs_pull < 2


@pytest.mark.xfail(strict=True, reason="F1: evaluate does not reject redundant (aliased) observables")
def test_F1_alias_observables_rejected():
    with pytest.raises(ValueError):
        ev.evaluate({k: 0.2252 for k in VUS_ALIASES}, "NO", use=VUS_ALIASES)


@pytest.mark.xfail(strict=True, reason="F1: m_c, m_s and m_c_over_m_s scored together count one ratio twice")
def test_F1_mass_ratio_alias_rejected():
    m = data.load_masses()
    pred = {k: m[k]["value"] for k in ("m_c", "m_s", "m_c_over_m_s")}
    with pytest.raises(ValueError):
        ev.evaluate(pred, "NO", use=tuple(pred))


# ------------------------------------------------------------ F2: input/prediction alias

def test_F2_input_alias_demo():
    s12 = CKM["s12"]["value"]
    r = ev.evaluate({"ckm_Vus": s12, "ckm_lambda": s12}, "NO", inputs=("ckm_s12",))
    assert r.N == 2 and r.k == 0 and r.chi2 < 1e-12 and r.p > 0.99


@pytest.mark.xfail(strict=True, reason="F2: overlap check is by name; ckm_s12 input vs ckm_Vus prediction passes")
def test_F2_input_alias_rejected():
    s12 = CKM["s12"]["value"]
    with pytest.raises(ValueError):
        ev.evaluate({"ckm_Vus": s12, "ckm_lambda": s12}, "NO", inputs=("ckm_s12",))


# ------------------------------------------------------------ F3: undeclared data leakage

def _leaky_pmns_model():
    """Reads the NuFIT best fit through the official loader and 'predicts' it."""
    return ev.pmns_best("NO")


def test_F3_leak_accepted_by_evaluate_demo():
    r = ev.evaluate(_leaky_pmns_model(), "NO", k=0)
    assert r.N == 4 and r.k == 0 and r.p > 0.99  # a perfect 'k = 0, N = 4' relation


@pytest.mark.xfail(strict=True, reason="F3: evaluate cannot detect undeclared data reads")
def test_F3_evaluate_detects_leak():
    with pytest.raises(ValueError):
        ev.evaluate(_leaky_pmns_model(), "NO", k=0, inputs=())


def test_F3_leak_probe_flags_leaky_model():
    keys = {k for k, _ in leak_probe(_leaky_pmns_model, files=("pmns.json",))}
    assert {"pmns_NO_s12sq", "pmns_NO_s13sq", "pmns_NO_s23sq", "pmns_NO_dcp_deg"} <= keys


def test_F3_leak_probe_clean_for_declared_inputs():
    b = BASELINES["A1s"]  # QLC solar, declares ckm_s12
    assert leak_probe(lambda: b.model({}, data_context()), declared_inputs=b.inputs) == []


def test_F3_leak_probe_flags_undeclared_read():
    b = BASELINES["A1s"]
    flagged = leak_probe(lambda: b.model({}, data_context()), declared_inputs=())
    assert [k for k, _ in flagged] == ["ckm_s12"]


# ------------------------------------------------------------ F4: use=None scores echoed data

@pytest.mark.xfail(strict=True, reason="F4: with use=None, echoed data keys are silently scored and counted in N")
def test_F4_use_none_does_not_count_echoed_inputs():
    pred = {"s12sq": 0.3, "ckm_s12": CKM["s12"]["value"]}  # model echoes its CKM input
    assert ev.evaluate(pred, "NO").N == 1


# ------------------------------------------------------------ F5: discrete-choice counting

@pytest.mark.xfail(strict=True, reason="F5: A4 picks 1 of 5 X x 3 orders = 15 options with k_d = 2; "
                                       "exp 007 counts log2(options) bits; conventions disagree")
def test_F5_k_discrete_counts_bits():
    assert BASELINES["A4b-BM"].k_discrete >= math.ceil(math.log2(5 * 3))


# ------------------------------------------------------------ F6: ordering choice

def test_F6_ordering_choice_not_counted_demo():
    """TM1: within-ordering p is 0.024 (NO) and 0.18 (IO); IO p_global is 0.07. Nothing in
    EvalResult records that IO was picked after seeing both."""
    b = BASELINES["B2-TM1"]
    no = ev.evaluate_baseline(b, "NO", n_starts=8).result
    io = ev.evaluate_baseline(b, "IO", n_starts=8).result
    assert no.p < 0.05 < io.p
    assert io.p_global < io.p
    assert io.k == no.k


# ------------------------------------------------------------ F9: CKM phase not wrapped

@pytest.mark.xfail(strict=True, reason="F9: ckm_delta_rad is a plain Gaussian, a 2 pi shift is penalized")
def test_F9_ckm_delta_wraps():
    d = CKM["delta_rad"]["value"]
    a = ev.evaluate({"ckm_delta_rad": d}, "NO").chi2
    b = ev.evaluate({"ckm_delta_rad": d + 2 * np.pi}, "NO").chi2
    assert np.isclose(a, b)


# ------------------------------------------------------------ criterion 4 arithmetic

def test_C4_clopper_pearson_thresholds():
    assert fake_universe_rate(0, 100)["upper"] > 0.029  # 100 fakes cannot certify < 1%
    assert fake_universe_rate(0, 298)["upper"] > 0.01
    assert fake_universe_rate(0, 300)["upper"] < 0.01
    assert fake_universe_rate(1, 300)["upper"] > 0.015  # one hit in 300 fails
    assert fake_universe_rate(4, 1000)["upper"] < 0.01
    assert fake_universe_rate(5, 1000)["upper"] > 0.01


# ------------------------------------------------------------ edge cases that are right

def test_delta_wrap_equivalence():
    for o in ("NO", "IO"):
        base = dict(s12sq=0.31, s13sq=0.0222, s23sq=0.5)
        a = ev.evaluate({**base, "dcp_deg": 210.0}, o).chi2
        assert np.isclose(a, ev.evaluate({**base, "dcp_deg": -150.0}, o).chi2)
        assert np.isclose(a, ev.evaluate({**base, "dcp_deg": 570.0}, o).chi2)


def test_octant_both_allowed_and_out_of_table_is_excluded():
    assert ev.evaluate({"s23sq": 0.45}, "NO").chi2 < 4
    assert ev.evaluate({"s23sq": 0.56}, "NO").chi2 < 4  # second octant not killed by a Gaussian
    r = ev.evaluate({"s23sq": 0.80}, "NO")
    assert math.isinf(r.chi2) and r.p == 0.0


def test_io_offset_added_once():
    r = ev.evaluate(ev.pmns_best("IO"), "IO")
    assert np.isclose(r.chi2_global - r.chi2, data.ordering_offset("IO"))
    assert data.ordering_offset("NO") == 0.0
    r2 = ev.evaluate({"ckm_Vus": CKM["Vus"]["value"]}, "IO")
    assert r2.chi2_global == r2.chi2  # no PMNS term, no offset

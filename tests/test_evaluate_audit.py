"""Exp 012 audit findings, fixed in exp 013. The first seven tests are the former strict
xfails of experiments/20261118-012-skeptic-infra/test_infra_findings.py, adapted to the
fixed API (use is explicit; leaks are caught by evaluate_model)."""
import math

import numpy as np
import pytest
from scipy import stats as st

from src import data, evaluate as ev
from src.baselines import BASELINES
from src.stats import discrete_bits, p_value

CKM = data.load_ckm()
VUS_ALIASES = ("ckm_lambda", "ckm_s12", "ckm_Vus", "ckm_Vcd", "ckm_direct_Vus", "ckm_direct_Vcd")
PMNS4 = ev.PMNS_KEYS


def _leaky_pmns_model():
    """Reads the NuFIT best fit through the official loader and 'predicts' it."""
    return ev.pmns_best("NO")


# ------------------------------------------------------------ former xfails (exp 012)


def test_F1_alias_observables_rejected():  # H2
    with pytest.raises(ValueError):
        ev.evaluate({k: 0.2252 for k in VUS_ALIASES}, "NO", use=VUS_ALIASES)


def test_F1_mass_ratio_alias_rejected():  # H2
    m = data.load_masses()
    pred = {k: m[k]["value"] for k in ("m_c", "m_s", "m_c_over_m_s")}
    with pytest.raises(ValueError):
        ev.evaluate(pred, "NO", use=tuple(pred))


def test_F2_input_alias_rejected():  # M1
    s12 = CKM["s12"]["value"]
    with pytest.raises(ValueError):
        ev.evaluate({"ckm_Vus": s12}, "NO", use=("ckm_Vus",), inputs=("ckm_s12",))
    with pytest.raises(ValueError):
        ev.evaluate({"ckm_Vus": s12, "ckm_lambda": s12}, "NO", use=("ckm_Vus", "ckm_lambda"),
                    inputs=("ckm_s12",))


def test_F3_evaluate_detects_leak():  # H3
    with pytest.raises(ValueError):
        ev.evaluate_model(_leaky_pmns_model, "NO", use=PMNS4, k=0, inputs=())
    with pytest.raises(ev.LeakError):
        ev.evaluate_model(_leaky_pmns_model, "NO", use=PMNS4, k=0, inputs=())


def test_F4_use_none_does_not_count_echoed_inputs():  # M6
    pred = {"s12sq": 0.3, "ckm_s12": CKM["s12"]["value"]}  # model echoes its CKM input
    with pytest.raises(ValueError, match="use=None"):
        ev.evaluate(pred, "NO")
    assert ev.evaluate(pred, "NO", use=("s12sq",), inputs=("ckm_s12",)).N == 1


def test_F5_k_discrete_counts_bits():  # M3
    assert BASELINES["A4b-BM"].k_discrete >= math.ceil(math.log2(5 * 3))


def test_F9_ckm_delta_wraps():  # L1
    d = CKM["delta_rad"]["value"]
    a = ev.evaluate({"ckm_delta_rad": d}, "NO", use=("ckm_delta_rad",)).chi2
    b = ev.evaluate({"ckm_delta_rad": d + 2 * np.pi}, "NO", use=("ckm_delta_rad",)).chi2
    c = ev.evaluate({"ckm_delta_rad": d - 0.05 - 4 * np.pi}, "NO", use=("ckm_delta_rad",))
    assert np.isclose(a, b)
    assert c.pulls["ckm_delta_rad"] == pytest.approx(-0.05 / CKM["delta_rad"]["err_minus"])


# ------------------------------------------------------------ H2 / M1 alias rules


def test_ckm_global_rules():
    ok = ev.evaluate({"ckm_lambda": 0.225, "ckm_A": 0.83, "ckm_rhobar": 0.16, "ckm_etabar": 0.36}, "NO",
                     use=("ckm_lambda", "ckm_A", "ckm_rhobar", "ckm_etabar"))
    assert ok.N == 4
    ev.check_observables(("ckm_s12", "ckm_s23", "ckm_s13", "ckm_delta_rad"))
    with pytest.raises(ev.AliasError, match="max 4"):
        ev.check_observables(("ckm_lambda", "ckm_A", "ckm_rhobar", "ckm_etabar", "ckm_J"))
    with pytest.raises(ev.AliasError, match="not independent"):
        ev.check_observables(("ckm_rhobar", "ckm_etabar", "ckm_delta_rad"))
    with pytest.raises(ev.AliasError, match="not independent"):  # J is fixed by the 4 inputs
        ev.check_observables(("ckm_J",), ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad"))
    for pair in (("ckm_s23", "ckm_Vcb"), ("ckm_s13", "ckm_Vub"), ("ckm_A", "ckm_Vts")):
        with pytest.raises(ev.AliasError):
            ev.check_observables(pair)


def test_global_and_direct_never_mixed():
    with pytest.raises(ev.AliasError, match="cannot be mixed"):
        ev.check_observables(("ckm_Vub", "ckm_direct_Vcb"))
    with pytest.raises(ev.AliasError, match="cannot be mixed"):
        ev.check_observables(("ckm_direct_Vcb",), ("ckm_s12",))
    ev.check_observables(("ckm_direct_Vus", "ckm_direct_Vcd"))  # independent measurements


def test_mass_alias_rules():
    ev.check_observables(("m_c", "m_s", "m_tau"))
    for bad in (("m_t_direct", "m_t_pole_xsec"), ("m_b_over_m_c", "m_b_over_m_s"), ("m_u", "m_u_over_m_d")):
        with pytest.raises(ev.AliasError):
            ev.check_observables(bad)
    with pytest.raises(ev.AliasError):
        ev.check_observables(("m_c_over_m_s",), ("m_c",))


def test_every_baseline_passes_alias_rules():
    for b in BASELINES.values():
        ev.check_observables(b.predicts, b.inputs)


def test_scorer_rechecks_inputs_given_at_call():
    sc = ev.Scorer(("ckm_Vus",), "NO")
    with pytest.raises(ev.AliasError):
        sc({"ckm_Vus": 0.225}, inputs=("ckm_s12",))


# ------------------------------------------------------------ M2 ordering bit


def test_best_ordering_adds_one_bit_and_uses_p_global():
    b = BASELINES["B2-TM1"]
    no = ev.evaluate_baseline(b, "NO", n_starts=8).result
    io = ev.evaluate_baseline(b, "IO", n_starts=8).result
    best = ev.evaluate_baseline(b, "best", n_starts=8).result
    assert best.ordering == min((no, io), key=lambda r: r.chi2_global).ordering == "IO"
    assert best.ordering_selected and best.k == io.k + 1 and best.k_discrete == io.k_discrete + 1
    assert best.k_continuous == io.k_continuous == 2
    assert best.chi2_global == pytest.approx(io.chi2_global, abs=1e-6)
    assert best.p_crit3 == best.p_global == pytest.approx(st.chi2.sf(best.chi2_global, best.N - 2))
    assert best.p_global < io.p  # the IO penalty is charged


def test_ordering_selected_flag_and_pred_best():
    pred = ev.pmns_best("IO")
    r = ev.evaluate(pred, "IO", use=PMNS4, ordering_selected=True)
    assert r.k == 1 and r.k_discrete == 1 and r.k_continuous == 0 and r.dof == 4
    rb = ev.evaluate(pred, "best", use=PMNS4)
    assert rb.ordering_selected and rb.k == 1
    rm = ev.evaluate({"m_tau": 1.777}, "best", use=("m_tau",))  # no PMNS key: no ordering choice
    assert rm.k == 0 and not rm.ordering_selected


# ------------------------------------------------------------ M3 counting convention


def test_discrete_bits_and_integer_counts():
    assert [discrete_bits(n) for n in (1, 2, 3, 4, 5, 15, 16, 17, 2784)] == [0, 1, 2, 2, 3, 4, 4, 5, 12]
    with pytest.raises(ValueError):
        discrete_bits(0)
    assert p_value(3.0, 4, 2.0) == pytest.approx(st.chi2.sf(3.0, 2))
    with pytest.raises(ValueError, match="not an integer"):
        p_value(3.0, 4, 1.5)  # used to truncate silently
    with pytest.raises(ValueError, match="not an integer"):
        ev.evaluate({"s12sq": 0.3}, "NO", use=("s12sq",), k=0.5)


def test_registry_k_discrete_from_options():
    for b in BASELINES.values():
        assert b.n_options >= 1, b.id
        assert b.k_discrete == discrete_bits(b.n_options), b.id


def test_p_uses_continuous_dof_and_criteria_flags():
    best = ev.pmns_best("NO")
    r = ev.evaluate(best, "NO", use=PMNS4, k=3, k_discrete=1)
    assert r.dof == 2 and r.p == pytest.approx(p_value(r.chi2, 4, 2))
    assert not r.compression_ok and r.fit_ok
    r = ev.evaluate(best, "NO", use=PMNS4, k=2, k_discrete=2)
    assert r.compression_ok and r.dof == 4


# ------------------------------------------------------------ H3 leak check in the fit path


def test_fit_and_evaluate_leak_check():
    b = BASELINES["A1s"]
    from src.baselines import data_context
    model = lambda th: b.model(th, data_context())  # noqa: E731  reads ckm_s12 at call time
    ev.fit_and_evaluate(model, {}, "NO", b.predicts, inputs=b.inputs, leak_check=True)
    with pytest.raises(ev.LeakError):
        ev.fit_and_evaluate(model, {}, "NO", b.predicts, inputs=(), leak_check=True)

import numpy as np

from src import data, evaluate as ev, rg
from src.baselines import BASELINES, data_context
from src.leakprobe import leak_probe, record_key


def _leaky_pmns_model():
    return ev.pmns_best("NO")


def test_flags_leaky_model():
    flagged = leak_probe(_leaky_pmns_model, files=("pmns.json",))
    keys = {f.key for f in flagged}
    assert {"s12sq", "s13sq", "s23sq", "dcp_deg"} <= keys
    assert all("/NO/" in f.record and "IC23_woSKatm" in f.record for f in flagged)


def test_qlc_clean_with_declared_input():
    b = BASELINES["A1s"]  # QLC solar, declares ckm_s12
    assert leak_probe(lambda: b.model({}, data_context()), declared_inputs=b.inputs) == []


def test_qlc_undeclared_read_flagged():
    b = BASELINES["A1s"]
    flagged = leak_probe(lambda: b.model({}, data_context()), declared_inputs=())
    assert [f.key for f in flagged] == ["ckm_s12"]
    assert flagged[0].moved == ("s12sq",)


def test_alias_of_declared_input_is_flagged():
    def model():
        return {"s12sq": data.load_ckm()["Vus"]["value"]}
    flagged = leak_probe(model, declared_inputs=("ckm_s12",), files=("ckm.json",))
    assert [f.key for f in flagged] == ["ckm_Vus"]


def test_cache_mediated_leak_is_caught_and_caches_restored():
    before = rg.quark_masses_mz()["m_s"]

    def model():  # reads m_s through rg's lru cache
        return {"x": rg.quark_masses_mz()["m_s"]}
    flagged = leak_probe(model, files=("masses.json",))
    assert "m_s" in {f.key for f in flagged}
    assert rg.quark_masses_mz()["m_s"] == before
    assert data.load_masses()["m_s"]["value"] == data._json.__wrapped__("masses.json")["quarks"]["m_s"]["value"]


def test_record_keys():
    assert record_key("ckm.json", ("magnitudes_direct", "Vus")) == "ckm_direct_Vus"
    assert record_key("ckm.json", ("wolfenstein", "lambda")) == "ckm_lambda"
    assert record_key("pmns.json", ("analyses", "IC23_woSKatm", "NO", "s12sq")) == "s12sq"
    assert record_key("masses.json", ("quarks", "m_c")) == "m_c"


def test_probe_leaves_scoring_unchanged():
    pred = {"s12sq": 0.3, "ckm_Vus": 0.2245}
    r0 = ev.evaluate(pred, "NO", use=tuple(pred))
    leak_probe(lambda: {"y": data.load_ckm()["Vus"]["value"]})
    r1 = ev.evaluate(pred, "NO", use=tuple(pred))
    assert r0.chi2 == r1.chi2 and np.isfinite(r1.chi2)

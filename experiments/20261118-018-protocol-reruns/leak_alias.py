"""Step 2: leak probe and alias check on each lead's model function (best-fit choices frozen)."""
import json
import time

import numpy as np

import pipelines as PL
from src import evaluate as ev
from src.leakprobe import leak_probe
from src.nulls import Universe

T1_J = 9412           # sin(3 sqrt(x)), up mapping, GUT (pipeline result on real data, repro.json)
F2_CFG = "GR2|c12=1|c23=1|c13=0|std"
F2_PHI = np.radians(287.08006786817066)


def t1_model():
    import symlib as S
    base = PL.t1_base(Universe.real("NO"), "GUT")
    lst = PL._t1_mapped(base, np.zeros((1, 10)))["up"]
    op, a, b, comp = PL.t1_templates()
    out = {}
    for k in (0, 1):
        v, _ = S.evaluate(op, a, b, comp, np.r_[lst[k][0], 0, 0, 0, 1, 1, 1, 1, 1, 1])
        out[PL.T1_ANG[k]] = float(np.sin(v[T1_J]) ** 2)
    return out


def f2_model():
    import f2lib as L
    ctx = Universe.real("NO").context()
    cfg = next(c for c in L.configs() if c["name"] == F2_CFG)
    U = L.full_matrix(cfg, (ctx["theta_C"], ctx["theta_ub"], ctx["theta_cb"], ctx["delta_q"]), F2_PHI)
    s12, s13, s23, d = L.obs_exact(U)
    return {"s12sq": s12, "s13sq": s13, "s23sq": s23, "dcp_deg": d}


def run(name, fn, declared, use, inputs):
    t0 = time.time()
    flags = leak_probe(fn, declared_inputs=declared)
    rows = [{"key": f.key, "record": f.record, "moved": list(f.moved)} for f in flags]
    pm = [r for r in rows if r["record"].startswith("pmns.json")]
    # size of each movement (relative), to tell numerical crosstalk from real dependence
    try:
        ev.check_observables(use, inputs)
        alias = "ok"
    except Exception as e:  # noqa: BLE001
        alias = f"{type(e).__name__}: {e}"
    return {"model": name, "pred": fn(), "declared": list(declared), "n_flags": len(rows), "flags": rows,
            "pmns_flags": pm, "alias_check": {"use": list(use), "inputs": list(inputs), "result": alias},
            "sec": time.time() - t0}


def magnitude(fn, record_path, n_sigma=1.0):
    """Relative output change when one record is shifted (same mechanism as leak_probe)."""
    import copy
    import json as js
    from src import data as D
    from src import leakprobe as LP
    f, *path = record_path.split("/")
    raw = js.load(open(D.DATA / f))
    mod = copy.deepcopy(raw)
    node = mod
    for p in path:
        node = node[p]
    node["value"] = node["value"] + n_sigma * abs(node["err_plus"]) + 1e-9 * abs(node["value"]) + 1e-300
    orig = D._json
    LP._clear_src_caches()
    ref = fn()
    D._json = lambda n: mod if n == f else orig(n)
    try:
        LP._clear_src_caches()
        out = fn()
    finally:
        D._json = orig
        LP._clear_src_caches()
    return {k: float(abs(out[k] / ref[k] - 1)) for k in ref}


def main():
    res = {}
    res["T1"] = run("T1", t1_model, ("m_u", "m_c", "m_t_direct"), ("s12sq", "s13sq"), PL.T1_INPUTS["up"])
    for r in res["T1"]["flags"]:
        r["rel_change"] = magnitude(t1_model, r["record"])
    res["F2LO"] = run("F2LO", f2_model, PL.F2_INPUTS, PL.PMNS4, PL.F2_INPUTS)
    for r in res["F2LO"]["flags"]:
        r["rel_change"] = magnitude(f2_model, r["record"])
    (PL.HERE / "leak_alias.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v["pred"], "flags", v["n_flags"], "pmns flags", len(v["pmns_flags"]), "alias", v["alias_check"]["result"])
        for r in v["flags"]:
            print("   ", r["key"], r["record"], {a: f"{b:.2e}" for a, b in r["rel_change"].items()})


if __name__ == "__main__":
    main()

"""Exp 041: lepton predictions of every grammar entry at the quark-only best fit (filter.json,
seed 1; all seeds converge to the same point). Parameter-free given the quark sector. -> quarkpoint.json"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grammar as g  # noqa: E402

F = json.load(open(HERE / "filter.json"))
th = F["quark_fits"][0]["params"]
out = dict(quark_params=th, quark_chi2=F["quark_fits"][0]["chi2"],
           Yd_23_block=dict(x=10 ** th["lxd"], w=g.w_of(th["vd"], 10 ** th["lxd"])), entries=[])
for e in g.ENTRIES:
    try:
        o = g.model(th, e)
    except ValueError as ex:
        out["entries"].append(dict(eid=e["id"], entry=g.name(e), invalid=str(ex)))
        continue
    out["entries"].append(dict(eid=e["id"], entry=g.name(e), in_G8=g.in_g8(e),
                               **{k: o[k] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio", "mr_e_mu",
                                                    "mr_mu_tau", "R_clebsch")}))
json.dump(out, open(HERE / "quarkpoint.json", "w"), indent=1)
print(out["Yd_23_block"])

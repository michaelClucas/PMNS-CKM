"""Logic checks of src/data.py against the committed tables and the NuFIT parameter table.
Run from repo root: .venv/Scripts/python experiments/20261118-005-data-verify/verify_loader.py
"""
import functools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src import data as D  # noqa: E402
from src.stats import Objective  # noqa: E402

ok_all = True


def check(name, cond, info=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")


for an in D.ANALYSES:
    for o in D.ORDERINGS:
        tag = f"{an} {o}"
        # grid ranges
        x, y = D._profile_1d(an, o, "DCP")
        check(f"{tag} DCP grid", x[0] == -180 and x[-1] == 180, f"{x[0]}..{x[-1]} step {x[1]-x[0]}")
        check(f"{tag} DCP periodic endpoints equal", y[0] == y[-1], f"{y[0]} {y[-1]}")
        xa, _ = D._profile_1d(an, o, "DMA")
        print(f"     {tag} DMA grid {xa[0]}..{xa[-1]}")
        # every tabulated 1D point reproduced exactly
        for t, kw, conv in [("T12", "s12sq", lambda v: v), ("T13", "s13sq", lambda v: v),
                            ("T23", "s23sq", lambda v: v), ("DCP", "dcp_deg", lambda v: v),
                            ("DMS", "dm21sq_eV2", lambda v: 10 ** v), ("DMA", "dm3l_sq_eV2", lambda v: v * 1e-3)]:
            xs, ys = D._profile_1d(an, o, t)
            idx = np.linspace(1, len(xs) - 2, 7).astype(int)
            got = np.array([D.pmns_delta_chi2(o, analysis=an, **{kw: conv(xs[i])}) for i in idx])
            want = ys[idx] - ys.min()
            check(f"{tag} {t} grid-point reproduction", np.allclose(got, want, rtol=1e-6, atol=1e-6),
                  f"max diff {np.abs(got-want).max():.1e}")
        # 2D table: min equals 1D floor, reproduces grid points
        gx, gy, gz = D._grid_2d(an, o, "T23-DCP")
        off = D.ordering_offset(o, an)
        check(f"{tag} T23-DCP min == 1D floor", abs(gz.min() - off) < 1e-3, f"{gz.min():.4f} vs {off:.4f}")
        i, j = 37, 50
        got = D.pmns_delta_chi2(o, s23sq=gx[i], dcp_deg=gy[j] + 360, analysis=an)
        check(f"{tag} 2D grid point (dcp+360 wrap)", abs(got - (gz[i, j] - gz.min())) < 1e-6,
              f"{got:.5f} vs {gz[i, j]-gz.min():.5f}")
        # 2D profiled over dcp must equal 1D T23 profile (consistency of projections)
        _, t23 = D._profile_1d(an, o, "T23")
        check(f"{tag} min_dcp(T23-DCP) == T23 profile", np.allclose(gz.min(axis=1), t23, atol=2e-3),
              f"max diff {np.abs(gz.min(axis=1)-t23).max():.1e}")
        _, dcp1 = D._profile_1d(an, o, "DCP")
        check(f"{tag} min_s23(T23-DCP) == DCP profile", np.allclose(gz.min(axis=0), dcp1, atol=2e-3),
              f"max diff {np.abs(gz.min(axis=0)-dcp1).max():.1e}")
        # best fit from table vs NuFIT parameter table
        P = D.load_pmns(o, an)
        for t, k in [("T12", "s12sq"), ("T13", "s13sq"), ("T23", "s23sq"), ("DCP", "dcp_deg")]:
            xs, ys = D._profile_1d(an, o, t)
            bf = xs[ys.argmin()]
            ref = P[k]["value"]
            if k == "dcp_deg":
                bf = bf % 360
            step = xs[1] - xs[0]
            check(f"{tag} {k} table argmin vs tbl-parameters", abs(bf - ref) <= step, f"{bf} vs {ref}")
        v = D.pmns_delta_chi2(o, s12sq=P["s12sq"]["value"], s13sq=P["s13sq"]["value"],
                              s23sq=P["s23sq"]["value"], dcp_deg=P["dcp_deg"]["value"],
                              dm21sq_eV2=P["dm21sq_1e-5eV2"]["value"] * 1e-5,
                              dm3l_sq_eV2=P["dm3l_sq_1e-3eV2"]["value"] * 1e-3, analysis=an)
        check(f"{tag} Delta chi2 at published best fit ~ 0", v < 0.1, f"{v:.4f}")
        # 1 sigma edges: 1D Delta chi2 ~ 1
        for k, kw, sc in [("s12sq", "s12sq", 1), ("s13sq", "s13sq", 1), ("s23sq", "s23sq", 1), ("dcp_deg", "dcp_deg", 1)]:
            r = P[k]
            lo = D.pmns_delta_chi2(o, analysis=an, **{kw: r["value"] - r["err_minus"]})
            hi = D.pmns_delta_chi2(o, analysis=an, **{kw: r["value"] + r["err_plus"]})
            print(f"     {tag} {k} Delta chi2 at -1sigma {lo:.2f}, +1sigma {hi:.2f}")
    print(f"ordering offset {an}: IO {D.ordering_offset('IO', an):.4f}, NO {D.ordering_offset('NO', an):.4f}, "
          f"table header {D.load_pmns('IO', an)['delta_chi2_IO_minus_NO']}")

# wrap helper
check("_wrap_dcp(180) maps into table", -180 <= D._wrap_dcp(180) <= 180, f"{D._wrap_dcp(180)}")
check("_wrap_dcp(-540)", D._wrap_dcp(-540) == -180)
check("relative_to=global IO adds offset once",
      abs(D.pmns_delta_chi2("IO", s12sq=0.31, s13sq=0.0225, relative_to="global")
          - D.pmns_delta_chi2("IO", s12sq=0.31, s13sq=0.0225) - D.ordering_offset("IO")) < 1e-12)

# Objective convention
P = D.load_pmns("IO")
pred = {"s12sq": P["s12sq"]["value"], "s13sq": P["s13sq"]["value"], "s23sq": P["s23sq"]["value"], "dcp_deg": P["dcp_deg"]["value"]}
o_local = Objective(pmns_delta_chi2=D.pmns_delta_chi2, pmns_ordering="IO", pmns_mode="sum1d")
o_glob = Objective(pmns_delta_chi2=functools.partial(D.pmns_delta_chi2, relative_to="global"),
                   pmns_ordering="IO", pmns_mode="sum1d")
o_joint = Objective(pmns_delta_chi2=functools.partial(D.pmns_delta_chi2, relative_to="global"),
                    pmns_ordering="IO", pmns_mode="joint")
print(f"Objective IO at IO best fit: sum1d local {o_local(pred).chi2:.3f}, "
      f"sum1d global {o_glob(pred).chi2:.3f}, joint global {o_joint(pred).chi2:.3f}, offset {D.ordering_offset('IO'):.3f}")
print("ALL OK" if ok_all else "SOME CHECKS FAILED")

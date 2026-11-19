"""Reproduce the published 2O best-fit points (arXiv:2409.15823 Tables 3-4) in our code.

Published parameters and published predictions are hardcoded here as the paper's numbers (not
measured data). Writes reproduce_2o.json.
"""
import json
import pathlib

import numpy as np

from model_2o import observables

HERE = pathlib.Path(__file__).parent

# Table 4 (combined fit) of 2409.15823
P_COMB = dict(re=-0.202, im=1.071, gE1=0.0707, rE2=0.698, rE3=86.4, nu_scale=0.174 ** 2,
              gu1=0.586, ru2=121.9, ru3=0.487, gd1=0.290, rd2=45.1, rd3=1.16, rd4=0.0132, rd5=0.00361)
P_LEP = dict(re=-0.192, im=1.083, gE1=0.0714, rE2=0.716, rE3=87.7, nu_scale=0.171 ** 2,
             gu1=0.586, ru2=121.9, ru3=0.487, gd1=0.290, rd2=45.1, rd3=1.16, rd4=0.0132, rd5=0.00361)
P_QRK = dict(re=-0.201, im=1.057, gE1=0.0707, rE2=0.698, rE3=86.4, nu_scale=0.174 ** 2,
             gu1=0.578, ru2=121.1, ru3=0.441, gd1=0.330, rd2=38.8, rd3=0.870, rd4=0.0101, rd5=0.00277)
# Table 3 published predictions (combined / leptons-only / quarks-only)
PUB = {
    "comb": dict(s12sq=0.344, s23sq=0.508, s13sq=0.0231, dm21=7.28e-5, DM2=0.00249, r_emu=0.00474,
                 r_mutau=0.0586, mtau=1.29, th12q=0.227, th13q=0.00351, th23q=0.0395, dq=1.23,
                 r_uc=0.00212, r_ct=0.00282, r_ds=0.0505, r_sb=0.0209, mb=0.965, mt=89.2),
    "lep": dict(s12sq=0.329, s23sq=0.506, s13sq=0.0219, dm21=7.43e-5, DM2=0.00248, r_emu=0.00474,
                r_mutau=0.0586, mtau=1.29),
    "qrk": dict(th12q=0.227, th13q=0.00349, th23q=0.0402, dq=1.21, r_uc=0.00196, r_ct=0.00282,
                r_ds=0.0505, r_sb=0.0183, mb=0.965, mt=89.2),
}
# 2409.15823 combined-fit lepton predictions quoted in text (Eq. 3.15 region)
PUB_PRED = dict(dl_rad=-2.01, m1_meV=15.05)


def main():
    out = {}
    for name, P, pub in (("comb", P_COMB, PUB["comb"]), ("lep", P_LEP, PUB["lep"]), ("qrk", P_QRK, PUB["qrk"])):
        for tr in (False, True):
            for cj in (False, True):
                o = observables(P, transpose=tr, conj_nu=cj)
                # gE1 is quoted in MeV in the paper; mtau is compared in GeV (unit check below)
                o = dict(o)
                o["mtau"] = o["mtau"]  # gE1 in GeV: the paper labels it MeV, a typo (MeV gives m_tau = 1.3 MeV)
                # dq: compare in the convention (-pi, pi]
                o["dq"] = (o["dq"] + np.pi) % (2 * np.pi) - np.pi
                dev = {k: o[k] / v - 1 for k, v in pub.items()}
                key = f"{name} transpose={tr} conj_nu={cj}"
                out[key] = {"max_abs_rel_dev": max(abs(x) for x in dev.values()),
                            "ours": {k: float(o[k]) for k in pub}, "pub": pub,
                            "dl_rad": float((o["dl"] + np.pi) % (2 * np.pi) - np.pi), "m1_meV": float(o["m1"] * 1e3)}
                print(f"{key:40s} max|rel dev| {out[key]['max_abs_rel_dev']:.3f}  dl {out[key]['dl_rad']:+.2f}"
                      f"  m1 {out[key]['m1_meV']:.2f} meV")
    best = min((k for k in out if k.startswith("comb")), key=lambda k: out[k]["max_abs_rel_dev"])
    print("\nbest convention (combined point):", best)
    for k, v in out[best]["pub"].items():
        print(f"  {k:8s} ours {out[best]['ours'][k]:.5g}  pub {v:.5g}  rel {out[best]['ours'][k] / v - 1:+.4f}")
    print(f"  delta_l ours {out[best]['dl_rad']:+.3f} rad, pub {PUB_PRED['dl_rad']} ; m1 ours {out[best]['m1_meV']:.2f} meV, pub {PUB_PRED['m1_meV']}")
    (HERE / "reproduce_2o.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Step 1a: evaluate the published 1306.4468 best-fit points (Appendix A) in our generator conventions.

MN no RGE (Eq. 15) is the model/scale we can compare directly (fit at M_GUT, Table 6 first column).
FN no RGE NH (Eqs. 23-24) and IH (Eq. 27) check the real 10+126+120 implementation the same way (Table 13 χ²).
For each point we try the 4 equivalent conventions (M or conj(M)) x (M or M^T): conj flips both δ's, the
transpose flips the sign of the antisymmetric 120 part. Output: dr_point.json.
"""
import json

import numpy as np

import so10fit as S

V = S.V_EW


def evaluate(H, F, r, s, rR, G=None, tu=0.0, tD=0.0, tl=0.0, ordering="NO"):
    out = {}
    base = S.dr_form_mats(H, F, r, s, G, tu, tD, tl)
    for conj in (False, True):
        for tr in (False, True):
            Ms = [np.conj(M) if conj else M for M in base[:3]]
            Ms = [np.swapaxes(M, 1, 2) if tr else M for M in Ms]
            # m_nu: rebuild from the transformed Dirac / Majorana pieces
            Gm = np.zeros((3, 3)) if G is None else G
            YD = r * (H - 3 * s * F + 1j * tD * Gm)
            YD = np.conj(YD) if conj else YD
            YD = YD.T if tr else YD
            Fm = np.conj(F) if conj else F
            Mnu = (YD.T @ np.linalg.solve(Fm, YD))[None]
            o = S.raw_obs(*Ms, Mnu)
            # Y_D already carries r; m_nu = v^2 r_R Y_D^T F^-1 Y_D (GeV) -> eV
            y = S.dr_vec(o, np.array([V]), np.array([V ** 2 * abs(rR) * 1e9]))[0]
            pull = (y - S.DR_VAL) / S.DR_ERR
            pull[9] = S.wrap(y[9] - S.DR_VAL[9]) / S.DR_ERR[9]
            if ordering == "IO":
                pull[11] = (y[11] - (-2.355e-3)) / 0.054e-3
                pull[13] = (y[13] - 0.59) / 0.022
            key = f"conj={int(conj)},T={int(tr)}"
            out[key] = dict(pred=dict(zip(S.DR_NAMES, map(float, y))), pull=dict(zip(S.DR_NAMES, map(float, pull))),
                            chi2_all18=float((pull ** 2).sum()), chi2_excl_lepton_masses=float((pull[:15] ** 2).sum()),
                            IO=bool(o["io"][0]), dl=float(o["dl"][0]))
    return out


def main():
    res = {}
    # ---- MN, no RGE (Eq. 15)
    H = np.diag([1.22387e-6, 5.92428e-5, 6.29473e-3]).astype(complex)
    F = np.array([[-2.95102e-6 - 3.48291e-6j, 1.27484e-5 - 7.53714e-8j, 1.07772e-4 + 6.02931e-5j],
                  [1.27484e-5 - 7.53714e-8j, -1.538e-4 + 6.75236e-5j, -2.67281e-4 + 2.48978e-4j],
                  [1.07772e-4 + 6.02931e-5j, -2.67281e-4 + 2.48978e-4j, -7.38503e-4 - 1.44559e-3j]])
    res["MN_noRGE"] = evaluate(H, F, 68.9624, 0.370726 + 0.063044j, 3.014e-16)
    res["MN_noRGE_published_pred"] = dict(zip(S.DR_NAMES, map(float, S.DR_MN_BEST)))
    res["MN_noRGE_published_chi2"] = 1.103
    # ---- FN, no RGE, NH (Eqs. 23-24)
    H = np.diag([-1.44349e-3, -2.12083e-4, 8.38498e-6])
    F = np.array([[-3.52616e-3, -1.87525e-5, -3.01471e-5], [-1.87525e-5, -4.43985e-4, -2.33814e-5],
                  [-3.01471e-5, -2.33814e-5, 1.94067e-6]])
    G = np.array([[0, 1.98673e-3, 1.36719e-4], [-1.98673e-3, 0, -1.67619e-5], [-1.36719e-4, 1.67619e-5, 0]])
    res["FN_noRGE_NH"] = evaluate(H, F, 67.1992, -2.0155, -1.54145e-16, G, -0.973721, -4.11394, 1.09375)
    res["FN_noRGE_NH_published_chi2"] = 6.6e-5
    # ---- FN, no RGE, IH (Eq. 27)
    H = np.diag([6.47261e-3, 2.64518e-4, 4.10329e-5])
    F = np.array([[-8.37936e-4, -8.11918e-4, 5.65936e-6], [-8.11918e-4, -2.65901e-4, 1.47332e-6],
                  [5.65936e-6, 1.47332e-6, -5.74189e-5]])
    G = np.array([[0, -9.67226e-8, -2.30032e-5], [9.67226e-8, 0, -2.9926e-5], [2.30032e-5, 2.9926e-5, 0]])
    res["FN_noRGE_IH"] = evaluate(H, F, -71.6954, 0.710962, 6.24335e-16, G, -0.049547, 21.5488, -11.9888, "IO")
    res["FN_noRGE_IH_published_chi2"] = 9.0e-10
    (S.HERE / "dr_point.json").write_text(json.dumps(res, indent=1))
    for k in ("MN_noRGE", "FN_noRGE_NH", "FN_noRGE_IH"):
        print(f"== {k}")
        for c, v in res[k].items():
            print(f"  {c}: chi2_18 {v['chi2_all18']:.3f} chi2_excl_lep {v['chi2_excl_lepton_masses']:.3f} IO {v['IO']} "
                  + " ".join(f"{n}={p:+.2f}" for n, p in v["pull"].items()))
    print("MN published pred vs ours (conj=0,T=0):")
    for n, a in zip(S.DR_NAMES, S.DR_MN_BEST):
        print(f"  {n:6s} pub {a:.5g}  ours {res['MN_noRGE']['conj=0,T=0']['pred'][n]:.5g}  "
              f"ours(conj) {res['MN_noRGE']['conj=1,T=0']['pred'][n]:.5g}")


if __name__ == "__main__":
    main()

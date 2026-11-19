"""Pati-Salam / SO(10) models where the left-handed charged-lepton rotation is the down-quark
rotation times Clebsch factors, on a fixed neutrino pattern.

Structures as read from the papers (PDFs kept outside the repo):
  FM   Frampton-Mohapatra hep-ph/0407139 (SU(2)L x SU(2)R x SU(4)c): M_nu = U_bm^* M_d U_bm^dag at LO,
       V_CKM^0 = 1, next order M_d ~ M_l in the 1-2 block, eq. (15): theta12^l ~ -3 theta_C m_s/m_mu x eta,
       with their benchmark eta ~ 1.5, m_s ~ m_mu/2 at v_R  ->  theta12^l ~ theta_C. U_nu = BM, real.
  AKM  Antusch-King-Mohapatra hep-ph/0504007 (G_422): eq. (26)-(30): Y_e/Y_d Clebsch -3 (12) and 2 (22)
       give theta12^e = (3/2) theta_C, theta23^e ~ lambda^2, theta13^e ~ lambda^4; U_nu: theta12 = pi/4,
       theta13 = 0, theta23 large (~pi/4); CP phases ignored. Prediction theta13 = 1.06 theta_C.
  TBC  King 1205.0506 / Antusch-Maurer 1107.3728: TBM neutrino mixing + theta12^e = c theta_C from PS/SU(5)
       Clebsch ratios (c = 1 gives tri-bimaximal-Cabibbo, theta13 = theta_C/sqrt2).
  PAT  Patel 1008.5061 SO(10) x S4, type-II: eq. (24): V_PMNS = U_e^T U_BM,
       U_e = R23(-3 theta_cb) R13(-3 theta_ub) R12(theta_C) (all quark-fixed: "the entire lepton sector
       emerges as the prediction of the model"; real parameters).
F2 coverage (exp 009): U = V_e(c12 th_C, c13 th_ub, c23 th_cb, delta_q)^dag diag(1, e^{i phi}, e^{i phi}) U_nu,
c in {0, 1/6, 2/9, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 9/2, 6}, U_nu in {TBM, BM, GR1, GR2, HEX, TM}. Each model is
compared below against the corresponding F2 lattice point (same |U_ij|, J).
"""
import numpy as np
from vlib import (V_of, r12, r13, r23, P, std_angles, std_U, report, dump, ckm_central,
                  same_up_to_rephasing, lo_matrix)


def f2(ckm, pattern, c12, c23=0.0, c13=0.0, phi=0.0, order="std"):
    a = std_angles(ckm)
    if order == "std":
        Ve = std_U(c12 * a["t12"], c13 * a["t13"], c23 * a["t23"], a["delta"])
    else:
        Ve = r12(c12 * a["t12"]) @ np.eye(3) @ r23(c23 * a["t23"])
    return Ve.conj().T @ P(0, phi, phi) @ lo_matrix(pattern)


def fm(ckm, phi=0.0):
    a = std_angles(ckm)
    return r12(a["t12"]).T @ P(0, phi, phi) @ lo_matrix("BM")


def akm(ckm, phi=0.0):
    a = std_angles(ckm)
    lam = np.sin(a["t12"])
    Ue = r23(lam ** 2) @ r13(lam ** 4) @ r12(1.5 * a["t12"])
    return Ue.T @ P(0, phi, phi) @ lo_matrix("BM")


def tbc(ckm, c=1.0, phi=0.0):
    a = std_angles(ckm)
    return r12(c * a["t12"]).T @ P(0, phi, phi) @ lo_matrix("TBM")


def patel(ckm, s23=-1.0):
    a = std_angles(ckm)
    Ue = r23(s23 * 3 * a["t23"]) @ r13(-3 * a["t13"]) @ r12(a["t12"])
    return Ue.T @ lo_matrix("BM")


if __name__ == "__main__":
    c = ckm_central()
    out = []
    for phi in (0.0, np.pi):
        out.append(report(f"FM (BM + th12e = th_C), phi={phi:.2f}", fm, {"phi": phi}))
        print("   = F2(BM, c12=1, phi):", same_up_to_rephasing(fm(c, phi), f2(c, "BM", 1.0, phi=phi)))
    for phi in (0.0, np.pi):
        out.append(report(f"AKM (BM + th12e = 1.5 th_C), phi={phi:.2f}", akm, {"phi": phi}))
        print("   vs F2(BM, c12=3/2, c23=0, phi): max|d|U_ij||^2 =",
              float(np.max(np.abs(np.abs(akm(c, phi)) ** 2 - np.abs(f2(c, 'BM', 1.5, phi=phi)) ** 2))))
    out.append(report("TBC (TBM + th12e = th_C), phi=pi/2", tbc, {"phi": np.pi / 2}))
    print("   = F2(TBM, c12=1, phi=pi/2):", same_up_to_rephasing(tbc(c, 1.0, np.pi / 2), f2(c, "TBM", 1.0, phi=np.pi / 2)))
    for s in (-1.0, 1.0):
        out.append(report(f"Patel SO(10)xS4 (paper sign s23={s:+.0f})", patel, {"s23": s}))
    # Patel vs F2 tier C (BM, c12=1, c23=3, c13=3): F2 fixes the mu-tau relative phase to 0, so a sign flip
    # of theta23e is outside the F2 lattice; the phi = 0/pi points are tested here.
    for phi in (0.0, np.pi):
        for s in (-1.0, 1.0):
            Uf = f2(c, "BM", 1.0, 3.0, 3.0, phi=phi)
            print(f"   Patel(s23={s:+.0f}) == F2(BM,1,3,3,phi={phi:.2f}):", same_up_to_rephasing(patel(c, s), Uf),
                  " max d|U|^2 =", float(np.max(np.abs(np.abs(patel(c, s)) ** 2 - np.abs(Uf) ** 2))))
    dump("verify_clebsch_family.json", out)

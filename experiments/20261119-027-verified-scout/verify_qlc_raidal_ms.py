"""QLC realizations: Raidal hep-ph/0404046 and Minakata-Smirnov hep-ph/0405088.

Paper structures (read from the PDFs, kept outside the repo):
  Raidal eq. (6), (8): V_MNS = V_e^dag V_nu. Bimaximal mixing sits in the charged-lepton left
    rotation V_e (via SU(5) Y_e = Y_d^T, maximal right-handed d rotations); the Cabibbo rotation
    is put into Y_u and, by SO(10) Y_nu = Y_u, also into V_nu. So U = U_BM^T-type x V_CKM-type.
    With V_e^dag = X = R23(pi/4) R12(pi/4) (so that U = X when V_CKM = 1) and V_nu = V_CKM^(dag),
    U = X V_CKM^(dag). We print both orientations; the paper's stated output theta12 = pi/4 - theta_C
    is the V^dag one (R12(pi/4) R12(-theta_C)).
  Minakata-Smirnov eq. (26)-(27): U_MNS = V_CKM^dag R23(pi/4) R12(pi/4) ("bimaximal minus CKM",
    U_l = V_CKM from V_l = V_d); eq. (40): U_nu = V_CKM, U_l = R12(pi/4) R23(pi/4)
    -> U_MNS = (R12 R23)^dag V_CKM.
Coverage: every form is W Phi X or X Phi W with W in {V, V^dag} and X in F1's BM class
(exp 007 enumerated these with fitted Phi, and with Phi on Z3/Z4 roots). The R12 R23 order of
eq. (40) is a permuted/transposed BM; checked below against F1-style X Phi W with X = BM^T.
"""
import numpy as np
from vlib import V_of, r12, r23, P, report, dump, ckm_central, same_up_to_rephasing, std_angles

PI4 = np.pi / 4
BM = r23(PI4) @ r12(PI4)


def raidal(ckm, a=0.0, b=0.0, dag=False):
    V = V_of(ckm)
    return BM @ P(0, a, b) @ (V.conj().T if dag else V)


def ms_27(ckm, a=0.0, b=0.0):
    return V_of(ckm).conj().T @ P(0, a, b) @ BM


def ms_40(ckm, a=0.0, b=0.0):
    return (r12(PI4) @ r23(PI4)).conj().T @ P(0, a, b) @ V_of(ckm)


if __name__ == "__main__":
    out = []
    tC = std_angles(ckm_central())["t12"]
    print("theta_C =", np.degrees(tC), " pi/4 - theta_C ->  s12sq =", np.sin(PI4 - tC) ** 2)
    for dag in (False, True):
        out.append(report(f"Raidal U = BM V{'^dag' if dag else ''}", raidal, {"dag": dag}))
    out.append(report("Minakata-Smirnov (27) U = V^dag BM", ms_27, {}))
    out.append(report("Minakata-Smirnov (40) U = (R12 R23)^dag V", ms_40, {}))
    # coverage check: MS (40) equals an F1 'X Phi W' with X = BM^T up to a fixed rephasing?
    c = ckm_central()
    U40 = ms_40(c)
    X = (r12(PI4) @ r23(PI4)).conj().T
    print("MS(40) equals X Phi V with X=(R12 R23)^T, Phi=1:", same_up_to_rephasing(U40, X @ V_of(c)))
    dump("verify_qlc_raidal_ms.json", out)

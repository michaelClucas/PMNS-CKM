"""Qin-Ma 1106.3284: 'quark-lepton complementarity in matrix form', V_CKM^dag U_PMNS V_CKM = U_TB, i.e.
U_PMNS = V U_TB V^dag (similarity transform; no free parameter). The PDF text extraction drops the dagger in eq. (7);
the orientation is fixed by reproducing the paper's numbers, eq. (12): theta12 31.80, theta23 44.66, theta13 7.67 deg
(the other five orientations are printed for comparison). The paper notes (eq. 15-16) that rephasing V -> Phi1 V Phi2
changes the prediction: U = V Phi2 U_TB Phi2^* V^dag, a hidden phase choice.
Coverage: F1 sandwich 'V Phi X V^dag' with X = TBM, Phi = 1 (exp 007, structured-phase variant includes 0, 0).
"""
import numpy as np
from vlib import V_of, ckm_central, obs, report, dump, P

U_TB = np.array([[2, np.sqrt(2), 0], [-1, np.sqrt(2), np.sqrt(3)], [1, -np.sqrt(2), np.sqrt(3)]]) / np.sqrt(6)


def qin_ma(ckm, a=0.0, b=0.0):
    V = V_of(ckm)
    return V @ P(0, a, b) @ U_TB @ P(0, -a, -b) @ V.conj().T


if __name__ == "__main__":
    V = V_of(ckm_central())
    for name, U in {"V T V^dag": V @ U_TB @ V.conj().T, "V^dag T V": V.conj().T @ U_TB @ V,
                    "V^dag T V^dag": V.conj().T @ U_TB @ V.conj().T, "V T V": V @ U_TB @ V}.items():
        o = obs(U)
        print(f"{name:14s} th12 {np.degrees(np.arcsin(np.sqrt(o['s12sq']))):6.2f}  th13 "
              f"{np.degrees(np.arcsin(np.sqrt(o['s13sq']))):5.2f}  th23 {np.degrees(np.arcsin(np.sqrt(o['s23sq']))):6.2f}  "
              f"delta {o['dcp_deg']:6.1f}   (paper: 31.80, 7.67, 44.66)")
    dump("verify_qin_ma.json", report("Qin-Ma U = V U_TB V^dag", qin_ma, {}))

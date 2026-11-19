"""'Bi-large' Cabibbo-seed lepton mixing: U = U_l^dag U_nu with U_l of CKM form and U_nu angles fixed
multiples of lambda.

Structures as read from the papers (PDFs kept outside the repo):
  CDSV-I   Chen-Ding-Srivastava-Valle 1902.08962, sec. II: U_nu = standard(th12, th13, th23, delta=pi) with
           sin th23 = 1 - lambda, sin th12 = 2 lambda, sin th13 = lambda (eq. 1); U_l = R23(th23^CKM) Phi R12(th12^CKM) Phi^*
           with sin th12^CKM = lambda, sin th23^CKM = A lambda^2, Phi = diag(e^{-i phi/2}, e^{i phi/2}, 1), phi free
           (eq. 3, SO(10)-motivated M_e ~ M_d). One free parameter phi. Published best fit phi/pi = 0.766:
           s23sq 0.573, s13sq 0.0216, s12sq 0.306, delta/pi 1.305 (eq. 6), with lambda 0.22453, A 0.836.
  CDSV-II  same paper sec. III: sin th13 = lambda, sin th12 = 2 lambda, sin th23 = 3 lambda, delta = pi (eq. 7);
           U_l = Phi R12^T Phi^* R23^T (eq. 9, SU(5)-motivated M_e ~ M_d^T). Best fit phi/pi = -0.232:
           s23sq 0.513, s13sq 0.0216, s12sq 0.320, delta/pi 1.270 (eq. 12).
  RMSV     Roy-Morisi-Singh-Valle 1410.3658: U_BL with sin th13 = lambda, sin th12 = sin th23 = psi lambda (psi free),
           U_l of type 1-4 (eqs. 3-6: R12 only or R23 R12, SO(10) or SU(5) orientation), one free phase phi.
           Two free parameters (psi, phi).
The CKM enters twice: through U_l (lambda, A) and through the numerical seed lambda in U_nu.
F1/F2 coverage: no. F1's X is a fixed group matrix and F2's U_nu is in {TBM, BM, GR1, GR2, HEX, TM}; here U_nu
itself depends on lambda and is not any of those.
"""
import numpy as np
from vlib import r12, r23, P, std_U, report, dump, obs

PI = np.pi


def Ul_I(lam, A, phi):
    return r23(np.arcsin(A * lam ** 2)) @ P(-phi / 2, phi / 2, 0) @ r12(np.arcsin(lam)) @ P(phi / 2, -phi / 2, 0)


def Ul_II(lam, A, phi):
    # phase sign chosen so that the paper's phi = -0.232 pi reproduces its delta/pi = 1.270 (phi is free)
    F = P(-phi / 2, phi / 2, 0)
    return F @ r12(np.arcsin(lam)).T @ F.conj() @ r23(np.arcsin(A * lam ** 2)).T


def cdsv(ckm, phi, pattern="I"):
    lam, A = ckm["lambda"], ckm["A"]
    if pattern == "I":
        Unu = std_U(np.arcsin(2 * lam), np.arcsin(lam), np.arcsin(1 - lam), PI)
        Ul = Ul_I(lam, A, phi)
    else:
        Unu = std_U(np.arcsin(2 * lam), np.arcsin(lam), np.arcsin(3 * lam), PI)
        Ul = Ul_II(lam, A, phi)
    return Ul.conj().T @ Unu


def rmsv(ckm, psi, phi, utype=2):
    lam, A = ckm["lambda"], ckm["A"]
    Unu = std_U(np.arcsin(psi * lam), np.arcsin(lam), np.arcsin(psi * lam), 0.0)
    Ul = {1: lambda: P(-phi / 2, phi / 2, 0) @ r12(np.arcsin(lam)) @ P(phi / 2, -phi / 2, 0),
          2: lambda: Ul_I(lam, A, phi),
          3: lambda: P(phi / 2, -phi / 2, 0) @ r12(np.arcsin(lam)).T @ P(-phi / 2, phi / 2, 0),
          4: lambda: Ul_II(lam, A, phi)}[utype]()
    return Ul.conj().T @ Unu


if __name__ == "__main__":
    out = {}
    # 1) reproduce the published benchmarks with the paper's inputs (lambda 0.22453, A 0.836)
    paper = {"lambda": 0.22453, "A": 0.836, "rhobar": 0.0, "etabar": 0.0}
    for pat, phi, pub in (("I", 0.766 * PI, (0.306, 0.0216, 0.573, 1.305)),
                          ("II", -0.232 * PI, (0.320, 0.0216, 0.513, 1.270))):
        o = obs(cdsv(paper, phi, pat))
        print(f"CDSV-{pat} at paper point: s12sq {o['s12sq']:.4f} s13sq {o['s13sq']:.5f} s23sq {o['s23sq']:.4f} "
              f"delta/pi {o['dcp_deg'] / 180:.4f}   published {pub}")
        out[f"repro_{pat}"] = {"ours": o, "published": pub}
    # 2) CKM dependence at the paper best-fit phi, our CKM
    out["dep"] = [report("CDSV-I", cdsv, {"phi": 0.766 * PI, "pattern": "I"}),
                  report("CDSV-II", cdsv, {"phi": -0.232 * PI, "pattern": "II"}),
                  report("RMSV type 2", rmsv, {"psi": 2.95, "phi": 1.764 * PI, "utype": 2})]
    # RMSV published Table I, type 2 (psi 2.9521, phi/pi 1.764, their lambda 0.22551, A 0.813): s23sq 0.4174,
    # delta/pi 1.216. Our angles agree; the paper's delta is the arcsin(J) branch (same sin delta, other quadrant).
    o = obs(rmsv({"lambda": 0.22551, "A": 0.813}, 2.9521, 1.764 * PI, 2))
    print("RMSV type 2 at paper point:", {k: round(float(v), 4) for k, v in o.items()}, "published s23sq 0.4174, delta/pi 1.216")
    out["repro_rmsv2"] = o
    dump("verify_bilarge.json", out)

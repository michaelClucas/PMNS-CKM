"""SU(5) + CSD2 flavour GUT class, Antusch-Hohl-Khosa-Susic 1808.09364 (PDF kept outside the repo).

Structure (their eq. 2.9-2.10, 2.13, B.3), GUT scale:
  Y_u = [[u1, i u2, 0], [i u2, u3, u4], [0, u4, u5]]      (theta13^uL = 0; theta12^uL theta23^uL = theta13^CKM)
  Y_d = [[0, z, 0], [y e^{i gamma}, x, 0], [0, 0, y_b]]   (block diagonal: theta12^dL ~ z/x)
  Y_e = [[0, c_y y e^{i omega}, 0], [c_z z, c_x x, 0], [0, 0, y_tau]]   (SU(5), Clebsches c_x, c_y, c_z)
  m_nu (CSD2, 102 variant) = m_a [[eps e^{i eta}, 0, 2 eps e^{i eta}], [0, 1, -1], [2 eps e^{i eta}, -1, 1 + 4 eps e^{i eta}]]
  x, y, z are fixed by m_e, m_mu and theta12^dL (their sec. 2.2: "the left angle theta12^eL ~ c_y y / (c_x x) is fixed
  by the Yukawa couplings and the CKM parameters"). theta12^dL follows from the CKM: theta12^uL = s13/s23 enters
  with a relative phase pi/2 (their right-angled unitarity triangle), so theta12^dL = sqrt(theta_C^2 - (s13/s23)^2).
Here: 2x2 (1-2) charged-lepton block solved exactly for (y, z) at x = 1 given theta12^dL and m_e/m_mu (low-scale
ratio used for both, a few-% approximation: the 1-2 ratios barely run, exp 021). CKM -> PMNS link is
theta12^eL ~ (c_y/c_x)(m_d/m_s)/theta12^dL, i.e. it goes as 1/theta_C times a mass ratio.
Coverage: CSD2 is TM1 with m1 = 0 (columns (0,1,-1) and (1,0,2) are orthogonal to (2,-1,-1)); F2's TM1 tier has
theta_nu continuous, phi_nu in {0, 90, 180, 270} deg and theta12e = c theta_C on a lattice. CSD2-GUT replaces the
lattice by (c_x, c_y, c_z) and makes the TM phase continuous (eta): same class, not the same points.
"""
import numpy as np
from scipy.optimize import brentq
from vlib import r12, P, report, dump, ckm_central, std_angles, obs, ROOT  # noqa: F401
from src.data import load_masses

CLEBSCH = (3.0, 1.5, 0.5)   # (c_x, c_y, c_z): a Table-2 tuple quoted in the paper's figures (choice = data-selected)


def lep_ratio():
    m = load_masses()
    return m["m_e"]["value"] / m["m_mu"]["value"]


def theta12dL(ckm):
    a = std_angles(ckm)
    tu = np.sin(a["t13"]) / np.sin(a["t23"])
    with np.errstate(invalid="ignore"):
        return np.sqrt(a["t12"] ** 2 - tu ** 2)


def left_angle(Y):
    w, v = np.linalg.eigh(Y @ Y.conj().T)       # ascending: light state first
    return np.arctan2(abs(v[0, 1]), abs(v[1, 1])), w


def Ue_left(ckm, omega, cl=CLEBSCH, ratio=None):
    cx, cy, cz = cl
    ratio = lep_ratio() if ratio is None else ratio
    z = np.tan(theta12dL(ckm))                  # Y_d = [[0, z], [y, 1]]: left angle tan = z exactly when y-col small
    def f(y):
        Ye = np.array([[0, cy * y * np.exp(1j * omega)], [cz * z, cx]], dtype=complex)
        w = np.linalg.svd(Ye, compute_uv=False)
        return w[1] / w[0] - ratio
    y = brentq(f, 1e-6, 5.0)
    Ye = np.array([[0, cy * y * np.exp(1j * omega)], [cz * z, cx]], dtype=complex)
    U, s, Vh = np.linalg.svd(Ye)               # Ye = U diag(s) Vh, s descending -> reorder light first
    U = U[:, ::-1]
    U3 = np.eye(3, dtype=complex)
    U3[:2, :2] = U
    return U3, y


def csd2_nu(eps, eta):
    e = eps * np.exp(1j * eta)
    M = np.array([[e, 0, 2 * e], [0, 1, -1], [2 * e, -1, 1 + 4 * e]], dtype=complex)
    w, v = np.linalg.eigh(M @ M.conj().T)
    return v, np.sqrt(np.clip(w, 0, None))     # ascending masses: m1 = 0


def csd2_gut(ckm, eps, eta, omega, cl=CLEBSCH):
    Ue, _ = Ue_left(ckm, omega, cl)
    Unu, _ = csd2_nu(eps, eta)
    return Ue.conj().T @ Unu


def dm_ratio(eps, eta):
    _, m = csd2_nu(eps, eta)
    return (m[1] ** 2 - m[0] ** 2) / (m[2] ** 2 - m[0] ** 2)


if __name__ == "__main__":
    c = ckm_central()
    t = theta12dL(c)
    Ue, y = Ue_left(c, 0.0)
    th_e = float(np.arctan2(abs(Ue[0, 1]), abs(Ue[1, 1])))
    print(f"theta12dL = {t:.4f} rad (theta_C = {std_angles(c)['t12']:.4f}); theta12eL = {th_e:.4f} for Clebsch {CLEBSCH}")
    # theta12eL vs theta12dL: power law check (expect slope -1)
    ts = []
    for s in (0.9, 1.1):
        cc = dict(c, **{"lambda": c["lambda"] * s})
        U2, _ = Ue_left(cc, 0.0)
        ts.append(np.arctan2(abs(U2[0, 1]), abs(U2[1, 1])))
    slope = np.log(ts[1] / ts[0]) / np.log(theta12dL(dict(c, **{"lambda": c["lambda"] * 1.1})) /
                                            theta12dL(dict(c, **{"lambda": c["lambda"] * 0.9})))
    print(f"d ln theta12eL / d ln theta12dL = {slope:.3f}")
    # a representative point: eps tuned to the dm ratio, eta near 270 deg (their promising models), omega free
    eps = brentq(lambda e: dm_ratio(e, 1.5 * np.pi) - 0.0295, 1e-3, 0.5)
    res = report("CSD2-GUT (3, 3/2, 1/2), eta=270deg, omega=90deg", csd2_gut,
                 {"eps": eps, "eta": 1.5 * np.pi, "omega": 0.5 * np.pi})
    res["theta12dL"], res["theta12eL"], res["slope_ln"] = t, th_e, slope
    dump("verify_csd2_gut.json", res)

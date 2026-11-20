"""Exp 039: universal-texture model classes (structures and paper references in model.md).

Every class builds four Hermitian (or symmetric) 3x3 mass matrices M_u, M_d, M_e and a neutrino
matrix from ONE texture pattern and a parameter set shared across sectors, and goes through the
shared engine src.yukawa.observables (LR convention, Y_f = M_f; Hermitian M_f so left = right).

Common conventions
  * Per sector f, two "mass" parameters set the sector's own mass ratios (fitted, counted in k).
  * Overall scales c_f cancel in every scored observable; mr_b_tau is therefore not predicted and
    not scored (N = 15: CKM 4, PMNS 4, six same-scale mass ratios, dm_ratio).
  * Phase sharing (declared before any fit, no bit): u and nu matrices are real; d and e carry the
    SAME shared phases (SU(5)-like d <-> e, u <-> nu correspondence).
  * Neutrino option (1 bit, data-selected): "dirac" (M_nu of the same pattern, Hermitian, Dirac
    masses: passed to the engine as U^* diag(y) U^dag) or "seesaw" (Dirac Yukawa M_D of the same
    pattern, M_R proportional to the identity, m_nu = -M_D^* M_D^dag).

Classes
  FZ4U   universal four-zero Hermitian texture (Fritzsch-Xing hep-ph/0212195 eqs (4)-(9)), shared
         shape a in (0, 1) and shared phases (phi1, phi2); universal eigenvalue sign eta (1 bit).
  FZ4F   diagnostic: as FZ4U with a separate shape a_f per sector (not universal; k_cont 14).
  DEMU   universal democratic texture: every sector M_f = c_f/3 [J + M1(delta_f) + M2(sigma_f, phi)]
         (review hep-ph/9912358 eqs (4.44), (4.47) with eps = delta, (4.52)); shared phase phi.
  DEMFX  quarks as DEMU; leptons as Fritzsch-Xing: M_l = c_l/3 [J + diag(-i delta_l, i delta_l,
         eps_l)], M_nu = c_nu [I + diag(-delta_nu, delta_nu, eps_nu)] (review eqs (5.4)-(5.6);
         real version: hep-ph/9808272 eq (5)). No shared continuous parameter between sectors
         other than phi of the quark sector; listed as a literature baseline.
  GEO<p> grammar class (part c): M_f[ij] = c_ij sqrt(mu_i mu_j) exp(i s_f phi_ij), c_ij in {0, 1}
         (6 Hermitian positions -> 64 patterns p), mu_f = (mu1, mu2, 1) per sector, shared phases
         phi_ij on the nonzero off-diagonal positions (s_f = 0 for u, nu; 1 for d, e).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from src.yukawa import left_rotation, observables  # noqa: E402

USE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "s12sq", "s13sq", "s23sq", "dcp_deg",
       "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau", "dm_ratio"]
QUSE = ["ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b",
        "mr_e_mu", "mr_mu_tau"]  # held-out test: quarks + charged leptons only
LUSE = ["s12sq", "s13sq", "s23sq", "dcp_deg", "dm_ratio"]
FLOOR = 0.02
SCALES = {"MZ": None, "GUT": {"mu": 2e16, "model": "SM"}}
SECTORS = ("u", "d", "e", "n")
TWO_PI = 2 * np.pi
J3 = np.ones((3, 3))


def dirac_mnu(M):
    """Dirac neutrinos: symmetric matrix whose Takagi factor is the left rotation of M."""
    U, y = left_rotation(M)
    return U.conj() @ np.diag(y) @ U.conj().T


def nu_matrix(Mn, nu):
    if nu == "dirac":
        return dirac_mnu(Mn)
    if nu == "seesaw":
        m = -Mn.conj() @ Mn.conj().T
        return 0.5 * (m + m.T)
    raise ValueError(nu)


def phases_diag(p1, p2):
    return np.diag([1.0, np.exp(1j * p1), np.exp(1j * (p1 + p2))])


# ---------------------------------------------------------------- (b) four-zero texture

def fz4_real(m1, m2, eta, a):
    """Real four-zero matrix of hep-ph/0212195 eq (5) with eigenvalues (lam1, lam2, lam3) =
    (-eta m1, eta m2, 1) and (3,3) element A = (1 - a) + a * max(0, lam1, lam2), a in (0, 1),
    eqs (7a)-(7c): Btilde = sum(lam) - A, |B|^2 = (A-l1)(A-l2)(l3-A)/A, |C|^2 = -l1 l2 l3 / A."""
    l1, l2, l3 = -eta * m1, eta * m2, 1.0
    lo = max(0.0, l1, l2)
    A = (1 - a) * l3 + a * lo
    Bt = l1 + l2 + l3 - A
    B2 = (A - l1) * (A - l2) * (l3 - A) / A
    C2 = -l1 * l2 * l3 / A
    if B2 < 0 or C2 < 0:
        raise ValueError("four-zero texture not realisable")
    B, C = np.sqrt(B2), np.sqrt(C2)
    return np.array([[0, C, 0], [C, Bt, B], [0, B, A]], dtype=complex)


def fz4_model(th, eta, nu, per_sector_a=False, flip_phase=False, ordering="NO", lshift=(0, 0), a_fixed=None):
    """th: lr12_f, lr23_f (log10 m1/m2, log10 m2/m3) for f in u, d, e, n; a (or a_f); phi1, phi2."""
    P = phases_diag(th["phi1"], th["phi2"])
    # FZ4S: fixed sign factors between the quark and lepton (1,2) / (2,3) entries, i.e. the lepton
    # phases are phi_k + pi * lshift_k (a Clebsch-like -1; each choice is a data-selected bit)
    PL = phases_diag(th["phi1"] + np.pi * lshift[0], th["phi2"] + np.pi * lshift[1])
    Ms = {}
    for f in SECTORS:
        if a_fixed and f in a_fixed:
            a = a_fixed[f]
        else:
            a = th["a_" + f] if per_sector_a else th["a"]
        m2 = 10 ** th["lr23_" + f]
        M = fz4_real(m2 * 10 ** th["lr12_" + f], m2, eta, a)
        carries = f == "d" or (f == "e" and not flip_phase) or (f == "n" and flip_phase)
        if carries:
            Q = PL if f in ("e", "n") else P
            M = Q @ M @ Q.conj().T
        Ms[f] = M
    return observables(Ms["u"], Ms["d"], Ms["e"], m_nu=nu_matrix(Ms["n"], nu), ordering=ordering)


def fz4_bounds(per_sector_a=False):
    b = {}
    for f in SECTORS:
        b["lr12_" + f] = (-5.0, -0.01)
        b["lr23_" + f] = (-6.0, -0.01)
    if per_sector_a:
        for f in SECTORS:
            b["a_" + f] = (1e-4, 1 - 1e-4)
    else:
        b["a"] = (1e-4, 1 - 1e-4)
    b["phi1"] = (0.0, TWO_PI)
    b["phi2"] = (0.0, TWO_PI)
    return b


# ---------------------------------------------------------------- (a) democratic textures

K1 = np.array([[1, 0, -1], [0, -1, 1], [-1, 1, 0]], float)
K2 = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], float)


def dem_q(delta, sigma, phi):
    """Review hep-ph/9912358 eq (4.44) + (4.47) with eps = delta (case (b), eq (4.51)) + (4.52),
    times 3/c_q: J + [[0,0,d],[0,0,d],[d,d,d]] + sigma (cos phi K1 + i sin phi K2). Hermitian."""
    M1 = np.array([[0, 0, delta], [0, 0, delta], [delta, delta, delta]], float)
    return J3 + M1 + sigma * (np.cos(phi) * K1 + 1j * np.sin(phi) * K2)


def dem_l_fx(delta, eps):
    """Review eqs (5.4)-(5.6): J + diag(-i delta, i delta, eps) (charged leptons, times 3/c_l)."""
    return J3 + np.diag([-1j * delta, 1j * delta, eps])


def dem_n_fx(delta, eps):
    """Review eqs (5.4)-(5.6): I + diag(-delta, delta, eps) (neutrinos, Majorana, times 1/c_nu)."""
    return np.eye(3) + np.diag([-delta, delta, eps])


def _dem_phi(th, f):
    """One universal phi, or (DEMU2) phi_u shared by u and nu, phi_d shared by d and e."""
    if "phi" in th:
        return th["phi"]
    return th["phi_u"] if f in ("u", "n") else th["phi_d"]


def demu_model(th, nu, ordering="NO"):
    """All four sectors dem_q(delta_f, sigma_f, phi_f): DEMU one shared phi; DEMU2 phi_u = phi_nu,
    phi_d = phi_e (the d <-> e, u <-> nu correspondence used for FZ4U)."""
    Ms = {f: dem_q(th["dl_" + f], th["sg_" + f], _dem_phi(th, f)) for f in SECTORS}
    return observables(Ms["u"], Ms["d"], Ms["e"], m_nu=nu_matrix(Ms["n"], nu), ordering=ordering)


def demu_bounds(two_phases=False):
    b = {}
    for f in SECTORS:
        b["dl_" + f] = (-0.6, 0.6)
        b["sg_" + f] = (-0.2, 0.2)
    if two_phases:
        b["phi_u"] = (0.0, TWO_PI)
        b["phi_d"] = (0.0, TWO_PI)
    else:
        b["phi"] = (0.0, TWO_PI)
    return b


def demfx_model(th, nu=None, ordering="NO"):
    Mu = dem_q(th["dl_u"], th["sg_u"], th.get("phi", th.get("phi_u")))
    Md = dem_q(th["dl_d"], th["sg_d"], th.get("phi", th.get("phi_d")))
    Me = dem_l_fx(th["dl_e"], th["ep_e"])
    mn = dem_n_fx(th["dl_n"], th["ep_n"]).astype(complex)
    return observables(Mu, Md, Me, m_nu=mn, ordering=ordering)


def demfx_bounds(two_phases=False):
    b = {"dl_u": (-0.6, 0.6), "sg_u": (-0.2, 0.2), "dl_d": (-0.6, 0.6), "sg_d": (-0.2, 0.2),
         "dl_e": (-0.3, 0.3), "ep_e": (-0.6, 0.6), "dl_n": (-0.2, 0.2), "ep_n": (-0.9, 0.9)}
    if two_phases:  # literature: phi_u and phi_d separate (review eq (4.53): phi_1 = phi_u - phi_d)
        b["phi_u"] = (0.0, TWO_PI)
        b["phi_d"] = (0.0, TWO_PI)
    else:
        b["phi"] = (0.0, TWO_PI)
    return b


# ---------------------------------------------------------------- (c) GEO grammar

POS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
OFF = [(0, 1), (0, 2), (1, 2)]


def pattern_mask(p):
    """Pattern id 0..63 -> 6 bits c_ij on POS (bit i set -> c = 1)."""
    return np.array([(p >> i) & 1 for i in range(6)], int)


def pattern_phases(p):
    m = pattern_mask(p)
    return [OFF[i] for i in range(3) if m[3 + i]]


def geo_matrix(mu, mask, phases):
    s = np.sqrt(np.asarray(mu, float))
    M = np.zeros((3, 3), complex)
    for b, (i, j) in zip(mask, POS):
        if b:
            v = s[i] * s[j] * np.exp(1j * phases.get((i, j), 0.0))
            M[i, j] = v
            M[j, i] = np.conj(v)
    return M


def geo_model(th, p, nu, ordering="NO"):
    mask = pattern_mask(p)
    ph = {ij: th[f"ph{ij[0]}{ij[1]}"] for ij in pattern_phases(p)}
    Ms = {}
    for f in SECTORS:
        mu = (10 ** th["lm1_" + f], 10 ** th["lm2_" + f], 1.0)
        Ms[f] = geo_matrix(mu, mask, ph if f in ("d", "e") else {})
    return observables(Ms["u"], Ms["d"], Ms["e"], m_nu=nu_matrix(Ms["n"], nu), ordering=ordering)


def geo_bounds(p):
    b = {}
    for f in SECTORS:
        b["lm1_" + f] = (-8.0, 1.0)
        b["lm2_" + f] = (-6.0, 1.0)
    for ij in pattern_phases(p):
        b[f"ph{ij[0]}{ij[1]}"] = (0.0, TWO_PI)
    return b

"""Known CKM-PMNS and within-sector relations as pure-numpy predictors.

Every function takes theory inputs (and, where the relation uses one sector to
predict the other, the measured inputs of that sector as arguments) and returns
predicted observables. No measured number is hardcoded here: the evaluator
passes values loaded through `src/`.

Convention (PDG standard parametrization, same for CKM and PMNS):
    U = R23(th23) . U13(th13, delta) . R12(th12)
    U_e3 = s13 exp(-i delta),  U_mu1 = -s12 c23 - c12 s23 s13 e^{i delta}, ...
    J = Im(U_e2 U_mu3 U_e3^* U_mu2^*) = c12 s12 c23 s23 c13^2 s13 sin(delta)
Angles in radians, delta in [0, 2pi). Majorana phases are not returned.
For CKM: th12 = theta_C (|V_us|), th23 = theta_cb (|V_cb|), th13 = |V_ub|.

Run `python baselines.py` for structural self-tests (no data needed).
"""
from __future__ import annotations

import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
PI = np.pi

# ---------------------------------------------------------------- utilities


def r12(t, phase=0.0):
    c, s = np.cos(t), np.sin(t)
    e = np.exp(-1j * phase)
    return np.array([[c, s * e, 0], [-s * np.conj(e), c, 0], [0, 0, 1]], dtype=complex)


def r23(t, phase=0.0):
    c, s = np.cos(t), np.sin(t)
    e = np.exp(-1j * phase)
    return np.array([[1, 0, 0], [0, c, s * e], [0, -s * np.conj(e), c]], dtype=complex)


def r13(t, phase=0.0):
    c, s = np.cos(t), np.sin(t)
    e = np.exp(-1j * phase)
    return np.array([[c, 0, s * e], [0, 1, 0], [-s * np.conj(e), 0, c]], dtype=complex)


def standard_matrix(th12, th13, th23, delta):
    """PDG standard parametrization."""
    return r23(th23) @ r13(th13, delta) @ r12(th12)


def angles_from_matrix(U):
    """Rephasing-invariant extraction of (th12, th13, th23, delta, J) from a 3x3 unitary."""
    A = np.abs(U)
    s13 = min(A[0, 2], 1.0)
    c13 = np.sqrt(max(1.0 - s13**2, 0.0))
    if c13 < 1e-12:
        raise ValueError("|U_e3| = 1: angles undefined")
    th13 = np.arcsin(s13)
    th12 = np.arctan2(A[0, 1], A[0, 0])
    th23 = np.arctan2(A[1, 2], A[2, 2])
    J = np.imag(U[0, 1] * U[1, 2] * np.conj(U[0, 2]) * np.conj(U[1, 1]))
    s12, c12, s23, c23 = np.sin(th12), np.cos(th12), np.sin(th23), np.cos(th23)
    denom = 2 * s12 * c12 * s23 * c23 * s13
    if denom < 1e-14:
        delta = 0.0
    else:
        # |U_tau1|^2 = s12^2 s23^2 + c12^2 c23^2 s13^2 - 2 s12 s23 c12 c23 s13 cos(delta)
        cosd = (s12**2 * s23**2 + c12**2 * c23**2 * s13**2 - A[2, 0] ** 2) / denom
        sind = J / (denom / 2 * c13**2)
        delta = np.arctan2(sind, cosd) % (2 * PI)
    return dict(th12=th12, th13=th13, th23=th23, delta=delta, J=J)


def jarlskog(th12, th13, th23, delta):
    return (np.cos(th12) * np.sin(th12) * np.cos(th23) * np.sin(th23)
            * np.cos(th13) ** 2 * np.sin(th13) * np.sin(delta))


def phase_diag(*phases):
    return np.diag(np.exp(1j * np.asarray(phases, dtype=float)))


def obs(th12, th13, th23, delta=np.nan):
    """Observable dict in the common output format."""
    J = jarlskog(th12, th13, th23, delta) if np.isfinite(delta) else np.nan
    return dict(th12=th12, th13=th13, th23=th23, delta=delta, J=J,
                s2_12=np.sin(th12) ** 2, s2_13=np.sin(th13) ** 2, s2_23=np.sin(th23) ** 2)


def obs_from_matrix(U):
    a = angles_from_matrix(U)
    o = obs(a["th12"], a["th13"], a["th23"], a["delta"])
    o["J"] = a["J"]
    return o


# --------------------------------------------- fixed leading-order patterns
# All have th13 = 0, th23 = pi/4 and differ only in th12. Real orthogonal,
# sign convention U = R23(-pi/4)^T-like so that U_mu3 = U_tau3 = 1/sqrt2.

LO_TH12 = {
    "TBM": np.arcsin(1 / np.sqrt(3)),       # sin^2 th12 = 1/3          (hep-ph/0202074)
    "BM": PI / 4,                           # sin^2 th12 = 1/2          (hep-ph/9806387)
    "GR1": np.arctan(1 / PHI),              # cot th12 = phi            (0705.4559)
    "GR2": PI / 5,                          # cos th12 = phi/2          (0810.5239)
    "HEX": PI / 6,                          # sin^2 th12 = 1/4          (1004.2798)
}


def lo_matrix(name):
    """Leading-order real mixing matrix: R23(pi/4) R12(th12)."""
    return r23(PI / 4) @ r12(LO_TH12[name])


def lo_pattern(name):
    """Prediction of an unperturbed pattern (N=3 angles, k=0 plus the pattern choice)."""
    return obs(LO_TH12[name], 0.0, PI / 4, np.nan)


# --------------------------------------------- quark-lepton complementarity


def qlc(theta_C, theta_cb):
    """QLC (hep-ph/0404046, hep-ph/0405088): th12 + theta_C = pi/4, th23 + theta_cb = pi/4."""
    return dict(th12=PI / 4 - theta_C, th23=PI / 4 - theta_cb)


def self_complementarity(th13):
    """Self-complementarity (1106.4040, 1202.4258): th12 + th13 = pi/4 (standard param.)."""
    return dict(th12=PI / 4 - th13)


def tbc_king(theta_C):
    """Tri-bimaximal-Cabibbo (1205.0506): s13 = sin(theta_C)/sqrt2, s12 = 1/sqrt3, s23 = 1/sqrt2."""
    return obs(np.arcsin(1 / np.sqrt(3)), np.arcsin(np.sin(theta_C) / np.sqrt(2)), PI / 4, np.nan)


def ckm_dagger_x(V_ckm, X="BM", phases=(0.0, 0.0), order="Vdag_X"):
    """U_PMNS = V_CKM^dagger . diag(1, e^{i a}, e^{i b}) . X  (Li-Ma hep-ph/0504161,
    Ludl-Smirnov 1507.03494, Smirnov-Xu 1803.07933, Giarnetti-Marciano-Meloni 2407.02487
    use V_CKM^* instead: pass order='Vconj_X').
    order='V_X_Vdag' gives Qin-Ma (1106.3284): U = V_CKM . X . V_CKM^dagger.
    X: name of an LO pattern or an explicit 3x3 unitary."""
    Xm = lo_matrix(X) if isinstance(X, str) else np.asarray(X, dtype=complex)
    P = phase_diag(0.0, *phases)
    V = np.asarray(V_ckm, dtype=complex)
    if order == "Vdag_X":
        U = V.conj().T @ P @ Xm
    elif order == "Vconj_X":
        U = V.conj() @ P @ Xm
    elif order == "V_X_Vdag":
        U = V @ P @ Xm @ V.conj().T
    else:
        raise ValueError(order)
    return obs_from_matrix(U)


# --------------------------------------------- charged-lepton corrections


def charged_lepton_corrected(U_nu, th12e, phi=0.0, th23e=0.0, phi23=0.0):
    """U_PMNS = R12(th12e, phi)^dagger R23(th23e, phi23)^dagger U_nu (Frampton-Petcov-Rodejohann
    hep-ph/0401206, Antusch-King hep-ph/0508044). 'Cabibbo haze' (hep-ph/0503222): th12e ~ theta_C."""
    Ue = r23(th23e, phi23) @ r12(th12e, phi)
    return obs_from_matrix(Ue.conj().T @ np.asarray(U_nu, dtype=complex))


def solar_sum_rule_cosdelta(th12, th13, th23, th12nu):
    """Exact solar sum rule for U = R12e^dag Phi R23(th23nu) R12(th12nu) (th13nu = th13e = 0):
    the tau row is untouched, |U_tau1|^2 = s12nu^2 (1 - c13^2 c23^2). Equivalent to Petcov
    1405.6006 / Girardi-Petcov-Titov 1410.8056. th12nu: 35.26 deg (TBM), 45 (BM), etc.
    Returns cos(delta); |cos|>1 means the pattern cannot fit those angles."""
    s12, c12, s13, c13 = np.sin(th12), np.cos(th12), np.sin(th13), np.cos(th13)
    s23, c23 = np.sin(th23), np.cos(th23)
    ut1 = np.sin(th12nu) ** 2 * (1 - c13**2 * c23**2)
    return (s12**2 * s23**2 + c12**2 * c23**2 * s13**2 - ut1) / (2 * s12 * s23 * c12 * c23 * s13)


def antusch_king_linear(th13, delta, th12nu):
    """Leading-order sum rule (hep-ph/0508044): th12 ~= th12nu + th13 cos(delta)
    (sign follows our convention; their eq. writes th12 + th13 cos(delta - pi) ~= th12nu)."""
    return dict(th12=th12nu + th13 * np.cos(delta))


# --------------------------------------------- trimaximal TM1 / TM2


def tm1(th13, th23=None):
    """TM1 (first TBM column kept; 0812.0436, 1308.4314).
    s12^2 = (1 - 3 s13^2) / (3 c13^2); if th23 given, also returns cos(delta)
    from |U_mu1|^2 = 1/6."""
    s13, c13 = np.sin(th13), np.cos(th13)
    th12 = np.arcsin(np.sqrt((1 - 3 * s13**2) / (3 * c13**2)))
    out = dict(th12=th12)
    if th23 is not None:
        s12, c12, s23, c23 = np.sin(th12), np.cos(th12), np.sin(th23), np.cos(th23)
        out["cos_delta"] = (1 / 6 - s12**2 * c23**2 - c12**2 * s13**2 * s23**2) / (
            2 * s12 * s13 * s23 * c12 * c23)
    return out


def tm2(th13, th23=None):
    """TM2 (second TBM column kept): s12^2 = 1 / (3 c13^2); cos(delta) from |U_mu2|^2 = 1/3."""
    s13, c13 = np.sin(th13), np.cos(th13)
    th12 = np.arcsin(np.sqrt(1 / (3 * c13**2)))
    out = dict(th12=th12)
    if th23 is not None:
        s12, c12, s23, c23 = np.sin(th12), np.cos(th12), np.sin(th23), np.cos(th23)
        out["cos_delta"] = (c12**2 * c23**2 + s12**2 * s23**2 * s13**2 - 1 / 3) / (
            2 * c12 * c23 * s12 * s23 * s13)
    return out


# --------------------------------------------- mass-mixing relations


def gst(m_d, m_s, m_u=None, m_c=None, phase=None):
    """Gatto-Sartori-Tonin (PLB 28 (1968) 128): |V_us| = sqrt(m_d/m_s);
    Fritzsch-type refinement |V_us| = |sqrt(m_d/m_s) - e^{i phase} sqrt(m_u/m_c)|."""
    v = np.sqrt(m_d / m_s)
    if phase is not None:
        v = np.abs(v - np.exp(1j * phase) * np.sqrt(m_u / m_c))
    return dict(Vus=v, th12=np.arcsin(v))


def lepton_gst(m_e, m_mu, U_nu="TBM", phi=0.0):
    """Leptonic GST analogue: charged-lepton 1-2 angle th12e = arcsin sqrt(m_e/m_mu)
    applied to a LO neutrino pattern. Gives s13 = sin(th12e)/sqrt2 for th23nu = pi/4."""
    th12e = np.arcsin(np.sqrt(m_e / m_mu))
    return charged_lepton_corrected(lo_matrix(U_nu), th12e, phi)


def koide_Q(m1, m2, m3, signs=(1, 1, 1)):
    """Koide ratio Q = sum m / (sum +-sqrt m)^2 (Q = 2/3 for charged leptons)."""
    r = np.array(signs) * np.sqrt(np.array([m1, m2, m3], dtype=float))
    return (m1 + m2 + m3) / r.sum() ** 2


def koide_predict_m3(m1, m2):
    """Solve Q(m1, m2, m3) = 2/3 for m3 (larger root). Used for m_tau from m_e, m_mu."""
    a, b = np.sqrt(m1), np.sqrt(m2)
    # (2/3)(a+b+x)^2 = a^2+b^2+x^2 with x = sqrt(m3), rearranged to A x^2 + B x + C = 0
    A = 1 - 2 / 3
    B = -(4 / 3) * (a + b)
    C = a**2 + b**2 - (2 / 3) * (a + b) ** 2
    x = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return x**2


def koide_neutrino_m1(dm21, dm31, signs=(1, 1, 1), lo=1e-8, hi=1.0):
    """Lightest NO neutrino mass m1 solving Q(m1, m2, m3) = 2/3 with m_i from splittings
    (Rodejohann-Zhang 1101.5525 allow sign flips of sqrt m). Returns list of roots in eV."""
    def f(m1):
        return koide_Q(m1, np.sqrt(m1**2 + dm21), np.sqrt(m1**2 + dm31), signs) - 2 / 3
    grid = np.logspace(np.log10(lo), np.log10(hi), 4000)
    vals = np.array([f(g) for g in grid])
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]:
        a, b = grid[i], grid[i + 1]
        for _ in range(80):
            m = 0.5 * (a + b)
            if np.sign(f(m)) == np.sign(f(a)):
                a = m
            else:
                b = m
        roots.append(0.5 * (a + b))
    return roots


def gut_yukawa_ratios(scheme="minimal_SU5"):
    """GUT-scale Yukawa ratios (y_e/y_d, y_mu/y_s, y_tau/y_b).
    minimal_SU5: Y_e = Y_d^T -> 1,1,1.  Georgi-Jarlskog (PLB 86 (1979) 297): 1/3, 3, 1.
    Antusch-Spinrath alternatives (0902.4644) e.g. y_mu/y_s in {9/2, 6}: pass explicitly."""
    table = {"minimal_SU5": (1.0, 1.0, 1.0), "GJ": (1 / 3, 3.0, 1.0)}
    return dict(zip(("e/d", "mu/s", "tau/b"), table[scheme]))


# --------------------------------------------- CP-phase sum rule (KM param.)


def km_matrix(t1, t2, t3, d):
    """Original Kobayashi-Maskawa parametrization."""
    c1, s1, c2, s2, c3, s3 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), np.cos(t3), np.sin(t3)
    e = np.exp(1j * d)
    return np.array([
        [c1, -s1 * c3, -s1 * s3],
        [s1 * c2, c1 * c2 * c3 - s2 * s3 * e, c1 * c2 * s3 + s2 * c3 * e],
        [s1 * s2, c1 * s2 * c3 + c2 * s3 * e, c1 * s2 * s3 - c2 * c3 * e]], dtype=complex)


def km_phase(U):
    """Rephasing-invariant KM phase. Pan-Sun-Ma-He (2003.09921): delta_KM^q + delta_KM^l ~ 0."""
    A = np.abs(U)
    s1 = np.sqrt(max(1 - A[0, 0] ** 2, 0.0))
    c1 = A[0, 0]
    c3 = A[0, 1] / s1
    s3 = A[0, 2] / s1
    s2 = A[2, 0] / s1
    c2 = A[1, 0] / s1
    cosd = (c1**2 * c2**2 * c3**2 + s2**2 * s3**2 - A[1, 1] ** 2) / (2 * c1 * c2 * c3 * s2 * s3)
    # J = s1^2 s2 s3 c1 c2 c3 sin(d) with the same J definition as the standard case
    J = np.imag(U[0, 1] * U[1, 2] * np.conj(U[0, 2]) * np.conj(U[1, 1]))
    Jkm = s1**2 * s2 * s3 * c1 * c2 * c3
    sind = J / Jkm
    return np.arctan2(sind, np.clip(cosd, -1, 1)) % (2 * PI)


# --------------------------------------------- null model


def haar_unitary(rng, n=3):
    """Haar-random U(n) (Mezzadri QR). Anarchy null hypothesis (hep-ph/9911341, 1204.1249)."""
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


# --------------------------------------------- self-tests (structure only)


def _selftest():
    rng = np.random.default_rng(0)
    # round trip of the standard parametrization incl. random rephasing
    for _ in range(200):
        t12, t13, t23 = rng.uniform(0.05, 1.5, 3)
        d = rng.uniform(0, 2 * PI)
        U = phase_diag(*rng.uniform(0, 2 * PI, 3)) @ standard_matrix(t12, t13, t23, d) \
            @ phase_diag(*rng.uniform(0, 2 * PI, 3))
        a = angles_from_matrix(U)
        assert np.allclose([a["th12"], a["th13"], a["th23"]], [t12, t13, t23], atol=1e-10)
        assert abs(np.angle(np.exp(1j * (a["delta"] - d)))) < 1e-8
        assert abs(a["J"] - jarlskog(t12, t13, t23, d)) < 1e-12
    # LO patterns
    assert np.isclose(lo_pattern("TBM")["s2_12"], 1 / 3)
    assert np.isclose(lo_pattern("HEX")["s2_12"], 1 / 4)
    assert np.isclose(1 / np.tan(LO_TH12["GR1"]), PHI)
    assert np.isclose(np.cos(LO_TH12["GR2"]), PHI / 2)
    # TM1 / TM2 against explicit constructions U_TBM R23 / U_TBM R13
    for _ in range(100):
        th, ph = rng.uniform(0.05, 0.3), rng.uniform(0, 2 * PI)
        o1 = obs_from_matrix(lo_matrix("TBM") @ r23(th, ph))
        t = tm1(o1["th13"], o1["th23"])
        assert np.isclose(t["th12"], o1["th12"]) and np.isclose(t["cos_delta"], np.cos(o1["delta"]))
        o2 = obs_from_matrix(lo_matrix("TBM") @ r13(th, ph))
        t = tm2(o2["th13"], o2["th23"])
        assert np.isclose(t["th12"], o2["th12"]) and np.isclose(t["cos_delta"], np.cos(o2["delta"]))
    # solar sum rule against explicit charged-lepton correction
    for _ in range(100):
        name = rng.choice(list(LO_TH12))
        th23nu = rng.uniform(0.6, 1.0)
        Unu = r23(th23nu) @ r12(LO_TH12[name]) @ phase_diag(0, *rng.uniform(0, 2 * PI, 2))
        o = charged_lepton_corrected(phase_diag(0, *rng.uniform(0, 2 * PI, 2)) @ Unu,
                                     rng.uniform(0.05, 0.25), rng.uniform(0, 2 * PI))
        cd = solar_sum_rule_cosdelta(o["th12"], o["th13"], o["th23"], LO_TH12[name])
        assert np.isclose(cd, np.cos(o["delta"]), atol=1e-8)
    # TBC: charged-lepton theta_C on TBM gives s13 = sin(theta_C)/sqrt2 exactly
    tc = 0.2
    o = charged_lepton_corrected(lo_matrix("TBM"), tc, PI)
    assert np.isclose(np.sin(o["th13"]), np.sin(tc) / np.sqrt(2))
    # KM phase round trip
    for _ in range(100):
        t1, t2, t3 = rng.uniform(0.05, 1.4, 3)
        d = rng.uniform(0, 2 * PI)
        U = phase_diag(*rng.uniform(0, 2 * PI, 3)) @ km_matrix(t1, t2, t3, d) \
            @ phase_diag(*rng.uniform(0, 2 * PI, 3))
        assert abs(np.angle(np.exp(1j * (km_phase(U) - d)))) < 1e-7
    # Koide: Q is 2/3 at the predicted m3
    m3 = koide_predict_m3(1.0, 200.0)
    assert np.isclose(koide_Q(1.0, 200.0, m3), 2 / 3)
    # V^dag X with V = 1 reproduces X
    o = ckm_dagger_x(np.eye(3), "TBM")
    assert np.isclose(o["s2_12"], 1 / 3) and np.isclose(o["th13"], 0, atol=1e-12)
    # Haar unitarity
    H = haar_unitary(rng)
    assert np.allclose(H @ H.conj().T, np.eye(3))
    print("baselines self-test: all passed")


if __name__ == "__main__":
    _selftest()

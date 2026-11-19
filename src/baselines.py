"""Known CKM-PMNS and within-sector relations, plus a scoring registry.

Ported from experiments/20261118-003-baseline-catalog/baselines.py (that copy is kept
unchanged as the record of exp 003). Differences from the port source:
  * angle extraction and the standard matrix come from src.mixing;
  * `obs` / `obs_from_matrix` also return the evaluator keys (s12sq, s13sq, s23sq,
    dcp_deg), and omit dcp_deg when delta is undefined;
  * `BASELINES` declares, for every scoreable relation, what it predicts, which measured
    numbers it consumes, its fitted parameters and its parameter counts, so that
    src.evaluate can score it the same way every time.

Convention (PDG standard parametrization, same for CKM and PMNS):
    U = R23(th23) . U13(th13, delta) . R12(th12),  U_e3 = s13 exp(-i delta).
Angles in radians, delta in [0, 2 pi). Majorana phases are not returned.
For CKM: th12 = theta_C, th23 = theta_cb, th13 = theta_ub.

No measured number is hardcoded. `data_context()` reads every input through src.data.

Parameter-counting convention for the registry (see also src/evaluate.py):
  predicts   observables scored against data; N = len(predicts).
  inputs     measured numbers plugged in at their central values. They are not scored
             and not counted in N. Only used when the input error is small next to the
             error of what it predicts (checked by input-shift tests in exp 006).
  params     fitted continuous parameters (k_continuous = len(params)). Includes
             nuisance parameters: when a relation takes a measured quantity with a
             non-negligible error as input (e.g. th13 for TM1, m_d/m_s for GST), that
             quantity is fitted as a parameter AND scored as an observable, so it adds
             1 to N and 1 to k (net zero dof) and its error propagates correctly.
  k_discrete data-selected discrete choices (pattern, sign, factor order, parametrization),
             in bits: k_discrete = stats.discrete_bits(n_options) = ceil(log2 n_options),
             where n_options is the number of alternatives the choice was made from
             (exp 012 M3; exp 013 changed A4 from 2 to 4, B and B1 from 1 to 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import data as _data
from .mixing import extract_angles
from .mixing import jarlskog as _jarlskog_matrix
from .mixing import standard_matrix as _standard_matrix
from .stats import discrete_bits

PHI = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
PI = np.pi
TWO_PI = 2.0 * np.pi
DEG = 180.0 / np.pi

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
    """PDG standard parametrization (src.mixing.standard_matrix)."""
    return _standard_matrix(th12, th13, th23, delta)


def angles_from_matrix(U):
    """(th12, th13, th23, delta, J) from a 3x3 unitary via src.mixing.extract_angles."""
    t12, t13, t23, d = (float(x) for x in extract_angles(np.asarray(U, dtype=complex)))
    return dict(th12=t12, th13=t13, th23=t23, delta=d, J=float(_jarlskog_matrix(U)))


def jarlskog(th12, th13, th23, delta):
    return (np.cos(th12) * np.sin(th12) * np.cos(th23) * np.sin(th23)
            * np.cos(th13) ** 2 * np.sin(th13) * np.sin(delta))


def phase_diag(*phases):
    return np.diag(np.exp(1j * np.asarray(phases, dtype=float)))


def obs(th12, th13, th23, delta=np.nan):
    """Observable dict. Old keys (th12, ..., s2_12) plus evaluator keys (s12sq, ...).

    dcp_deg is present only when delta is finite (an undefined phase is not a prediction).
    """
    J = jarlskog(th12, th13, th23, delta) if np.isfinite(delta) else np.nan
    s12sq, s13sq, s23sq = np.sin(th12) ** 2, np.sin(th13) ** 2, np.sin(th23) ** 2
    o = dict(th12=th12, th13=th13, th23=th23, delta=delta, J=J,
             s2_12=s12sq, s2_13=s13sq, s2_23=s23sq,
             s12sq=s12sq, s13sq=s13sq, s23sq=s23sq)
    if np.isfinite(delta):
        o["dcp_deg"] = float(delta * DEG)
    return o


def obs_from_matrix(U):
    a = angles_from_matrix(U)
    o = obs(a["th12"], a["th13"], a["th23"], a["delta"])
    o["J"] = a["J"]
    return o


# --------------------------------------------- fixed leading-order patterns

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
    """Prediction of an unperturbed pattern (3 angles; delta undefined)."""
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
    """U_PMNS = V_CKM^dagger . diag(1, e^{i a}, e^{i b}) . X  (order='Vdag_X'; Li-Ma
    hep-ph/0504161, Ludl-Smirnov 1507.03494, Smirnov-Xu 1803.07933).
    order='Vconj_X': V_CKM^* . P . X (Giarnetti-Marciano-Meloni 2407.02487).
    order='V_X_Vdag': V_CKM . P . X . V_CKM^dagger (Qin-Ma 1106.3284; lead L1 with X = BM).
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
    """U_PMNS = (R23(th23e, phi23) R12(th12e, phi))^dagger U_nu (Frampton-Petcov-Rodejohann
    hep-ph/0401206, Antusch-King hep-ph/0508044). 'Cabibbo haze' (hep-ph/0503222)."""
    Ue = r23(th23e, phi23) @ r12(th12e, phi)
    return obs_from_matrix(Ue.conj().T @ np.asarray(U_nu, dtype=complex))


def solar_sum_rule_cosdelta(th12, th13, th23, th12nu):
    """Exact solar sum rule for U = R12e^dag Phi R23(th23nu) R12(th12nu):
    |U_tau1|^2 = s12nu^2 (1 - c13^2 c23^2). Returns cos(delta); |cos| > 1 means the
    pattern cannot fit those angles."""
    s12, c12, s13, c13 = np.sin(th12), np.cos(th12), np.sin(th13), np.cos(th13)
    s23, c23 = np.sin(th23), np.cos(th23)
    ut1 = np.sin(th12nu) ** 2 * (1 - c13**2 * c23**2)
    return (s12**2 * s23**2 + c12**2 * c23**2 * s13**2 - ut1) / (2 * s12 * s23 * c12 * c23 * s13)


def antusch_king_linear(th13, delta, th12nu):
    """Leading-order sum rule (hep-ph/0508044): th12 ~= th12nu + th13 cos(delta)."""
    return dict(th12=th12nu + th13 * np.cos(delta))


# --------------------------------------------- trimaximal TM1 / TM2


def tm1(th13, th23=None):
    """TM1 (first TBM column kept): s12^2 = (1 - 3 s13^2) / (3 c13^2); with th23, also
    cos(delta) from |U_mu1|^2 = 1/6."""
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
    """Gatto-Sartori-Tonin: |V_us| = sqrt(m_d/m_s); Fritzsch-type refinement
    |V_us| = |sqrt(m_d/m_s) - e^{i phase} sqrt(m_u/m_c)|."""
    v = np.sqrt(m_d / m_s)
    if phase is not None:
        v = np.abs(v - np.exp(1j * phase) * np.sqrt(m_u / m_c))
    return dict(Vus=v, th12=np.arcsin(v))


def lepton_gst(m_e, m_mu, U_nu="TBM", phi=0.0):
    """Leptonic GST analogue: th12e = arcsin sqrt(m_e/m_mu) on a LO neutrino pattern."""
    th12e = np.arcsin(np.sqrt(m_e / m_mu))
    return charged_lepton_corrected(lo_matrix(U_nu), th12e, phi)


def koide_Q(m1, m2, m3, signs=(1, 1, 1)):
    """Koide ratio Q = sum m / (sum +-sqrt m)^2."""
    r = np.array(signs) * np.sqrt(np.array([m1, m2, m3], dtype=float))
    return (m1 + m2 + m3) / r.sum() ** 2


def koide_predict_m3(m1, m2):
    """Solve Q(m1, m2, m3) = 2/3 for m3 (larger root)."""
    a, b = np.sqrt(m1), np.sqrt(m2)
    A = 1 - 2 / 3
    B = -(4 / 3) * (a + b)
    C = a**2 + b**2 - (2 / 3) * (a + b) ** 2
    x = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return x**2


def koide_neutrino_m1(dm21, dm31, signs=(1, 1, 1), lo=1e-8, hi=1.0):
    """Lightest NO neutrino mass m1 solving Q = 2/3 (Rodejohann-Zhang 1101.5525). Roots in eV."""
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
    """GUT-scale Yukawa ratios (y_e/y_d, y_mu/y_s, y_tau/y_b): minimal SU(5) or Georgi-Jarlskog."""
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
    """Rephasing-invariant KM phase (Pan-Sun-Ma-He 2003.09921: delta_KM^q + delta_KM^l ~ 0)."""
    A = np.abs(U)
    s1 = np.sqrt(max(1 - A[0, 0] ** 2, 0.0))
    c1 = A[0, 0]
    c3 = A[0, 1] / s1
    s3 = A[0, 2] / s1
    s2 = A[2, 0] / s1
    c2 = A[1, 0] / s1
    cosd = (c1**2 * c2**2 * c3**2 + s2**2 * s3**2 - A[1, 1] ** 2) / (2 * c1 * c2 * c3 * s2 * s3)
    J = np.imag(U[0, 1] * U[1, 2] * np.conj(U[0, 2]) * np.conj(U[1, 1]))
    Jkm = s1**2 * s2 * s3 * c1 * c2 * c3
    sind = J / Jkm
    return np.arctan2(sind, np.clip(cosd, -1, 1)) % (2 * PI)


# --------------------------------------------- null model


def haar_unitary(rng, n=3):
    """Haar-random U(n) (Mezzadri QR). For production use src.haar."""
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


# ================================================================ data context

#: input name -> (loader, record name). Names match src.evaluate observable keys.
INPUT_RECORDS = {
    "ckm_s12": ("ckm", "s12"),
    "ckm_s13": ("ckm", "s13"),
    "ckm_s23": ("ckm", "s23"),
    "ckm_delta_rad": ("ckm", "delta_rad"),
    "m_e": ("masses", "m_e"),
    "m_mu": ("masses", "m_mu"),
}


def data_context(shifts=None) -> dict:
    """Measured inputs for the registry, read through src.data.

    shifts: optional {input name: n_sigma}; shifts that input by n_sigma (err_plus for
    n > 0, err_minus for n < 0). Used to test sensitivity to plugged-in inputs.
    Returns dict with ckm_s12 ... m_mu values plus derived theta_C, theta_cb, theta_ub,
    delta_q (radians) and V_ckm (standard-form 3x3 complex).
    """
    shifts = dict(shifts or {})
    bad = set(shifts) - set(INPUT_RECORDS)
    if bad:
        raise KeyError(f"unknown inputs {bad}")
    src = {"ckm": _data.load_ckm(), "masses": _data.load_masses()}
    ctx = {}
    for name, (grp, rec) in INPUT_RECORDS.items():
        r = src[grp][rec]
        n = float(shifts.get(name, 0.0))
        ctx[name] = r["value"] + n * (r["err_plus"] if n > 0 else abs(r["err_minus"]))
    ctx["theta_C"] = float(np.arcsin(ctx["ckm_s12"]))
    ctx["theta_ub"] = float(np.arcsin(ctx["ckm_s13"]))
    ctx["theta_cb"] = float(np.arcsin(ctx["ckm_s23"]))
    ctx["delta_q"] = float(ctx["ckm_delta_rad"])
    ctx["V_ckm"] = standard_matrix(ctx["theta_C"], ctx["theta_ub"], ctx["theta_cb"], ctx["delta_q"])
    return ctx


# ================================================================ registry

PMNS4 = ("s12sq", "s13sq", "s23sq", "dcp_deg")
CKM_INPUTS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")


@dataclass(frozen=True)
class Baseline:
    """One scoreable relation.

    model(theta: dict, ctx: dict) -> pred dict with at least the `predicts` keys.
    theta holds the fitted parameters (empty if params is empty).
    """
    id: str
    name: str
    relation: str
    predicts: tuple
    inputs: tuple
    k_discrete: int
    model: Callable
    params: dict = field(default_factory=dict)      # name -> (lo, hi)
    nuisance: tuple = ()                             # subset of params that are fitted data
    periodic: tuple = ()                             # params with period (hi - lo)
    catalog: str = ""                                # exp 003 baselines.md entry
    note: str = ""
    n_options: int = 0                               # alternatives behind k_discrete (0 = not stated)

    @property
    def N(self):
        return len(self.predicts)

    @property
    def k_continuous(self):
        return len(self.params)

    @property
    def k(self):
        return self.k_continuous + self.k_discrete


def _pmns_from_U(U, keys=PMNS4):
    o = obs_from_matrix(U)
    return {k: o[k] for k in keys if k in o}


def _qlc_solar(th, c):
    return {"s12sq": np.sin(PI / 4 - c["theta_C"]) ** 2}


def _qlc_full(sign):
    def f(th, c):
        return {"s12sq": np.sin(PI / 4 - c["theta_C"]) ** 2,
                "s23sq": np.sin(PI / 4 + sign * c["theta_cb"]) ** 2}
    return f


def _sc(th, c):
    t13 = np.arcsin(np.sqrt(th["s13sq"]))
    return {"s12sq": np.sin(PI / 4 - t13) ** 2, "s13sq": th["s13sq"]}


def _tbc(th, c):
    o = tbc_king(c["theta_C"])
    return {k: o[k] for k in ("s12sq", "s13sq", "s23sq")}


def _kms(th, c):
    return {"s13sq": c["ckm_s12"] ** 2 / 2}


def _ckm_x(X, order):
    def f(th, c):
        return {k: v for k, v in ckm_dagger_x(c["V_ckm"], X, (th["alpha"], th["beta"]), order).items()
                if k in PMNS4}
    return f


def _lo(name):
    def f(th, c):
        o = lo_pattern(name)
        return {k: o[k] for k in ("s12sq", "s13sq", "s23sq")}
    return f


def _solar(name):
    def f(th, c):
        U_nu = r23(th["th23nu"]) @ r12(LO_TH12[name])
        o = charged_lepton_corrected(U_nu, th["th12e"], th["phi"])
        return {k: o[k] for k in PMNS4}
    return f


def _tm(which):
    def f(th, c):
        R = r23 if which == 1 else r13
        return _pmns_from_U(lo_matrix("TBM") @ R(th["theta"], th["phi"]))
    return f


def _gst(th, c):
    return {"ckm_Vus": np.sqrt(th["m_d"] / th["m_s"]), "m_d": th["m_d"], "m_s": th["m_s"]}


def _lgst(th, c):
    return {"s13sq": lepton_gst(c["m_e"], c["m_mu"])["s13sq"]}


def _koide(th, c):
    return {"m_tau": koide_predict_m3(c["m_e"], c["m_mu"])}


def _km(th, c):
    d_l = (-km_phase(c["V_ckm"])) % TWO_PI
    return _pmns_from_U(km_matrix(th["t1"], th["t2"], th["t3"], d_l))


def _build():
    H = PI / 2
    out = [
        Baseline("A1s", "QLC solar", "th12 + theta_C = pi/4", ("s12sq",), ("ckm_s12",), 0,
                 _qlc_solar, catalog="A1", n_options=1),
        Baseline("A1-", "QLC, th23 = pi/4 - theta_cb", "th12 + theta_C = pi/4, th23 + theta_cb = pi/4",
                 ("s12sq", "s23sq"), ("ckm_s12", "ckm_s23"), 1, _qlc_full(-1), catalog="A1", n_options=2,
                 note="k_d = sign of the atmospheric relation"),
        Baseline("A1+", "QLC, th23 = pi/4 + theta_cb", "th12 + theta_C = pi/4, th23 - theta_cb = pi/4",
                 ("s12sq", "s23sq"), ("ckm_s12", "ckm_s23"), 1, _qlc_full(+1), catalog="A1", n_options=2,
                 note="k_d = sign of the atmospheric relation"),
        Baseline("A2", "Self-complementarity", "th12 + th13 = pi/4", ("s12sq", "s13sq"), (), 1, _sc,
                 params={"s13sq": (0.0, 0.07)}, nuisance=("s13sq",), catalog="A2", n_options=2,
                 note="th13 fitted as nuisance (N and k both +1)"),
        Baseline("A3", "TBC (King)", "s13 = s_C/sqrt2, s12^2 = 1/3, s23^2 = 1/2",
                 ("s12sq", "s13sq", "s23sq"), ("ckm_s12",), 1, _tbc, catalog="A3", n_options=2),
        Baseline("A3k", "KMS th13 only", "s13 = s_C/sqrt2", ("s13sq",), ("ckm_s12",), 1, _kms,
                 catalog="A3", n_options=2),
    ]
    for order, tag, form in (("Vdag_X", "A4a", "U = V_CKM^dag Phi X"),
                             ("Vconj_X", "A4c", "U = V_CKM^* Phi X"),
                             ("V_X_Vdag", "A4b", "U = V_CKM Phi X V_CKM^dag")):
        for X in LO_TH12:
            out.append(Baseline(
                f"{tag}-{X}", f"{form}, X = {X}", f"{form}, Phi = diag(1, e^ia, e^ib), X = {X}",
                PMNS4, CKM_INPUTS, discrete_bits(len(LO_TH12) * 3), _ckm_x(X, order),
                params={"alpha": (0.0, TWO_PI), "beta": (0.0, TWO_PI)}, periodic=("alpha", "beta"),
                catalog="A4", n_options=len(LO_TH12) * 3,
                note="k_d = ceil(log2(5 X x 3 factor orders))" + ("; lead L1" if tag == "A4b" and X == "BM" else "")))
    for X in LO_TH12:
        out.append(Baseline(f"B-{X}", f"LO {X}", f"{X}: th13 = 0, th23 = pi/4, fixed th12",
                            ("s12sq", "s13sq", "s23sq"), (), discrete_bits(len(LO_TH12)), _lo(X), catalog="B",
                            n_options=len(LO_TH12), note="k_d = ceil(log2 5 patterns)"))
    for X in LO_TH12:
        out.append(Baseline(
            f"B1-{X}", f"solar sum rule, {X}", f"U = R12(th12e, phi)^dag R23(th23nu) R12(th12nu = {X})",
            PMNS4, (), discrete_bits(len(LO_TH12)), _solar(X),
            params={"th12e": (0.0, 0.6), "phi": (0.0, TWO_PI), "th23nu": (0.3, 1.3)}, periodic=("phi",),
            catalog="B1", n_options=len(LO_TH12),
            note="full model fit; equivalent to the exact solar sum rule; k_d = ceil(log2 5 patterns)"))
    out += [
        Baseline("B2-TM1", "TM1", "U = U_TBM R23(theta, phi)", PMNS4, (), 1, _tm(1),
                 params={"theta": (0.0, 0.6), "phi": (0.0, TWO_PI)}, periodic=("phi",), catalog="B2", n_options=2),
        Baseline("B2-TM2", "TM2", "U = U_TBM R13(theta, phi)", PMNS4, (), 1, _tm(2),
                 params={"theta": (0.0, 0.6), "phi": (0.0, TWO_PI)}, periodic=("phi",), catalog="B2", n_options=2),
        Baseline("C1", "GST", "|V_us| = sqrt(m_d/m_s)", ("ckm_Vus", "m_d", "m_s"), (), 0, _gst,
                 params={"m_d": (0.003, 0.007), "m_s": (0.07, 0.12)}, nuisance=("m_d", "m_s"), catalog="C1", n_options=1,
                 note="m_d, m_s fitted as nuisance; input errors dominate"),
        Baseline("C2", "leptonic GST (TBM)", "th12e = asin sqrt(m_e/m_mu) on TBM, th13 only",
                 ("s13sq",), ("m_e", "m_mu"), 1, _lgst, catalog="C2", n_options=2),
        Baseline("C3", "Koide m_tau", "Q(m_e, m_mu, m_tau) = 2/3", ("m_tau",), ("m_e", "m_mu"), 1, _koide,
                 catalog="C3", n_options=2, note="strict k_d = 1 (2/3 chosen after data)"),
        Baseline("E1", "KM-phase sum rule", "delta_KM^l = -delta_KM^q (KM parametrization)",
                 PMNS4, CKM_INPUTS, 1, _km,
                 params={"t1": (0.0, H), "t2": (0.0, H), "t3": (0.0, H)}, catalog="E1", n_options=2,
                 note="PMNS in KM form with fixed phase; 3 angles fitted"),
    ]
    return {b.id: b for b in out}


#: id -> Baseline. Not scored (no measured target or need GUT-scale running / model fit):
#: C3' Koide-type neutrino (predicts unmeasured m1), C4 SU(5)/GJ (GUT-scale Yukawas),
#: C5 Haba-Yamada, D (covered by B/B2 patterns), E2 zero 1-3 hypothesis, F anarchy (null).
BASELINES = _build()

#: The lead L1 from CANDIDATES.md.
L1_ID = "A4b-BM"

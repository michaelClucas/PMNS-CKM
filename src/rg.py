"""Renormalization-group running of quark/lepton Yukawas and the Weinberg operator.

Models: SM, and MSSM above a single threshold M_SUSY (SM below), tree-level matching.
Loop order: gauge couplings two-loop (incl. one-loop Yukawa traces in the two-loop term),
Yukawas, Higgs quartic and kappa one-loop.

Conventions (all couplings MSbar; DRbar/MSbar conversion in the MSSM ignored)
---------------------------------------------------------------------------
-L = Qbar Y_u u_R Htilde + Qbar Y_d d_R H + Lbar Y_e e_R H
     - (1/4) kappa_{gf} (L_g^c eps H)(L_f eps H) + h.c.
The left index of every Yukawa is the doublet index, so Y Y^dag is the left-handed
combination. Masses: M_f = Y_f v / sqrt2 (SM), m_nu = kappa v^2 / 4.
Diagonalisation: Y_f Y_f^dag = U_f diag(y^2) U_f^dag; V_CKM = U_u^dag U_d.
kappa = U_nu^* diag(m) U_nu^dag (up to v^2/4); U_PMNS = U_e^dag U_nu.
g1 is GUT-normalised, g1^2 = (5/3) g'^2. lam is defined by V = lam (H^dag H)^2,
so m_H^2 = 2 lam v^2 at tree level (REAP's lambda = 4 lam).
t = ln(mu / GeV).

One-loop RGEs, 16 pi^2 dX/dt (SM: Machacek-Vaughn; kappa: Antusch et al.
hep-ph/0108005 and hep-ph/0305273):
  SM   Y_u: [ 3/2 (H_u - H_d) + T - (17/20 g1^2 + 9/4 g2^2 + 8 g3^2) ] Y_u
       Y_d: [ 3/2 (H_d - H_u) + T - ( 1/4 g1^2 + 9/4 g2^2 + 8 g3^2) ] Y_d
       Y_e: [ 3/2 H_e         + T - ( 9/4 g1^2 + 9/4 g2^2)           ] Y_e
       T = Tr(3 H_u + 3 H_d + H_e), H_f = Y_f Y_f^dag
       kappa: -3/2 (kappa H_e + H_e^T kappa) + (2 T - 3 g2^2 + 4 lam) kappa
       lam: 24 lam^2 + 4 lam T - 2 Tr(3 H_u^2 + 3 H_d^2 + H_e^2)
            - 3 lam (3 g2^2 + 3/5 g1^2) + 3/8 (2 g2^4 + (g2^2 + 3/5 g1^2)^2)
  MSSM Y_u: [ 3 H_u + H_d + 3 Tr H_u - (13/15 g1^2 + 3 g2^2 + 16/3 g3^2) ] Y_u
       Y_d: [ 3 H_d + H_u + Tr(3 H_d + H_e) - (7/15 g1^2 + 3 g2^2 + 16/3 g3^2) ] Y_d
       Y_e: [ 3 H_e + Tr(3 H_d + H_e) - (9/5 g1^2 + 3 g2^2) ] Y_e
       kappa: (kappa H_e + H_e^T kappa) + (6 Tr H_u - 6/5 g1^2 - 6 g2^2) kappa
       lam is frozen above M_SUSY (it is not an MSSM parameter; freezing it makes
       up-then-down runs exactly invertible).
Gauge: 16 pi^2 dg_i/dt = b_i g_i^3 + g_i^3/(16 pi^2) [sum_j b_ij g_j^2 - sum_f c_if Tr H_f].

Matching at M_SUSY (tree level, no tan-beta-enhanced threshold corrections):
  Y_u^MSSM = Y_u^SM / sin(beta), Y_{d,e}^MSSM = Y_{d,e}^SM / cos(beta),
  kappa^MSSM = kappa^SM / sin(beta)^2, gauge couplings continuous.
Observables are always reported in SM normalisation (y_u sin(beta), y_d cos(beta), ...),
as in arXiv:2510.01312 Tables 5-8.

Boundary values at M_Z (build_mz_state)
--------------------------------------
g1, g2, g3, lam, v(M_Z), v(M_t): data/sm_inputs.json block 'ahs2025_msbar_MZ'.
Quark masses: data/masses.json via src.data.load_masses(), run with four-loop QCD
(beta and gamma functions below) and two-loop decoupling at mu = m_q(m_q):
  u, d, s from 2 GeV (nf = 4); c from m_c(m_c) (nf = 4); b from m_b(m_b) (nf = 5),
  all to M_Z with nf = 5. y_q = sqrt2 m_q(M_Z) / v(M_Z).
Top: pole -> MSbar with three-loop QCD, y_t(M_t) = sqrt2 m_t(m_t) / v(M_t), then y_t(M_Z)
is solved for with the SM RGEs of this module (secant on the M_Z value).
Charged leptons: one-loop QED pole -> MSbar at M_Z, m(mu) = M [1 - alpha/pi (1 + 3/2 ln(mu/M))]
with alpha = alpha(0) (good to ~0.2% vs the two-loop numbers of arXiv:2510.01312).
CKM: PDG standard parametrisation from data/ckm.json (treated as the M_Z value).
PMNS: NuFIT from data/pmns.json, masses from m_lightest + splittings, Majorana phases
(phi1, phi2) enter as kappa = (4/v^2) U^* diag(m1 e^{i phi1}, m2 e^{i phi2}, m3) U^dag.
"""
from __future__ import annotations

import copy
import functools
import json
import pathlib

import numpy as np
from scipy.integrate import solve_ivp

from src import data as _data
from src.mixing import extract_angles, jarlskog, standard_matrix

__all__ = [
    "load_sm_inputs", "qcd_alpha_s", "quark_masses_mz", "derived_yukawas_mz", "build_mz_state",
    "run", "run_sm", "run_mssm", "observables", "ckm_at", "pmns_at", "neutrino_masses",
]

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
LOOP = 16.0 * np.pi ** 2
ZETA3, ZETA4, ZETA5 = 1.2020569031595942, np.pi ** 4 / 90, 1.0369277551433699
EV = 1e-9  # GeV per eV


@functools.lru_cache(maxsize=None)
def load_sm_inputs() -> dict:
    with open(DATA / "sm_inputs.json") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ QCD running

def _beta(nf):
    """beta_i for a = alpha_s/pi, da/dln(mu^2) = -sum beta_i a^(i+2) (four loops)."""
    return (
        (11 - 2 * nf / 3) / 4,
        (102 - 38 * nf / 3) / 16,
        (2857 / 2 - 5033 * nf / 18 + 325 * nf ** 2 / 54) / 64,
        ((149753 / 6 + 3564 * ZETA3) - (1078361 / 162 + 6508 * ZETA3 / 27) * nf
         + (50065 / 162 + 6472 * ZETA3 / 81) * nf ** 2 + 1093 / 729 * nf ** 3) / 256,
    )


def _gamma(nf):
    """gamma_i for dln m/dln(mu^2) = -sum gamma_i a^(i+1) (four loops)."""
    return (
        1.0,
        (202 / 3 - 20 * nf / 9) / 16,
        (1249 + (-2216 / 27 - 160 * ZETA3 / 3) * nf - 140 * nf ** 2 / 81) / 64,
        (4603055 / 162 + 135680 * ZETA3 / 27 - 8800 * ZETA5
         + (-91723 / 27 - 34192 * ZETA3 / 9 + 880 * ZETA4 + 18400 * ZETA5 / 9) * nf
         + (5242 / 243 + 800 * ZETA3 / 9 - 160 * ZETA4 / 3) * nf ** 2
         + (-332 / 243 + 64 * ZETA3 / 27) * nf ** 3) / 256,
    )


def _qcd_run(a0, m0, mu0, mu1, nf):
    """Run (a = alpha_s/pi, m) from mu0 to mu1 at fixed nf. m0 may be an array."""
    b, g = _beta(nf), _gamma(nf)
    m0 = np.atleast_1d(np.asarray(m0, dtype=float))

    def f(L, y):  # L = ln mu^2
        a = y[0]
        da = -sum(b[i] * a ** (i + 2) for i in range(4))
        gm = -sum(g[i] * a ** (i + 1) for i in range(4))
        return np.concatenate(([da], gm * y[1:]))

    sol = solve_ivp(f, (2 * np.log(mu0), 2 * np.log(mu1)), np.concatenate(([a0], m0)),
                    method="DOP853", rtol=1e-11, atol=1e-14)
    return sol.y[0, -1], sol.y[1:, -1]


# two-loop decoupling at mu = m_h(m_h): a^(nf-1) = a^(nf) (1 + 11/72 a^2), m^(nf-1) = m^(nf) (1 + 89/432 a^2)
def _down(a, m):
    return a * (1 + 11 / 72 * a ** 2), m * (1 + 89 / 432 * a ** 2)


def _up(a, m):
    return a / (1 + 11 / 72 * a ** 2), m / (1 + 89 / 432 * a ** 2)


def qcd_alpha_s(mu, nf_target=None):
    """alpha_s(mu) (MSbar) from PDG alpha_s(M_Z) with thresholds at m_c(m_c), m_b(m_b). mu in [1.3, 1000] GeV."""
    sm = load_sm_inputs()["pdg2026"]
    ms = _data.load_masses()
    mz, a = sm["M_Z"]["value"], sm["alpha_s_MZ"]["value"] / np.pi
    mb, mc = ms["m_b"]["value"], ms["m_c"]["value"]
    if mu >= mb:
        return np.pi * _qcd_run(a, 1.0, mz, mu, 5)[0]
    a, _ = _qcd_run(a, 1.0, mz, mb, 5)
    a, _ = _down(a, 1.0)
    if mu >= mc:
        return np.pi * _qcd_run(a, 1.0, mb, mu, 4)[0]
    a, _ = _qcd_run(a, 1.0, mb, mc, 4)
    a, _ = _down(a, 1.0)
    return np.pi * _qcd_run(a, 1.0, mc, mu, 3)[0]


# ------------------------------------------------------------------ input overrides

def _freeze_masses(masses):
    """masses override -> hashable ((name, value), ...). Accepts floats or records with 'value'."""
    if not masses:
        return ()
    base = _data.load_masses()
    out = []
    for k, v in dict(masses).items():
        if isinstance(v, dict) and "value" not in v:
            continue  # e.g. 'meta', 'neutrino_splittings' in a full load_masses() copy
        if k not in base or not isinstance(base[k], dict) or "value" not in base[k]:
            raise KeyError(f"unknown mass record {k!r}")
        out.append((k, float(v["value"] if isinstance(v, dict) else v)))
    return tuple(sorted(out))


def _freeze_sm(sm_inputs):
    """sm_inputs override {block: {name: value or record}} -> hashable ((block, name, value), ...)."""
    if not sm_inputs:
        return ()
    base = load_sm_inputs()
    out = []
    for blk, entries in dict(sm_inputs).items():
        if blk not in base or not isinstance(entries, dict):
            raise KeyError(f"unknown sm_inputs block {blk!r}")
        for k, v in entries.items():
            if isinstance(v, dict) and "value" not in v:
                continue
            if k not in base[blk]:
                raise KeyError(f"unknown sm_inputs entry {blk}/{k}")
            out.append((blk, k, float(v["value"] if isinstance(v, dict) else v)))
    return tuple(sorted(out))


def _thaw_sm(fs):
    if not fs:
        return None
    out = {}
    for blk, k, v in fs:
        out.setdefault(blk, {})[k] = v
    return out


def _merged_masses(frozen=()):
    ms = _data.load_masses()
    if not frozen:
        return ms
    ms = dict(ms)
    for k, v in frozen:
        ms[k] = dict(ms[k], value=v)
    return ms


def _merged_sm(frozen=()):
    sm = load_sm_inputs()
    if not frozen:
        return sm
    sm = {blk: dict(e) if isinstance(e, dict) else e for blk, e in sm.items()}
    for blk, k, v in frozen:
        old = sm[blk][k]
        sm[blk][k] = dict(old, value=v) if isinstance(old, dict) else v
    return sm


def _quark_masses_impl(ms, sm_all):
    sm = sm_all["pdg2026"]
    mz, aZ = sm["M_Z"]["value"], sm["alpha_s_MZ"]["value"] / np.pi
    mb, mc = ms["m_b"]["value"], ms["m_c"]["value"]
    # alpha_s at m_b (nf=5), then nf=4 below
    a_b5, _ = _qcd_run(aZ, 1.0, mz, mb, 5)
    a_b4, _ = _down(a_b5, 1.0)
    a_2, _ = _qcd_run(a_b4, 1.0, mb, 2.0, 4)
    a_c4, _ = _qcd_run(a_b4, 1.0, mb, mc, 4)
    # light quarks: 2 GeV -> m_b (nf=4), decouple, -> M_Z (nf=5)
    light = np.array([ms["m_u"]["value"], ms["m_d"]["value"], ms["m_s"]["value"]])
    _, l_b = _qcd_run(a_2, light, 2.0, mb, 4)
    _, l_b = _up(a_b4, l_b)
    _, l_z = _qcd_run(a_b5, l_b, mb, mz, 5)
    # charm: m_c(m_c) nf=4 -> m_b, -> M_Z
    _, c_b = _qcd_run(a_c4, [mc], mc, mb, 4)
    _, c_b = _up(a_b4, c_b)
    _, c_z = _qcd_run(a_b5, c_b, mb, mz, 5)
    _, b_z = _qcd_run(a_b5, [mb], mb, mz, 5)
    # top: pole -> MSbar, three-loop QCD (nf_light = 5), alpha_s^(6)(m_t) ~ alpha_s^(5)(m_t) at this order
    Mt = ms["m_t_direct"]["value"]
    mt = Mt
    for _ in range(30):
        a_t, _ = _qcd_run(aZ, 1.0, mz, mt, 5)
        nl = 5
        r = 1 + 4 / 3 * a_t + (13.4434 - 1.0414 * nl) * a_t ** 2 \
            + (190.595 - 26.655 * nl + 0.6527 * nl ** 2) * a_t ** 3
        mt = Mt / r
    return {"m_u": l_z[0], "m_d": l_z[1], "m_s": l_z[2], "m_c": c_z[0], "m_b": b_z[0],
            "m_t_mt": mt, "M_t": Mt, "alpha_s_2GeV": np.pi * a_2}


@functools.lru_cache(maxsize=256)
def _quark_masses_cached(fm, fs):
    return _quark_masses_impl(_merged_masses(fm), _merged_sm(fs))


def quark_masses_mz(masses=None, sm_inputs=None) -> dict:
    """MSbar m_q(M_Z) in GeV for u, d, s, c, b (nf = 5, QCD only, four loops), and m_t(m_t) from the pole mass.

    masses: optional overrides {record name: value or record} on top of data/masses.json
        (e.g. {"m_s": 0.095}); sm_inputs: optional {block: {name: value}} on top of
        data/sm_inputs.json (e.g. {"pdg2026": {"alpha_s_MZ": 0.1185}}). Results are cached
        per override set; a fresh dict is returned on every call.
    """
    return dict(_quark_masses_cached(_freeze_masses(masses), _freeze_sm(sm_inputs)))


# Backward compatibility: quark_masses_mz used to be an lru_cache'd function, and exp 010
# calls quark_masses_mz.__wrapped__() after monkeypatching the loaders.
quark_masses_mz.__wrapped__ = lambda: _quark_masses_impl(_data.load_masses(), load_sm_inputs())
quark_masses_mz.cache_clear = _quark_masses_cached.cache_clear
quark_masses_mz.cache_info = _quark_masses_cached.cache_info


def _lepton_msbar(M, mu, sm=None):
    a0 = 1.0 / (sm or load_sm_inputs())["pdg2026"]["alpha_0_inv"]["value"]
    return M * (1 - a0 / np.pi * (1 + 1.5 * np.log(mu / M)))


# ------------------------------------------------------------------ state packing

N_REAL = 3 + 3 * 18 + 1 + 18  # g, Yu, Yd, Ye, lam, kappa


def _pack(s):
    c = lambda M: np.concatenate((M.real.ravel(), M.imag.ravel()))
    return np.concatenate((s["g"], c(s["Yu"]), c(s["Yd"]), c(s["Ye"]), [s["lam"]],
                           c(s["kappa"] / s["kscale"])))


def _unpack(y, kscale):
    m = lambda a: (a[:9] + 1j * a[9:18]).reshape(3, 3)
    return {"g": y[:3].copy(), "Yu": m(y[3:21]), "Yd": m(y[21:39]), "Ye": m(y[39:57]),
            "lam": y[57], "kappa": m(y[58:76]) * kscale, "kscale": kscale}


B1_SM = np.array([41 / 10, -19 / 6, -7.0])
B2_SM = np.array([[199 / 50, 27 / 10, 44 / 5], [9 / 10, 35 / 6, 12], [11 / 10, 9 / 2, -26]])
C_SM = np.array([[17 / 10, 1 / 2, 3 / 2], [3 / 2, 3 / 2, 1 / 2], [2, 2, 0]])
B1_MSSM = np.array([33 / 5, 1.0, -3.0])
B2_MSSM = np.array([[199 / 25, 27 / 5, 88 / 5], [9 / 5, 25, 24], [11 / 5, 9, 14]])
C_MSSM = np.array([[26 / 5, 14 / 5, 18 / 5], [6, 6, 2], [4, 4, 0]])


def _rhs(model, kscale):
    def f(t, y):
        s = _unpack(y, 1.0)  # kappa kept in scaled units inside the integrator
        g = s["g"]
        g2 = g ** 2
        Yu, Yd, Ye, k, lam = s["Yu"], s["Yd"], s["Ye"], s["kappa"], s["lam"]
        Hu, Hd, He = Yu @ Yu.conj().T, Yd @ Yd.conj().T, Ye @ Ye.conj().T
        tu, td, te = np.trace(Hu).real, np.trace(Hd).real, np.trace(He).real
        if model == "SM":
            b1, b2, cc = B1_SM, B2_SM, C_SM
            T = 3 * tu + 3 * td + te
            dYu = (1.5 * (Hu - Hd) + (T - (17 / 20 * g2[0] + 9 / 4 * g2[1] + 8 * g2[2])) * np.eye(3)) @ Yu
            dYd = (1.5 * (Hd - Hu) + (T - (1 / 4 * g2[0] + 9 / 4 * g2[1] + 8 * g2[2])) * np.eye(3)) @ Yd
            dYe = (1.5 * He + (T - (9 / 4 * g2[0] + 9 / 4 * g2[1])) * np.eye(3)) @ Ye
            dk = -1.5 * (k @ He + He.T @ k) + (2 * T - 3 * g2[1] + 4 * lam) * k
            gp2 = 0.6 * g2[0]
            dlam = (24 * lam ** 2 + 4 * lam * T
                    - 2 * np.trace(3 * Hu @ Hu + 3 * Hd @ Hd + He @ He).real
                    - 3 * lam * (3 * g2[1] + gp2) + 3 / 8 * (2 * g2[1] ** 2 + (g2[1] + gp2) ** 2))
        else:
            b1, b2, cc = B1_MSSM, B2_MSSM, C_MSSM
            Td = 3 * td + te
            dYu = (3 * Hu + Hd + (3 * tu - (13 / 15 * g2[0] + 3 * g2[1] + 16 / 3 * g2[2])) * np.eye(3)) @ Yu
            dYd = (3 * Hd + Hu + (Td - (7 / 15 * g2[0] + 3 * g2[1] + 16 / 3 * g2[2])) * np.eye(3)) @ Yd
            dYe = (3 * He + (Td - (9 / 5 * g2[0] + 3 * g2[1])) * np.eye(3)) @ Ye
            dk = (k @ He + He.T @ k) + (6 * tu - 6 / 5 * g2[0] - 6 * g2[1]) * k
            dlam = 0.0
        tr = np.array([tu, td, te])
        dg = g ** 3 * (b1 + (b2 @ g2 - cc @ tr) / LOOP)
        out = {"g": dg / LOOP, "Yu": dYu / LOOP, "Yd": dYd / LOOP, "Ye": dYe / LOOP,
               "lam": dlam / LOOP, "kappa": dk / LOOP, "kscale": 1.0}
        return _pack(out)
    return f


def _integrate(state, t0, t1, model, rtol):
    meta = {k: v for k, v in state.items() if k not in ("g", "Yu", "Yd", "Ye", "lam", "kappa", "kscale")}
    if t0 == t1:
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    else:
        ks = state["kscale"]
        sol = solve_ivp(_rhs(model, ks), (t0, t1), _pack(state), method="DOP853", rtol=rtol, atol=1e-22)
        if not sol.success:
            raise RuntimeError(sol.message)
        out = {**meta, **_unpack(sol.y[:, -1], ks)}
    out["mu"] = float(np.exp(t1))
    return out


def _match(state, tan_beta, to_mssm):
    sb, cb = np.sin(np.arctan(tan_beta)), np.cos(np.arctan(tan_beta))
    s = dict(state)
    if to_mssm:
        s["Yu"], s["Yd"], s["Ye"], s["kappa"] = state["Yu"] / sb, state["Yd"] / cb, state["Ye"] / cb, state["kappa"] / sb ** 2
    else:
        s["Yu"], s["Yd"], s["Ye"], s["kappa"] = state["Yu"] * sb, state["Yd"] * cb, state["Ye"] * cb, state["kappa"] * sb ** 2
    return s


def run(mu_from, mu_to, state, model="SM", tan_beta=None, m_susy=3000.0, rtol=1e-10):
    """Evolve `state` from mu_from to mu_to (GeV).

    model="SM": SM RGEs throughout. model="MSSM": SM below m_susy, MSSM above, tree-level
    matching at m_susy with tan_beta. The state carries MSSM-normalised Yukawas/kappa when
    it sits above m_susy (state["normalisation"] says which). Returns a new state dict.
    """
    s = dict(state)
    s.setdefault("normalisation", "SM")
    if model == "SM":
        out = _integrate(s, np.log(mu_from), np.log(mu_to), "SM", rtol)
        out["normalisation"] = "SM"
        return out
    if tan_beta is None:
        raise ValueError("MSSM running needs tan_beta")
    t0, t1, ts = np.log(mu_from), np.log(mu_to), np.log(m_susy)
    above0 = mu_from > m_susy or (mu_from == m_susy and s["normalisation"] == "MSSM")
    expect = "MSSM" if above0 else "SM"
    if s["normalisation"] != expect:
        raise ValueError(f"state normalisation {s['normalisation']} inconsistent with mu_from={mu_from}")
    if t1 >= t0:
        if t1 <= ts:
            out = _integrate(s, t0, t1, "SM", rtol); out["normalisation"] = "SM"; return out
        if not above0:
            s = _integrate(s, t0, ts, "SM", rtol)
            s = _match(s, tan_beta, True)
            t0 = ts
        out = _integrate(s, t0, t1, "MSSM", rtol); out["normalisation"] = "MSSM"; return out
    if t1 >= ts and above0:
        out = _integrate(s, t0, t1, "MSSM", rtol); out["normalisation"] = "MSSM"; return out
    if above0:
        s = _integrate(s, t0, ts, "MSSM", rtol)
        s = _match(s, tan_beta, False)
        t0 = ts
    out = _integrate(s, t0, t1, "SM", rtol); out["normalisation"] = "SM"; return out


def run_sm(mu_from, mu_to, inputs, rtol=1e-10):
    """SM running of a state dict (g, Yu, Yd, Ye, lam, kappa, kscale) from mu_from to mu_to (GeV)."""
    return run(mu_from, mu_to, inputs, model="SM", rtol=rtol)


def run_mssm(mu_from, mu_to, inputs, tan_beta, m_susy=3000.0, rtol=1e-10):
    return run(mu_from, mu_to, inputs, model="MSSM", tan_beta=tan_beta, m_susy=m_susy, rtol=rtol)


# ------------------------------------------------------------------ boundary values at M_Z

def neutrino_masses(ordering, m_lightest, analysis=_data.DEFAULT_ANALYSIS):
    """(m1, m2, m3) in eV from NuFIT splittings."""
    sp = _data.load_masses()["neutrino_splittings"][analysis][ordering]
    dm21 = sp["dm21sq"]["value"]
    if ordering == "NO":
        m1 = m_lightest
        return np.array([m1, np.sqrt(m1 ** 2 + dm21), np.sqrt(m1 ** 2 + sp["dm31sq"]["value"])])
    m3 = m_lightest
    m2 = np.sqrt(m3 ** 2 - sp["dm32sq"]["value"])
    return np.array([np.sqrt(m2 ** 2 - dm21), m2, m3])


def _pmns_input(ordering, analysis):
    p = _data.load_pmns(ordering, analysis)
    d = np.pi / 180
    return (p["theta12_deg"]["value"] * d, p["theta13_deg"]["value"] * d,
            p["theta23_deg"]["value"] * d, p["dcp_deg"]["value"] * d)


def _ckm_input():
    c = _data.load_ckm()
    return (np.arcsin(c["s12"]["value"]), np.arcsin(c["s13"]["value"]),
            np.arcsin(c["s23"]["value"]), c["delta_rad"]["value"])


def build_mz_state(ordering="NO", m_lightest=0.0, majorana=(0.0, 0.0), analysis=_data.DEFAULT_ANALYSIS,
                   yukawas=None, ckm=None, pmns=None, masses=None, sm_inputs=None):
    """SM state at M_Z.

    yukawas: optional dict y_u..y_tau overriding the values derived from data/ (used for
    validation against arXiv:2510.01312 Eq. (4)). ckm, pmns: optional (t12, t13, t23, delta).
    masses, sm_inputs: input overrides as in quark_masses_mz. masses only matters when
    yukawas is None (derived Yukawas); sm_inputs also sets g, lambda and v.
    """
    sm = _merged_sm(_freeze_sm(sm_inputs)) if sm_inputs else load_sm_inputs()
    ahs = sm["ahs2025_msbar_MZ"]
    mz = sm["pdg2026"]["M_Z"]["value"]
    v = ahs["v_MZ"]["value"]
    g = np.array([ahs["g1"]["value"], ahs["g2"]["value"], ahs["g3"]["value"]])
    lam = ahs["lambda_reap"]["value"] / 4
    if yukawas is None:
        yukawas = derived_yukawas_mz(masses=masses, sm_inputs=sm_inputs)
    V = standard_matrix(*(ckm if ckm is not None else _ckm_input()))
    U = standard_matrix(*(pmns if pmns is not None else _pmns_input(ordering, analysis)))
    Yu = np.diag([yukawas["y_u"], yukawas["y_c"], yukawas["y_t"]]).astype(complex)
    Yd = V @ np.diag([yukawas["y_d"], yukawas["y_s"], yukawas["y_b"]])
    Ye = np.diag([yukawas["y_e"], yukawas["y_mu"], yukawas["y_tau"]]).astype(complex)
    m = neutrino_masses(ordering, m_lightest, analysis) * EV
    D = np.diag(m * np.exp(1j * np.array([majorana[0], majorana[1], 0.0])))
    kappa = 4 / v ** 2 * U.conj() @ D @ U.conj().T
    kscale = 4 / v ** 2 * max(m.max(), 1e-3 * EV)
    return {"g": g, "Yu": Yu, "Yd": Yd, "Ye": Ye, "lam": lam, "kappa": kappa, "kscale": kscale,
            "mu": mz, "normalisation": "SM", "ordering": ordering}


def _derived_yukawas_impl(q, sm, lep, sm_override=None):
    ahs = sm["ahs2025_msbar_MZ"]
    mz = sm["pdg2026"]["M_Z"]["value"]
    v, vt = ahs["v_MZ"]["value"], ahs["v_Mt"]["value"]
    r2 = np.sqrt(2)
    y = {"y_u": r2 * q["m_u"] / v, "y_d": r2 * q["m_d"] / v, "y_s": r2 * q["m_s"] / v,
         "y_c": r2 * q["m_c"] / v, "y_b": r2 * q["m_b"] / v,
         "y_e": r2 * _lepton_msbar(lep["m_e"]["value"], mz, sm) / v,
         "y_mu": r2 * _lepton_msbar(lep["m_mu"]["value"], mz, sm) / v,
         "y_tau": r2 * _lepton_msbar(lep["m_tau"]["value"], mz, sm) / v}
    # solve y_t(M_Z) such that SM running gives y_t(M_t) = sqrt2 m_t(m_t) / v(M_t)
    target = r2 * q["m_t_mt"] / vt
    Mt = q["M_t"]

    def yt_at_mt(yt):
        yy = dict(y, y_t=yt)
        s = build_mz_state(yukawas=yy, sm_inputs=sm_override)
        s = run_sm(mz, Mt, s, rtol=1e-9)
        return np.linalg.svd(s["Yu"], compute_uv=False)[0]

    a, b = 0.95, 0.98
    fa, fb = yt_at_mt(a) - target, yt_at_mt(b) - target
    for _ in range(20):
        c = b - fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b, fb = c, yt_at_mt(c) - target
        if abs(fb) < 1e-12:
            break
    y["y_t"] = b
    return tuple(sorted(y.items()))


@functools.lru_cache(maxsize=None)
def _derived_yukawas_tuple():
    # no-argument form kept: exp 010 calls _derived_yukawas_tuple.__wrapped__() after
    # monkeypatching rg.quark_masses_mz and rg.load_sm_inputs
    return _derived_yukawas_impl(quark_masses_mz(), load_sm_inputs(), _data.load_masses())


@functools.lru_cache(maxsize=64)
def _derived_yukawas_override(fm, fs):
    return _derived_yukawas_impl(dict(_quark_masses_cached(fm, fs)), _merged_sm(fs), _merged_masses(fm),
                                 sm_override=_thaw_sm(fs))


def derived_yukawas_mz(masses=None, sm_inputs=None) -> dict:
    """SM MSbar Yukawas at M_Z derived from data/masses.json (see module docstring).

    masses, sm_inputs: optional input overrides, as in quark_masses_mz. Cached per override set.
    """
    fm, fs = _freeze_masses(masses), _freeze_sm(sm_inputs)
    if not fm and not fs:
        return dict(_derived_yukawas_tuple())
    return dict(_derived_yukawas_override(fm, fs))


def ahs2025_yukawas_mz() -> dict:
    """Not measured data: the M_Z Yukawas of arXiv:2510.01312 Eq. (4) used only for validation."""
    return dict(AHS_EQ4_YUKAWAS)


# Literature values (arXiv:2510.01312v2 Eq. (4) and Eq. (6)), for validation only.
AHS_EQ4_YUKAWAS = {"y_u": 7.04e-6, "y_d": 1.54e-5, "y_s": 3.06e-4, "y_c": 3.56e-3, "y_b": 1.630e-2,
                   "y_t": 0.967, "y_e": 2.77713e-6, "y_mu": 5.85042e-4, "y_tau": 0.99378e-2}
AHS_EQ6_CKM = (0.2270, 3.70e-3, 4.194e-2, 1.139)  # (t12, t13, t23, delta)


# ------------------------------------------------------------------ observables

def _left_basis(Y):
    w, U = np.linalg.eigh(Y @ Y.conj().T)
    return np.sqrt(np.clip(w, 0, None)), U


def observables(state, tan_beta=None) -> dict:
    """Mixing angles, delta, J and SM-normalised Yukawa eigenvalues / neutrino masses (eV)."""
    sb = cb = 1.0
    if state.get("normalisation", "SM") == "MSSM":
        sb, cb = np.sin(np.arctan(tan_beta)), np.cos(np.arctan(tan_beta))
    yu, Uu = _left_basis(state["Yu"])
    yd, Ud = _left_basis(state["Yd"])
    ye, Ue = _left_basis(state["Ye"])
    V = Uu.conj().T @ Ud
    t12, t13, t23, d = extract_angles(V)
    out = {"ckm": {"theta12": float(t12), "theta13": float(t13), "theta23": float(t23),
                   "delta": float(d), "J": float(jarlskog(V))},
           "y": {"y_u": yu[0] * sb, "y_c": yu[1] * sb, "y_t": yu[2] * sb,
                 "y_d": yd[0] * cb, "y_s": yd[1] * cb, "y_b": yd[2] * cb,
                 "y_e": ye[0] * cb, "y_mu": ye[1] * cb, "y_tau": ye[2] * cb},
           "g": state["g"].copy()}
    v = load_sm_inputs()["ahs2025_msbar_MZ"]["v_MZ"]["value"]
    k = Ue.T @ state["kappa"] @ Ue  # charged-lepton mass basis
    w, Un = np.linalg.eigh(k.conj().T @ k)
    m = np.sqrt(np.clip(w, 0, None)) * v ** 2 / 4 * sb ** 2 / EV
    if state.get("ordering", "NO") == "IO":
        Un, m = Un[:, [1, 2, 0]], m[[1, 2, 0]]
    t12, t13, t23, d = extract_angles(Un)
    out["pmns"] = {"theta12": float(t12), "theta13": float(t13), "theta23": float(t23),
                   "delta": float(d), "J": float(jarlskog(Un)), "m": m}
    return out


def _run_to(mu, state, model, tan_beta, m_susy):
    mz = state["mu"]
    if model == "SM":
        return run_sm(mz, mu, state)
    return run_mssm(mz, mu, state, tan_beta, m_susy)


def _freeze_seq(x):
    return None if x is None else tuple(float(v) for v in np.asarray(x, dtype=float).ravel())


def _freeze_dict(d):
    return None if d is None else tuple(sorted((k, float(v)) for k, v in dict(d).items()))


@functools.lru_cache(maxsize=1024)
def _ckm_at_cached(mu, model, tan_beta, m_susy, fy, fc, fm, fs):
    s = build_mz_state(yukawas=None if fy is None else dict(fy), ckm=fc,
                       masses=dict(fm) or None, sm_inputs=_thaw_sm(fs))
    o = observables(_run_to(mu, s, model, tan_beta, m_susy), tan_beta)
    y = o["y"]
    r = {"mu_over_mc": y["y_u"] / y["y_c"], "mc_over_mt": y["y_c"] / y["y_t"],
         "md_over_ms": y["y_d"] / y["y_s"], "ms_over_mb": y["y_s"] / y["y_b"],
         "me_over_mmu": y["y_e"] / y["y_mu"], "mmu_over_mtau": y["y_mu"] / y["y_tau"],
         "mb_over_mtau": y["y_b"] / y["y_tau"]}
    return {**o["ckm"], "yukawas": y, "ratios": r, "g": o["g"]}


def ckm_at(mu, model="SM", tan_beta=None, m_susy=3000.0, yukawas=None, ckm=None,
           masses=None, sm_inputs=None, cache=True) -> dict:
    """CKM angles/delta/J and quark mass ratios at scale mu (GeV), running from M_Z.

    yukawas, ckm: as in build_mz_state. masses, sm_inputs: input overrides as in
    quark_masses_mz (masses is ignored when yukawas is given; sm_inputs still sets g,
    lambda, v). Results are memoised per argument set (cache=False bypasses the memo); the
    returned dict is a fresh deep copy. ckm_at.cache_clear() drops the memo, which is
    needed only if you monkeypatch the loaders in-process.
    """
    key = (float(mu), model, None if tan_beta is None else float(tan_beta), float(m_susy),
           _freeze_dict(yukawas), _freeze_seq(ckm), _freeze_masses(masses), _freeze_sm(sm_inputs))
    r = _ckm_at_cached(*key) if cache else _ckm_at_cached.__wrapped__(*key)
    return copy.deepcopy(r)


ckm_at.cache_clear = _ckm_at_cached.cache_clear
ckm_at.cache_info = _ckm_at_cached.cache_info


def pmns_at(mu, ordering="NO", m_lightest=0.0, majorana=(0.0, 0.0), model="SM", tan_beta=None,
            m_susy=3000.0, analysis=_data.DEFAULT_ANALYSIS) -> dict:
    """PMNS angles/delta and neutrino masses (eV, m_nu = kappa v(M_Z)^2/4) at scale mu."""
    s = build_mz_state(ordering, m_lightest, majorana, analysis)
    o = observables(_run_to(mu, s, model, tan_beta, m_susy), tan_beta)
    p = o["pmns"]
    m = p["m"]
    return {**p, "m21_ratio": m[0] / m[1] if m[1] else np.nan,
            "dm21_over_dm3l": (m[1] ** 2 - m[0] ** 2) / abs(m[2] ** 2 - (m[0] ** 2 if ordering == "NO" else m[1] ** 2))}
